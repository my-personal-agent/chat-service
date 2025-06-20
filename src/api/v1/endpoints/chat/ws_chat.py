import json
import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.tool import ToolMessage

from config.settings_config import get_settings
from core.redis_manager import get_redis
from enums.chat_role import ChatRole
from services.v1.chat_service import save_bot_messages, save_user_message, upsert_chat

router = APIRouter()
logger = logging.getLogger(__name__)


def _merge_token_content(token: AIMessageChunk) -> str:
    if isinstance(token.content, list):
        return "".join(str(item) for item in token.content)
    return str(token.content)


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    redis_client = await get_redis()

    logger.info("🔌 WebSocket connected")

    try:
        user_id = "user_id"  # TODO: replace with real authentication

        while True:
            data_text = await websocket.receive_text()
            try:
                data = json.loads(data_text)
            except Exception:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            event_type = data.get("type")
            logger.debug(f"Event Type: {event_type}")

            # Heartbeat
            if event_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            # 🟡 Resume connection (reconnect, page reload, etc.)
            if event_type == "resume":
                chat_id = data.get("chat_id")
                if not chat_id:
                    await websocket.send_json(
                        {"type": "error", "message": "Missing chat_id"}
                    )
                    continue

                # Ensure chat exists
                _, chat = await upsert_chat(user_id, chat_id)

                # Restore stream state from Redis
                redis_key = f"in_progress:{chat_id}"
                raw_state = await redis_client.get(redis_key)
                if raw_state:
                    stream_state = json.loads(raw_state)
                    resume_current = stream_state["current"]
                    thinking = stream_state["thinking"]

                    await websocket.send_json(
                        {
                            **(resume_current or {}),
                            "type": (
                                "resume_thinking" if thinking else "resume_messaging"
                            ),
                        }
                    )

                await websocket.send_json({"type": "resume_ack", "chat_id": chat_id})
                continue

            # 🔵 Incoming user message
            if event_type == "user_message":
                chat_id = data.get("chat_id")
                message = data.get("message")

                # Upsert chat
                is_created, chat = await upsert_chat(user_id, chat_id)
                chat_id = chat.id

                if is_created:
                    await websocket.send_json(
                        {
                            "type": "create",
                            "chat_id": chat_id,
                            "timestamp": datetime.now(timezone.utc).timestamp(),
                        }
                    )

                user_msg = await save_user_message(chat_id, message)

                await websocket.send_json(
                    {
                        "type": "init",
                        "id": user_msg.id,
                        "chat_id": chat_id,
                        "role": ChatRole.USER.value,
                        "timestamp": datetime.now(timezone.utc).timestamp(),
                        "content": user_msg.content,
                    }
                )

                buffered: list[dict] = []
                current: Optional[dict] = None
                thinking = False

                config = {
                    "configurable": {
                        "thread_id": "thread_id",
                        "chat_id": chat_id,
                        "user_id": user_id,
                    }
                }

                async for (
                    stream_mode,
                    chunk,
                ) in websocket.app.state.supervisor_agent.astream(
                    {"messages": [{"role": "user", "content": message}]},
                    stream_mode=["updates", "messages"],
                    config=config,
                ):
                    if stream_mode != "messages" or not isinstance(chunk, tuple):
                        continue

                    token, _ = chunk
                    if isinstance(token, AIMessageChunk) and not token.tool_calls:
                        content = _merge_token_content(token)

                        # Thinking state
                        if content == "<think>":
                            if current:
                                await websocket.send_json(
                                    {**current, "type": "end_messaging"}
                                )
                                buffered.append(current)
                                current = None

                            thinking = True
                            current = {
                                "id": str(uuid4()),
                                "chat_id": chat_id,
                                "role": ChatRole.SYSTEM.value,
                                "timestamp": datetime.now(timezone.utc).timestamp(),
                                "content": "",
                            }

                            await websocket.send_json(
                                {**current, "type": "start_thinking"}
                            )

                            # Cache to Redis
                            await redis_client.setex(
                                f"in_progress:{chat_id}",
                                get_settings().stream_cache_ttl,
                                json.dumps({"current": current, "thinking": True}),
                            )
                            continue

                        if content == "</think>":
                            thinking = False
                            if current:
                                await websocket.send_json(
                                    {**current, "type": "end_thinking"}
                                )
                                buffered.append(current)
                            current = None
                            await redis_client.delete(f"in_progress:{chat_id}")
                            continue

                        if thinking and current:
                            current["timestamp"] = datetime.now(
                                timezone.utc
                            ).timestamp()
                            current["content"] += content
                            await websocket.send_json({**current, "type": "thinking"})

                            # Cache to Redis
                            await redis_client.setex(
                                f"in_progress:{chat_id}",
                                get_settings().stream_cache_ttl,
                                json.dumps({"current": current, "thinking": True}),
                            )
                            continue

                        # Bot messaging (not thinking)
                        if not thinking:
                            if current is None:
                                if not content.strip():
                                    continue
                                current = {
                                    "id": str(uuid4()),
                                    "chat_id": chat_id,
                                    "role": ChatRole.BOT.value,
                                    "timestamp": datetime.now(timezone.utc).timestamp(),
                                    "content": content,
                                }
                                await websocket.send_json(
                                    {**current, "type": "start_messaging"}
                                )
                            else:
                                current["timestamp"] = datetime.now(
                                    timezone.utc
                                ).timestamp()
                                current["content"] += content
                                await websocket.send_json(
                                    {**current, "type": "messaging"}
                                )

                            # Cache to Redis
                            await redis_client.setex(
                                f"in_progress:{chat_id}",
                                get_settings().stream_cache_ttl,
                                json.dumps({"current": current, "thinking": False}),
                            )

                    elif isinstance(token, ToolMessage):
                        logger.info(token)

                # End stream
                if current:
                    buffered.append(current)
                    await websocket.send_json({**current, "type": "end_messaging"})

                await save_bot_messages(buffered)
                await redis_client.delete(f"in_progress:{chat_id}")

                await websocket.send_json({"type": "complete", "chat_id": chat_id})
                continue

            # 🔴 Stop streaming
            elif event_type == "stop":
                await websocket.send_json({"type": "complete"})
                continue

            else:
                await websocket.send_json(
                    {"type": "error", "message": f"Unknown type '{event_type}'"}
                )

    except WebSocketDisconnect:
        logger.info("❌ WebSocket disconnected")
