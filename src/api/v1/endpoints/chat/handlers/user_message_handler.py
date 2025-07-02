import asyncio
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Optional, Union
from uuid import uuid4

import ollama
import redis.asyncio as redis
from fastapi import WebSocket
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.tool import ToolMessage
from langgraph.types import Command

from api.v1.schema.chat import (
    ChatMessage,
    ConfirmationChatMessage,
    StreamChat,
    StreamChatMessage,
    StreamChatTitle,
)
from config.settings_config import get_settings
from db.prisma.generated.models import Chat as PrismaChat
from enums.chat import ApproveType, ChatRole, StreamType
from services.v1.chat_service import (
    get_connectors,
    get_user_fullname,
    save_bot_messages,
    save_user_message,
    update_chat_title,
    update_confirmation_message_approve,
    upsert_chat,
)

logger = logging.getLogger(__name__)


def _merge_token_content(token: AIMessageChunk) -> str:
    if isinstance(token.content, list):
        return "".join(str(item) for item in token.content)
    return str(token.content)


def _is_greeting(message: str) -> bool:
    prompt = f"""
Determine whether the user's message is only a greeting (e.g. 'hi', 'hello', 'good morning', etc.).
If yes, respond only with "yes". If not, respond only with "no".

Message: "{message.strip()}"
Answer:""".strip()

    response = ollama.generate(model=get_settings().chat_title_model, prompt=prompt)
    answer = (
        re.sub(r"<think>.*?</think>", "", response["response"], flags=re.DOTALL)
        .strip()
        .lower()
    )
    return answer.startswith("yes")


async def _cache_stream_to_redis(redis_client, chat_id, current, thinking) -> None:
    await redis_client.setex(
        f"chat_messages_in_progress:{chat_id}",
        get_settings().stream_cache_ttl,
        json.dumps({"current": current, "thinking": thinking}),
    )


async def _handle_chat(
    websocket: WebSocket, user_id: str, chat_id: Optional[str] = None
) -> PrismaChat:
    is_chat_created, chat = await upsert_chat(user_id, chat_id)
    chat_id = chat.id

    if is_chat_created:
        await websocket.send_json(
            {
                "type": "create_chat",
                "chat_id": chat_id,
                "content": chat.title,
                "timestamp": chat.timestamp,
            }
        )
    else:
        await websocket.send_json(
            {
                "type": "update_chat",
                "chat_id": chat_id,
                "timestamp": chat.timestamp,
            }
        )

    return chat


async def _handle_init_user_message(
    websocket: WebSocket, chat_id: str, group_id: str, message: str
) -> None:
    user_msg = await save_user_message(chat_id, group_id, message)

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


async def _get_config(chat_id: str, user_id: str) -> dict:
    config = {
        "configurable": {
            "thread_id": chat_id,
            "chat_id": chat_id,
            "user_id": user_id,
        }
    }

    user_fullname = await get_user_fullname(user_id)
    if user_fullname:
        config["configurable"]["user_fullname"] = user_fullname

    connectors = await get_connectors(user_id)
    for connector in connectors:
        config["configurable"][
            f"{connector.connector_type}_user_id"
        ] = connector.connector_id

    return config


def _generate_title(buffered: list[ChatMessage]) -> str:
    messages = [
        {"role": buffer["role"], "content": buffer["content"]}
        for buffer in buffered
        if buffer["role"] in [ChatRole.USER, ChatRole.ASSISTANT]
    ]

    instruction = (
        "Generate a short and relevant title (max 5 words) for the following conversation "
        "between a user and an assistant. Respond with only the title.\n\n"
    )

    dialogue = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content'].strip()}" for msg in messages
    )

    prompt = f"{instruction}{dialogue}\n\nTitle:"

    response = ollama.generate(
        model=get_settings().chat_title_model,
        prompt=prompt,
    )
    title = re.sub(
        r"<think>.*?</think>", "", response["response"], flags=re.DOTALL
    ).strip()

    return title


async def _generate_chat_title(
    websocket: WebSocket,
    user_id: str,
    chat: PrismaChat,
    message: str,
    buffered: list[ChatMessage],
) -> None:
    # Generate title
    if not chat.isTitleSet:
        stream_chat: StreamChat = {
            "type": StreamType.CHECKING_TITLE,
            "chat_id": chat.id,
        }
        await websocket.send_json(stream_chat)
        await asyncio.sleep(0)

        if _is_greeting(message):
            stream_chat_title: StreamChatTitle = {
                "type": StreamType.GENERATED_TITLE,
                "chat_id": chat.id,
                "content": chat.title,
                "timestamp": chat.timestamp,
            }
            await websocket.send_json(stream_chat_title)
        else:
            title = _generate_title(buffered)
            updated_chat = await update_chat_title(user_id, chat.id, title)
            stream_chat_title: StreamChatTitle = {
                "type": StreamType.GENERATED_TITLE,
                "chat_id": chat.id,
                "content": updated_chat.title,
                "timestamp": updated_chat.timestamp,
            }
            await websocket.send_json(stream_chat_title)


async def _is_completed(
    websocket: WebSocket,
    redis_client: redis.Redis,
    config: dict,
    chat: PrismaChat,
    group_id: str,
    buffered: list[ChatMessage],
) -> bool:
    state = await websocket.app.state.supervisor_agent.aget_state(
        config, subgraphs=True
    )

    if hasattr(state, "tasks") and len(state.tasks) > 0:
        sub_state = state.tasks[0].state

        if sub_state.next and "tools" in sub_state.next:
            last_message = sub_state.values["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                tool_call = last_message.tool_calls[0]
                confirmation: ConfirmationChatMessage = {
                    "name": tool_call.get("name"),
                    "args": tool_call.get("args"),
                    "approve": None,
                }
                current: ChatMessage = {
                    "id": str(uuid4()),
                    "chat_id": chat.id,
                    "role": ChatRole.CONFIRMATION,
                    "timestamp": datetime.now(timezone.utc).timestamp(),
                    "content": confirmation,
                    "group_id": group_id,
                }
                buffered.append(
                    {
                        **current,
                        "content": {
                            **confirmation,
                            "approve": False,
                        },
                    }
                )
                stream_msg: StreamChatMessage = {
                    **current,
                    "type": StreamType.CONFIRMATION,
                }
                await websocket.send_json(stream_msg)

                await redis_client.setex(
                    f"chat_messages_in_confirmation:{current['id']}",
                    get_settings().stream_cache_ttl,
                    json.dumps({"group_id": group_id}),
                )

                return False

    return True


async def _stream_messages(
    websocket: WebSocket,
    redis_client: redis.Redis,
    chat: PrismaChat,
    group_id: str,
    message: Union[str, dict],
    config: dict,
    msg_id: Optional[str] = None,
) -> tuple[bool, list[ChatMessage]]:
    buffered: list[ChatMessage] = []
    current: Optional[ChatMessage] = None
    thinking = False
    input = None
    is_approved = False

    if isinstance(message, dict):
        approve = message.get("approve")
        if approve == ApproveType.ACCEPT:
            input = Command(resume={"type": message.get("approve")})
            is_approved = True

        elif approve == ApproveType.DENY:
            updated = await update_confirmation_message_approve(
                chat.id, group_id, str(msg_id), False
            )
            stream_msg: StreamChatMessage = {
                "id": updated.id,
                "chat_id": updated.chatId,
                "role": ChatRole(updated.role),
                "timestamp": updated.timestamp,
                "content": updated.content,
                "group_id": group_id,
                "type": StreamType.END_CONFIRMATION,
            }
            await websocket.send_json(stream_msg)

    else:
        input = {"messages": [{"role": "user", "content": message}]}

    if input:
        async for stream_mode, chunk in websocket.app.state.supervisor_agent.astream(
            input, stream_mode=["updates", "messages"], config=config
        ):
            if stream_mode != "messages" or not isinstance(chunk, tuple):
                continue

            token, _ = chunk
            if isinstance(token, AIMessageChunk) and not token.tool_calls:
                content = _merge_token_content(token)

                if content == "<think>":
                    if current:
                        stream_msg: StreamChatMessage = {
                            **current,
                            "type": StreamType.END_MESSAGING,
                        }
                        await websocket.send_json(stream_msg)
                        buffered.append(current)
                        current = None

                    thinking = True
                    current = {
                        "id": str(uuid4()),
                        "chat_id": chat.id,
                        "role": ChatRole.SYSTEM,
                        "timestamp": datetime.now(timezone.utc).timestamp(),
                        "content": "",
                        "group_id": group_id,
                    }
                    stream_msg: StreamChatMessage = {
                        **current,
                        "type": StreamType.START_THINKING,
                    }
                    await websocket.send_json(stream_msg)
                    await _cache_stream_to_redis(
                        redis_client, chat.id, current, thinking
                    )
                    continue

                if content == "</think>":
                    thinking = False
                    if current:
                        stream_msg: StreamChatMessage = {
                            **current,
                            "type": StreamType.END_THINKING,
                        }
                        await websocket.send_json(stream_msg)
                        buffered.append(current)
                    current = None
                    await redis_client.delete(f"chat_messages_in_progress:{chat.id}")
                    continue

                if thinking and current:
                    current["timestamp"] = datetime.now(timezone.utc).timestamp()
                    current["content"] = str(current["content"]) + content
                    stream_msg: StreamChatMessage = {
                        **current,
                        "type": StreamType.THINKING,
                    }
                    await websocket.send_json(stream_msg)
                    await _cache_stream_to_redis(
                        redis_client, chat.id, current, thinking
                    )
                    continue

                if not thinking:
                    if current is None:
                        if not content.strip():
                            continue
                        current = {
                            "id": str(uuid4()),
                            "chat_id": chat.id,
                            "role": ChatRole.ASSISTANT,
                            "timestamp": datetime.now(timezone.utc).timestamp(),
                            "content": content,
                            "group_id": group_id,
                        }
                        stream_msg: StreamChatMessage = {
                            **current,
                            "type": StreamType.START_MESSAGING,
                        }
                        await websocket.send_json(stream_msg)
                    else:
                        current["timestamp"] = datetime.now(timezone.utc).timestamp()
                        current["content"] = str(current["content"]) + content
                        stream_msg: StreamChatMessage = {
                            **current,
                            "type": StreamType.MESSAGING,
                        }
                        await websocket.send_json(stream_msg)

                    await _cache_stream_to_redis(
                        redis_client, chat.id, current, thinking
                    )

            elif isinstance(token, ToolMessage):
                logger.info(token)

    if current:
        buffered.append(current)
        stream_msg: StreamChatMessage = {**current, "type": StreamType.END_MESSAGING}
        await websocket.send_json(stream_msg)
        current = None

    if is_approved:
        updated = await update_confirmation_message_approve(
            chat.id, group_id, str(msg_id), True
        )
        stream_msg: StreamChatMessage = {
            "id": updated.id,
            "chat_id": updated.chatId,
            "role": ChatRole(updated.role),
            "timestamp": updated.timestamp,
            "content": updated.content,
            "group_id": group_id,
            "type": StreamType.END_CONFIRMATION,
        }
        await websocket.send_json(stream_msg)

    if input is None:
        is_completed = True
    else:
        is_completed = await _is_completed(
            websocket, redis_client, config, chat, group_id, buffered
        )

    return is_completed, buffered


async def handle_user_message(
    websocket: WebSocket,
    redis_client: redis.Redis,
    user_id: str,
    data: Any,
) -> None:
    chat_id = data.get("chat_id")
    message: Union[str, dict] = data.get("message")

    chat = await _handle_chat(websocket, user_id, chat_id)

    group_id = None
    msg_id = None
    is_completed = True
    buffered = None

    if isinstance(message, dict):
        msg_id = data.get("msg_id")
        redis_key = f"chat_messages_in_confirmation:{msg_id}"
        raw_state = await redis_client.get(redis_key)
        if raw_state:
            stream_state = json.loads(raw_state)
            group_id = stream_state["group_id"]
            await redis_client.delete(redis_key)
    else:
        group_id = str(uuid.uuid4())
        await _handle_init_user_message(websocket, chat.id, group_id, message)

    if group_id:
        config = await _get_config(chat.id, user_id)

        is_completed, buffered = await _stream_messages(
            websocket, redis_client, chat, group_id, message, config, msg_id
        )

        await save_bot_messages(buffered)

    await redis_client.delete(f"chat_messages_in_progress:{chat.id}")

    if buffered and isinstance(message, str):
        await _generate_chat_title(websocket, user_id, chat, message, buffered)

    if is_completed:
        await websocket.send_json({"type": "complete", "chat.id": chat.id})
