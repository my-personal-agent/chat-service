import json
import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Query, Request
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.tool import ToolMessage
from sse_starlette import EventSourceResponse

from enums.chat_role import ChatRole
from services.v1.chat_service import (
    save_bot_messages,
    save_user_message,
    upsert_conversation,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/stream", response_class=EventSourceResponse)
async def chat_stream(
    request: Request,
    message: str = Query(...),
    conversation_id: Optional[str] = Query(None),
):
    user_id = "user_id"  # 🔒 Replace with real user

    conversation = await upsert_conversation(user_id, conversation_id)
    user_message = await save_user_message(conversation.id, message)

    async def event_generator():
        yield {
            "event": "init",
            "data": json.dumps(
                {
                    "id": user_message.id,
                    "conversation_id": conversation.id,
                    "role": ChatRole.USER.value,
                    "timestamp": datetime.now(timezone.utc).timestamp(),
                    "content": "",
                }
            ),
        }

        config = {
            "configurable": {
                "thread_id": "thread_id",
                "conversation_id": conversation.id,
                "user_id": user_id,
            }
        }

        is_thinking = False
        buffered_messages: list[dict] = []
        current: Optional[dict] = None

        async for stream_mode, chunk in request.app.state.supervisor_agent.astream(
            {"messages": [{"role": "user", "content": message}]},
            stream_mode=["updates", "messages"],
            config=config,
        ):
            if stream_mode == "messages":
                token, _ = chunk
                if isinstance(token, AIMessageChunk):
                    if len(token.tool_calls) > 0 or len(token.tool_call_chunks) > 0:
                        logger.info(token)
                        continue

                    content = _merge_token_content(token)

                    # start thinking
                    if content == "<think>":
                        if current is not None:
                            buffered_messages.append(current)
                            yield {
                                "event": "end_messaging",
                                "data": json.dumps(current),
                            }
                            current = None

                        is_thinking = True
                        current = {
                            "id": str(uuid4()),
                            "conversation_id": conversation.id,
                            "role": ChatRole.SYSTEM.value,
                            "timestamp": datetime.now(timezone.utc).timestamp(),
                            "content": "",
                        }
                        yield {"event": "start_thinking", "data": json.dumps(current)}
                        continue

                    # end thinking
                    if content == "</think>":
                        is_thinking = False
                        yield {"event": "end_thinking", "data": json.dumps(current)}
                        if current:
                            buffered_messages.append(current)
                        current = None
                        continue

                    # thinking
                    if is_thinking and current:
                        current["timestamp"] = datetime.now(timezone.utc).timestamp()
                        current["content"] += content
                        yield {"event": "thinking", "data": json.dumps(current)}
                        continue

                    # bot message
                    if not is_thinking:
                        if current is None:
                            if content.strip() == "":
                                continue

                            current = {
                                "id": str(uuid4()),
                                "conversation_id": conversation.id,
                                "role": ChatRole.BOT.value,
                                "content": content,
                                "timestamp": datetime.now(timezone.utc).timestamp(),
                            }
                            yield {
                                "event": "start_messaging",
                                "data": json.dumps(current),
                            }
                        else:
                            current["timestamp"] = datetime.now(
                                timezone.utc
                            ).timestamp()
                            current["content"] += content
                            yield {"event": "messaging", "data": json.dumps(current)}

                elif isinstance(token, ToolMessage):
                    logger.info(token)

        if current is not None:
            buffered_messages.append(current)

        yield {"event": "end_messaging", "data": json.dumps(current)}
        current = None

        # store after stream ends
        await save_bot_messages(buffered_messages)

        yield {
            "event": "complete",
            "data": json.dumps(
                {
                    "conversation_id": conversation.id,
                }
            ),
        }

    return EventSourceResponse(event_generator())


def _merge_token_content(token: AIMessageChunk):
    if isinstance(token.content, list):
        return "".join(str(item) for item in token.content)

    return str(token.content)
