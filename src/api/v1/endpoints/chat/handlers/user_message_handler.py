import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, List, Optional, Union
from uuid import uuid4

import redis.asyncio as redis
from fastapi import WebSocket
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StateSnapshot, StateUpdate
from ollama import AsyncClient

from api.v1.schema.chat import (
    ChatMessage,
    ChatMessageUploadFile,
    ConfirmationChatMessage,
    StreamChat,
    StreamChatMessage,
    StreamChatTitle,
)
from config.settings_config import get_settings
from db.prisma.generated.models import Chat as PrismaChat
from enums.chat import ApproveType, ChatRole, StreamType
from services.v1.chat_service import (
    get_asked_files,
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


async def _is_greeting(message: str) -> bool:
    prompt = f"""
Determine whether the user's message is only a greeting (e.g. 'hi', 'hello', 'good morning', etc.).
If yes, respond only with "yes". If not, respond only with "no".

Message: "{message.strip()}"
Answer:""".strip()

    client = AsyncClient(host=str(get_settings().ollama_base_url))
    response = await client.generate(
        model=get_settings().chat_title_model, prompt=prompt, think=False
    )

    answer = response.response.strip()

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
    websocket: WebSocket,
    chat_id: str,
    group_id: str,
    message: str,
    upload_files: List[ChatMessageUploadFile],
) -> None:
    user_msg = await save_user_message(chat_id, group_id, message, upload_files)

    strem_message: StreamChatMessage = {
        "type": StreamType.INIT,
        "id": user_msg.id,
        "chat_id": chat_id,
        "role": ChatRole.USER,
        "group_id": group_id,
        "timestamp": datetime.now(timezone.utc).timestamp(),
        "content": user_msg.content,
        "upload_files": upload_files,
    }

    await websocket.send_json(strem_message)


async def _get_config(
    chat_id: str, user_id: str, upload_files: List[ChatMessageUploadFile]
) -> RunnableConfig:
    config: RunnableConfig = {
        "configurable": {
            "thread_id": chat_id,
            "chat_id": chat_id,
            "user_id": user_id,
            "upload_files": upload_files,
        }
    }

    # name
    user_fullname = await get_user_fullname(user_id)
    if user_fullname:
        config["configurable"]["user_fullname"] = user_fullname

    # connectors
    connectors = await get_connectors(user_id)
    for connector in connectors:
        config["configurable"][f"{connector.connector_type}_user_id"] = (
            connector.connector_id
        )

    # asked files
    asked_files = await get_asked_files(chat_id)
    config["configurable"]["asked_files"] = asked_files

    return config


async def _generate_title(message: str, last_message: ChatMessage) -> str:
    instruction = (
        "Generate a short and relevant title (max 5 words) for the following conversation "
        "between a user and an assistant. Respond with only the title.\n\n"
    )

    content = last_message["content"]
    if isinstance(content, str):
        content_str = content.strip()
    else:
        content_str = str(content)

    dialogue = f"USER: {message} \n{last_message['role'].capitalize()}: {content_str}"
    prompt = f"{instruction}{dialogue}\n\nTitle:"

    client = AsyncClient(host=str(get_settings().ollama_base_url))
    response = await client.generate(
        model=get_settings().chat_title_model, prompt=prompt, think=False
    )

    title = response.response.strip()

    return title


async def _generate_chat_title(
    websocket: WebSocket,
    user_id: str,
    chat: PrismaChat,
    message: Optional[str],
    last_message: ChatMessage,
) -> None:
    # Generate title
    if not chat.isTitleSet and message is not None:
        stream_chat: StreamChat = {
            "type": StreamType.CHECKING_TITLE,
            "chat_id": chat.id,
        }
        await websocket.send_json(stream_chat)
        await asyncio.sleep(0)

        is_greeting = await _is_greeting(message)

        if is_greeting:
            greeting_title: StreamChatTitle = {
                "type": StreamType.GENERATED_TITLE,
                "chat_id": chat.id,
                "content": chat.title,
                "timestamp": chat.timestamp,
            }
            await websocket.send_json(greeting_title)
        else:
            title = await _generate_title(message, last_message)
            updated_chat = await update_chat_title(user_id, chat.id, title)
            generated_title: StreamChatTitle = {
                "type": StreamType.GENERATED_TITLE,
                "chat_id": chat.id,
                "content": updated_chat.title,
                "timestamp": updated_chat.timestamp,
            }
            await websocket.send_json(generated_title)


async def _get_sub_graph_state(
    websocket: WebSocket,
    config: RunnableConfig,
) -> Union[None, StateSnapshot]:
    state = await websocket.app.state.supervisor_agent.aget_state(
        config, subgraphs=True
    )

    if hasattr(state, "tasks") and len(state.tasks) > 0:
        return state.tasks[0].state

    return None


async def _is_completed(
    websocket: WebSocket,
    redis_client: redis.Redis,
    config: RunnableConfig,
    chat: PrismaChat,
    group_id: str,
    buffered: List[ChatMessage],
    user_msg: Optional[str] = None,
) -> bool:
    sub_state = await _get_sub_graph_state(websocket, config)

    if sub_state is None:
        return True

    confirm_tools = websocket.app.state.confirm_tools

    if sub_state.next and "tools" in sub_state.next:
        last_message = sub_state.values["messages"][-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            tool_call = last_message.tool_calls[0]
            if (
                sub_state.metadata is not None
                and sub_state.metadata.get("langgraph_node") in confirm_tools
                and tool_call.get("name")
                in confirm_tools[sub_state.metadata.get("langgraph_node")]
            ):
                confirmation: ConfirmationChatMessage = {
                    "name": tool_call.get("name"),
                    "args": tool_call.get("args"),
                    "approve": ApproveType.ASKING,
                }
                current: ChatMessage = {
                    "id": str(uuid4()),
                    "chat_id": chat.id,
                    "role": ChatRole.CONFIRMATION,
                    "timestamp": datetime.now(timezone.utc).timestamp(),
                    "content": confirmation,
                    "group_id": group_id,
                    "upload_files": [],
                }
                buffered.append(
                    {
                        **current,
                        "content": {
                            **confirmation,
                            "approve": ApproveType.CANCEL,
                        },
                    }
                )
                stream_msg: StreamChatMessage = {
                    **current,
                    "type": StreamType.CONFIRMATION,
                }
                await websocket.send_json(stream_msg)

                last_user_message = sub_state.values["messages"][-2]
                await redis_client.setex(
                    f"chat_messages_in_confirmation:{current['id']}",
                    get_settings().stream_cache_ttl,
                    json.dumps(
                        {
                            "group_id": group_id,
                            "tool_call_id": tool_call.get("id"),
                            "tool_call_name": tool_call.get("name"),
                            "tool_call_args": tool_call.get("args"),
                            "user_msg": user_msg,
                            "sub_last_user_msg_id": last_user_message.id,
                            "sub_last_msg_id": last_message.id,
                        }
                    ),
                )

                return False

        # continue with the next state
        await _send_stream_messages(
            websocket,
            redis_client,
            chat,
            group_id,
            config,
            None,
            buffered,
        )

        return await _is_completed(
            websocket,
            redis_client,
            config,
            chat,
            group_id,
            buffered,
            user_msg,
        )

    return True


async def _send_stream_messages(
    websocket: WebSocket,
    redis_client: redis.Redis,
    chat: PrismaChat,
    group_id: str,
    config: RunnableConfig,
    message: Union[BaseMessage, None] = None,
    buffered: list[ChatMessage] = [],
) -> None:
    current: Optional[ChatMessage] = None
    thinking = False

    input = None
    if isinstance(message, BaseMessage):
        input = {"messages": [message]}

    supervisor_agent: CompiledStateGraph = websocket.app.state.supervisor_agent
    async for _, stream_mode, chunk in supervisor_agent.astream(
        input, stream_mode=["messages"], config=config, subgraphs=True
    ):
        if stream_mode != "messages" or not isinstance(chunk, tuple):
            continue

        token, _ = chunk
        if isinstance(token, AIMessageChunk) and not token.tool_calls:
            content = _merge_token_content(token)

            if content == "<think>":
                if current:
                    end_msg: StreamChatMessage = {
                        **current,
                        "type": StreamType.END_MESSAGING,
                    }
                    await websocket.send_json(end_msg)
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
                    "upload_files": [],
                }
                start_thinking_msg: StreamChatMessage = {
                    **current,
                    "type": StreamType.START_THINKING,
                }
                await websocket.send_json(start_thinking_msg)
                await _cache_stream_to_redis(redis_client, chat.id, current, thinking)
                continue

            if content == "</think>":
                thinking = False
                if current:
                    end_thinking_msg: StreamChatMessage = {
                        **current,
                        "type": StreamType.END_THINKING,
                    }
                    await websocket.send_json(end_thinking_msg)
                    buffered.append(current)
                current = None
                await redis_client.delete(f"chat_messages_in_progress:{chat.id}")
                continue

            if thinking and current:
                current["timestamp"] = datetime.now(timezone.utc).timestamp()
                current["content"] = str(current["content"]) + content
                thinking_msg: StreamChatMessage = {
                    **current,
                    "type": StreamType.THINKING,
                }
                await websocket.send_json(thinking_msg)
                await _cache_stream_to_redis(redis_client, chat.id, current, thinking)
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
                        "upload_files": [],
                    }
                    start_msg: StreamChatMessage = {
                        **current,
                        "type": StreamType.START_MESSAGING,
                    }
                    await websocket.send_json(start_msg)
                else:
                    current["timestamp"] = datetime.now(timezone.utc).timestamp()
                    current["content"] = str(current["content"]) + content
                    messaging_msg: StreamChatMessage = {
                        **current,
                        "type": StreamType.MESSAGING,
                    }
                    await websocket.send_json(messaging_msg)

                await _cache_stream_to_redis(redis_client, chat.id, current, thinking)

        elif isinstance(token, ToolMessage):
            logger.debug(token)

    if current:
        buffered.append(current)
        final_msg: StreamChatMessage = {**current, "type": StreamType.END_MESSAGING}
        await websocket.send_json(final_msg)
        current = None


async def _stream_user_messages(
    websocket: WebSocket,
    redis_client: redis.Redis,
    chat: PrismaChat,
    message: str,
    config: RunnableConfig,
    upload_files: List[ChatMessageUploadFile],
) -> tuple[bool, list[ChatMessage], str]:
    group_id = str(uuid.uuid4())
    await _handle_init_user_message(websocket, chat.id, group_id, message, upload_files)

    buffered: list[ChatMessage] = []
    await _send_stream_messages(
        websocket,
        redis_client,
        chat,
        group_id,
        config,
        HumanMessage(content=message),
        buffered,
    )

    is_completed = await _is_completed(
        websocket, redis_client, config, chat, group_id, buffered, message
    )

    return is_completed, buffered, message


async def _stream_confirm_messages(
    websocket: WebSocket,
    redis_client: redis.Redis,
    chat: PrismaChat,
    data: Any,
    config: RunnableConfig,
) -> tuple[bool, list[ChatMessage], Optional[str]]:
    group_id = None
    user_msg = None
    sub_last_user_msg_id = None
    sub_last_msg_id = None
    tool_call_id = None
    tool_call_name = None
    tool_call_args = None

    msg_id: str = data.get("msg_id")
    message: dict = data.get("message")

    redis_key = f"chat_messages_in_confirmation:{msg_id}"
    raw_state = await redis_client.get(redis_key)
    if raw_state:
        stream_state = json.loads(raw_state)
        group_id = stream_state["group_id"]
        sub_last_user_msg_id = stream_state.get("sub_last_user_msg_id")
        sub_last_msg_id = stream_state.get("sub_last_msg_id")
        user_msg = stream_state["user_msg"]
        tool_call_id = stream_state["tool_call_id"]
        tool_call_name = stream_state["tool_call_name"]
        tool_call_args = stream_state["tool_call_args"]
        await redis_client.delete(redis_key)

    buffered: list[ChatMessage] = []
    is_completed = True

    sub_graph = await _get_sub_graph_state(websocket, config)

    if (
        sub_graph
        and group_id
        and sub_last_user_msg_id
        and sub_last_msg_id
        and tool_call_id
        and tool_call_name
        and tool_call_args
    ):
        approve_value = message.get("approve")
        if approve_value is None:
            raise ValueError("Missing 'approve' value in message")

        approve: ApproveType = ApproveType(approve_value)

        update_data: Optional[dict] = message.get("data")
        if (
            approve == ApproveType.UPDATE or approve == ApproveType.FEEDBACK
        ) and update_data is None:
            raise ValueError("Missing 'data' for update in message")

        updated_chat_message = await update_confirmation_message_approve(
            chat.id, group_id, str(msg_id), approve, update_data
        )
        confirm_msg: StreamChatMessage = {
            "id": updated_chat_message.id,
            "chat_id": updated_chat_message.chatId,
            "role": ChatRole(updated_chat_message.role),
            "timestamp": updated_chat_message.timestamp,
            "content": updated_chat_message.content,
            "group_id": updated_chat_message.groupId,
            "upload_files": [],
            "type": StreamType.END_CONFIRMATION,
        }
        await websocket.send_json(confirm_msg)

        if approve == ApproveType.ACCEPT or approve == ApproveType.UPDATE:
            if approve == ApproveType.UPDATE:
                if (
                    update_data is None
                    or "args" not in update_data
                    or update_data["args"] is None
                ):
                    raise ValueError("Missing 'args' for update in message")

                update_args = update_data["args"]
                ai_message = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": tool_call_name,
                            "args": {
                                **tool_call_args,
                                **update_args,
                            },
                            "id": tool_call_id,
                        }
                    ],
                )

                await websocket.app.state.supervisor_agent.abulk_update_state(
                    sub_graph.config, [[StateUpdate(values={"messages": [ai_message]})]]
                )

            await _send_stream_messages(
                websocket, redis_client, chat, group_id, config, None, buffered
            )

        elif approve == ApproveType.FEEDBACK:
            if update_data is None or "message" not in update_data:
                raise ValueError("Missing 'message' for feedback in message")

            feedback_message = ToolMessage(
                tool_call_id=tool_call_id,
                content=update_data["message"],
            )
            await _send_stream_messages(
                websocket,
                redis_client,
                chat,
                group_id,
                config,
                feedback_message,
                buffered,
            )

        elif approve == ApproveType.CANCEL:
            await websocket.app.state.supervisor_agent.abulk_update_state(
                sub_graph.config,
                [
                    [
                        StateUpdate(
                            values={"messages": [RemoveMessage(id=sub_last_msg_id)]}
                        )
                    ]
                ],
            )
            await websocket.app.state.supervisor_agent.abulk_update_state(
                config,
                [
                    [
                        StateUpdate(
                            values={
                                "messages": [RemoveMessage(id=sub_last_user_msg_id)]
                            }
                        )
                    ]
                ],
            )

        is_completed = await _is_completed(
            websocket, redis_client, config, chat, group_id, buffered, user_msg
        )

    return is_completed, buffered, user_msg


async def handle_user_message(
    websocket: WebSocket,
    redis_client: redis.Redis,
    user_id: str,
    data: Any,
) -> None:
    chat_id = data.get("chat_id")
    message: Union[str, dict] = data.get("message")

    upload_files = data.get("upload_files", [])

    chat = await _handle_chat(websocket, user_id, chat_id)
    config = await _get_config(chat.id, user_id, upload_files)

    if isinstance(message, dict):
        is_completed, buffered, user_message = await _stream_confirm_messages(
            websocket, redis_client, chat, data, config
        )
    elif isinstance(message, str):
        is_completed, buffered, user_message = await _stream_user_messages(
            websocket,
            redis_client,
            chat,
            message,
            config,
            upload_files,
        )

    await save_bot_messages(buffered)

    await redis_client.delete(f"chat_messages_in_progress:{chat.id}")

    if is_completed:
        if len(buffered) > 0:
            await _generate_chat_title(
                websocket, user_id, chat, user_message, buffered[-1]
            )
        await websocket.send_json({"type": "complete", "chat.id": chat.id})
