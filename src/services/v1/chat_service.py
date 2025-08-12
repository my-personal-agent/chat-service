import json
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import HTTPException, status

from api.v1.schema.chat import (
    Agent,
    ChatMessage,
    ChatMessageResponse,
    ChatMessagesResponse,
    ChatMessageUploadFile,
    ChatResponse,
    ChatsResponse,
    ConfirmationChatMessage,
)
from db.prisma.generated._fields import Json
from db.prisma.generated.enums import Role
from db.prisma.generated.models import Chat
from db.prisma.generated.models import ChatMessage as PrismaChatMessage
from db.prisma.generated.models import Connector
from db.prisma.utils import get_db
from enums.chat import ApproveType, ChatRole


async def get_chat(user_id: str, chat_id: str) -> Chat:
    db = await get_db()

    chat = await db.chat.find_first(where={"id": chat_id, "userId": user_id})
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    return chat


async def update_chat_title(user_id: str, chat_id: str, title: str) -> Chat:
    db = await get_db()

    updated_chat = await db.chat.update(
        where={"id": chat_id},
        data={
            "title": title.strip(),
            "timestamp": datetime.now(timezone.utc).timestamp(),
            "isTitleSet": True,
        },
    )
    if not updated_chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    return updated_chat


async def upsert_chat(user_id: str, chat_id: Optional[str] = None) -> tuple[bool, Chat]:
    db = await get_db()

    if chat_id:
        updated_chat = await db.chat.update(
            where={"id": chat_id},
            data={"timestamp": datetime.now(timezone.utc).timestamp()},
        )
        if not updated_chat:
            raise HTTPException(status_code=404, detail="Chat not found")

        return False, updated_chat

    return True, await db.chat.create(
        data={
            "title": "New Chat",
            "timestamp": datetime.now(timezone.utc).timestamp(),
            "userId": user_id,
        },
    )


async def save_user_message(
    chat_id: str, group_id: str, message: str, upload_files: List[ChatMessageUploadFile]
) -> PrismaChatMessage:
    db = await get_db()

    message = message.strip()
    return await db.chatmessage.create(
        data={
            "chatId": chat_id,
            "content": Json(message),
            "role": Role.user,
            "groupId": group_id,
            "timestamp": datetime.now(timezone.utc).timestamp(),
            "uploadFiles": (
                {"connect": [{"id": file["id"]} for file in upload_files]}
                if upload_files
                else {"connect": []}
            ),
        }
    )


def _is_non_empty_content(content: str | ConfirmationChatMessage) -> bool:
    if isinstance(content, str):
        return content.strip() != ""
    if isinstance(content, dict):
        return json.dumps(content).strip() != "{}"
    return False


async def save_bot_messages(messages: list[ChatMessage]) -> None:
    if not messages:
        return

    db = await get_db()

    await db.chatmessage.create_many(
        data=[
            {
                "id": str(msg["id"]),
                "chatId": msg["chat_id"],
                "content": Json(msg["content"]),  # type: ignore
                "role": Role(msg["role"]),
                "groupId": msg["group_id"],
                "timestamp": msg["timestamp"],
                "agent": Json(msg["agent"]) if msg.get("agent") else None,  # type: ignore
            }
            for msg in messages
            if _is_non_empty_content(msg["content"])
        ]
    )


async def update_confirmation_message_approve(
    chat_id: str,
    group_id: str,
    msg_id: str,
    approve: ApproveType,
    data: Optional[dict] = None,
) -> PrismaChatMessage:
    db = await get_db()

    message = await db.chatmessage.find_first(
        where={"id": msg_id, "chatId": chat_id, "groupId": group_id}
    )
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")

    updated = await db.chatmessage.update(
        where={"id": msg_id},
        data={"content": Json({**message.content, "approve": approve, **(data or {})})},
    )

    if not updated:
        raise HTTPException(status_code=404, detail="Message not found")

    return updated


async def get_messages_by_chat_id(
    user_id: str, chat_id: str, limit: int, cursor: Optional[str] = None
) -> ChatMessagesResponse:
    db = await get_db()

    chat = await db.chat.find_first(where={"id": chat_id, "userId": user_id})
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    total = await db.chatmessage.count(where={"chatId": chat_id})
    query_args = {
        "where": {"chatId": chat_id},
        "order": {"timestamp": "desc"},
        "take": limit + 1,  # Fetch one extra to check for next page
    }

    if cursor:
        query_args["cursor"] = {"id": cursor}
        query_args["skip"] = 1  # Skip the cursor itself

    messages = await db.chatmessage.find_many(
        **query_args, include={"uploadFiles": True}
    )

    has_next_page = len(messages) > limit
    next_cursor = messages[-1].id if has_next_page else None
    paginated_messages = messages[:limit]

    return ChatMessagesResponse(
        total=total,
        next_cursor=next_cursor,
        messages=[
            ChatMessageResponse(
                id=mes.id,
                content=mes.content,
                agent=(
                    Agent(id=str(mes.agent["id"]), name=str(mes.agent["name"]))
                    if mes.agent
                    else None
                ),
                role=ChatRole(mes.role),
                timestamp=mes.timestamp,
                chat_id=mes.chatId,
                group_id=mes.groupId,
                upload_files=[
                    ChatMessageUploadFile(
                        id=file.id, filename=file.filename, description=file.description
                    )
                    for file in (mes.uploadFiles or [])
                ],
            )
            for mes in paginated_messages
        ],
    )


async def get_chat_list(
    user_id: str, limit: int, cursor: Optional[str] = None
) -> ChatsResponse:
    db = await get_db()

    total = await db.chat.count(where={"userId": user_id})
    query_args = {
        "where": {"userId": user_id},
        "order": {"timestamp": "desc"},
        "take": limit + 1,  # Fetch one extra to check for next page
    }

    if cursor:
        query_args["cursor"] = {"id": cursor}
        query_args["skip"] = 1  # Skip the cursor itself

    chats = await db.chat.find_many(**query_args)

    has_next_page = len(chats) > limit
    next_cursor = chats[-1].id if has_next_page else None
    paginated_chats = chats[:limit]

    return ChatsResponse(
        total=total,
        next_cursor=next_cursor,
        chats=[
            ChatResponse(
                id=chat.id,
                title=chat.title,
                timestamp=chat.timestamp,
            )
            for chat in paginated_chats
        ],
    )


async def delete_chat_of_user(user_id: str, chat_id: str) -> None:
    db = await get_db()

    deleted = await db.chat.delete_many(where={"id": chat_id, "userId": user_id})
    if deleted == 0:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Chat Not Found")


async def get_connectors(user_id: str) -> List[Connector]:
    db = await get_db()

    return await db.connector.find_many(where={"userId": user_id})


async def get_user_fullname(user_id: str) -> str:
    db = await get_db()

    user = await db.user.find_first(where={"id": user_id})
    if not user:
        return ""

    if user.lastName:
        return f"{user.firstName} {user.lastName}"

    return user.firstName


async def get_asked_files(chat_id: str) -> List[ChatMessageUploadFile]:
    db = await get_db()

    results = await db.chatmessage.find_many(
        where={"chatId": chat_id}, include={"uploadFiles": True}
    )

    files_dict = {
        file.id: file
        for chatmessage in results
        for file in (chatmessage.uploadFiles or [])
    }

    return [
        {"id": file.id, "filename": file.filename, "description": file.description}
        for file in files_dict.values()
    ]
