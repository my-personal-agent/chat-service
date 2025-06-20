from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException

from db.prisma.generated.enums import Role
from db.prisma.generated.models import Chat, ChatMessage
from db.prisma.utils import get_db


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


async def save_user_message(chat_id: str, message: str) -> ChatMessage:
    db = await get_db()

    return await db.chatmessage.create(
        data={
            "chatId": chat_id,
            "content": message.strip(),
            "role": Role.user,
            "timestamp": datetime.now(timezone.utc).timestamp(),
        }
    )


async def save_bot_messages(messages: list[dict]):
    if not messages:
        return

    db = await get_db()

    await db.chatmessage.create_many(
        data=[
            {
                "id": str(msg["id"]),
                "chatId": msg["chat_id"],
                "content": msg["content"].strip(),
                "role": msg["role"],
                "timestamp": msg["timestamp"],
            }
            for msg in messages
            if msg["content"].strip() != ""
        ]
    )


async def get_messages_by_chat_id(
    chat_id: str, limit: int, cursor: Optional[str] = None
):
    db = await get_db()

    total = await db.chatmessage.count(where={"chatId": chat_id})
    query_args = {
        "where": {"chatId": chat_id},
        "order": {"timestamp": "desc"},
        "take": limit + 1,  # Fetch one extra to check for next page
    }

    if cursor:
        query_args["cursor"] = {"id": cursor}
        query_args["skip"] = 1  # Skip the cursor itself

    messages = await db.chatmessage.find_many(**query_args)

    has_next_page = len(messages) > limit
    next_cursor = messages[-1].id if has_next_page else None
    paginated_messages = messages[:limit]

    return {
        "total": total,
        "nextCursor": next_cursor,
        "messages": [
            {
                "id": mes.id,
                "content": mes.content,
                "role": mes.role,
                "timestamp": mes.timestamp,
                "chat_id": mes.chatId,
            }
            for mes in paginated_messages
        ],
    }


async def get_chat_list(user_id: str, limit: int, cursor: Optional[str] = None):
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

    return {
        "total": total,
        "nextCursor": next_cursor,
        "messages": [
            {
                "id": mes.id,
                "title": mes.title,
                "timestamp": mes.timestamp,
            }
            for mes in paginated_chats
        ],
    }
