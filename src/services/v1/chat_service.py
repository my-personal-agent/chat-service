from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException

from db.prisma.generated.enums import Role
from db.prisma.generated.models import Conversation, ConversationMessage
from db.prisma.utils import get_db


async def upsert_conversation(
    user_id: str, conversation_id: Optional[str] = None
) -> tuple[bool, Conversation]:
    db = await get_db()

    if conversation_id:
        updated_conversation = await db.conversation.update(
            where={"id": conversation_id},
            data={"timestamp": datetime.now(timezone.utc).timestamp()},
        )
        if not updated_conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return False, updated_conversation

    return True, await db.conversation.create(
        data={
            "title": "New Chat",
            "timestamp": datetime.now(timezone.utc).timestamp(),
            "userId": user_id,
        },
    )


async def save_user_message(conversation_id: str, message: str) -> ConversationMessage:
    db = await get_db()

    return await db.conversationmessage.create(
        data={
            "conversationId": conversation_id,
            "content": message.strip(),
            "role": Role.user,
            "timestamp": datetime.now(timezone.utc).timestamp(),
        }
    )


async def save_bot_messages(messages: list[dict]):
    if not messages:
        return

    db = await get_db()

    await db.conversationmessage.create_many(
        data=[
            {
                "id": str(msg["id"]),
                "conversationId": msg["conversation_id"],
                "content": msg["content"].strip(),
                "role": msg["role"],
                "timestamp": msg["timestamp"],
            }
            for msg in messages
            if msg["content"].strip() != ""
        ]
    )


async def get_messages_by_conversation_id(
    conversation_id: str, limit: int, cursor: Optional[str] = None
):
    db = await get_db()

    total = await db.conversationmessage.count(
        where={"conversationId": conversation_id}
    )
    query_args = {
        "where": {"conversationId": conversation_id},
        "order": {"timestamp": "desc"},
        "take": limit + 1,  # Fetch one extra to check for next page
    }

    if cursor:
        query_args["cursor"] = {"id": cursor}
        query_args["skip"] = 1  # Skip the cursor itself

    messages = await db.conversationmessage.find_many(**query_args)

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
                "conversation_id": mes.conversationId,
            }
            for mes in paginated_messages
        ],
    }


async def get_conversation_list(user_id: str, limit: int, cursor: Optional[str] = None):
    db = await get_db()

    total = await db.conversation.count(where={"userId": user_id})
    query_args = {
        "where": {"userId": user_id},
        "order": {"timestamp": "desc"},
        "take": limit + 1,  # Fetch one extra to check for next page
    }

    if cursor:
        query_args["cursor"] = {"id": cursor}
        query_args["skip"] = 1  # Skip the cursor itself

    conversations = await db.conversation.find_many(**query_args)

    has_next_page = len(conversations) > limit
    next_cursor = conversations[-1].id if has_next_page else None
    paginated_conversations = conversations[:limit]

    return {
        "total": total,
        "nextCursor": next_cursor,
        "messages": [
            {
                "id": mes.id,
                "title": mes.title,
                "timestamp": mes.timestamp,
            }
            for mes in paginated_conversations
        ],
    }
