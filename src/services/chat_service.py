from datetime import datetime, timezone
from typing import Optional

from core.prisma.db import get_db
from core.prisma.generated.enums import Role
from core.prisma.generated.models import Conversation, ConversationMessage


async def upsert_conversation(
    user_id: str, conversation_id: Optional[str] = None
) -> Conversation:
    db = await get_db()

    if conversation_id:
        return await db.conversation.find_unique_or_raise(where={"id": conversation_id})

    return await db.conversation.create(
        data={
            "title": "",
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
        ]
    )
