from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException

from db.prisma.generated.enums import Role
from db.prisma.generated.models import Conversation, ConversationMessage
from db.prisma.utils import get_db


async def upsert_conversation(
    user_id: str, conversation_id: Optional[str] = None
) -> Conversation:
    db = await get_db()

    if conversation_id:
        updated_conversation = await db.conversation.update(
            where={"id": conversation_id},
            data={"timestamp": datetime.now(timezone.utc).timestamp()},
        )
        if not updated_conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return updated_conversation

    return await db.conversation.create(
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


async def get_messages_by_conversation_id(conversation_id: str):
    # todo: add user id to condition
    db = await get_db()
    messages = await db.conversationmessage.find_many(
        where={"conversationId": conversation_id}, order={"updatedAt": "desc"}
    )

    return [
        {
            "id": mes.id,
            "content": mes.content,
            "role": mes.role,
            "timestamp": mes.timestamp,
            "conversation_id": mes.conversationId,
        }
        for mes in messages
    ]
