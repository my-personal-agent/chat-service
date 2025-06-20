from typing import Optional
from uuid import UUID

from pydantic import BaseModel

from enums.chat_role import ChatRole


class ChatsMessage(BaseModel):
    id: UUID
    content: str
    role: ChatRole
    timestamp: float
    chat_id: UUID


class ChatMessagesResponse(BaseModel):
    total: int
    nextCursor: Optional[str]
    messages: list[ChatsMessage]
