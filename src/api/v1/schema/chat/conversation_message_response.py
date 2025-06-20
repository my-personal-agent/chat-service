from typing import Optional
from uuid import UUID

from pydantic import BaseModel

from enums.chat_role import ChatRole


class ConversationMessage(BaseModel):
    id: UUID
    content: str
    role: ChatRole
    timestamp: float
    conversation_id: UUID


class ConversationMessageResponse(BaseModel):
    total: int
    nextCursor: Optional[str]
    messages: list[ConversationMessage]
