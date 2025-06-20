from typing import Optional
from uuid import UUID

from pydantic import BaseModel


class Conversation(BaseModel):
    id: UUID
    title: str
    timestamp: float


class ConversationResponse(BaseModel):
    total: int
    nextCursor: Optional[str]
    conversations: list[Conversation]
