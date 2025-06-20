from typing import Optional
from uuid import UUID

from pydantic import BaseModel


class Chat(BaseModel):
    id: UUID
    title: str
    timestamp: float


class ChatsResponse(BaseModel):
    total: int
    nextCursor: Optional[str]
    chats: list[Chat]
