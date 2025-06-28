from typing import Optional

from pydantic import BaseModel

from enums.chat_role import ChatRole


class ChatResponse(BaseModel):
    id: str
    title: str
    timestamp: float


class ChatsResponse(BaseModel):
    total: int
    next_cursor: Optional[str]
    chats: list[ChatResponse]


class ChatMessageResponse(BaseModel):
    id: str
    content: str
    role: ChatRole
    timestamp: float
    chat_id: str


class ChatMessagesResponse(BaseModel):
    total: int
    nextCursor: Optional[str]
    messages: list[ChatMessageResponse]
