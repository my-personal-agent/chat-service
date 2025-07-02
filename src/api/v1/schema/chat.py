from typing import Optional, TypedDict, Union

from pydantic import BaseModel

from enums.chat import ChatRole, StreamType


class ConfirmationChatMessage(TypedDict):
    name: str
    args: dict
    approve: Optional[bool]


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
    content: Union[str, ConfirmationChatMessage]
    role: ChatRole
    timestamp: float
    chat_id: str


class ChatMessagesResponse(BaseModel):
    total: int
    next_cursor: Optional[str]
    messages: list[ChatMessageResponse]


class ChatMessage(TypedDict):
    id: str
    chat_id: str
    role: ChatRole
    timestamp: float
    content: Union[str, ConfirmationChatMessage]
    group_id: str


class StreamChat(TypedDict):
    type: StreamType
    chat_id: str


class StreamChatTitle(TypedDict):
    type: StreamType
    chat_id: str
    content: str
    timestamp: float


class StreamChatMessage(ChatMessage):
    type: StreamType
