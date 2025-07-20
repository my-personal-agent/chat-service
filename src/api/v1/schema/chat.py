from typing import List, Optional, TypedDict, Union

from pydantic import BaseModel

from enums.chat import ApproveType, ChatRole, StreamType


class ConfirmationChatMessage(TypedDict):
    name: str
    args: dict
    approve: ApproveType


class ChatResponse(BaseModel):
    id: str
    title: str
    timestamp: float


class ChatsResponse(BaseModel):
    total: int
    next_cursor: Optional[str]
    chats: list[ChatResponse]


class ChatMessageUploadFile(TypedDict):
    id: str
    filename: str
    description: str


class ChatMessageResponse(BaseModel):
    id: str
    content: Union[str, ConfirmationChatMessage]
    role: ChatRole
    timestamp: float
    chat_id: str
    group_id: str
    upload_files: List[ChatMessageUploadFile]


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
    upload_files: List[ChatMessageUploadFile]


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


class UploadFileChunkResponse(BaseModel):
    file_id: str
    file_name: str
    complete: bool
