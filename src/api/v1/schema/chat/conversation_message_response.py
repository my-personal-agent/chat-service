from uuid import UUID

from pydantic import BaseModel

from enums.chat_role import ChatRole


class ConversationMessageResponse(BaseModel):
    id: UUID
    content: str
    role: ChatRole
    timestamp: float
    conversation_id: UUID
