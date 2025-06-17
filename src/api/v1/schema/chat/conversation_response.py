from pydantic import BaseModel


class ConversationResponse(BaseModel):
    id: str
    title: str
    timestamp: float
