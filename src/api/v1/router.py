from fastapi import APIRouter

from api.v1.endpoints.chat.conversation import router as get_conversation_router
from api.v1.endpoints.chat.stream import router as stream_router

api_router = APIRouter()

# chat
chat_prefix = "/chat"
api_router.include_router(stream_router, prefix=chat_prefix)
api_router.include_router(get_conversation_router, prefix=chat_prefix)
