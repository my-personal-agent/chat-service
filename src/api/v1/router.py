from fastapi import APIRouter

from api.v1.endpoints.chat.conversation import router as get_conversation_router
from api.v1.endpoints.chat.ws_chat import router as ws_chat_router

api_router = APIRouter()

# chat
api_router.include_router(ws_chat_router)
api_router.include_router(get_conversation_router)
