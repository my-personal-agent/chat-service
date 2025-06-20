from fastapi import APIRouter

from api.v1.endpoints.chat.chats import router as chats_router
from api.v1.endpoints.chat.ws_chat import router as ws_chat_router

api_router = APIRouter()

# chat
api_router.include_router(ws_chat_router)
api_router.include_router(chats_router)
