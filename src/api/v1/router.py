from fastapi import APIRouter

from api.v1.endpoints.chat.chat import router as chat_router
from api.v1.endpoints.chat.upload import router as chat_upload_router
from api.v1.endpoints.chat.ws_chat import router as ws_chat_router
from api.v1.endpoints.connector import router as connectors_router
from api.v1.endpoints.profile import router as profile_router

api_router = APIRouter()

# chat
api_router.include_router(ws_chat_router)
api_router.include_router(chat_router)
api_router.include_router(chat_upload_router)
api_router.include_router(profile_router)
api_router.include_router(connectors_router)
