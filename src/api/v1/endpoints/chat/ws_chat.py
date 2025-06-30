import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.v1.endpoints.chat.handlers.ping_handler import handle_ping
from api.v1.endpoints.chat.handlers.resume_handler import handle_resume
from api.v1.endpoints.chat.handlers.stop_handler import handle_stop
from api.v1.endpoints.chat.handlers.unknown_handler import handle_unknown
from api.v1.endpoints.chat.handlers.user_message_handler import handle_user_message
from core.redis_manager import get_redis

router = APIRouter()
logger = logging.getLogger(__name__)


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    redis_client = await get_redis()

    logger.info("🔌 Chat WebSocket connected")

    try:
        user_id = "user_id"  # TODO: replace with real authentication

        while True:
            data_text = await websocket.receive_text()
            try:
                data = json.loads(data_text)
            except Exception:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            event_type = data.get("type")
            logger.debug(f"Event Type: {event_type}")

            if event_type == "ping":
                await handle_ping(websocket)
            elif event_type == "resume":
                await handle_resume(websocket, redis_client, user_id, data)
            elif event_type == "user_message":
                await handle_user_message(websocket, redis_client, user_id, data)
            elif event_type == "stop":
                await handle_stop(websocket)
            else:
                await handle_unknown(websocket, event_type)

    except WebSocketDisconnect:
        logger.info("❌ Chat WebSocket disconnected")
