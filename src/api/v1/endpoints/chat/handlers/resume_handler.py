import json

import redis.asyncio as redis
from fastapi import WebSocket


async def handle_resume(
    websocket: WebSocket, redis_client: redis.Redis, user_id: str, data: dict
) -> None:
    chat_id = data.get("chat_id")
    if not chat_id:
        await websocket.send_json({"type": "error", "message": "Missing chat_id"})
        return

    # TODO
    # await get_chat(user_id, chat_id)

    redis_key = f"chat_messages_in_progress:{chat_id}"
    raw_state = await redis_client.get(redis_key)
    if raw_state:
        stream_state = json.loads(raw_state)
        resume_current = stream_state["current"]
        thinking = stream_state["thinking"]

        await websocket.send_json(
            {
                **(resume_current or {}),
                "type": "resume_thinking" if thinking else "resume_messaging",
            }
        )

    await websocket.send_json({"type": "resume_ack", "chat_id": chat_id})
