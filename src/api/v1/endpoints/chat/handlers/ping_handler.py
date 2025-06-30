from fastapi import WebSocket


async def handle_ping(websocket: WebSocket):
    await websocket.send_json({"type": "pong"})
