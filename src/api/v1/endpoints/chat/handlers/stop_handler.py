from fastapi import WebSocket


async def handle_stop(websocket: WebSocket):
    await websocket.send_json({"type": "complete"})
