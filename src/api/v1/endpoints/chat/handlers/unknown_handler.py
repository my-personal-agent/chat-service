from fastapi import WebSocket


async def handle_unknown(websocket: WebSocket, event_type: str):
    await websocket.send_json(
        {"type": "error", "message": f"Unknown type '{event_type}'"}
    )
