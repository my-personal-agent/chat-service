import logging

from fastapi import APIRouter, Query, Request

from api.v1.schema.chat.chat_messages_response import ChatMessagesResponse
from api.v1.schema.chat.chats_response import ChatsResponse
from services.v1.chat_service import get_chat_list, get_messages_by_chat_id

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/chats/{chat_id}/messages",
    response_model=ChatMessagesResponse,
)
async def get_chat_messages(
    request: Request,
    chat_id: str,
    limit: int = Query(20, ge=1, le=100),
    cursor: str = Query(None),
):
    return await get_messages_by_chat_id(chat_id, limit, cursor)


@router.get(
    "/chats",
    response_model=list[ChatsResponse],
)
async def get_chats(
    request: Request,
    limit: int = Query(20, ge=1, le=100),
    cursor: str = Query(None),
):
    # todo: add user id to condition
    return await get_chat_list("user_id", limit, cursor)
