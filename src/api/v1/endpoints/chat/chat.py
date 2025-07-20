import logging

from fastapi import APIRouter, Query, Request, status

from api.v1.schema.chat import ChatMessagesResponse, ChatsResponse
from services.v1.chat_service import (
    delete_chat_of_user,
    get_chat_list,
    get_messages_by_chat_id,
)

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
    # todo
    user_id = "user_id"
    return await get_messages_by_chat_id(user_id, chat_id, limit, cursor)


@router.get(
    "/chats",
    response_model=ChatsResponse,
)
async def get_chats(
    request: Request,
    limit: int = Query(30, ge=1, le=100),
    cursor: str = Query(None),
):
    # todo: add user id to condition
    return await get_chat_list("user_id", limit, cursor)


@router.delete("/chats/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat(request: Request, chat_id: str):
    # todo
    user_id = "user_id"
    await delete_chat_of_user(user_id, chat_id)
