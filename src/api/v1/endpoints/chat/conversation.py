import logging

from fastapi import APIRouter, Query, Request

from api.v1.schema.chat.conversation_message_response import ConversationMessageResponse
from api.v1.schema.chat.conversation_response import ConversationResponse
from services.v1.chat_service import (
    get_conversation_list,
    get_messages_by_conversation_id,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/chat/conversations/{conversation_id}/messages",
    response_model=ConversationMessageResponse,
)
async def get_conversation_messages(
    request: Request,
    conversation_id: str,
    limit: int = Query(20, ge=1, le=100),
    cursor: str = Query(None),
):
    return await get_messages_by_conversation_id(conversation_id, limit, cursor)


@router.get(
    "/conversations",
    response_model=list[ConversationResponse],
)
async def get_conversations(
    request: Request,
    limit: int = Query(20, ge=1, le=100),
    cursor: str = Query(None),
):
    # todo: add user id to condition
    return await get_conversation_list("user_id", limit, cursor)
