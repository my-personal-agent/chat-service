import logging

from fastapi import APIRouter, Request

from api.v1.schema.chat.conversation_message_response import ConversationMessageResponse
from api.v1.schema.chat.conversation_response import ConversationResponse
from db.prisma.utils import get_db
from services.chat_service import get_messages_by_conversation_id

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=list[ConversationMessageResponse],
)
async def get_conversation_messages(
    request: Request,
    conversation_id: str,
):
    return await get_messages_by_conversation_id(conversation_id=conversation_id)


@router.get(
    "/conversations",
    response_model=list[ConversationResponse],
)
async def get_conversations(
    request: Request,
):
    db = await get_db()

    # todo: add user id to condition
    return await db.conversation.find_many(order={"updatedAt": "desc"})
