import logging
from typing import Optional

from fastapi import APIRouter, Request, status
from fastapi.responses import RedirectResponse

from api.v1.schema.connector import ConnectorsResponse
from services.v1.connectors_service import (
    get_connectors_of_user,
    upsert_connector_of_user,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/connectors/callback",
    response_class=RedirectResponse,
    status_code=status.HTTP_302_FOUND,
)
async def callback(
    request: Request,
    google_id: Optional[str],
    auth_type: Optional[str],
    current_uri: Optional[str],
):
    # todo
    user_id = "user_id"
    return await upsert_connector_of_user(user_id, google_id, auth_type, current_uri)


@router.get("/connectors", response_model=ConnectorsResponse)
async def get_connectors(request: Request):
    # todo
    user_id = "user_id"
    return await get_connectors_of_user(user_id)
