import logging

from fastapi import APIRouter, Request

from api.v1.schema.profile import ProfileResponse, UpdateProfileRequest
from services.v1.profile_service import get_profile_of_user, update_profile_of_user

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/profile",
    response_model=ProfileResponse,
)
async def get_profile(request: Request):
    # todo
    user_id = "user_id"
    return await get_profile_of_user(user_id)


@router.patch("/profile", response_model=ProfileResponse)
async def update_profile(reqiest: Request, updateRequest: UpdateProfileRequest):
    # todo
    user_id = "user_id"
    return await update_profile_of_user(user_id, updateRequest.updates)
