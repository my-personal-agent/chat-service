from typing import Any, Dict

from fastapi import HTTPException

from api.v1.schema.profile import ProfileResponse
from db.prisma.generated.types import UserUpdateInput
from db.prisma.utils import get_db


async def get_profile_of_user(user_id: str) -> ProfileResponse:
    db = await get_db()

    user = await db.user.find_first_or_raise(where={"id": user_id})

    return ProfileResponse(
        first_name=user.firstName,
        last_name=user.lastName,
        nick_name=user.nickName,
        timezone=user.timezone,
        language=user.language,
    )


async def update_profile_of_user(
    user_id: str, updates: Dict[str, Any]
) -> ProfileResponse:
    db = await get_db()

    # Allowed fields and their allowed types
    field_types: Dict[str, tuple[type, ...]] = {
        "firstName": (str,),
        "lastName": (str, type(None)),
        "nickName": (str,),
        "timezone": (str,),
        "language": (str,),
    }

    update_data: Dict[str, Any] = {}

    for field, value in updates.items():
        if field not in field_types:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot update field '{field}'. Allowed fields: {list(field_types)}",
            )

        allowed_types = field_types[field]
        if not isinstance(value, allowed_types):
            allowed_names = ", ".join(t.__name__ for t in allowed_types)
            raise HTTPException(
                status_code=400,
                detail=f"Field '{field}' expects types ({allowed_names}), got {type(value).__name__}",
            )

        update_data[field] = value

    if not update_data:
        raise HTTPException(
            status_code=400, detail="No valid fields were provided to update."
        )

    updated_user = await db.user.update(
        where={"id": user_id},
        data=UserUpdateInput(**update_data),
    )

    if not updated_user:
        raise HTTPException(status_code=404, detail="User not found")

    return ProfileResponse(
        first_name=updated_user.firstName,
        last_name=updated_user.lastName,
        nick_name=updated_user.nickName,
        timezone=updated_user.timezone,
        language=updated_user.language,
    )
