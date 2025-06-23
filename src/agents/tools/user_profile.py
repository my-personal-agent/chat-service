from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from db.prisma.utils import get_db


@tool
async def get_profile(config: RunnableConfig):
    """
    Get the current user's profile information.

    Retrieves the user profile for the currently authenticated user.

    Returns:
        Dictionary containing user's firstName, lastName, and nickName
    """
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id")
    if not user_id:
        raise KeyError("user_id is required to to get profile")

    db = await get_db()

    user = await db.user.find_first_or_raise(where={"id": user_id})

    return {
        "firstName": user.firstName,
        "lastName": user.lastName,
        "nickName": user.nickName,
    }


@tool
async def update_profile(field: str, value: str, config: RunnableConfig) -> str:
    """
    Update a user profile field.

    Args:
        field: The field to update (firstName, lastName, or nickName)
        value: The new value for the field

    Returns:
        Success message confirming the update
    """
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id")
    if not user_id:
        raise KeyError("user_id is required to to get profile")

    db = await get_db()

    # Ensure only allowed fields are updated and types match
    allowed_fields = {"firstName", "lastName", "nickName"}
    if field not in allowed_fields:
        raise ValueError(
            f"Cannot update field '{field}'. Allowed fields: {allowed_fields}"
        )

    await db.user.update(where={"id": user_id}, data={field: value})  # type: ignore

    return f"Updated {field} to '{value}'."
