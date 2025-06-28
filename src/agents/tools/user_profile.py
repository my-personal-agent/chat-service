from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from services.v1.profile_service import get_profile_of_user, update_profile_of_user


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

    return get_profile_of_user(user_id)


@tool
async def update_profile(field: str, value: str, config: RunnableConfig) -> str:
    """
    Update a user profile field.

    Args:
        field: The field to update (firstName, lastName, nickName, langauge, timezone)
        value: The new value for the field

    Returns:
        Success message confirming the update
    """
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id")
    if not user_id:
        raise KeyError("user_id is required to to get profile")

    await update_profile_of_user(user_id, {field: value})

    return f"Updated {field} to '{value}'."
