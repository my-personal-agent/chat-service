from typing import Any, Dict, Optional

from pydantic import BaseModel


class ProfileResponse(BaseModel):
    first_name: str
    last_name: Optional[str]
    nick_name: str
    timezone: str
    language: str


class UpdateProfileRequest(BaseModel):
    updates: Dict[str, Any]
