from typing import Annotated, Optional

from pydantic import BaseModel, BeforeValidator, Field


class MessageRequest(BaseModel):
    conservation_id: Annotated[
        Optional[str], BeforeValidator(str.strip), Field(default=None)
    ] = None
    message: Annotated[str, BeforeValidator(str.strip), Field(min_length=1)]
