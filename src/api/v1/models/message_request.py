from typing import Annotated

from pydantic import BaseModel, BeforeValidator, Field


class MessageRequest(BaseModel):
    message: Annotated[str, BeforeValidator(str.strip), Field(min_length=1)]
