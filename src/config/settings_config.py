import logging
from functools import lru_cache
from typing import Annotated, List

from pydantic import AnyHttpUrl, BeforeValidator, Field, ValidationError, computed_field
from pydantic_settings import BaseSettings

from enums.mcp_transport import McpTransport

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    env: Annotated[str, BeforeValidator(str.strip), Field(min_length=1)]

    project_name: Annotated[str, BeforeValidator(str.strip), Field(min_length=1)]
    project_version: Annotated[str, BeforeValidator(str.strip), Field(min_length=1)]

    backend_cors_origins: List[AnyHttpUrl]
    allowed_hosts: List[AnyHttpUrl]

    react_agent_model: Annotated[str, BeforeValidator(str.strip), Field(min_length=1)]
    supervisor_agent_model: Annotated[
        str, BeforeValidator(str.strip), Field(min_length=1)
    ]
    ollama_base_url: AnyHttpUrl

    mcp_server_weather_url: AnyHttpUrl
    mcp_server_weather_transport: McpTransport

    class ConfigDict:
        env_file = ".env"
        env_file_encoding = "utf-8"
        use_enum_values = True

    @computed_field
    def project_info(self) -> str:
        return f"{self.project_name} - {self.project_version}"


@lru_cache()
def get_settings() -> Settings:
    try:
        return Settings()  # type: ignore
    except ValidationError as e:
        logger.error("Environment configuration error:")
        logger.error(e)
        raise RuntimeError("Shutting down due to bad config")
        raise RuntimeError("Shutting down due to bad config")
