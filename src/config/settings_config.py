import logging
from functools import lru_cache
from typing import Annotated, List

from pydantic import AnyHttpUrl, BeforeValidator, Field, ValidationError, computed_field
from pydantic_settings import BaseSettings

from enums.mcp_transport import McpTransport

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    env: Annotated[str, BeforeValidator(str.strip), Field(min_length=1)]

    # API
    project_name: Annotated[str, BeforeValidator(str.strip), Field(min_length=1)]
    project_version: Annotated[str, BeforeValidator(str.strip), Field(min_length=1)]
    host: Annotated[str, BeforeValidator(str.strip), Field(min_length=1)]
    port: Annotated[int, Field(ge=0)]
    backend_cors_origins: List[AnyHttpUrl]
    allowed_hosts: List[AnyHttpUrl]

    # agents
    ollama_base_url: AnyHttpUrl
    supervisor_agent_model: Annotated[
        str, BeforeValidator(str.strip), Field(min_length=1)
    ]
    weather_agent_model: Annotated[str, BeforeValidator(str.strip), Field(min_length=1)]
    user_profile_agent_model: Annotated[
        str, BeforeValidator(str.strip), Field(min_length=1)
    ]
    code_agent_model: Annotated[str, BeforeValidator(str.strip), Field(min_length=1)]
    translator_agent_model: Annotated[
        str, BeforeValidator(str.strip), Field(min_length=1)
    ]
    google_agent_model: Annotated[str, BeforeValidator(str.strip), Field(min_length=1)]

    supervisor_rag_agent_model: Annotated[
        str, BeforeValidator(str.strip), Field(min_length=1)
    ]
    upload_file_agent_model: Annotated[
        str, BeforeValidator(str.strip), Field(min_length=1)
    ]

    # models
    chat_title_model: Annotated[str, BeforeValidator(str.strip), Field(min_length=1)]
    chat_upload_file_description_model: Annotated[
        str, BeforeValidator(str.strip), Field(min_length=1)
    ]
    memory_summarizer_model: Annotated[
        str, BeforeValidator(str.strip), Field(min_length=1)
    ]
    memory_summarizer_model_max_tokens: Annotated[int, Field(ge=0)]
    memory_summarizer_max_tokens: Annotated[int, Field(ge=0)]

    # mcp servers
    mcp_server_weather_url: AnyHttpUrl
    mcp_server_weather_transport: McpTransport
    mcp_server_google_url: AnyHttpUrl
    mcp_server_google_transport: McpTransport

    # postgres
    postgres_database_url: Annotated[
        str, BeforeValidator(str.strip), Field(min_length=1)
    ]
    postgres_pool_min_size: Annotated[int, Field(ge=0)]
    postgres_pool_max_size: Annotated[int, Field(ge=0)]

    # qdrant
    qdrant_url: AnyHttpUrl
    qdrant_embeddings_model: Annotated[
        str, BeforeValidator(str.strip), Field(min_length=1)
    ]
    qdrant_embeddings_model_dims: Annotated[int, Field(ge=0)]
    qdrant_upload_collection_name: Annotated[
        str, BeforeValidator(str.strip), Field(min_length=1)
    ]

    # dir
    rag_agent_upload_temp_dir: Annotated[
        str, BeforeValidator(str.strip), Field(min_length=1)
    ]

    # semantic and store
    lang_store_embeddings_model: Annotated[
        str, BeforeValidator(str.strip), Field(min_length=1)
    ]
    lang_store_embeddings_model_dims: Annotated[int, Field(ge=0)]

    # redis
    redis_url: Annotated[str, BeforeValidator(str.strip), Field(min_length=1)]
    redis_max_connection: Annotated[int, Field(ge=0)]

    # stream
    stream_cache_ttl: Annotated[int, Field(ge=0)]

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
