import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from agents.supervisor_agent import get_supervisor_agent
from agents.weather_agent import get_weather_agent
from config.settings_config import get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for FastAPI app.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info(f"Starting up {get_settings().project_info}...")

    # Load agents
    await get_weather_agent()  # type: ignore
    await get_supervisor_agent()  # type: ignore

    logger.info(f"{get_settings().project_info} completely loaded")

    app.state.ready = True

    yield

    # Shutdown
    logger.info(f"Shutting down {get_settings().project_info}...")

    # Add cleanup tasks here
    # Example: Close database connections, cleanup resources, etc.

    logger.info(f"{get_settings().project_info} completely shutdown")
