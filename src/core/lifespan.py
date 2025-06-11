import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from config.settings_config import get_settings
from agent.agent import get_agent

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for FastAPI app.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info(f"Starting up {get_settings().project_info}...")

    # Load agent
    await get_agent()

    logger.info(f"{get_settings().project_info} completely loaded")

    app.state.ready = True

    yield

    # Shutdown
    logger.info(f"Shutting down {get_settings().project_info}...")

    # Add cleanup tasks here
    # Example: Close database connections, cleanup resources, etc.

    logger.info(f"{get_settings().project_info} completely shutdown")
