import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from langgraph.store.postgres.base import PoolConfig

from agents.embeddings import get_lang_store_embeddings
from agents.supervisor_agent import build_supervisor_agent
from config.settings_config import get_settings
from core.redis_manager import redis_manager
from db.prisma.utils import get_db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for FastAPI app.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info(f"Starting up {get_settings().project_info}...")

    # load db
    db = await get_db()

    # redis
    await redis_manager.connect()

    # embeddings
    embeddings, dims = get_lang_store_embeddings()

    # load lang store and checkpointer
    async with (
        AsyncPostgresStore.from_conn_string(
            get_settings().postgres_database_url,
            pool_config=PoolConfig(
                min_size=get_settings().postgres_pool_min_size,
                max_size=get_settings().postgres_pool_max_size,
            ),
            index={"embed": embeddings, "dims": dims},
        ) as store,
        AsyncPostgresSaver.from_conn_string(
            get_settings().postgres_database_url
        ) as checkpointer,
    ):
        await store.setup()
        await checkpointer.setup()

        # build agents
        supervisor_agent = await build_supervisor_agent(store, checkpointer)  # type: ignore

        # set data
        app.state.supervisor_agent = supervisor_agent
        app.state.ready = True

        # log
        logger.info(f"{get_settings().project_info} completely loaded")

        yield

    # Shutdown
    logger.info(f"Shutting down {get_settings().project_info}...")

    # Add cleanup tasks
    await db.disconnect()
    await redis_manager.connect()

    logger.info(f"{get_settings().project_info} completely shutdown")
