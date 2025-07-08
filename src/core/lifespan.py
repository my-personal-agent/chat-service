import logging
from contextlib import AsyncExitStack, asynccontextmanager
from typing import AsyncGenerator, Optional, Tuple, cast

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


class PersistentAsyncResources:
    """Context manager that keeps async resources alive for the entire app lifecycle."""

    def __init__(self, postgres_url: str, pool_config, embeddings, dims):
        self.postgres_url = postgres_url
        self.pool_config = pool_config
        self.embeddings = embeddings
        self.dims = dims
        self.store: Optional[AsyncPostgresStore] = None
        self.checkpointer: Optional[AsyncPostgresSaver] = None
        self._exit_stack: Optional[AsyncExitStack] = None

    async def __aenter__(self) -> Tuple[AsyncPostgresStore, AsyncPostgresSaver]:
        # Create exit stack to manage multiple context managers
        self._exit_stack = AsyncExitStack()

        try:
            # Create and enter context managers with type casting
            self.store = cast(
                AsyncPostgresStore,
                await self._exit_stack.enter_async_context(
                    AsyncPostgresStore.from_conn_string(
                        self.postgres_url,
                        pool_config=self.pool_config,
                        index={"embed": self.embeddings, "dims": self.dims},
                    )
                ),
            )

            self.checkpointer = cast(
                AsyncPostgresSaver,
                await self._exit_stack.enter_async_context(
                    AsyncPostgresSaver.from_conn_string(self.postgres_url)
                ),
            )

            # Setup - now mypy knows these are not None
            await self.store.setup()
            await self.checkpointer.setup()

            return self.store, self.checkpointer

        except Exception:
            # If anything fails, clean up what we've created
            await self._exit_stack.aclose()
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._exit_stack:
            try:
                await self._exit_stack.aclose()
            except Exception as e:
                logger.error(f"Error closing resources: {e}")

        # Reset references
        self.store = None
        self.checkpointer = None
        self._exit_stack = None


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

    # Use our custom context manager for persistent resources
    async with PersistentAsyncResources(
        get_settings().postgres_database_url,
        PoolConfig(
            min_size=get_settings().postgres_pool_min_size,
            max_size=get_settings().postgres_pool_max_size,
        ),
        embeddings,
        dims,
    ) as (store, checkpointer):
        # build agents
        supervisor_agent, confirm_tools = await build_supervisor_agent(
            store, checkpointer
        )  # type: ignore

        # set data
        app.state.supervisor_agent = supervisor_agent
        app.state.confirm_tools = confirm_tools
        app.state.store = store
        app.state.checkpointer = checkpointer
        app.state.ready = True

        # log
        logger.info(f"{get_settings().project_info} completely loaded")

        yield

    # Shutdown
    logger.info(f"Shutting down {get_settings().project_info}...")

    # Add cleanup tasks
    await db.disconnect()
    await redis_manager.disconnect()

    logger.info(f"{get_settings().project_info} completely shutdown")
