import logging
from functools import lru_cache
from typing import Optional

import redis.asyncio as redis
from fastapi import HTTPException
from redis.asyncio import ConnectionPool

from config.settings_config import get_settings

logger = logging.getLogger(__name__)


class RedisManager:
    """Singleton Redis manager"""

    _instance: Optional["RedisManager"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.pool = None
            cls._instance.client = None
        return cls._instance

    async def connect(self):
        """Initialize Redis connection"""
        if self.client is not None:
            return  # Already connected

        try:
            self.pool = ConnectionPool.from_url(
                get_settings().redis_url,
                max_connections=get_settings().redis_max_connection,
                decode_responses=True,
                retry_on_timeout=True,
            )

            self.client = redis.Redis(connection_pool=self.pool)
            await self.client.ping()
            logger.info("Redis connected")

        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise

    async def disconnect(self):
        """Close Redis connection"""
        if self.client:
            await self.client.close()
        if self.pool:
            await self.pool.disconnect()
        self.client = None
        self.pool = None
        logger.info("🔌 Redis disconnected")

    def get_client(self) -> redis.Redis:
        """Get Redis client"""
        if not self.client:
            raise HTTPException(status_code=500, detail="Redis not connected")
        return self.client


# Global instance
redis_manager = RedisManager()


@lru_cache()
def get_redis() -> redis.Redis:
    """FastAPI dependency to get Redis client"""
    return redis_manager.get_client()
