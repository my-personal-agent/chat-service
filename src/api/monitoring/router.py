import logging
import os

import psutil
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from core.monitoring import cpu_usage, memory_usage

logger = logging.getLogger(__name__)
api_router = APIRouter()


@api_router.get("/healthz")
async def healthz() -> JSONResponse:
    """
    Liveness check — confirms that the Chat API is running.
    """
    logger.debug("Health check called")
    return JSONResponse(status_code=200, content={"status": "alive"})


@api_router.get("/readyz")
async def readyz(request: Request) -> JSONResponse:
    """
    Readiness check — later hook this with Redis, DB, or VectorDB health checks.
    Currently returns 200 to indicate basic readiness.
    """
    logger.debug("Readiness check called")
    try:
        if not getattr(request.app.state, "ready", False):
            return JSONResponse(status_code=503, content={"status": "unready"})

        return JSONResponse(status_code=200, content={"status": "ready"})
    except Exception as e:
        logger.warning(f"Readiness failed: {e}")
        return JSONResponse(status_code=503, content={"status": "unready"})


@api_router.get("/metrics")
async def metrics_endpoint() -> PlainTextResponse:
    """
    Exposes Prometheus-compatible metrics from `prometheus_client`.
    Also updates CPU and memory usage just-in-time.
    """
    logger.debug("Metrics endpoint called")
    process = psutil.Process(os.getpid())
    memory_usage.set(process.memory_info().rss)
    cpu_usage.set(process.cpu_percent(interval=0.1))

    return PlainTextResponse(
        generate_latest(), headers={"Content-Type": CONTENT_TYPE_LATEST}
    )
