import logging
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from core.monitoring import api_calls_counter, api_duration_histogram

logger = logging.getLogger(__name__)


class LoggingMetricMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and responses"""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # request info
        method = request.method
        url = request.url

        # Log request
        logger.info(f"Request: {method} {url}")

        # Process request
        response = await call_next(request)

        # response info
        status_code = response.status_code

        # Calculate processing time
        process_time = time.time() - start_time

        # Prometheus metrics
        api_calls_counter.labels(
            method=method, endpoint=url, status=str(status_code)
        ).inc()
        api_duration_histogram.labels(method=method, endpoint=url).observe(process_time)

        # Log response
        logger.info(f"Response: {status_code} - Processed in {process_time:.4f}s")

        # Add processing time to response headers
        response.headers["X-Process-Time"] = str(process_time)

        return response
