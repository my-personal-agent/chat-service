from fastapi import FastAPI

from api.monitoring.router import api_router as monitoring_api_router
from api.v1.router import api_router as v1_api_router
from config.settings_config import get_settings
from core.exceptions import setup_exception_handlers
from core.lifespan import lifespan
from middleware.logging_metric_middleware import LoggingMetricMiddleware


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application
    """
    # Create FastAPI application
    app = FastAPI(
        title=get_settings().project_name,
        version=get_settings().project_version,
        lifespan=lifespan,
    )

    # Set up CORS middleware
    # if get_settings().backend_cors_origins:
    #     app.add_middleware(
    #         CORSMiddleware,
    #         allow_origins=[
    #             str(origin) for origin in get_settings().backend_cors_origins
    #         ],
    #         allow_credentials=True,
    #         allow_methods=["*"],
    #         allow_headers=["*"],
    #     )

    # Add trusted host middleware for security
    # if get_settings().allowed_hosts:
    #     app.add_middleware(
    #         TrustedHostMiddleware,
    #         allowed_hosts=[str(origin) for origin in get_settings().allowed_hosts],
    #     )

    # Add logging middleware
    app.add_middleware(LoggingMetricMiddleware)

    # Setup exception handlers
    setup_exception_handlers(app)

    # Include API routers
    app.include_router(v1_api_router, prefix="/api/v1", tags=["API v1"])
    app.include_router(
        monitoring_api_router, prefix="/api/monitoring", tags=["API Monitoring"]
    )

    return app
