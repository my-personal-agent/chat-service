import logging

import uvicorn

from config.logging_config import setup_logging
from config.settings_config import get_settings
from core.app_factory import create_app

setup_logging()

logger = logging.getLogger(__name__)

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=get_settings().host,
        port=get_settings().port,
        lifespan="on",
        reload=get_settings().env == "local",
    )
