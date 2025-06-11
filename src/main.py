import logging

from config.logging_config import setup_logging
from core.app_factory import create_app

setup_logging()

logger = logging.getLogger(__name__)

app = create_app()
