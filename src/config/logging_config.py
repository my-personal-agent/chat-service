import logging.config
import os
from pathlib import Path

import yaml

from config.settings_config import get_settings
from core.utils import deep_merge


def setup_logging():
    """
    Sets up logging configuration for the application.

    Loads the base logging configuration from a YAML file, and if an environment-specific
    override exists, merges it with the base configuration. Ensures that any directories
    for file handlers exist before applying the logging configuration.

    Raises:
        FileNotFoundError: If the base logging configuration file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML files.
    """
    # Determine the base directory of the project
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    # Path to the base logging configuration YAML file
    base_config_path = os.path.join(base_dir, "config", "logging", "logging.yaml")
    # Path to the environment-specific logging configuration YAML file
    env_config_path = os.path.join(
        base_dir, "config", "logging", f"logging.{get_settings().env}.yaml"
    )

    # Load the base logging configuration
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    # If an environment-specific config exists, load and merge it
    if Path(env_config_path).exists():
        with open(env_config_path, "r") as f:
            override_config = yaml.safe_load(f)
        config = deep_merge(base_config, override_config)
    else:
        config = base_config

    # Apply the logging configuration
    logging.config.dictConfig(config)
