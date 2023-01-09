import os

from ..yaml_utils import load_yaml

LOGGER_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "logger.yaml")

LOGGER_CONFIG = load_yaml(LOGGER_CONFIG_PATH)
