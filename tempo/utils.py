import logging
import os


def _get_logger():
    logger = logging.getLogger("tempo")
    logger.setLevel(logging.INFO)
    return logger


def _get_env():
    if os.environ.get("a", None):
        pass


logger = _get_logger()
