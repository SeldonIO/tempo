import logging
import os
import contextvars

try:
    insights_context
except:
    insights_context = contextvars.ContextVar("insights_manager", default=None)

def _get_logger():
    logger = logging.getLogger("tempo")
    logger.setLevel(logging.INFO)
    return logger


def _get_env():
    if os.environ.get("a", None):
        pass


logger = _get_logger()
