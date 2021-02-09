import logging


def _get_logger():
    logger = logging.getLogger("tempo")
    logger.setLevel(logging.INFO)
    return logger


logger = _get_logger()
