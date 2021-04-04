import logging
import os


def _get_logger():
    logger = logging.getLogger("tempo")
    logger.setLevel(logging.INFO)
    return logger


def _get_env():
    if os.environ.get("a",None):
        pass

logger = _get_logger()

class TempoSettings():

    def __init__(self):
        self.kubernetes = False

    def remote_kubernetes(self, val: bool):
        self.kubernetes = val

    def use_kubernetes(self):
        return self.kubernetes


tempo_settings = TempoSettings()
