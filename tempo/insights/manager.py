import asyncio

from .worker import start_insights_worker_from_async, start_insights_worker_from_sync
from ..utils import logger

class InsightsManager:
    def __init__(
            self,
            worker_endpoint: str = "",
            batch_size: int = 1,
            parallelism: int = 1,
            retries: int = 3,
            window_time: int = None,
        ):
        args = (
            worker_endpoint,
            batch_size,
            parallelism,
            retries,
            window_time,
        )
        logger.info(f"Initialising logger with {args}")
        try:
            asyncio.get_running_loop()
        except:
            logger.debug("Initialising sync insights worker")
            self._q = start_insights_worker_from_sync(*args)
            def log(self, data):
                self._q.put(data)
            self.log = log.__get__(self, self.__class__) # pylint: disable=E1120,E1111
            logger.debug("Sync worker set up")
        else:
            logger.debug("Initialising async insights worker")
            self._q = start_insights_worker_from_async(*args)
            def log(self, data):
                asyncio.create_task(self._q.put(data))
            self.log = log.__get__(self, self.__class__)  # pylint: disable=E1120,E1111
            logger.debug("Async worker set up")

    def log(self, data): # pylint: disable=E0202
        raise Exception("Not implemented")

