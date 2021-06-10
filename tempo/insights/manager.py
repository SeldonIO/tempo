import asyncio

from .worker import start_insights_worker_from_async, start_insights_worker_from_sync

class InsightsManager:
    def __init__(
            self,
            worker_endpoint: str = "",
            batch_size: int = 1,
            parallelism: int = 1,
            retries: int = 3,
            output_file_path: str = None,
            window_time: int = None,
        ):
        args = (
            worker_endpoint,
            batch_size,
            parallelism,
            retries,
            output_file_path,
            window_time,
        )
        try:
            asyncio.get_running_loop()
        except:
            self._q = start_insights_worker_from_sync(*args)
        else:
            self._q = start_insights_worker_from_async(*args)

    def log(self, data):
        self._q.put(data)

