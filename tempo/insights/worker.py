import requests
import os
import json
import asyncio
import aiohttp
import janus
import threading
from ..utils import logger


async def start_worker(
    q_in: janus.Queue,
    worker_endpoint: str,
    parallelism: int = 1,
    batch_size: int = 1,  # TODO
    retries: int = 3,  # TODO
    window_time: int = None,  # TODO
):
    logger.debug("Insights Worker Starting Requests Functions")

    async def _start_request_worker():
        async with aiohttp.ClientSession() as session:
            while True:
                data = await q_in.get()
                async with session.post(worker_endpoint, json=data) as response:
                    text = await response.text()
                    q_in.task_done()

    for _ in range(parallelism):
        asyncio.create_task(_start_request_worker())

    logger.debug("Insights Worker Waiting for worker tasks")
    await asyncio.gather(*asyncio.all_tasks())


def start_insights_worker_from_async(
    worker_endpoint: str,
    parallelism: int = 1,
    batch_size: int = 1,  # TODO
    retries: int = 3,  # TODO
    window_time: int = None,  # TODO
) -> janus.Queue:

    queue = janus.Queue()

    args = (
        queue.async_q,
        worker_endpoint,
        parallelism,
        batch_size,
        retries,
        window_time,
    )
    logger.debug(f"Insights Worker starting insights worker from ASYNC with params {args}")

    asyncio.create_task(start_worker(*args))

    return queue.async_q


def sync_init_loop_queue(
    event,
    worker_endpoint,
    parallelism,
    batch_size,
    retries,
    window_time,
):
    async def inner_loop():
        event.queue = janus.Queue()
        event.set()

        await start_worker(
            event.queue.async_q,
            worker_endpoint,
            parallelism,
            batch_size,
            retries,
            window_time,
        )

    asyncio.run(inner_loop())


def start_insights_worker_from_sync(
    worker_endpoint: str = "",
    batch_size: int = 1,
    parallelism: int = 1,
    retries: int = 3,
    output_file_path: str = None,
    window_time: int = None,
) -> janus.Queue:

    event = threading.Event()
    args = (
        event,
        worker_endpoint,
        parallelism,
        batch_size,
        retries,
        window_time,
    )
    logger.debug(f"Insights Worker starting insights worker from sync with params {args}")
    thread = threading.Thread(target=sync_init_loop_queue, args=args)
    thread.start()
    event.wait()

    queue = event.queue  # pylint: disable=no-member

    logger.debug("Insights Worker successful creation worker from sync")

    return queue.sync_q
