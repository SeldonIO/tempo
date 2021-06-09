
import requests
import os
import json
import asyncio
import aiohttp
import aiofiles
import janus
import time
import threading

async def start_worker(
    q_in: janus.Queue,
    worker_endpoint: str = "",
    batch_size: int = 1, # TODO
    parallelism: int = 1,
    retries: int = 3, # TODO
    output_file_path: str = None,
    window_time: int = None,
):
    if not worker_endpoint and not output_file_path:
        raise Exception("One of worker_endpoint or file_path needs to be provided")

    q_request = None
    q_file = None

    if worker_endpoint:
        q_request = asyncio.Queue()
        async def _start_request_worker():
            async with aiohttp.ClientSession() as session:
                while True:
                    data = await q_request.get()
                    async with  session.post(worker_endpoint, json=data) as response:
                        text = await response.text()
                        q_in.task_done()
        for _ in range(parallelism):
            asyncio.create_task(_start_request_worker())

    if output_file_path:
        q_file = asyncio.Queue()
        async def _start_file_worker():
            async with aiofiles.open(output_file_path, "w", buffering=1) as output_data_file:
                while True:
                    line = await q_file.get()
                    await output_data_file.writelines(f"{line}\n")
                    q_file.task_done()
        asyncio.create_task(_start_file_worker())

    async def _start_queues_worker():
        while True:
            data = await q_in.get()
            if output_file_path:
                await q_file.put(data)
            if worker_endpoint:
                await q_request.put(data)

    asyncio.create_task(_start_queues_worker())

    await asyncio.gather(*asyncio.all_tasks())

async def start_insights_worker_from_async(
    worker_endpoint: str = "http://localhost:3333/",
    batch_size: int = 1,
    parallelism: int = 1,
    retries: int = 3,
    output_file_path: str = None,
    window_time: int = None,
) -> janus.Queue:

    loop = asyncio.get_event_loop()

    queue = janus.Queue(loop_override=loop)

    loop.create_task(
        start_worker(
            queue,
            worker_endpoint,
            batch_size,
            parallelism,
            retries,
            output_file_path,
            window_time,
        )
    )
    return queue.async_q

def sync_init_loop_queue(
        event,
        worker_endpoint,
        batch_size,
        parallelism,
        retries,
        output_file_path,
        window_time,
):
    async def inner_loop():
        event.queue = janus.Queue()
        event.set()

        await start_worker(
            event.queue.async_q,
            worker_endpoint,
            batch_size,
            parallelism,
            retries,
            output_file_path,
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


    print("creating threading")
    event = threading.Event()
    args = (event, worker_endpoint, batch_size, parallelism,
            retries, output_file_path, window_time)
    thread = threading.Thread(target=sync_init_loop_queue, args=args)
    thread.start()
    event.wait()
    print("waiting for threading")

    queue = event.queue # pylint: disable=no-member

    return queue.sync_q

if __name__ == "__main__":
    print("creating insights worker")
    q = start_insights_worker_from_sync(output_file_path="out.txt")
    while True:
        time.sleep(1)
        print("adding")
        q.put({"hello": "world"})

