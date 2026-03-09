# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import contextlib
import inspect
import logging
import os
import time

import uvicorn
from fastapi import FastAPI

from verl.utils.net_utils import get_free_port

logger = logging.getLogger(__file__)


def get_max_position_embeddings(hf_config) -> int:
    max_len = getattr(hf_config, "max_position_embeddings", None)
    if max_len is None:
        text_config = getattr(hf_config, "text_config", None)
        if text_config is not None:
            max_len = getattr(text_config, "max_position_embeddings", None)

    if max_len is None:
        raise ValueError("max_position_embeddings not found in HFModelConfig!")
    return int(max_len)


async def run_unvicorn(
    app: FastAPI,
    server_args,
    server_address,
    max_retries=5,
    startup_timeout_s: float = 30.0,
) -> tuple[int, asyncio.Task]:
    """Start uvicorn and only return after the server is actually listening."""
    for attempt in range(1, max_retries + 1):
        server_port = None
        sock = None
        server = None
        server_task = None
        started = False

        try:
            # Reserve a free port first to avoid races in multi-actor startup.
            server_port, sock = get_free_port(server_address)
            app.server_args = server_args
            config = uvicorn.Config(app, host=server_address, port=server_port, log_level="warning")
            server = uvicorn.Server(config)

            serve_sig = inspect.signature(server.serve)
            if "sockets" in serve_sig.parameters:
                server_task = asyncio.create_task(server.serve(sockets=[sock]))
            else:
                sock.close()
                sock = None
                server_task = asyncio.create_task(server.serve())

            deadline = time.monotonic() + startup_timeout_s
            while time.monotonic() < deadline:
                if getattr(server, "started", False):
                    started = True
                    logger.info(f"HTTP server started on port {server_port}")
                    return server_port, server_task

                if server_task.done():
                    exc = server_task.exception()
                    if exc is None:
                        raise RuntimeError("Uvicorn server exited before startup.")
                    raise exc

                await asyncio.sleep(0.05)

            raise TimeoutError(f"Uvicorn did not report started=True within {startup_timeout_s}s.")
        except Exception as e:  # noqa: BLE001
            logger.error(
                "Failed to start HTTP server on port %s at try %d/%d, error: %r",
                server_port,
                attempt,
                max_retries,
                e,
            )
            if server is not None:
                server.should_exit = True
            if server_task is not None and not server_task.done():
                server_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await server_task
        finally:
            # Keep the socket open only on success; otherwise release it.
            if not started and sock is not None:
                with contextlib.suppress(OSError):
                    sock.close()

    logger.error(f"Failed to start HTTP server after {max_retries} retries, exiting...")
    os._exit(-1)


async def ensure_async_iterator(iterable):
    """Convert an iterable to an async iterator."""
    if hasattr(iterable, "__aiter__"):
        async for item in iterable:
            yield item
    else:
        for item in iterable:
            yield item
