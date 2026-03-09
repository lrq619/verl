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
import logging
import os
import time

import aiohttp
from verl.single_controller.ray.base import RayResourcePool, split_resource_pool
from verl.workers.config import HFModelConfig, RewardModelConfig
from verl.workers.rollout.replica import get_rollout_replica_class

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class RewardModelManager:
    """Reward model manager."""

    def __init__(
        self,
        config: RewardModelConfig,
        resource_pool: RayResourcePool = None,
    ):
        """
        Initialize the reward model manager.

        Args:
            config (RewardModelConfig): Reward model configuration.
            resource_pool (RayResourcePool, optional): Resource pool. Defaults to None.
        """
        self.config = config
        self.resource_pool = resource_pool
        logger.info("=== [RM_STARTUP] BEGIN RewardModelManager init ===")
        self._initialize_llm_servers()
        self._initialize_router()
        self._verify_servers_ready()
        assert self.config.rollout.skip_tokenizer_init is False, "Reward model should not skip tokenizer init."
        if self.config.rollout.free_cache_engine:
            self.sleep()
        logger.info("=== [RM_STARTUP] END RewardModelManager init ===")

    def _initialize_llm_servers(self):
        logger.info("=== [RM_STARTUP] Step 1/3: launching reward model rollout replicas ===")
        rollout_world_size = self.config.rollout.tensor_model_parallel_size
        world_size = (
            self.resource_pool.world_size
            if self.resource_pool  # colocate mode
            else self.config.n_gpus_per_node * self.config.nnodes  # standalone mode
        )
        num_replicas = world_size // rollout_world_size
        logger.info(
            "Initializing reward model servers: model_path=%s rollout_name=%s world_size=%d tp=%d num_replicas=%d "
            "resource_pool=%s",
            self.config.model_path,
            self.config.rollout.name,
            world_size,
            rollout_world_size,
            num_replicas,
            self.resource_pool is not None,
        )

        rollout_replica_class = get_rollout_replica_class(self.config.rollout.name)
        rollout_config = self.config.rollout
        model_config = HFModelConfig(path=self.config.model_path)
        self.tokenizer = model_config.get_processor()
        self.rollout_replicas = [
            rollout_replica_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=self.config.n_gpus_per_node,
                is_reward_model=True,
            )
            for replica_rank in range(num_replicas)
        ]
        if self.resource_pool:
            split_resource_pools = split_resource_pool(self.resource_pool, split_size=rollout_world_size)
            assert len(split_resource_pools) == len(self.rollout_replicas)
            self._run_all(
                [
                    server.init_colocated(resource_pool)
                    for server, resource_pool in zip(self.rollout_replicas, split_resource_pools, strict=True)
                ]
            )
        else:
            self._run_all([server.init_standalone() for server in self.rollout_replicas])
        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]
        logger.info("Reward model server addresses: %s", self.server_addresses)
        logger.info("=== [RM_STARTUP] Step 1/3 done: rollout replicas launched ===")

    def _initialize_router(self):
        logger.info("=== [RM_STARTUP] Step 2/3: launching reward router ===")
        worker_urls = [f"http://{server_address}" for server_address in self.server_addresses]
        logger.info("Launching reward router with worker URLs: %s", worker_urls)

        # TODO (dyy): sglang router is not ready yet.
        # if self.config.rollout.name == "sglang":
        #     from .router.inner_sglang_router import launch_router_process
        # else:
        #     from .router.naive_router import launch_router_process

        from .router.naive_router import launch_router_process

        self.router_address, _ = launch_router_process(worker_urls=worker_urls)
        logger.info("Reward router launched at %s", self.router_address)
        logger.info("=== [RM_STARTUP] Step 2/3 done: reward router launched ===")

    def _verify_servers_ready(self):
        """Probe reward workers and router with simple HTTP GET /health requests."""
        enabled = os.getenv("VERL_RM_STARTUP_PROBE", "1").lower() not in {"0", "false", "off"}
        if not enabled:
            logger.info("=== [RM_STARTUP] Step 3/3 skipped: probe disabled by VERL_RM_STARTUP_PROBE ===")
            return

        timeout_s = float(os.getenv("VERL_RM_STARTUP_PROBE_TIMEOUT_S", "120"))
        interval_s = float(os.getenv("VERL_RM_STARTUP_PROBE_INTERVAL_S", "2"))
        logger.info(
            "=== [RM_STARTUP] Step 3/3: probing HTTP readiness (timeout=%.1fs interval=%.1fs) ===",
            timeout_s,
            interval_s,
        )

        worker_urls = [f"http://{server_address}" for server_address in self.server_addresses]
        router_url = f"http://{self.router_address}"
        asyncio.run(
            self._verify_servers_ready_async(
                worker_urls=worker_urls,
                router_url=router_url,
                timeout_s=timeout_s,
                interval_s=interval_s,
            )
        )
        logger.info("=== [RM_STARTUP] Step 3/3 done: HTTP probe passed for all workers and router ===")

    async def _verify_servers_ready_async(
        self,
        worker_urls: list[str],
        router_url: str,
        timeout_s: float,
        interval_s: float,
    ):
        timeout = aiohttp.ClientTimeout(total=min(timeout_s, 10.0), connect=min(timeout_s, 5.0))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for idx, url in enumerate(worker_urls):
                await self._probe_health_with_retry(
                    session=session,
                    base_url=url,
                    server_name=f"reward_worker[{idx}]",
                    timeout_s=timeout_s,
                    interval_s=interval_s,
                )
            await self._probe_health_with_retry(
                session=session,
                base_url=router_url,
                server_name="reward_router",
                timeout_s=timeout_s,
                interval_s=interval_s,
            )

    async def _probe_health_with_retry(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        server_name: str,
        timeout_s: float,
        interval_s: float,
    ):
        health_url = f"{base_url}/health"
        deadline = time.monotonic() + timeout_s
        attempt = 0
        last_err = "unknown"
        while time.monotonic() < deadline:
            attempt += 1
            try:
                async with session.get(health_url) as resp:
                    body = await resp.text()
                    if resp.status == 200:
                        logger.info(
                            "[RM_STARTUP][probe] %s is ready: url=%s status=%s attempt=%d",
                            server_name,
                            health_url,
                            resp.status,
                            attempt,
                        )
                        return

                    # Some servers may not expose /health but still indicate process is alive.
                    if resp.status in (404, 405):
                        logger.warning(
                            "[RM_STARTUP][probe] %s returned status=%s on /health; treating as reachable: url=%s "
                            "attempt=%d",
                            server_name,
                            resp.status,
                            health_url,
                            attempt,
                        )
                        return

                    preview = body[:200].replace("\n", "\\n")
                    last_err = f"status={resp.status} body_preview={preview}"
            except Exception as e:  # noqa: BLE001
                last_err = repr(e)

            logger.warning(
                "[RM_STARTUP][probe] %s not ready yet: url=%s attempt=%d last_err=%s",
                server_name,
                health_url,
                attempt,
                last_err,
            )
            await asyncio.sleep(interval_s)

        raise RuntimeError(
            f"[RM_STARTUP][probe] {server_name} failed readiness check: url={health_url}, "
            f"timeout_s={timeout_s}, last_err={last_err}"
        )

    def get_router_address(self):
        return self.router_address

    def wake_up(self):
        """Wake up all rollout replica instances."""
        self._run_all([replica.wake_up() for replica in self.rollout_replicas])

    def sleep(self):
        """Sleep all rollout replica instances."""
        self._run_all([replica.sleep() for replica in self.rollout_replicas])

    def _run_all(self, tasks: list[asyncio.Task]):
        async def run_all():
            await asyncio.gather(*tasks)

        asyncio.run(run_all())
