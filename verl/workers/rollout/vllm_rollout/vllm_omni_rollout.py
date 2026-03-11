# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""
The vllm_omni_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import asyncio
import logging
import os
import time
from typing import Any, Optional

import ray
import zmq
from torch.distributed.device_mesh import DeviceMesh

from verl.utils.device import get_device_id, is_support_ipc
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.vllm_rollout.utils import get_device_uuid
from verl.workers.rollout.vllm_rollout.vllm_rollout import ServerAdapter

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _omni_client_log(stage: str, message: str):
    logger.warning("[OMNI_CLIENT][%s] %s", stage, message)


class vLLMOmniServerAdapter(ServerAdapter):
    """
    vLLM-Omni server adapter used in native async mode, serve as a client to request vLLM-Omni server
    to resume/release/update weights and kv_cache.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super(ServerAdapter, self).__init__(config, model_config, device_mesh)
        self.server_handle: ray.actor.ActorHandle = None

        rank = int(os.environ["RANK"])
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        rollout_world_size = (
            self.config.tensor_model_parallel_size
            * self.config.data_parallel_size
            * self.config.pipeline_model_parallel_size
        )
        self.replica_rank = rank // rollout_world_size
        self.rollout_rank = rank % rollout_world_size
        self.node_rank = self.rollout_rank // local_world_size

        self.sleep_level = 1
        self.device_uuid = get_device_uuid(get_device_id())
        self.zmq_context = zmq.Context()
        self.zmq_handle = f"ipc:///tmp/rl-colocate-zmq-{self.device_uuid}.sock"

        self.use_shm = not is_support_ipc()
        if self.use_shm:
            logger.warning(
                "IPC is not supported on your devices. Falling back to shared memory for weight transfer, "
                "which may cause performance degradation. If you are using Ascend NPUs, please ensure that "
                "your software and CANN toolkit versions meet the requirements for IPC support. (Ascend HDK version "
                ">= 25.3.rc1 and CANN toolkit version >= 8.3.RC1)"
            )

    async def _execute_method(
        self,
        method: str,
        non_block: bool = False,
        timeout: Optional[float] = None,
        args: tuple = (),
        kwargs: Optional[dict] = None,
    ) -> Any:
        """Execute method on inference engine via ray.

        Args:
            method: The method name to execute on the server.
            non_block: If True, execute the method asynchronously and return immediately.
            timeout: Timeout for the collective_rpc call.
            args: Positional arguments for the method.
            kwargs: Keyword arguments for the method.

        Returns:
            The result of the method execution, or None if non_block=True.
        """
        if self.rollout_rank != 0:
            _omni_client_log(
                "_execute_method.skip",
                f"method={method} rollout_rank={self.rollout_rank} replica_rank={self.replica_rank} node_rank={self.node_rank}",
            )
            return None

        # Lazy init http server adapter because http server is launched after hybrid engine.
        if self.server_handle is None:
            actor_name = f"vllm_omni_server_{self.replica_rank}_{self.node_rank}"
            _omni_client_log("_execute_method.get_actor.START", f"method={method} actor={actor_name}")
            self.server_handle = ray.get_actor(actor_name)
            _omni_client_log("_execute_method.get_actor.END", f"method={method} actor={actor_name}")

        t0 = time.monotonic()
        _omni_client_log(
            "_execute_method.dispatch.START",
            f"method={method} non_block={non_block} timeout={timeout} args_len={len(args)} kwargs_keys={sorted((kwargs or {}).keys())}",
        )
        future = self.server_handle.collective_rpc.remote(method, timeout=timeout, args=args, kwargs=kwargs)
        _omni_client_log(
            "_execute_method.dispatch.END",
            f"method={method} non_block={non_block} elapsed={time.monotonic() - t0:.3f}s",
        )
        if non_block:
            _omni_client_log("_execute_method.return_future", f"method={method}")
            return future

        rpc_timeout = timeout if timeout is not None else float(os.getenv("VERL_VLLM_RPC_TIMEOUT_S", "1800"))
        try:
            _omni_client_log("_execute_method.await.START", f"method={method} rpc_timeout={rpc_timeout}")
            result = await asyncio.wait_for(future, timeout=rpc_timeout)
            _omni_client_log(
                "_execute_method.await.END",
                f"method={method} elapsed={time.monotonic() - t0:.3f}s result_type={type(result).__name__}",
            )
            return result
        except TimeoutError as e:
            _omni_client_log(
                "_execute_method.await.TIMEOUT",
                f"method={method} rpc_timeout={rpc_timeout} elapsed={time.monotonic() - t0:.3f}s",
            )
            raise TimeoutError(
                f"Timed out waiting for vLLM-Omni rpc method `{method}` after {rpc_timeout}s. "
                "This usually means rollout server init/wake_up is stuck."
            ) from e
