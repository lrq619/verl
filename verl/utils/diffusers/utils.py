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

import numpy as np
from diffusers import SchedulerMixin

from verl.utils.device import get_device_name
from verl.workers.config import DiffusersModelConfig


def _is_qwen_image_model(model_config: DiffusersModelConfig) -> bool:
    paths = [
        getattr(model_config, "path", None),
        getattr(model_config, "local_path", None),
        getattr(model_config, "tokenizer_path", None),
        getattr(model_config, "local_tokenizer_path", None),
    ]
    for path in paths:
        if not isinstance(path, str):
            continue
        normalized = path.replace("\\", "/").lower()
        if normalized.endswith("qwen-image") or "/qwen-image/" in normalized:
            return True
        # HuggingFace local cache path style:
        # .../models--Qwen--Qwen-Image/snapshots/<sha>/
        if "models--qwen--qwen-image" in normalized:
            return True
    return False


def set_timesteps(scheduler: SchedulerMixin, model_config: DiffusersModelConfig):
    # TODO (mike): using path name is not robust, refactor later
    if _is_qwen_image_model(model_config):
        from diffusers.pipelines.qwenimage.pipeline_qwenimage import calculate_shift

        vae_scale_factor = getattr(model_config, "vae_scale_factor", 8)
        latent_height, latent_width = (
            model_config.image_height // vae_scale_factor // 2,
            model_config.image_width // vae_scale_factor // 2,
        )
        num_inference_steps = model_config.num_inference_steps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        mu = calculate_shift(
            latent_height * latent_width,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.15),
        )
        scheduler.set_timesteps(num_inference_steps, device=get_device_name(), sigmas=sigmas, mu=mu)
    else:
        raise NotImplementedError(
            f"unsupported model for custom scheduler settings: path={model_config.path}, local_path={model_config.local_path}"
        )
