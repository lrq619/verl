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
import torch

from verl.utils.diffusers.schedulers import FlowMatchSDEDiscreteScheduler


def test_index_for_timestep_accepts_approximate_float_values() -> None:
    scheduler = FlowMatchSDEDiscreteScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(
        sigmas=np.array([1.0, 0.87321, 0.73111, 0.61234, 0.50123, 0.41234, 0.30123, 0.20123, 0.12345, 0.06123])
    )

    expected_index = 3
    timestep = scheduler.timesteps[expected_index]
    approximate_timestep = timestep.to(torch.bfloat16)

    assert scheduler.index_for_timestep(approximate_timestep) == expected_index
