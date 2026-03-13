# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.ray_diffusion_trainer import RayFlowGRPOTrainer


class DummyTokenizer:
    def __init__(self, prompts: list[str]) -> None:
        self.prompts = prompts

    def batch_decode(self, token_ids, skip_special_tokens: bool = True) -> list[str]:
        return self.prompts[: len(token_ids)]


class TestFlowGRPOLowRewardDump(unittest.TestCase):
    def test_dump_low_reward_rollout_samples_respects_threshold_and_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = RayFlowGRPOTrainer.__new__(RayFlowGRPOTrainer)
            trainer.config = OmegaConf.create(
                {
                    "trainer": {
                        "low_reward_rollout_dump": {
                            "enabled": True,
                            "base_dir": tmpdir,
                            "score_threshold": 0.05,
                            "max_images_per_step": 1,
                        }
                    }
                }
            )
            trainer.tokenizer = DummyTokenizer(["prompt_0", "prompt_1", "prompt_2"])
            trainer.global_steps = 7
            trainer._low_reward_rollout_dump_root = f"{tmpdir}/20260313_120000"

            batch = SimpleNamespace(
                batch={
                    "prompts": torch.ones((3, 1), dtype=torch.long),
                    "responses": torch.tensor(
                        [
                            [[[0.1, 0.1], [0.1, 0.1]]] * 3,
                            [[[0.5, 0.5], [0.5, 0.5]]] * 3,
                            [[[0.9, 0.9], [0.9, 0.9]]] * 3,
                        ],
                        dtype=torch.float32,
                    ),
                },
                non_tensor_batch={"uid": np.array(["uid_0", "uid_1", "uid_2"], dtype=object)},
            )
            reward_tensor = torch.tensor([[0.03], [0.07], [0.01]], dtype=torch.float32)

            trainer._maybe_dump_low_reward_rollout_samples(batch, reward_tensor, timing_raw={})

            step_dir = trainer._low_reward_rollout_dump_root + "/step_7"
            metadata_path = step_dir + "/metadata.jsonl"

            self.assertTrue(os.path.isdir(step_dir))
            self.assertTrue(os.path.isfile(metadata_path))

            image_paths = sorted(path for path in os.listdir(step_dir) if path.endswith(".jpg"))
            self.assertEqual(image_paths, ["000_sample_0002.jpg"])

            with open(metadata_path) as f:
                metadata_lines = f.read().strip().splitlines()

            self.assertEqual(len(metadata_lines), 1)

            metadata = json.loads(metadata_lines[0])
            self.assertEqual(metadata["step"], 7)
            self.assertEqual(metadata["batch_index"], 2)
            self.assertEqual(metadata["image"], "000_sample_0002.jpg")
            self.assertEqual(metadata["prompt"], "prompt_2")
            self.assertAlmostEqual(metadata["score"], 0.01, places=6)
            self.assertEqual(metadata["uid"], "uid_2")


if __name__ == "__main__":
    unittest.main()
