#!/usr/bin/env python3
# Copyright 2026

import argparse
import asyncio
import base64
import glob
import json
import os
import re
from io import BytesIO
from pathlib import Path
from urllib import request

import numpy as np
import ray
import torch
from hydra import compose, initialize_config_dir
from PIL import Image

from verl.experimental.reward_loop import RewardLoopManager
from verl.protocol import DataProto
from verl.utils import hf_tokenizer

DEFAULT_GROUND_TRUTHS = [
    "Page 666",
    "System Override Active",
    "Photosynthesis Process",
    "First Steps",
    "Take With Food",
]


def _pil_image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image;base64,{encoded}"


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    if len(a) > len(b):
        a, b = b, a

    prev = list(range(len(a) + 1))
    for i, char_b in enumerate(b, start=1):
        cur = [i]
        for j, char_a in enumerate(a, start=1):
            insert_cost = cur[j - 1] + 1
            delete_cost = prev[j] + 1
            replace_cost = prev[j - 1] + (char_a != char_b)
            cur.append(min(insert_cost, delete_cost, replace_cost))
        prev = cur
    return prev[-1]


def _chat_complete_blocking(router_address: str, chat_complete_request: dict) -> dict:
    url = f"http://{router_address}/v1/chat/completions"
    payload = json.dumps(chat_complete_request).encode("utf-8")
    req = request.Request(url=url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


async def _chat_complete(router_address: str, chat_complete_request: dict) -> dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _chat_complete_blocking, router_address, chat_complete_request)


async def compute_score_ocr_simple(
    data_source: str,
    solution_image: Image.Image | np.ndarray | torch.Tensor,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: str,
    reward_model_tokenizer=None,
    model_name: str | None = None,
):
    del data_source, extra_info, reward_model_tokenizer

    image = solution_image
    if isinstance(image, torch.Tensor):
        image = image.float().permute(1, 2, 0).cpu().numpy()
    if isinstance(image, np.ndarray):
        assert image.shape[-1] == 3, "Image must be HWC RGB."
        image = (image * 255).round().clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)
    assert isinstance(image, Image.Image)

    image_base64 = _pil_image_to_base64(image)
    query = [
        {"type": "image_url", "image_url": {"url": image_base64}},
        {
            "type": "text",
            "text": "Please output only the text content from the image without any additional descriptions.",
        },
    ]
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query},
    ]
    chat_complete_request = {
        "messages": messages,
        "model": model_name,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 512,
    }
    result = await _chat_complete(router_address=reward_router_address, chat_complete_request=chat_complete_request)
    text = result["choices"][0]["message"]["content"]

    gt = re.sub(r"\s+", "", ground_truth).lower()
    pred = re.sub(r"\s+", "", text).lower()
    if gt in pred:
        dist = 0
    else:
        dist = _levenshtein_distance(pred, gt)
    dist = min(dist, len(gt))
    score = 1.0 - (dist / len(gt))

    return {"score": float(score), "acc": float(score == 1.0), "genrm_response": text}


def _extract_num_suffix(path: str) -> tuple[int, str]:
    name = Path(path).stem
    match = re.search(r"(\d+)$", name)
    if match is None:
        return (10**9, path)
    return (int(match.group(1)), path)


def _build_data(tokenizer, image_paths: list[str], ground_truths: list[str], data_source: str) -> DataProto:
    pil_images = [np.array(Image.open(path).convert("RGB")) for path in image_paths]
    responses = [torch.tensor(img).permute(2, 0, 1).float() / 255.0 for img in pil_images]
    responses = torch.stack(responses)

    prompt_length = 128
    prompt_ids = []
    pad_token_id = tokenizer.pad_token_id
    for gt in ground_truths:
        prompt_tokens = tokenizer.encode(f'Please read the image text: "{gt}"')
        prompt_tokens = prompt_tokens[-prompt_length:]
        padded_prompt = [pad_token_id] * (prompt_length - len(prompt_tokens)) + prompt_tokens
        prompt_ids.append(torch.tensor(padded_prompt))
    prompt_ids = torch.stack(prompt_ids)

    reward_info = [{"ground_truth": gt} for gt in ground_truths]
    extra_info = [{} for _ in ground_truths]
    data_sources = [data_source for _ in ground_truths]

    return DataProto.from_dict(
        tensors={
            "input_ids": prompt_ids,
            "responses": responses,
        },
        non_tensors={
            "data_source": data_sources,
            "reward_model": reward_info,
            "extra_info": extra_info,
        },
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Start flowgrpo reward worker and score OCR images.")
    parser.add_argument(
        "--model-path",
        default="/tmp/models/Qwen/Qwen3-VL-8B-Instruct",
        help="Reward model path.",
    )
    parser.add_argument(
        "--image-glob",
        default="~/proj/ROLL/output/response_*.png",
        help="Glob for the 5 generated response images.",
    )
    parser.add_argument(
        "--data-source",
        default="ocr",
        help="Data source string passed to reward function.",
    )
    parser.add_argument(
        "--output",
        default="./reward.json",
        help="Output json path.",
    )
    parser.add_argument(
        "--rollout-name",
        default=os.getenv("ROLLOUT_NAME", "vllm"),
        choices=["vllm", "sglang"],
        help="Reward model rollout backend.",
    )
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size for reward model.")
    parser.add_argument(
        "--n-gpus-per-node",
        type=int,
        default=1,
        help="Number of GPUs on node available to reward model workers.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = os.path.expanduser(args.model_path)
    image_glob = os.path.expanduser(args.image_glob)
    image_paths = sorted(glob.glob(image_glob), key=_extract_num_suffix)

    if len(image_paths) != len(DEFAULT_GROUND_TRUTHS):
        raise ValueError(
            f"Expected {len(DEFAULT_GROUND_TRUTHS)} images by pattern {image_glob}, but found {len(image_paths)}: "
            f"{image_paths}"
        )

    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            }
        }
    )

    try:
        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
            config = compose(config_name="ppo_trainer")

        config.actor_rollout_ref.model.path = model_path
        config.actor_rollout_ref.model.tokenizer_path = model_path
        config.reward.custom_reward_function.path = os.path.abspath(__file__)
        config.reward.custom_reward_function.name = "compute_score_ocr_simple"
        config.reward.num_workers = 1
        config.reward.reward_manager.name = "image"
        config.reward.reward_model.enable = True
        config.reward.reward_model.enable_resource_pool = True
        config.reward.reward_model.n_gpus_per_node = args.n_gpus_per_node
        config.reward.reward_model.nnodes = 1
        config.reward.reward_model.model_path = model_path
        config.reward.reward_model.rollout.name = args.rollout_name
        config.reward.reward_model.rollout.gpu_memory_utilization = 0.9
        config.reward.reward_model.rollout.tensor_model_parallel_size = args.tp_size
        config.reward.reward_model.rollout.skip_tokenizer_init = False
        config.reward.reward_model.rollout.prompt_length = 2048
        config.reward.reward_model.rollout.response_length = 1024

        tokenizer = hf_tokenizer(config.actor_rollout_ref.model.tokenizer_path)
        data = _build_data(tokenizer, image_paths=image_paths, ground_truths=DEFAULT_GROUND_TRUTHS, data_source=args.data_source)

        reward_loop_manager = RewardLoopManager(config)
        outputs = reward_loop_manager.compute_rm_score(data)
        scores = outputs.batch["rm_scores"].squeeze(-1).tolist()
        identified_words = outputs.non_tensor_batch.get(
            "genrm_response",
            np.array([""] * len(DEFAULT_GROUND_TRUTHS), dtype=object),
        ).tolist()

        result = {
            gt: {
                "reward": float(score),
                "identified word": identified_word,
            }
            for gt, score, identified_word in zip(DEFAULT_GROUND_TRUTHS, scores, identified_words, strict=True)
        }
        output_path = Path(args.output).expanduser().resolve()
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"Saved reward json to: {output_path}")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
