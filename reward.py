#!/usr/bin/env python3
"""Launch verl reward model stack once and probe endpoints.

This script uses the same startup path as training:
`RewardModelManager -> rollout replicas -> naive_router`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate verl RewardModelManager startup (real RM + router launch)."
    )
    parser.add_argument("--model-path", required=True, help="Reward model path used by reward.reward_model.model_path")
    parser.add_argument("--rollout-name", default="vllm", choices=["vllm", "sglang", "vllm_omni", "trtllm"])
    parser.add_argument("--tp", type=int, default=1, help="reward.reward_model.rollout.tensor_model_parallel_size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--n-gpus-per-node", type=int, default=None)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=1024)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--load-format", default="auto")
    parser.add_argument(
        "--probe-kind",
        default="health",
        choices=["health", "classify", "chat", "none"],
        help="Endpoint probe after launch. Use chat for GenRM-style OpenAI probing.",
    )
    parser.add_argument("--probe-timeout", type=float, default=20.0)
    parser.add_argument("--chat-max-tokens", type=int, default=8)
    parser.add_argument(
        "--hold-seconds",
        type=int,
        default=30,
        help="Keep servers alive for this many seconds. Set 0 to exit immediately, -1 to wait forever.",
    )
    parser.add_argument("--ray-address", default=None, help="Optional Ray cluster address. Default starts local Ray.")
    return parser.parse_args()


def infer_visible_gpus() -> int | None:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cuda_visible:
        return None
    entries = [x.strip() for x in cuda_visible.split(",") if x.strip()]
    return len(entries) if entries else None


def build_config(args: argparse.Namespace):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.abspath(REPO_ROOT / "verl/trainer/config")):
        config = compose(config_name="ppo_trainer")

    inferred_gpus = infer_visible_gpus()
    n_gpus_per_node = args.n_gpus_per_node if args.n_gpus_per_node is not None else inferred_gpus or 8

    config.reward.reward_model.enable = True
    config.reward.reward_model.enable_resource_pool = False
    config.reward.reward_model.model_path = args.model_path
    config.reward.reward_model.n_gpus_per_node = n_gpus_per_node
    config.reward.reward_model.nnodes = args.nnodes
    config.reward.reward_model.rollout.name = args.rollout_name
    config.reward.reward_model.rollout.dtype = args.dtype
    config.reward.reward_model.rollout.gpu_memory_utilization = args.gpu_memory_utilization
    config.reward.reward_model.rollout.tensor_model_parallel_size = args.tp
    config.reward.reward_model.rollout.max_num_seqs = args.max_num_seqs
    config.reward.reward_model.rollout.max_num_batched_tokens = args.max_num_batched_tokens
    config.reward.reward_model.rollout.load_format = args.load_format
    config.reward.reward_model.rollout.skip_tokenizer_init = False
    if args.max_model_len is not None:
        config.reward.reward_model.rollout.max_model_len = args.max_model_len

    return config, n_gpus_per_node


def http_json_request(
    url: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 20.0,
) -> tuple[int, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url=url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body) if body else {}
        except json.JSONDecodeError:
            parsed = body
        return resp.status, parsed


def probe_one(
    base_url: str,
    probe_kind: str,
    model_path: str,
    timeout: float,
    chat_max_tokens: int,
) -> None:
    if probe_kind == "none":
        return
    if probe_kind == "health":
        endpoint = "/health"
        method = "GET"
        payload = None
    elif probe_kind == "classify":
        endpoint = "/classify"
        method = "POST"
        payload = {
            "model": model_path,
            "input": "hello",
            "use_activation": False,
        }
    else:
        endpoint = "/v1/chat/completions"
        method = "POST"
        payload = {
            "model": model_path,
            "messages": [{"role": "user", "content": "Say ok"}],
            "max_tokens": chat_max_tokens,
            "temperature": 0.0,
        }

    url = f"{base_url.rstrip('/')}{endpoint}"
    status, body = http_json_request(url=url, method=method, payload=payload, timeout=timeout)
    print(f"[probe:{probe_kind}] {url} -> HTTP {status}")
    if isinstance(body, dict):
        keys = sorted(body.keys())
        print(f"[probe:{probe_kind}] response keys: {keys}")
    else:
        preview = str(body)
        if len(preview) > 300:
            preview = preview[:300] + "...(truncated)"
        print(f"[probe:{probe_kind}] response: {preview}")


def main() -> None:
    args = parse_args()

    import ray

    from verl.experimental.reward_loop.reward_model import RewardModelManager

    if args.tp <= 0:
        raise ValueError("--tp must be > 0")

    if not ray.is_initialized():
        ray.init(address=args.ray_address)

    config, n_gpus_per_node = build_config(args)
    world_size = n_gpus_per_node * args.nnodes
    expected_replicas = world_size // args.tp
    if world_size % args.tp != 0:
        print(
            f"[warn] world_size={world_size} is not divisible by tp={args.tp}; "
            "RewardModelManager computes replicas using floor division."
        )

    print("[launch] RewardModelManager starting")
    print(f"[launch] model_path={args.model_path}")
    print(f"[launch] rollout={args.rollout_name} tp={args.tp} world_size={world_size}")
    print(f"[launch] expected_replicas={expected_replicas} (world_size // tp)")

    manager = RewardModelManager(config.reward.reward_model)

    worker_urls = [f"http://{addr}" for addr in manager.server_addresses]
    router_url = f"http://{manager.get_router_address()}"
    print(f"[launch] actual_replicas={len(worker_urls)}")
    for idx, url in enumerate(worker_urls):
        print(f"[worker {idx}] {url}")
    print(f"[router] {router_url}")

    if args.probe_kind != "none":
        print(f"[probe] kind={args.probe_kind}")
        for idx, url in enumerate(worker_urls):
            try:
                probe_one(url, args.probe_kind, args.model_path, args.probe_timeout, args.chat_max_tokens)
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                print(f"[probe:{args.probe_kind}] worker[{idx}] HTTPError {e.code}: {body}")
            except Exception as e:  # noqa: BLE001
                print(f"[probe:{args.probe_kind}] worker[{idx}] failed: {type(e).__name__}: {e}")

        try:
            probe_one(router_url, args.probe_kind, args.model_path, args.probe_timeout, args.chat_max_tokens)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            print(f"[probe:{args.probe_kind}] router HTTPError {e.code}: {body}")
        except Exception as e:  # noqa: BLE001
            print(f"[probe:{args.probe_kind}] router failed: {type(e).__name__}: {e}")

    if args.hold_seconds == -1:
        print("[hold] waiting forever; Ctrl+C to exit")
        while True:
            time.sleep(3600)
    elif args.hold_seconds > 0:
        print(f"[hold] sleeping for {args.hold_seconds}s")
        time.sleep(args.hold_seconds)

    print("[exit] shutting down Ray")
    ray.shutdown()


if __name__ == "__main__":
    main()
