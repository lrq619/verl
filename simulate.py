#!/usr/bin/env python3
"""Simulate the vLLM-Omni launch path used by verl FlowGRPO.

This script reproduces the same parser/validation path used in
`verl/workers/rollout/vllm_rollout/vllm_omni_async_server.py`.

By default it only validates CLI args via the vllm_omni parser.
Optionally it can initialize AsyncOmni or run the HTTP server.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import asyncio
import inspect
import json
import shlex
import sys
from typing import Any

import vllm_omni.entrypoints.cli.serve
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm_omni.engine.arg_utils import AsyncOmniEngineArgs
from vllm_omni.entrypoints import AsyncOmni
from vllm_omni.entrypoints.openai.api_server import build_app, omni_init_app_state

from verl.workers.rollout.utils import run_unvicorn
from verl.workers.rollout.vllm_rollout.utils import build_cli_args_from_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulate verl vLLM-Omni launch path")

    p.add_argument("--model", required=True, help="Model path or HF id")
    p.add_argument("--host", default="127.0.0.1", help="Server host when --serve is enabled")

    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--load-format", default="safetensors")
    p.add_argument("--max-model-len", type=int, default=1058)
    p.add_argument("--max-num-seqs", type=int, default=1024)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.2)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scheduling-policy", default="fcfs")
    p.add_argument("--compilation-config", default='{"cudagraph_mode":"NONE"}')

    # These are the fields you asked about.
    p.add_argument("--stage-init-timeout", type=int, default=None)
    p.add_argument("--init-timeout", type=int, default=None)

    # Extra engine kwargs to merge into args (same as **engine_kwargs in verl).
    p.add_argument(
        "--engine-kwargs-json",
        default="{}",
        help="JSON dict merged into launch args, e.g. '{\"stage_init_timeout\":900}'",
    )

    p.add_argument(
        "--mode",
        choices=["validate", "init", "serve"],
        default="validate",
        help="validate: parse/validate only; init: create AsyncOmni; serve: run API server",
    )

    return p.parse_args()


def _build_verl_like_args(args: argparse.Namespace) -> list[str]:
    try:
        engine_kwargs = json.loads(args.engine_kwargs_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid --engine-kwargs-json: {e}") from e

    if not isinstance(engine_kwargs, dict):
        raise ValueError("--engine-kwargs-json must decode to a JSON object")

    config: dict[str, Any] = {
        "dtype": args.dtype,
        "load_format": args.load_format,
        "skip_tokenizer_init": False,
        "distributed_executor_backend": "mp",
        "worker_extension_cls": "verl.workers.rollout.vllm_rollout.utils.vLLMOmniColocateWorkerExtension",
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_prefix_caching": True,
        "enable_sleep_mode": True,
        "logprobs_mode": "processed_logprobs",
        "enforce_eager": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "disable_log_stats": True,
        "tensor_parallel_size": args.tensor_parallel_size,
        "seed": args.seed,
        "override_generation_config": "{}",
        "hf_overrides": {},
        "scheduling_policy": args.scheduling_policy,
        "compilation_config": args.compilation_config,
    }

    if args.stage_init_timeout is not None:
        config["stage_init_timeout"] = args.stage_init_timeout
    if args.init_timeout is not None:
        config["init_timeout"] = args.init_timeout

    # Exactly like verl's `args = {..., **engine_kwargs}`.
    config.update(engine_kwargs)

    return ["serve", args.model] + build_cli_args_from_config(config)


def _parse_and_validate_with_vllm_omni(cli_args: list[str]) -> argparse.Namespace:
    cmd_modules = [vllm_omni.entrypoints.cli.serve]
    parser = FlexibleArgumentParser(description="vLLM-Omni CLI")
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    cmds = {}

    for module in cmd_modules:
        for cmd in module.cmd_init():
            cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
            cmds[cmd.name] = cmd

    ns = parser.parse_args(args=cli_args)
    ns.model = ns.model_tag

    if ns.subparser in cmds:
        cmds[ns.subparser].validate(ns)

    return ns


async def _init_or_serve(ns: argparse.Namespace, mode: str, host: str) -> None:
    engine_args = AsyncOmniEngineArgs.from_cli_args(ns)
    engine_args = asdict(engine_args)
    engine_client = AsyncOmni(**engine_args)

    print("[OK] AsyncOmni initialized")

    if mode != "serve":
        return

    app = build_app(ns)
    if len(inspect.signature(omni_init_app_state).parameters) >= 4:
        await omni_init_app_state(engine_client, None, app.state, ns)
    else:
        await omni_init_app_state(engine_client, app.state, ns)

    port, _ = await run_unvicorn(app, ns, host)
    print(f"[OK] Serving at http://{host}:{port}")
    await asyncio.Event().wait()


def main() -> None:
    args = parse_args()
    cli_args = _build_verl_like_args(args)

    print("[verl-like cli args list]")
    print(cli_args)
    print("\n[equivalent command string]")
    print("python -m vllm_omni.entrypoints.cli.serve " + shlex.join(cli_args))

    try:
        ns = _parse_and_validate_with_vllm_omni(cli_args)
    except SystemExit as e:
        print(
            "\n[PARSE ERROR] vllm_omni parser rejected args. "
            "This is exactly what verl would hit on this package version.",
            file=sys.stderr,
        )
        raise e

    print("\n[OK] vllm_omni parser accepted args")

    if args.mode in {"init", "serve"}:
        asyncio.run(_init_or_serve(ns, args.mode, args.host))


if __name__ == "__main__":
    main()
