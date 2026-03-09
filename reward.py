#!/usr/bin/env python3
"""Simulate reward-model vLLM launch path used by verl.

This reproduces how verl builds vLLM CLI args for reward-model rollout
(`vLLMHttpServer.launch_server`), prints the exact arg list, validates it with
the same vLLM parser, and can optionally run real `serve`.
"""

from __future__ import annotations

import argparse
import json
import pprint
import shlex
import sys
from typing import Any


def _parse_json_dict(raw: str, name: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON for {name}: {e}") from e
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object, got: {type(value).__name__}")
    return value


def build_cli_args_from_config(config: dict[str, Any]) -> list[str]:
    """Convert a config dictionary to vLLM CLI args."""
    cli_args: list[str] = []
    for key, value in config.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                cli_args.append(f"--{key}")
            continue
        if isinstance(value, list):
            if not value:
                continue
            cli_args.append(f"--{key}")
            cli_args.extend([str(item) for item in value])
            continue
        cli_args.append(f"--{key}")
        cli_args.append(json.dumps(value) if isinstance(value, dict) else str(value))
    return cli_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate verl reward-model vLLM launch args, validate, and optionally serve."
    )
    parser.add_argument("--model", required=True, help="Reward model path/id.")

    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--load-format", default="auto")
    parser.add_argument("--max-model-len", type=int, default=128000)
    parser.add_argument("--max-num-seqs", type=int, default=1024)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scheduling-policy", default="fcfs")
    parser.add_argument("--logprobs-mode", default="processed_logprobs")

    parser.add_argument("--enable-chunked-prefill", action="store_true", default=True)
    parser.add_argument("--disable-enable-chunked-prefill", action="store_true")
    parser.add_argument("--enable-prefix-caching", action="store_true", default=True)
    parser.add_argument("--disable-enable-prefix-caching", action="store_true")
    parser.add_argument("--enable-sleep-mode", action="store_true", default=True)
    parser.add_argument("--disable-enable-sleep-mode", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true", default=True)
    parser.add_argument("--disable-enforce-eager", action="store_true")
    parser.add_argument("--disable-log-stats", action="store_true", default=True)
    parser.add_argument("--enable-log-stats", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true", default=False)

    parser.add_argument(
        "--override-generation-config-json",
        default='{"temperature":1.0,"top_k":-1,"top_p":1.0,"repetition_penalty":1.0,"max_new_tokens":2048}',
        help="JSON object passed as --override_generation_config.",
    )
    parser.add_argument(
        "--hf-overrides-json",
        default="{}",
        help="JSON object passed as --hf_overrides.",
    )
    parser.add_argument(
        "--compilation-config-json",
        default='{"cudagraph_mode":"FULL_AND_PIECEWISE"}',
        help="JSON object passed as --compilation_config.",
    )
    parser.add_argument(
        "--engine-kwargs-json",
        default="{}",
        help="Extra vLLM CLI args as JSON object (merged into args last).",
    )

    parser.add_argument(
        "--mode",
        choices=["print", "validate", "serve"],
        default="validate",
        help="print: only print args; validate: parse/validate; serve: actually start vLLM serve.",
    )
    return parser.parse_args()


def _resolve_bool_switches(args: argparse.Namespace) -> dict[str, bool]:
    enable_chunked_prefill = args.enable_chunked_prefill and not args.disable_enable_chunked_prefill
    enable_prefix_caching = args.enable_prefix_caching and not args.disable_enable_prefix_caching
    enable_sleep_mode = args.enable_sleep_mode and not args.disable_enable_sleep_mode
    enforce_eager = args.enforce_eager and not args.disable_enforce_eager

    # default True unless explicitly enabled log stats
    disable_log_stats = args.disable_log_stats and not args.enable_log_stats

    return {
        "enable_chunked_prefill": enable_chunked_prefill,
        "enable_prefix_caching": enable_prefix_caching,
        "enable_sleep_mode": enable_sleep_mode,
        "enforce_eager": enforce_eager,
        "disable_log_stats": disable_log_stats,
    }


def build_reward_like_cli_args(args: argparse.Namespace) -> list[str]:
    override_generation_config = _parse_json_dict(
        args.override_generation_config_json, "--override-generation-config-json"
    )
    hf_overrides = _parse_json_dict(args.hf_overrides_json, "--hf-overrides-json")
    compilation_config = _parse_json_dict(args.compilation_config_json, "--compilation-config-json")
    engine_kwargs = _parse_json_dict(args.engine_kwargs_json, "--engine-kwargs-json")
    bools = _resolve_bool_switches(args)

    config: dict[str, Any] = {
        "dtype": args.dtype,
        "load_format": args.load_format,
        "distributed_executor_backend": "mp",
        "worker_extension_cls": "verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension",
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "enable_chunked_prefill": bools["enable_chunked_prefill"],
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_prefix_caching": bools["enable_prefix_caching"],
        "enable_sleep_mode": bools["enable_sleep_mode"],
        "logprobs_mode": args.logprobs_mode,
        "enforce_eager": bools["enforce_eager"],
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "disable_log_stats": bools["disable_log_stats"],
        "tensor_parallel_size": args.tensor_parallel_size,
        "seed": args.seed,
        "override_generation_config": override_generation_config,
        "hf_overrides": hf_overrides,
        "scheduling_policy": args.scheduling_policy,
        "compilation_config": compilation_config,
        "trust_remote_code": args.trust_remote_code,
    }

    # Keep the same behavior as verl: user-provided engine kwargs override defaults.
    config.update(engine_kwargs)
    return ["serve", args.model] + build_cli_args_from_config(config)


def parse_and_validate_with_vllm(cli_args: list[str]) -> tuple[argparse.Namespace, dict[str, Any]]:
    import vllm.entrypoints.cli.serve

    try:
        from vllm.utils.argparse_utils import FlexibleArgumentParser
    except Exception:
        from vllm.utils import FlexibleArgumentParser

    cmd_modules = [vllm.entrypoints.cli.serve]
    parser = FlexibleArgumentParser(description="vLLM CLI")
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    cmds: dict[str, Any] = {}
    for module in cmd_modules:
        new_cmds = module.cmd_init()
        for cmd in new_cmds:
            cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
            cmds[cmd.name] = cmd

    ns = parser.parse_args(args=cli_args)
    if hasattr(ns, "model_tag"):
        ns.model = ns.model_tag
    if ns.subparser in cmds:
        cmds[ns.subparser].validate(ns)
    return ns, cmds


def run_serve(ns: argparse.Namespace) -> None:
    if not hasattr(ns, "dispatch_function") or ns.dispatch_function is None:
        raise RuntimeError("No dispatch_function on parsed namespace; cannot run serve.")
    ns.dispatch_function(ns)


def main() -> None:
    args = parse_args()
    cli_args = build_reward_like_cli_args(args)

    print("[reward-like vLLM args list]")
    pprint.pprint(cli_args, width=120)
    print("\n[equivalent command]")
    print("vllm " + shlex.join(cli_args))

    if args.mode == "print":
        return

    try:
        ns, _ = parse_and_validate_with_vllm(cli_args)
    except SystemExit as e:
        print("\n[PARSE ERROR] vLLM parser rejected args. This is what verl would hit too.", file=sys.stderr)
        raise e

    print("\n[OK] vLLM parser accepted args.")
    if args.mode == "serve":
        print("[RUN] starting vLLM serve...")
        run_serve(ns)


if __name__ == "__main__":
    main()
