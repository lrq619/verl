#!/usr/bin/env python3
"""Simulate reward-model vLLM launch path used by verl.

This reproduces how verl builds vLLM CLI args for reward-model rollout
(`vLLMHttpServer.launch_server`), prints the exact arg list, validates it with
the same vLLM parser, and can optionally run real `serve`.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import pprint
import shlex
import sys
from typing import Any

import torch


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
    parser.add_argument("--host", default="127.0.0.1", help="Server host when --mode serve is used.")

    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--load-format", default="auto")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
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
        choices=["print", "validate", "init", "serve"],
        default="validate",
        help="print: only print args; validate: parse/validate; init: initialize engine then sleep/wake; "
        "serve: initialize engine, sleep/wake, then start vLLM serve.",
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


def _format_gpu_memory_used_total_gb() -> str:
    """Return visible GPU memory usage as `used/total` in GB for each local device."""
    try:
        if not torch.cuda.is_available():
            return "cuda_unavailable"

        stats = []
        for idx in range(torch.cuda.device_count()):
            with torch.cuda.device(idx):
                free_bytes, total_bytes = torch.cuda.mem_get_info()
            used_bytes = total_bytes - free_bytes
            stats.append(f"cuda:{idx}={used_bytes / (1024**3):.2f}/{total_bytes / (1024**3):.2f}GB")
        return ", ".join(stats) if stats else "no_visible_gpu"
    except Exception as e:  # noqa: BLE001
        return f"gpu_mem_error={e!r}"


async def _sleep_engine(engine_client: Any, level: int = 2) -> None:
    """Sleep engine with compatibility across vLLM API variants."""
    if hasattr(engine_client, "collective_rpc"):
        await engine_client.collective_rpc(method="sleep", kwargs={"level": level})
        return
    if hasattr(engine_client, "sleep"):
        await engine_client.sleep(level=level)
        return
    raise AttributeError("Engine has neither `collective_rpc` nor `sleep` method")


async def _wake_up_engine(engine_client: Any, tags: list[str] | None = None) -> None:
    """Wake engine with compatibility across vLLM API variants."""
    if hasattr(engine_client, "wake_up"):
        await engine_client.wake_up(tags=tags)
    elif hasattr(engine_client, "wakeup"):
        wakeup = getattr(engine_client, "wakeup")
        try:
            await wakeup(tags=tags)
        except TypeError:
            await wakeup()
    elif hasattr(engine_client, "collective_rpc"):
        try:
            await engine_client.collective_rpc(method="wake_up", kwargs={"tags": tags})
        except Exception:
            await engine_client.collective_rpc(method="wakeup", kwargs={"tags": tags})
    else:
        raise AttributeError("Engine has neither `wake_up`/`wakeup` nor `collective_rpc` method")

    if hasattr(engine_client, "reset_prefix_cache"):
        await engine_client.reset_prefix_cache()


async def _init_or_serve(ns: argparse.Namespace, mode: str, host: str) -> None:
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.entrypoints.openai.api_server import build_app, init_app_state
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.async_llm import AsyncLLM

    from verl.workers.rollout.utils import run_unvicorn

    engine_args = AsyncEngineArgs.from_cli_args(ns)
    usage_context = UsageContext.OPENAI_API_SERVER
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)

    fn_args = set(dict(inspect.signature(AsyncLLM.from_vllm_config).parameters).keys())
    kwargs = {}
    if "enable_log_requests" in fn_args:
        kwargs["enable_log_requests"] = engine_args.enable_log_requests
    if "disable_log_stats" in fn_args:
        kwargs["disable_log_stats"] = engine_args.disable_log_stats

    engine_client = AsyncLLM.from_vllm_config(vllm_config=vllm_config, usage_context=usage_context, **kwargs)

    if hasattr(engine_client, "reset_mm_cache"):
        await engine_client.reset_mm_cache()

    print("[OK] AsyncLLM initialized")
    print(f"[SLEEP_MEM][REWARD][BEFORE_WAIT] gpu_mem={_format_gpu_memory_used_total_gb()}")
    print("[SLEEP_MEM][REWARD] waiting 3 seconds before calling engine sleep")
    await asyncio.sleep(3)
    print(f"[SLEEP_MEM][REWARD][BEFORE_SLEEP] gpu_mem={_format_gpu_memory_used_total_gb()}")
    await _sleep_engine(engine_client, level=2)
    print(f"[SLEEP_MEM][REWARD][AFTER_SLEEP] gpu_mem={_format_gpu_memory_used_total_gb()}")
    print("[SLEEP_MEM][REWARD] waiting 3 seconds before calling engine wake_up")
    await asyncio.sleep(3)
    print(f"[SLEEP_MEM][REWARD][BEFORE_WAKE_UP] gpu_mem={_format_gpu_memory_used_total_gb()}")
    await _wake_up_engine(engine_client, tags=["kv_cache", "weights"])
    print(f"[SLEEP_MEM][REWARD][AFTER_WAKE_UP] gpu_mem={_format_gpu_memory_used_total_gb()}")

    if mode != "serve":
        return

    build_app_sig = inspect.signature(build_app)
    supported_tasks: tuple[Any, ...] = ()
    if "supported_tasks" in build_app_sig.parameters:
        supported_tasks = await engine_client.get_supported_tasks()
        app = build_app(ns, supported_tasks)
    else:
        app = build_app(ns)

    init_app_sig = inspect.signature(init_app_state)
    if "vllm_config" in init_app_sig.parameters:
        await init_app_state(engine_client, vllm_config, app.state, ns)
    elif "supported_tasks" in init_app_sig.parameters:
        await init_app_state(engine_client, app.state, ns, supported_tasks)
    else:
        await init_app_state(engine_client, app.state, ns)

    port, _ = await run_unvicorn(app, ns, host)
    print(f"[OK] Serving at http://{host}:{port}")
    await asyncio.Event().wait()


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
    if args.mode in {"init", "serve"}:
        asyncio.run(_init_or_serve(ns, args.mode, args.host))


if __name__ == "__main__":
    main()
