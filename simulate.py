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

import torch
import vllm_omni.entrypoints.cli.serve
from vllm.sampling_params import SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm_omni.engine.arg_utils import AsyncOmniEngineArgs
from vllm_omni.entrypoints import AsyncOmni
from vllm_omni.entrypoints.openai.api_server import build_openai_app, omni_init_app_state
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

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
    p.add_argument("--ulysses_degree", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scheduling-policy", default="fcfs")
    p.add_argument("--compilation-config", default='{"cudagraph_mode":"NONE"}')
    p.add_argument("--enable-sleep-mode", default=True)

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
    cli_args = ["serve", args.model] + build_cli_args_from_config(config)

    # vLLM-Omni expects hyphenated diffusion SP flags. Preserve verl-like config
    # building for standard args, then append the exact spellings the parser accepts.
    if args.ulysses_degree is not None:
        cli_args.extend(["--ulysses-degree", str(args.ulysses_degree)])

    return cli_args


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


async def _sleep_engine(engine_client: AsyncOmni, level: int = 2) -> None:
    """Sleep engine with compatibility across vLLM-Omni API variants."""
    if hasattr(engine_client, "sleep"):
        await engine_client.sleep(level=level)
        return
    if hasattr(engine_client, "collective_rpc"):
        await engine_client.collective_rpc(method="sleep", kwargs={"level": level})
        return
    raise AttributeError("Engine has neither `sleep` nor `collective_rpc` method")


async def _wake_up_engine(engine_client: AsyncOmni, tags: list[str] | None = None) -> None:
    """Wake engine with compatibility across vLLM-Omni API variants."""
    if hasattr(engine_client, "wake_up"):
        await engine_client.wake_up(tags=tags)
        return
    if hasattr(engine_client, "wakeup"):
        wakeup = getattr(engine_client, "wakeup")
        try:
            await wakeup(tags=tags)
        except TypeError:
            await wakeup()
        return
    if hasattr(engine_client, "collective_rpc"):
        try:
            await engine_client.collective_rpc(method="wake_up", kwargs={"tags": tags})
        except Exception:
            await engine_client.collective_rpc(method="wakeup", kwargs={"tags": tags})
        return
    raise AttributeError("Engine has neither `wake_up`/`wakeup` nor `collective_rpc` method")


def _build_dummy_sampling_params_list(engine_client: AsyncOmni) -> list[Any]:
    """Build lightweight per-stage sampling params for a quick dummy request."""
    params_list: list[Any] = []
    for stage in getattr(engine_client, "stage_list", []):
        stage_type = str(getattr(stage, "stage_type", "")).lower()
        if stage_type == "diffusion":
            params_list.append(
                OmniDiffusionSamplingParams(
                    num_inference_steps=1,
                    num_outputs_per_prompt=1,
                    height=256,
                    width=256,
                )
            )
        else:
            params_list.append(SamplingParams(max_tokens=1, temperature=0.0))
    return params_list


def _safe_sorted_keys(value: Any) -> list[str] | None:
    if isinstance(value, dict):
        return sorted(str(k) for k in value.keys())
    return None


async def _dummy_request_and_report(engine_client: AsyncOmni) -> None:
    """Send a dummy internal request and print multimodal output schema checks."""
    request_id = f"sim-dummy-{int(asyncio.get_running_loop().time() * 1000)}"
    sampling_params_list = _build_dummy_sampling_params_list(engine_client)
    if not sampling_params_list:
        print("[DUMMY_CHECK][SKIP] empty stage_list; cannot send dummy request")
        return

    async def _collect_last_output() -> Any:
        last_output = None
        async for output in engine_client.generate(
            prompt="a tiny white cat",
            request_id=request_id,
            sampling_params_list=sampling_params_list,
        ):
            last_output = output
        return last_output

    try:
        final_output = await asyncio.wait_for(_collect_last_output(), timeout=300)
    except Exception as e:  # noqa: BLE001
        print(f"[DUMMY_CHECK][ERROR] failed to run dummy request: {e!r}")
        return

    if final_output is None:
        print("[DUMMY_CHECK][ERROR] dummy request returned no output")
        return

    output_dict = final_output.to_dict() if hasattr(final_output, "to_dict") else {}
    print(f"[DUMMY_CHECK] final_output_type={type(final_output).__name__}")
    print(f"[DUMMY_CHECK] output_dict_keys={sorted(output_dict.keys())}")
    multimodal_output = final_output.multimodal_output
    print(f"[DUMMY_CHECK] has_output_field_multimodal_output={hasattr(final_output, "multimodal_output")}")
    print(f"[DUMMY_CHECK] multimodal_output keys: {multimodal_output.keys()}")





async def _init_or_serve(ns: argparse.Namespace, mode: str, host: str) -> None:
    engine_args = AsyncOmniEngineArgs.from_cli_args(ns)
    engine_args = asdict(engine_args)

    # AsyncOmniEngineArgs does not currently retain diffusion-only SP fields,
    # but AsyncOmni still accepts them as kwargs.
    if getattr(ns, "ulysses_degree", None) is not None:
        engine_args["ulysses_degree"] = ns.ulysses_degree
    if getattr(ns, "ring_degree", None) is not None:
        engine_args["ring_degree"] = ns.ring_degree

    print(
        f"tp_size: {engine_args['tensor_parallel_size']}, "
        f"ulysses_size: {engine_args.get('ulysses_degree')}"
    )
    engine_client = AsyncOmni(**engine_args)

    print("[OK] AsyncOmni initialized")
    print(f"[SLEEP_MEM][SIM][BEFORE_WAIT] gpu_mem={_format_gpu_memory_used_total_gb()}")
    print("[SLEEP_MEM][SIM] waiting 3 seconds before calling engine.sleep(level=1)")
    await asyncio.sleep(3)
    print(f"[SLEEP_MEM][SIM][BEFORE_SLEEP] gpu_mem={_format_gpu_memory_used_total_gb()}")
    await _sleep_engine(engine_client, level=1)
    print(f"[SLEEP_MEM][SIM][AFTER_SLEEP] gpu_mem={_format_gpu_memory_used_total_gb()}")
    print("[SLEEP_MEM][SIM] waiting 3 seconds before calling engine.wake_up(...)")
    await asyncio.sleep(3)
    print(f"[SLEEP_MEM][SIM][BEFORE_WAKE_UP] gpu_mem={_format_gpu_memory_used_total_gb()}")
    await _wake_up_engine(engine_client)
    print(f"[SLEEP_MEM][SIM][AFTER_WAKE_UP] gpu_mem={_format_gpu_memory_used_total_gb()}")
    await _dummy_request_and_report(engine_client)

    if mode != "serve":
        return

    app = build_openai_app(ns)
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
