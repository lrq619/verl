#!/usr/bin/env python3
import argparse
import asyncio
import inspect
import json

import vllm_omni.entrypoints.cli.serve
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm_omni.engine.arg_utils import AsyncOmniEngineArgs
from vllm_omni.entrypoints import AsyncOmni
from vllm_omni.entrypoints.openai.api_server import build_app, omni_init_app_state

from verl.workers.rollout.utils import run_unvicorn
from verl.workers.rollout.vllm_rollout.utils import build_cli_args_from_config


def parse_args():
    p = argparse.ArgumentParser(description="Start vLLM-Omni with the same code path used by verl.")
    p.add_argument("--model", required=True, help="Model id or local path.")
    p.add_argument("--tokenizer", default=None, help="Optional tokenizer path/id.")
    p.add_argument("--custom-pipeline", default=None, help="Optional custom pipeline class path.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--load-format", default="safetensors")
    p.add_argument("--max-model-len", type=int, default=1058)
    p.add_argument("--max-num-seqs", type=int, default=1024)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compilation-config", default='{"cudagraph_mode":"NONE"}')
    p.add_argument("--enable-lora", action="store_true", default=True)
    p.add_argument("--max-loras", type=int, default=1)
    p.add_argument("--max-lora-rank", type=int, default=64)
    p.add_argument("--init-only", action="store_true", help="Only validate initialization, then exit.")
    return p.parse_args()


def build_server_namespace(args):
    config = {
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
        "hf_overrides": "{}",
        "scheduling_policy": "fcfs",
        "compilation_config": args.compilation_config,
        "enable_lora": args.enable_lora,
        "max_loras": args.max_loras,
        "max_lora_rank": args.max_lora_rank,
    }
    if args.tokenizer:
        config["tokenizer"] = args.tokenizer
    server_args = ["serve", args.model] + build_cli_args_from_config(config)

    cmd_modules = [vllm_omni.entrypoints.cli.serve]
    parser = FlexibleArgumentParser(description="vLLM-Omni CLI")
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    cmds = {}
    for mod in cmd_modules:
        for cmd in mod.cmd_init():
            cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
            cmds[cmd.name] = cmd

    ns = parser.parse_args(args=server_args)
    ns.model = ns.model_tag
    if ns.subparser in cmds:
        cmds[ns.subparser].validate(ns)
    return ns


async def main_async(args):
    ns = build_server_namespace(args)
    engine_args = AsyncOmniEngineArgs.from_cli_args(ns)

    kwargs = {
        "model": engine_args.model,
        "enable_sleep_mode": engine_args.enable_sleep_mode,
        "worker_extension_cls": engine_args.worker_extension_cls,
        "enforce_eager": engine_args.enforce_eager,
    }
    if args.custom_pipeline:
        kwargs["enable_dummy_pipeline"] = True
        kwargs["custom_pipeline_args"] = {"pipeline_class": args.custom_pipeline}

    engine_client = AsyncOmni(**kwargs)
    app = build_app(ns)
    if len(inspect.signature(omni_init_app_state).parameters) >= 4:
        await omni_init_app_state(engine_client, None, app.state, ns)
    else:
        await omni_init_app_state(engine_client, app.state, ns)

    print("[OK] AsyncOmni initialized successfully.")
    if args.init_only:
        return

    port, _ = await run_unvicorn(app, ns, args.host)
    print(f"[OK] Server started at http://{args.host}:{port}")
    await asyncio.Event().wait()


def main():
    args = parse_args()
    # Validate compilation config early for cleaner errors.
    json.loads(args.compilation_config)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
