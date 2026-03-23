#!/usr/bin/env python3
"""Benchmark a standalone FlowGRPO rollout server with verl-like requests."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_SHELL_VAR_RE = re.compile(r"\$(\w+|\{[^}]+\})")
_ASSIGNMENT_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


@dataclass
class FlowGRPOConfig:
    script_path: Path
    train_files: list[str]
    prompt_key: str
    train_batch_size: int
    rollout_n: int
    model_path: str
    temperature: float | None
    top_p: float | None
    response_length: int | None
    max_model_len: int | None
    image_height: int | None
    image_width: int | None
    num_inference_steps: int | None
    noise_level: float | None
    guidance_scale: float | None
    sde_type: str | None
    sde_window_size: int | None
    sde_window_range: Any
    tp_size: int | None
    trainer_gpus_per_node: int | None
    trainer_nnodes: int | None
    rollout_data_parallel_size: int | None

    @property
    def current_dp_size(self) -> int | None:
        if self.rollout_data_parallel_size is not None:
            return self.rollout_data_parallel_size
        if self.tp_size and self.trainer_gpus_per_node and self.trainer_nnodes:
            total_gpus = self.trainer_gpus_per_node * self.trainer_nnodes
            if total_gpus % self.tp_size == 0:
                return total_gpus // self.tp_size
        return None

    @property
    def prompts_per_replica(self) -> int:
        dp_size = self.current_dp_size or 1
        return math.ceil(self.train_batch_size / dp_size)


def _expand_shell_vars(value: str, env: dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        key = key[1:-1] if key.startswith("{") and key.endswith("}") else key
        return env.get(key, os.environ.get(key, match.group(0)))

    return _SHELL_VAR_RE.sub(repl, value)


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _parse_value(raw: str, env: dict[str, str]) -> Any:
    expanded = _expand_shell_vars(_strip_quotes(raw), env).strip()
    if expanded == "":
        return ""

    lowered = expanded.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None

    try:
        return int(expanded)
    except ValueError:
        pass

    try:
        return float(expanded)
    except ValueError:
        pass

    if expanded.startswith(("{", "[")) or expanded.startswith(("(",)):
        try:
            return json.loads(expanded)
        except json.JSONDecodeError:
            pass

    return expanded


def _normalize_override_key(key: str) -> str:
    return key.lstrip("+")


def _extract_overrides(tokens: list[str], env: dict[str, str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for token in tokens:
        if "=" not in token or token.startswith("--"):
            continue
        key, raw_value = token.split("=", 1)
        if not key:
            continue
        overrides[_normalize_override_key(key)] = _parse_value(raw_value, env)
    return overrides


def _find_launch_command(script_text: str) -> str | None:
    logical_lines: list[str] = []
    current = ""

    for raw_line in script_text.splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        if current:
            current += " " + stripped.removesuffix("\\").strip()
        else:
            current = stripped.removesuffix("\\").strip()

        if stripped.endswith("\\"):
            continue

        logical_lines.append(current)
        current = ""

    if current:
        logical_lines.append(current)

    for line in logical_lines:
        if "python -m verl.trainer.main_ppo" in line or re.search(r"(^|\s)bash\s+\S+\.sh(\s|$)", line):
            return line
    return None


def _parse_shell_assignments(script_text: str, env: dict[str, str]) -> dict[str, str]:
    merged = dict(env)
    for raw_line in script_text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _ASSIGNMENT_RE.match(stripped)
        if not match:
            continue
        key, raw_value = match.groups()
        if any(ch in raw_value for ch in ("`", "$(")):
            continue
        merged[key] = _expand_shell_vars(_strip_quotes(raw_value), merged)
    return merged


def _parse_script_recursive(script_path: Path, inherited_env: dict[str, str]) -> tuple[dict[str, Any], dict[str, str]]:
    script_text = script_path.read_text(encoding="utf-8")
    local_env = _parse_shell_assignments(script_text, inherited_env)
    launch_command = _find_launch_command(script_text)
    if launch_command is None:
        raise ValueError(f"Could not find launch command in {script_path}")

    tokens = shlex.split(_expand_shell_vars(launch_command, local_env))
    token_env = dict(local_env)
    cursor = 0
    while cursor < len(tokens):
        match = _ASSIGNMENT_RE.match(tokens[cursor])
        if not match:
            break
        key, raw_value = match.groups()
        token_env[key] = _expand_shell_vars(_strip_quotes(raw_value), token_env)
        cursor += 1

    if cursor >= len(tokens):
        raise ValueError(f"Could not parse executable from launch command in {script_path}")

    command = tokens[cursor]
    rest = tokens[cursor + 1 :]

    if command == "bash":
        if not rest:
            raise ValueError(f"Expected nested script after bash in {script_path}")
        nested_path = (script_path.parent / rest[0]).resolve()
        nested_overrides, nested_env = _parse_script_recursive(nested_path, token_env)
        nested_overrides.update(_extract_overrides(rest[1:], {**token_env, **nested_env}))
        return nested_overrides, {**token_env, **nested_env}

    if command == "python" and rest[:2] == ["-m", "verl.trainer.main_ppo"]:
        return _extract_overrides(rest[2:], token_env), token_env

    raise ValueError(f"Unsupported launch command `{command}` in {script_path}")


def parse_flowgrpo_config(script_path: Path) -> FlowGRPOConfig:
    overrides, _ = _parse_script_recursive(script_path.resolve(), {})

    train_files = overrides.get("data.train_files")
    if train_files is None:
        raise ValueError("Could not find `data.train_files` in launcher script")

    if isinstance(train_files, str):
        train_files = [train_files]
    elif not isinstance(train_files, list):
        raise ValueError(f"Unsupported `data.train_files` type: {type(train_files).__name__}")

    def resolve_path_like(value: str) -> str:
        return os.path.expanduser(os.path.expandvars(str(value)))

    return FlowGRPOConfig(
        script_path=script_path.resolve(),
        train_files=[resolve_path_like(path) for path in train_files],
        prompt_key=str(overrides.get("data.prompt_key", "prompt")),
        train_batch_size=int(overrides.get("data.train_batch_size", 1024)),
        rollout_n=int(overrides.get("actor_rollout_ref.rollout.n", 1)),
        model_path=resolve_path_like(overrides["actor_rollout_ref.model.path"]),
        temperature=_maybe_float(overrides.get("actor_rollout_ref.rollout.temperature", 1.0)),
        top_p=_maybe_float(overrides.get("actor_rollout_ref.rollout.top_p", 1.0)),
        response_length=_maybe_int(overrides.get("actor_rollout_ref.rollout.response_length")),
        max_model_len=_maybe_int(overrides.get("actor_rollout_ref.rollout.max_model_len")),
        image_height=_maybe_int(overrides.get("actor_rollout_ref.rollout.image_height", 512)),
        image_width=_maybe_int(overrides.get("actor_rollout_ref.rollout.image_width", 512)),
        num_inference_steps=_maybe_int(overrides.get("actor_rollout_ref.rollout.num_inference_steps", 10)),
        noise_level=_maybe_float(overrides.get("actor_rollout_ref.rollout.noise_level", 0.7)),
        guidance_scale=_maybe_float(overrides.get("actor_rollout_ref.rollout.guidance_scale", 4.5)),
        sde_type=_maybe_str(overrides.get("actor_rollout_ref.rollout.sde_type", "sde")),
        sde_window_size=_maybe_int(overrides.get("actor_rollout_ref.rollout.sde_window_size")),
        sde_window_range=overrides.get("actor_rollout_ref.rollout.sde_window_range"),
        tp_size=_maybe_int(overrides.get("actor_rollout_ref.rollout.tensor_model_parallel_size")),
        trainer_gpus_per_node=_maybe_int(overrides.get("trainer.n_gpus_per_node")),
        trainer_nnodes=_maybe_int(overrides.get("trainer.nnodes", 1)),
        rollout_data_parallel_size=_maybe_int(overrides.get("actor_rollout_ref.rollout.data_parallel_size")),
    )


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _maybe_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _to_plain_python(value: Any) -> Any:
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except TypeError:
            pass
    if isinstance(value, dict):
        return {k: _to_plain_python(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain_python(v) for v in value]
    return value


def load_first_prompts(train_files: list[str], prompt_key: str, batch_size: int) -> list[Any]:
    try:
        import datasets
    except ImportError as exc:
        raise RuntimeError("`datasets` is required to read parquet prompts. Please run in the verl environment.") from exc

    prompts: list[Any] = []
    for train_file in train_files:
        dataset = datasets.load_dataset("parquet", data_files=train_file, split="train")
        for row in dataset:
            prompts.append(_to_plain_python(row[prompt_key]))
            if len(prompts) >= batch_size:
                return prompts

    if not prompts:
        raise ValueError("No prompts found in training dataset")

    repeated: list[Any] = []
    while len(repeated) < batch_size:
        remaining = batch_size - len(repeated)
        repeated.extend(prompts[:remaining])
    return repeated


class ProgressTracker:
    def __init__(self, total_requests: int):
        self.total_requests = total_requests
        self._lock = threading.Lock()
        self._bench_start: float | None = None
        self._started = 0
        self._completed = 0

    def mark_benchmark_start(self) -> None:
        with self._lock:
            self._bench_start = time.perf_counter()
            print(
                f"[loadgen] benchmark started: total_requests={self.total_requests}",
                flush=True,
            )

    def log_request_start(self, request_idx: int, prompt_idx: int, sample_idx: int) -> None:
        with self._lock:
            self._started += 1
            print(
                f"[loadgen] -> request {request_idx + 1}/{self.total_requests} "
                f"(prompt={prompt_idx}, sample={sample_idx}) started; in_flight={self._started - self._completed}",
                flush=True,
            )

    def log_request_done(self, request_idx: int, prompt_idx: int, sample_idx: int, elapsed_s: float, status: int) -> None:
        with self._lock:
            self._completed += 1
            bench_elapsed = None
            current_tps = None
            if self._bench_start is not None:
                bench_elapsed = time.perf_counter() - self._bench_start
                if bench_elapsed > 0:
                    current_tps = self._completed / bench_elapsed

            throughput_msg = "n/a" if current_tps is None else f"{current_tps:.3f} req/s"
            elapsed_msg = "n/a" if bench_elapsed is None else f"{bench_elapsed:.3f}s"
            print(
                f"[loadgen] <- request {request_idx + 1}/{self.total_requests} "
                f"(prompt={prompt_idx}, sample={sample_idx}) done "
                f"status={status} latency={elapsed_s:.3f}s completed={self._completed}/{self.total_requests} "
                f"benchmark_elapsed={elapsed_msg} current_throughput={throughput_msg}",
                flush=True,
            )

    def log_request_error(self, request_idx: int, prompt_idx: int, sample_idx: int, error: Exception) -> None:
        with self._lock:
            print(
                f"[loadgen] !! request {request_idx + 1}/{self.total_requests} "
                f"(prompt={prompt_idx}, sample={sample_idx}) failed: {error}",
                flush=True,
            )


def build_payload(messages: Any, config: FlowGRPOConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": config.model_path,
        "messages": messages,
    }

    if config.temperature is not None:
        payload["temperature"] = config.temperature
    if config.top_p is not None:
        payload["top_p"] = config.top_p
    if config.response_length is not None:
        payload["max_tokens"] = config.response_length

    if config.max_model_len is not None:
        payload["max_sequence_length"] = config.max_model_len
    if config.image_height is not None:
        payload["height"] = config.image_height
    if config.image_width is not None:
        payload["width"] = config.image_width
    if config.guidance_scale is not None:
        payload["true_cfg_scale"] = config.guidance_scale
    if config.num_inference_steps is not None:
        payload["num_inference_steps"] = config.num_inference_steps
    if config.noise_level is not None:
        payload["noise_level"] = config.noise_level
    if config.sde_type is not None:
        payload["sde_type"] = config.sde_type
    if config.sde_window_size is not None:
        payload["sde_window_size"] = config.sde_window_size
    if config.sde_window_range is not None:
        payload["sde_window_range"] = config.sde_window_range

    return payload


def _post_chat_completion(
    url: str,
    payload: dict[str, Any],
    request_idx: int,
    prompt_idx: int,
    sample_idx: int,
    tracker: ProgressTracker | None = None,
) -> tuple[float, int]:
    if tracker is not None:
        tracker.log_request_start(request_idx, prompt_idx, sample_idx)

    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=body,
        headers={
            "Authorization": "Bearer token-abc123",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=None) as response:
            response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        error = RuntimeError(f"Request failed with status {exc.code}: {error_body[:500]}")
        if tracker is not None:
            tracker.log_request_error(request_idx, prompt_idx, sample_idx, error)
        raise error from exc
    elapsed = time.perf_counter() - started
    if tracker is not None:
        tracker.log_request_done(request_idx, prompt_idx, sample_idx, elapsed, status)
    return elapsed, status


def run_benchmark(
    base_url: str,
    config: FlowGRPOConfig,
    prompts: list[Any],
    concurrency: int,
    warmup_prompts: int,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    request_items = [
        {
            "request_idx": request_idx,
            "prompt_idx": prompt_idx,
            "sample_idx": sample_idx,
            "payload": build_payload(prompt, config),
        }
        for request_idx, (prompt_idx, prompt, sample_idx) in enumerate(
            (prompt_idx, prompt, sample_idx)
            for prompt_idx, prompt in enumerate(prompts)
            for sample_idx in range(config.rollout_n)
        )
    ]
    warmup_items = request_items[: min(len(request_items), warmup_prompts)]
    tracker = ProgressTracker(total_requests=len(request_items))

    if warmup_items:
        print(f"[loadgen] warmup started: warmup_requests={len(warmup_items)}", flush=True)
        with ThreadPoolExecutor(max_workers=min(concurrency, len(warmup_items))) as executor:
            futures = [
                executor.submit(
                    _post_chat_completion,
                    url,
                    item["payload"],
                    item["request_idx"],
                    item["prompt_idx"],
                    item["sample_idx"],
                    None,
                )
                for item in warmup_items
            ]
            for future in as_completed(futures):
                future.result()
        print("[loadgen] warmup finished", flush=True)

    tracker.mark_benchmark_start()
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                _post_chat_completion,
                url,
                item["payload"],
                item["request_idx"],
                item["prompt_idx"],
                item["sample_idx"],
                tracker,
            )
            for item in request_items
        ]
        results = [future.result() for future in as_completed(futures)]
    wall_elapsed = time.perf_counter() - wall_start

    latencies = [elapsed for elapsed, _ in results]
    total_requests = len(request_items)
    unique_prompts = len(prompts)
    return {
        "url": url,
        "requests": total_requests,
        "unique_prompts": unique_prompts,
        "wall_time_s": wall_elapsed,
        "samples_per_s": total_requests / wall_elapsed,
        "prompts_per_s": unique_prompts / wall_elapsed,
        "mean_latency_s": statistics.fmean(latencies),
        "p50_latency_s": _percentile(latencies, 0.50),
        "p95_latency_s": _percentile(latencies, 0.95),
        "max_latency_s": max(latencies),
    }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    frac = index - lower
    return ordered[lower] * (1 - frac) + ordered[upper] * frac


def estimate_step_times(config: FlowGRPOConfig, measured_samples_per_s: float, dp_sizes: list[int]) -> list[dict[str, Any]]:
    estimates: list[dict[str, Any]] = []
    for dp_size in dp_sizes:
        prompts_per_replica = math.ceil(config.train_batch_size / dp_size)
        replica_samples = prompts_per_replica * config.rollout_n
        estimates.append(
            {
                "dp_size": dp_size,
                "prompts_per_replica": prompts_per_replica,
                "samples_per_replica": replica_samples,
                "estimated_step_time_s": replica_samples / measured_samples_per_s,
            }
        )
    return estimates


def _default_dp_sizes(current_dp_size: int | None) -> list[int]:
    sizes = {1, 2, 4, 8}
    if current_dp_size is not None:
        sizes.add(current_dp_size)
    return sorted(size for size in sizes if size > 0)


def _parse_dp_sizes(raw: str | None, current_dp_size: int | None) -> list[int]:
    if raw is None:
        return _default_dp_sizes(current_dp_size)
    sizes = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not sizes:
        raise ValueError("`--dp-sizes` must contain at least one integer")
    return sizes


def _default_measure_prompts(config: FlowGRPOConfig) -> int:
    target_requests = 8
    prompts = math.ceil(target_requests / max(1, config.rollout_n))
    return max(1, min(config.prompts_per_replica, prompts))


def _format_table(rows: list[dict[str, Any]]) -> str:
    headers = list(rows[0].keys())
    widths = {
        header: max(len(header), *(len(_stringify(row[header])) for row in rows))
        for header in headers
    }
    line = " | ".join(header.ljust(widths[header]) for header in headers)
    sep = "-+-".join("-" * widths[header] for header in headers)
    body = [" | ".join(_stringify(row[header]).ljust(widths[header]) for header in headers) for row in rows]
    return "\n".join([line, sep, *body])


def _stringify(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a standalone FlowGRPO rollout server.")
    parser.add_argument("--script", default="run_flowgrpo.sh", help="Launcher script to parse.")
    parser.add_argument("--base-url", required=True, help="Base URL of the simulate server, e.g. http://127.0.0.1:8000")
    parser.add_argument(
        "--measure-prompts",
        type=int,
        default=None,
        help="How many prompts to actually send. Defaults to a small prompt count that keeps total requests around 8.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Max in-flight requests. Defaults to min(total_requests, 4).",
    )
    parser.add_argument(
        "--warmup-prompts",
        type=int,
        default=1,
        help="Number of requests to warm up before timing starts.",
    )
    parser.add_argument(
        "--dp-sizes",
        default=None,
        help="Comma-separated DP sizes to estimate, e.g. 1,2,4,8. Defaults to 1,2,4,8 plus current DP size.",
    )
    parser.add_argument(
        "--dump-json",
        default=None,
        help="Optional path to write the parsed config, benchmark result, and DP estimates as JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_path = Path(args.script)
    config = parse_flowgrpo_config(script_path)

    unresolved_paths = [path for path in [config.model_path, *config.train_files] if "$" in path]
    if unresolved_paths:
        print("Warning: some parsed paths still contain shell variables; export them before running the benchmark.")
        for path in unresolved_paths:
            print(f"  unresolved: {path}")
        print()

    measure_prompts = args.measure_prompts or _default_measure_prompts(config)
    prompts = load_first_prompts(config.train_files, config.prompt_key, measure_prompts)
    total_requests = len(prompts) * config.rollout_n
    concurrency = args.concurrency or min(total_requests, 4)
    dp_sizes = _parse_dp_sizes(args.dp_sizes, config.current_dp_size)

    print(
        "[loadgen] parsed workload: "
        f"global_train_batch_size={config.train_batch_size}, "
        f"current_dp_size={config.current_dp_size}, "
        f"prompts_per_replica={config.prompts_per_replica}, "
        f"default_measure_prompts={_default_measure_prompts(config)}, "
        f"measure_prompts={len(prompts)}, "
        f"rollout_n={config.rollout_n}, "
        f"total_requests={total_requests}, "
        f"concurrency={concurrency}",
        flush=True,
    )

    benchmark = run_benchmark(
        base_url=args.base_url,
        config=config,
        prompts=prompts,
        concurrency=concurrency,
        warmup_prompts=args.warmup_prompts,
    )
    estimates = estimate_step_times(config, benchmark["samples_per_s"], dp_sizes)

    print("Parsed FlowGRPO config")
    print(
        json.dumps(
            {
                "script": str(config.script_path),
                "train_files": config.train_files,
                "prompt_key": config.prompt_key,
                "train_batch_size": config.train_batch_size,
                "prompts_per_replica": config.prompts_per_replica,
                "measured_prompts": len(prompts),
                "rollout_n": config.rollout_n,
                "tp_size": config.tp_size,
                "current_dp_size": config.current_dp_size,
                "model_path": config.model_path,
                "num_inference_steps": config.num_inference_steps,
                "guidance_scale": config.guidance_scale,
                "noise_level": config.noise_level,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print()

    print("Measured throughput")
    print(
        _format_table(
            [
                {
                    "requests": benchmark["requests"],
                    "unique_prompts": benchmark["unique_prompts"],
                    "wall_time_s": benchmark["wall_time_s"],
                    "samples_per_s": benchmark["samples_per_s"],
                    "prompts_per_s": benchmark["prompts_per_s"],
                    "mean_latency_s": benchmark["mean_latency_s"],
                    "p50_latency_s": benchmark["p50_latency_s"],
                    "p95_latency_s": benchmark["p95_latency_s"],
                    "max_latency_s": benchmark["max_latency_s"],
                }
            ]
        )
    )
    print()

    print("Estimated rollout step time by DP size")
    print(_format_table(estimates))
    print()
    print("Assumption: one simulate server ~= one rollout replica, and throughput scales linearly with DP size.")

    if args.dump_json:
        output_path = Path(args.dump_json)
        output_path.write_text(
            json.dumps(
                {
                    "config": config.__dict__,
                    "benchmark": benchmark,
                    "estimates": estimates,
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"Saved report to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
