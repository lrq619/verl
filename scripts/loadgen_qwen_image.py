#!/usr/bin/env python3
"""Visualize Qwen-Image generations from a parquet dataset via vLLM-Omni.

This mirrors the FlowGRPO OCR setup in this repo as closely as possible:
- system prompt from `examples/data_preprocess/qwenimage_ocr.py`
- negative prompt user content is a single space
- generation goes through `/v1/chat/completions`, which is the serving path
  documented for Qwen-Image in vLLM-Omni

The Qwen-Image serving pipeline wraps a plain user prompt with the same
hardcoded system template used during training, so the HTTP request only sends
the user text while the manifest preserves the full training prompt messages.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


TRAIN_SYSTEM_PROMPT = (
    "Describe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
)
DEFAULT_NEGATIVE_PROMPT = " "
DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 4.0
DEFAULT_TRUE_CFG_SCALE = 4.0
DEFAULT_TIMEOUT_S = 300
DATA_URL_RE = re.compile(r"^data:image/([^;]+);base64,(.+)$", re.DOTALL)
REQUEST_KEY_CANDIDATES = (
    "request",
    "text",
    "query",
    "instruction",
    "input",
    "question",
    "caption",
    "prompt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", required=True, help="Base URL of the vLLM-Omni server, e.g. http://127.0.0.1:42195")
    parser.add_argument("--data-path", default="/data/train.parquet", help="Parquet file to read requests from")
    parser.add_argument("--count", type=int, default=5, help="Number of rows to process")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to scripts/output/loadgen_<timestamp>")
    parser.add_argument("--model", default=None, help="Optional model id/path. Defaults to the first entry from /v1/models")
    parser.add_argument("--prompt-key", default="prompt", help="Column containing full prompt messages")
    parser.add_argument("--negative-prompt-key", default="negative_prompt", help="Column containing negative prompt messages")
    parser.add_argument("--request-key", default=None, help="Optional column containing the raw user request text")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Output image height")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Output image width")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=DEFAULT_NUM_INFERENCE_STEPS,
        help="Diffusion inference steps. FlowGRPO validation commonly uses 50.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=DEFAULT_GUIDANCE_SCALE,
        help="`guidance_scale` sent in `extra_body`",
    )
    parser.add_argument(
        "--true-cfg-scale",
        type=float,
        default=DEFAULT_TRUE_CFG_SCALE,
        help="`true_cfg_scale` sent in `extra_body`",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional base seed. If set, row i uses seed+i")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S, help="HTTP timeout in seconds")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_first_rows(Path(args.data_path), args.count)
    server_url = args.server_url.rstrip("/")
    model = args.model or fetch_default_model(server_url, args.timeout)

    manifest: list[dict[str, Any]] = []
    warnings: list[str] = []

    for index, row in enumerate(rows):
        seed = None if args.seed is None else args.seed + index
        try:
            resolved = resolve_prompt_row(
                row=row,
                row_index=index,
                prompt_key=args.prompt_key,
                negative_prompt_key=args.negative_prompt_key,
                request_key=args.request_key,
                warnings=warnings,
            )

            response = generate_image(
                server_url=server_url,
                model=model,
                user_prompt=resolved["user_prompt"],
                negative_prompt=resolved["negative_prompt_user"],
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                true_cfg_scale=args.true_cfg_scale,
                seed=seed,
                timeout=args.timeout,
            )
            image_path = save_image_from_response(response, output_dir / f"{index:04d}.png")

            manifest.append(
                {
                    "row_index": index,
                    "status": "ok",
                    "image_filename": image_path.name,
                    "server_prompt": resolved["user_prompt"],
                    "negative_prompt": resolved["negative_prompt_user"],
                    "training_prompt_messages": resolved["training_prompt_messages"],
                    "training_negative_prompt_messages": resolved["training_negative_prompt_messages"],
                    "prompt_source": resolved["prompt_source"],
                    "negative_prompt_source": resolved["negative_prompt_source"],
                    "request_key_used": resolved["request_key_used"],
                    "system_prompt_matches_qwen_image_training": resolved["system_prompt_matches_training"],
                    "seed": seed,
                }
            )
        except Exception as exc:  # noqa: BLE001
            manifest.append(
                {
                    "row_index": index,
                    "status": "error",
                    "image_filename": None,
                    "error": str(exc),
                    "seed": seed,
                }
            )

    if warnings:
        (output_dir / "warnings.txt").write_text("\n".join(warnings) + "\n", encoding="utf-8")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Saved {len(manifest)} image(s) to {output_dir}")
    print(f"Manifest: {manifest_path}")
    if warnings:
        print(f"Warnings: {output_dir / 'warnings.txt'}")

    return 0


def resolve_output_dir(output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir)
    stamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    return Path("scripts/output") / f"loadgen_{stamp}"


def read_first_rows(path: Path, count: int) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("--count must be positive")
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pyarrow is required to read parquet files. Run this script in the same Python environment "
            "you use for your Qwen/vLLM setup."
        ) from exc

    parquet_file = pq.ParquetFile(path)
    rows: list[dict[str, Any]] = []
    for batch in parquet_file.iter_batches(batch_size=count):
        rows.extend(pa.Table.from_batches([batch]).to_pylist())
        if len(rows) >= count:
            break
    return rows[:count]


def fetch_default_model(server_url: str, timeout: int) -> str:
    models = get_json(f"{server_url}/v1/models", timeout=timeout)
    data = models.get("data") or []
    if not data:
        raise RuntimeError("No models returned from /v1/models")
    model = data[0].get("id")
    if not isinstance(model, str) or not model:
        raise RuntimeError(f"Unexpected /v1/models payload: {json.dumps(models, ensure_ascii=False)}")
    return model


def get_json(url: str, timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(url, method="GET")
    return request_json(request, timeout=timeout)


def post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    return request_json(request, timeout=timeout)


def request_json(request: urllib.request.Request, timeout: int) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {request.full_url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request failed for {request.full_url}: {exc}") from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON from {request.full_url}: {body[:500]}") from exc


def resolve_prompt_row(
    row: dict[str, Any],
    row_index: int,
    prompt_key: str,
    negative_prompt_key: str,
    request_key: str | None,
    warnings: list[str],
) -> dict[str, Any]:
    prompt_value = row.get(prompt_key)
    negative_prompt_value = row.get(negative_prompt_key)

    prompt_messages = normalize_messages(prompt_value)
    negative_prompt_messages = normalize_messages(negative_prompt_value)

    request_key_used = request_key
    user_prompt = None

    if request_key and request_key in row:
        user_prompt = extract_textish_value(row[request_key])
    if user_prompt is None and prompt_messages is not None:
        user_prompt = extract_last_user_text(prompt_messages)
        request_key_used = prompt_key
    if user_prompt is None:
        for candidate in REQUEST_KEY_CANDIDATES:
            if candidate in row:
                user_prompt = extract_textish_value(row[candidate])
                if user_prompt is not None:
                    request_key_used = candidate
                    break
    if user_prompt is None:
        raise RuntimeError(
            f"Could not infer request text for row {row_index}. Available keys: {sorted(str(k) for k in row.keys())}"
        )

    if prompt_messages is None:
        prompt_messages = build_training_prompt_messages(user_prompt)

    if negative_prompt_messages is None:
        negative_prompt_user = DEFAULT_NEGATIVE_PROMPT
        negative_prompt_messages = build_training_prompt_messages(negative_prompt_user)
        negative_prompt_source = "hardcoded_training_default"
    else:
        negative_prompt_user = extract_last_user_text(negative_prompt_messages) or DEFAULT_NEGATIVE_PROMPT
        negative_prompt_source = negative_prompt_key

    system_prompt = extract_first_system_text(prompt_messages)
    system_prompt_matches_training = system_prompt == TRAIN_SYSTEM_PROMPT if system_prompt is not None else True
    if system_prompt is not None and not system_prompt_matches_training:
        warnings.append(
            f"Row {row_index}: dataset system prompt differs from Qwen-Image OCR training prompt. "
            "The server still receives only the user prompt, because Qwen-Image serving wraps it with the "
            "hardcoded training system template."
        )

    return {
        "user_prompt": user_prompt,
        "negative_prompt_user": negative_prompt_user,
        "training_prompt_messages": prompt_messages,
        "training_negative_prompt_messages": negative_prompt_messages,
        "prompt_source": prompt_key if prompt_value is not None else "rebuilt_from_request",
        "negative_prompt_source": negative_prompt_source,
        "request_key_used": request_key_used,
        "system_prompt_matches_training": system_prompt_matches_training,
    }


def normalize_messages(value: Any) -> list[dict[str, Any]] | None:
    if value is None:
        return None
    if isinstance(value, list):
        if all(isinstance(item, dict) for item in value):
            return value
        return None
    return None


def build_training_prompt_messages(user_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": TRAIN_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]


def extract_textish_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        messages = normalize_messages(value)
        if messages is not None:
            return extract_last_user_text(messages)
    if isinstance(value, dict):
        if "text" in value and isinstance(value["text"], str):
            return value["text"]
        if "content" in value:
            return extract_content_text(value["content"])
    return None


def extract_last_user_text(messages: list[dict[str, Any]]) -> str | None:
    for message in reversed(messages):
        if message.get("role") == "user":
            return extract_content_text(message.get("content"))
    return None


def extract_first_system_text(messages: list[dict[str, Any]]) -> str | None:
    for message in messages:
        if message.get("role") == "system":
            return extract_content_text(message.get("content"))
    return None


def extract_content_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    chunks.append(item["text"])
                elif isinstance(item.get("content"), str):
                    chunks.append(item["content"])
        if chunks:
            return "\n".join(chunks)
    return None


def generate_image(
    server_url: str,
    model: str,
    user_prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    true_cfg_scale: float,
    seed: int | None,
    timeout: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": user_prompt}],
        "stream": False,
        "extra_body": {
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "true_cfg_scale": true_cfg_scale,
            "negative_prompt": negative_prompt,
        },
    }
    if seed is not None:
        payload["extra_body"]["seed"] = seed
    return post_json(f"{server_url}/v1/chat/completions", payload, timeout=timeout)


def save_image_from_response(response: dict[str, Any], image_path: Path) -> Path:
    data_url = extract_first_image_data_url(response)
    match = DATA_URL_RE.match(data_url)
    if not match:
        raise RuntimeError("Expected a base64 image data URL in chat completion response")
    image_format, encoded = match.groups()
    image_bytes = base64.b64decode(encoded)

    expected_suffix = format_to_suffix(image_format)
    if image_path.suffix.lower() != expected_suffix:
        image_path = image_path.with_suffix(expected_suffix)
    image_path.write_bytes(image_bytes)
    return image_path


def extract_first_image_data_url(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"Unexpected chat completion response: {json.dumps(response, ensure_ascii=False)[:1000]}")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str) and content.startswith("data:image/"):
        return content
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image_url":
                image_url = item.get("image_url") or {}
                url = image_url.get("url")
                if isinstance(url, str) and url.startswith("data:image/"):
                    return url
    raise RuntimeError(f"Could not find image data URL in response: {json.dumps(response, ensure_ascii=False)[:1000]}")


def format_to_suffix(image_format: str) -> str:
    normalized = image_format.lower()
    if normalized in {"jpeg", "jpg"}:
        return ".jpg"
    if normalized == "svg+xml":
        return ".svg"
    return f".{normalized}"


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
