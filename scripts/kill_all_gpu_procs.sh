#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: kill_all_gpu_procs.sh [--dry-run] [--grace-seconds N]

Options:
  --dry-run          Only print target PIDs, do not kill.
  --grace-seconds N  Seconds to wait after SIGTERM before SIGKILL (default: 3).
  -h, --help         Show this help.
EOF
}

dry_run=0
grace_seconds=3

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      dry_run=1
      shift
      ;;
    --grace-seconds)
      grace_seconds="${2:-}"
      if [[ -z "$grace_seconds" || ! "$grace_seconds" =~ ^[0-9]+$ ]]; then
        echo "Invalid --grace-seconds value: ${grace_seconds:-<empty>}" >&2
        exit 1
      fi
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found." >&2
  exit 1
fi

mapfile -t raw_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)

declare -A uniq=()
pids=()
for p in "${raw_pids[@]}"; do
  p="${p//[[:space:]]/}"
  if [[ "$p" =~ ^[0-9]+$ ]] && [[ "$p" -gt 0 ]] && [[ -z "${uniq[$p]+x}" ]]; then
    uniq["$p"]=1
    pids+=("$p")
  fi
done

if [[ "${#pids[@]}" -eq 0 ]]; then
  echo "No GPU compute processes found."
  exit 0
fi

echo "GPU process PIDs: ${pids[*]}"
echo "Process details:"
ps -fp "$(IFS=,; echo "${pids[*]}")" || true

if [[ "$dry_run" -eq 1 ]]; then
  echo "Dry run only. No process killed."
  exit 0
fi

echo "Sending SIGTERM..."
for pid in "${pids[@]}"; do
  kill -TERM "$pid" 2>/dev/null || true
done

sleep "$grace_seconds"

remaining=()
for pid in "${pids[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    remaining+=("$pid")
  fi
done

if [[ "${#remaining[@]}" -eq 0 ]]; then
  echo "All GPU processes exited after SIGTERM."
  exit 0
fi

echo "Still alive after ${grace_seconds}s: ${remaining[*]}"
echo "Sending SIGKILL..."
for pid in "${remaining[@]}"; do
  kill -KILL "$pid" 2>/dev/null || true
done

sleep 1

survivors=()
for pid in "${remaining[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    survivors+=("$pid")
  fi
done

if [[ "${#survivors[@]}" -eq 0 ]]; then
  echo "Done. All target GPU processes killed."
  exit 0
fi

echo "Warning: some PIDs still alive: ${survivors[*]}" >&2
exit 1
