#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Benchmark docker push throughput for:
- Docker Hub official registry
- GitHub Container Registry (GHCR)
- Aliyun Container Registry (ACR)
- Google Artifact Registry (GAR)

No sweep args are needed.

Usage:
  scripts/bench_docker_push_mirrors.sh
  scripts/bench_docker_push_mirrors.sh --help

Optional env vars:
  SIZE_MB        Payload layer size in MiB. Default: 256
  REPO_NAME      Repo/image base name. Default: docker-push-bench
  TAG_PREFIX     Tag prefix. Default: pushbench
  ENABLE_DOCKERHUB  1/0. Default: 1
  ENABLE_GHCR       1/0. Default: 1
  ENABLE_ACR        1/0. Default: 1
  ENABLE_GAR        1/0. Default: 1

  DOCKERHUB_USER Docker Hub username. Auto-detect if unset.
  GHCR_OWNER     GHCR owner (user or org). Auto-detect/fallback if unset.

  ACR_PREFIX     Full ACR repo prefix to enable ACR test.
                 Example: registry.cn-hangzhou.aliyuncs.com/<namespace>
                 Example: <instance>.cr.cn-hangzhou.aliyuncs.com/<namespace>

  GAR_PREFIX     Full GAR repo prefix to enable GAR test.
                 Example: us-docker.pkg.dev/<project>/<repository>

Notes:
  1) "用户仓库" means a writable path like docker.io/<you>/<repo>.
  2) ACR/GAR require explicit prefixes because path formats vary by account.
  3) Script does not delete remote tags/images.
USAGE
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

read_auth_user_from_config() {
  local key="$1"
  local cfg="${HOME}/.docker/config.json"
  local auth_b64=""

  if [[ ! -f "$cfg" ]]; then
    return 1
  fi
  if ! command -v jq >/dev/null 2>&1; then
    return 1
  fi
  if ! command -v base64 >/dev/null 2>&1; then
    return 1
  fi

  auth_b64="$(jq -r --arg k "$key" '.auths[$k].auth // empty' "$cfg" 2>/dev/null || true)"
  if [[ -z "$auth_b64" ]]; then
    return 1
  fi

  printf '%s' "$auth_b64" | base64 -d 2>/dev/null | cut -d: -f1
}

detect_dockerhub_user() {
  local u=""

  if [[ -n "${DOCKERHUB_USER:-}" ]]; then
    echo "${DOCKERHUB_USER}"
    return 0
  fi

  u="$(docker info --format '{{.Username}}' 2>/dev/null || true)"
  if [[ -n "$u" && "$u" != "<no value>" ]]; then
    echo "$u"
    return 0
  fi

  for key in "https://index.docker.io/v1/" "index.docker.io" "registry-1.docker.io" "docker.io"; do
    u="$(read_auth_user_from_config "$key" || true)"
    if [[ -n "$u" ]]; then
      echo "$u"
      return 0
    fi
  done

  return 1
}

detect_ghcr_owner() {
  local fallback_user="${1:-}"
  local u=""

  if [[ -n "${GHCR_OWNER:-}" ]]; then
    echo "${GHCR_OWNER}"
    return 0
  fi

  if [[ -n "${GITHUB_USER:-}" ]]; then
    echo "${GITHUB_USER}"
    return 0
  fi

  for key in "ghcr.io" "https://ghcr.io/v1/" "https://ghcr.io"; do
    u="$(read_auth_user_from_config "$key" || true)"
    if [[ -n "$u" ]]; then
      echo "$u"
      return 0
    fi
  done

  if [[ -n "$fallback_user" ]]; then
    echo "$fallback_user"
    return 0
  fi

  return 1
}

add_target() {
  TARGET_NAMES+=("$1")
  TARGET_REPOS+=("$2")
}

add_skip() {
  SKIP_NAMES+=("$1")
  SKIP_REASONS+=("$2")
}

if [[ $# -gt 0 ]]; then
  case "${1:-}" in
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "This script takes no sweep args. Run with no args, or --help." >&2
      exit 1
      ;;
  esac
fi

SIZE_MB="${SIZE_MB:-256}"
REPO_NAME="${REPO_NAME:-docker-push-bench}"
TAG_PREFIX="${TAG_PREFIX:-pushbench}"
ENABLE_DOCKERHUB="${ENABLE_DOCKERHUB:-1}"
ENABLE_GHCR="${ENABLE_GHCR:-1}"
ENABLE_ACR="${ENABLE_ACR:-1}"
ENABLE_GAR="${ENABLE_GAR:-1}"
ACR_PREFIX="${ACR_PREFIX:-}"
GAR_PREFIX="${GAR_PREFIX:-}"

if ! [[ "$SIZE_MB" =~ ^[0-9]+$ ]] || [[ "$SIZE_MB" -le 0 ]]; then
  echo "SIZE_MB must be a positive integer." >&2
  exit 1
fi

for flag in "$ENABLE_DOCKERHUB" "$ENABLE_GHCR" "$ENABLE_ACR" "$ENABLE_GAR"; do
  if [[ "$flag" != "0" && "$flag" != "1" ]]; then
    echo "ENABLE_* flags must be 0 or 1." >&2
    exit 1
  fi
done

require_cmd docker
require_cmd dd
require_cmd awk
require_cmd mktemp
require_cmd stat
require_cmd date
require_cmd cut

RUN_ID="$(date +%Y%m%d%H%M%S)"
TMP_DIR="$(mktemp -d -t docker-push-bench.XXXXXX)"
RESULT_FILE="/tmp/docker-push-bench-${RUN_ID}.tsv"
BASE_IMAGE="local/push-bench:${RUN_ID}"

cleanup() {
  docker image rm -f "$BASE_IMAGE" >/dev/null 2>&1 || true
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

rm -f "$RESULT_FILE"

declare -a TARGET_NAMES=()
declare -a TARGET_REPOS=()
declare -a SKIP_NAMES=()
declare -a SKIP_REASONS=()

HUB_USER=""
if [[ "$ENABLE_DOCKERHUB" == "1" || "$ENABLE_GHCR" == "1" ]]; then
  HUB_USER="$(detect_dockerhub_user || true)"
fi

if [[ "$ENABLE_DOCKERHUB" == "1" && -n "$HUB_USER" ]]; then
  add_target "dockerhub-official" "docker.io/${HUB_USER}/${REPO_NAME}"
elif [[ "$ENABLE_DOCKERHUB" == "1" ]]; then
  add_skip "dockerhub-official" "set DOCKERHUB_USER or docker login docker.io"
else
  add_skip "dockerhub-official" "disabled by ENABLE_DOCKERHUB=0"
fi

GHCR_USER=""
if [[ "$ENABLE_GHCR" == "1" ]]; then
  GHCR_USER="$(detect_ghcr_owner "$HUB_USER" || true)"
fi
if [[ "$ENABLE_GHCR" == "1" && -n "$GHCR_USER" ]]; then
  add_target "ghcr" "ghcr.io/${GHCR_USER}/${REPO_NAME}"
elif [[ "$ENABLE_GHCR" == "1" ]]; then
  add_skip "ghcr" "set GHCR_OWNER or GITHUB_USER (and docker login ghcr.io)"
else
  add_skip "ghcr" "disabled by ENABLE_GHCR=0"
fi

if [[ "$ENABLE_ACR" == "1" && -n "$ACR_PREFIX" ]]; then
  add_target "aliyun-acr" "${ACR_PREFIX%/}/${REPO_NAME}"
elif [[ "$ENABLE_ACR" == "1" ]]; then
  add_skip "aliyun-acr" "set ACR_PREFIX to enable"
else
  add_skip "aliyun-acr" "disabled by ENABLE_ACR=0"
fi

if [[ "$ENABLE_GAR" == "1" && -n "$GAR_PREFIX" ]]; then
  add_target "google-artifact-registry" "${GAR_PREFIX%/}/${REPO_NAME}"
elif [[ "$ENABLE_GAR" == "1" ]]; then
  add_skip "google-artifact-registry" "set GAR_PREFIX to enable"
else
  add_skip "google-artifact-registry" "disabled by ENABLE_GAR=0"
fi

if [[ ${#TARGET_REPOS[@]} -eq 0 ]]; then
  echo "No enabled targets to test."
  for i in "${!SKIP_NAMES[@]}"; do
    echo "- ${SKIP_NAMES[$i]}: ${SKIP_REASONS[$i]}"
  done
  exit 1
fi

echo "Preparing payload (${SIZE_MB} MiB) ..."
dd if=/dev/urandom of="$TMP_DIR/payload.bin" bs=1M count="$SIZE_MB" status=none

cat > "$TMP_DIR/Dockerfile" <<'DOCKERFILE'
FROM scratch
COPY payload.bin /payload.bin
DOCKERFILE

echo "Building temporary image: $BASE_IMAGE"
docker build --no-cache -t "$BASE_IMAGE" "$TMP_DIR" >/dev/null

PAYLOAD_BYTES="$(stat -c%s "$TMP_DIR/payload.bin")"
PAYLOAD_MIB="$(awk -v b="$PAYLOAD_BYTES" 'BEGIN { printf "%.2f", b/1024/1024 }')"

echo
echo "Push benchmark started (payload layer: ${PAYLOAD_MIB} MiB)"
echo
printf "target\tstatus\tseconds\test_mib_per_sec\timage_or_reason\n" > "$RESULT_FILE"

for i in "${!SKIP_NAMES[@]}"; do
  printf "%s\tskip\t0\t0\t%s\n" "${SKIP_NAMES[$i]}" "${SKIP_REASONS[$i]}" >> "$RESULT_FILE"
done

for i in "${!TARGET_REPOS[@]}"; do
  name="${TARGET_NAMES[$i]}"
  target_repo="${TARGET_REPOS[$i]}"
  image_ref="${target_repo}:${TAG_PREFIX}-${RUN_ID}"

  docker image tag "$BASE_IMAGE" "$image_ref"
  echo "Pushing -> [$name] $image_ref"

  start_ns="$(date +%s%N)"
  if push_output="$(docker push "$image_ref" 2>&1)"; then
    end_ns="$(date +%s%N)"
    elapsed_ns="$((end_ns - start_ns))"
    elapsed_s="$(awk -v ns="$elapsed_ns" 'BEGIN { printf "%.3f", ns/1000000000 }')"
    mibps="$(awk -v m="$PAYLOAD_MIB" -v s="$elapsed_s" 'BEGIN { if (s > 0) printf "%.2f", m/s; else print "inf" }')"
    printf "%s\tok\t%s\t%s\t%s\n" "$name" "$elapsed_s" "$mibps" "$image_ref" >> "$RESULT_FILE"
    echo "  ok: ${elapsed_s}s, ~${mibps} MiB/s"
  else
    end_ns="$(date +%s%N)"
    elapsed_ns="$((end_ns - start_ns))"
    elapsed_s="$(awk -v ns="$elapsed_ns" 'BEGIN { printf "%.3f", ns/1000000000 }')"
    err_last_line="$(printf "%s\n" "$push_output" | tail -n 1)"
    printf "%s\tfail\t%s\t0\t%s\n" "$name" "$elapsed_s" "$err_last_line" >> "$RESULT_FILE"
    echo "  fail after ${elapsed_s}s: $err_last_line"
  fi
done

echo
echo "Summary (sorted by est_mib_per_sec desc):"
if command -v column >/dev/null 2>&1; then
  {
    head -n 1 "$RESULT_FILE"
    tail -n +2 "$RESULT_FILE" | sort -t$'\t' -k4,4nr
  } | column -t -s $'\t'
else
  {
    head -n 1 "$RESULT_FILE"
    tail -n +2 "$RESULT_FILE" | sort -t$'\t' -k4,4nr
  }
fi

echo
echo "Raw result file: $RESULT_FILE"
