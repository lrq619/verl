#!/usr/bin/env bash
set -euo pipefail

# 0) Paths
RAW_DIR="/workspace/data/ocr"
OUT_DIR="/workspace/data/ocr"

# 1) Install tools
# uv pip install -U "datasets>=2.20" "pyarrow>=19" pandas

# 2) Download raw OCR dataset from flow_grpo repo
# Source: https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr
mkdir -p "${RAW_DIR%/ocr}"
if [ ! -d "${RAW_DIR}" ]; then
  git clone --depth 1 https://github.com/yifan123/flow_grpo.git /tmp/flow_grpo_repo
  mkdir -p "${RAW_DIR}"
  cp -r /tmp/flow_grpo_repo/dataset/ocr/* "${RAW_DIR}/"
  rm -rf /tmp/flow_grpo_repo
fi

# 3) Convert to verl parquet format
mkdir -p "${OUT_DIR}"
python examples/data_preprocess/qwenimage_ocr.py \
  --local_dataset_path "${RAW_DIR}" \
  --local_save_dir "${OUT_DIR}"

# 4) Verify outputs
ls -lh "${OUT_DIR}/train.parquet" "${OUT_DIR}/test.parquet"
echo "Done. Parquet files are in ${OUT_DIR}"

