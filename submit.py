"""
submit.py
设置环境变量，然后启动 ./run_flowgrpo.sh
"""

import os
import subprocess
import sys


def main():
    # ============================================================
    # 环境变量配置（请根据实际情况填写）
    # ============================================================

    # HuggingFace 缓存根目录，run_flowgrpo.sh 中大量路径依赖此变量
    os.environ["HF_HOME"] = "/data/oss_bucket_0/borui/hf_home"

    # 控制可见的 GPU 设备
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    os.environ["CKP_DIR"] = "/data/oss_bucket_0/borui/ckp_1"

    # vLLM RPC 超时时间（秒）
    os.environ["VERL_VLLM_RPC_TIMEOUT_S"] = "600"

    # WandB 相关（如果 logger 包含 wandb）
    os.environ["WANDB_API_KEY"] = "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"
    os.environ["WANDB_PROJECT"] = "qwen-image-ocr-nebula"

    # ============================================================
    # 启动训练脚本
    # ============================================================
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_flowgrpo_tp1_nebul.sh")

    print(f"[submit] Launching {script_path} ...")
    result = subprocess.run(
        ["bash", script_path],
        env=os.environ,
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()