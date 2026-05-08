#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"

create_clone_if_missing() {
  local target_env="$1"
  local base_env="$2"
  if conda env list | awk '{print $1}' | grep -Fxq "$target_env"; then
    echo "[跳过] 环境已存在: $target_env"
    return
  fi
  echo "[创建] conda create -n $target_env --clone $base_env"
  conda create -n "$target_env" --clone "$base_env" -y
}

pip_install_quiet() {
  local env_name="$1"
  shift
  conda activate "$env_name"
  pip install -q "$@"
  conda deactivate
}

echo "[1/3] 创建系列隔离环境（按厂商+系列）..."
create_clone_if_missing "eriq-qwen2.5-vl-3b" "eriq-qwen"
create_clone_if_missing "eriq-qwen3-vl-2b" "eriq-qwen"
create_clone_if_missing "eriq-qwen3.5-vl-0.8b" "eriq-qwen"
create_clone_if_missing "eriq-llava-0.5b" "eriq-qwen"
create_clone_if_missing "eriq-minicpm-v-2.5" "eriq-qwen"
create_clone_if_missing "eriq-smolvlm" "eriq-qwen"
create_clone_if_missing "eriq-phi-vision" "eriq-qwen"
create_clone_if_missing "eriq-internvl2.5-1b" "eriq-internvl"
create_clone_if_missing "eriq-internvl3-1b" "eriq-internvl"
create_clone_if_missing "eriq-internvl3.5-1b" "eriq-internvl"

echo "[2/3] 按官方文档固定关键依赖版本..."
# InternVL2.5/3 README: transformers>=4.37.2；InternVL3 config 标注 4.48.3，优先对齐实测稳定版本
pip_install_quiet "eriq-internvl2.5-1b" "transformers==4.48.3" "einops" "timm"
pip_install_quiet "eriq-internvl3-1b" "transformers==4.48.3" "einops" "timm"
# InternVL3.5 README: transformers>=4.52.1（20B 需 >=4.55.0），此处统一使用 4.55.0
pip_install_quiet "eriq-internvl3.5-1b" "transformers==4.55.0" "einops" "timm"

# LLaVA OneVision README: transformers >= 4.45.0
pip_install_quiet "eriq-llava-0.5b" "transformers>=4.45.0"
# MiniCPM-V2.5 README 示例环境: transformers==4.40.0
pip_install_quiet "eriq-minicpm-v-2.5" "transformers==4.40.0"
# SmolVLM/Molmo/Phi-3(.5)-Vision 依赖较新 transformers，顺带对齐 huggingface_hub 避免导入冲突
pip_install_quiet "eriq-smolvlm" "transformers==4.57.1" "huggingface_hub>=0.34.0" "accelerate>=0.34.0" "sentencepiece"
pip_install_quiet "eriq-phi-vision" "transformers==4.57.1" "huggingface_hub>=0.34.0" "accelerate>=0.34.0" "sentencepiece"

echo "[3/3] 完成。当前可用隔离环境："
conda env list
