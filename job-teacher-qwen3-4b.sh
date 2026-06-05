#!/bin/bash
#SBATCH -J teacher_qwen35_27b
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -o logs/teacher_%j.out
#SBATCH -e logs/teacher_%j.err
#SBATCH -w gpu01

set -euo pipefail

echo "======================================================"
echo "🚀 正在计算节点 $(hostname) 上初始化 Teacher 服务..."
echo "======================================================"

# 1. 加载运行环境 (按需取消注释)
ml miniconda/26.1.1
ml cuda/12.8

eval "$(conda shell.bash hook)"
conda activate vllm19

# 2. 配置参数
TEACHER_MODEL_NAME="/data/zrliu/models/Qwen/Qwen3-4B-Instruct-2507"
PORT=8000
HOST="0.0.0.0" 

echo "📦 模型路径: $TEACHER_MODEL_NAME"
echo "🖥️  分配 GPU: $CUDA_VISIBLE_DEVICES"

vllm serve "$TEACHER_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --gpu-memory-utilization 0.85 \
  --served-model-name "Qwen3-4B-Instruct-2507" \
  --max-model-len 32768
