#!/bin/bash

# export PATH="/path/to/vllm-env/bin:$PATH"

if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SRC_DIR="${ROOT_DIR}/src"
PROJECT_DIR="${ROOT_DIR}/experiments/table1_main_cot"
cd "$PROJECT_DIR"

MODEL_PATH=${1:-"./output/final_model"}
DATA_FILE=${2:-""}
OUTPUT_FILE=${3:-"predictions.jsonl"}
MAX_SAMPLES=${4:-500}

echo "========================================"
echo "Atom Task 预测 - CoT (vLLM加速)"
echo "模型: $MODEL_PATH"
echo "数据: $DATA_FILE"
echo "输出: $OUTPUT_FILE"
echo "最大样本数: $MAX_SAMPLES"
echo "GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "========================================"

if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型路径不存在: $MODEL_PATH"
    exit 1
fi

if [ -z "$DATA_FILE" ]; then
    echo "错误: 请指定数据文件"
    exit 1
fi

if [ ! -f "$DATA_FILE" ]; then
    echo "错误: 数据文件不存在: $DATA_FILE"
    exit 1
fi

echo ""
echo "开始预测..."
echo "========================================"

mkdir -p "$(dirname "$OUTPUT_FILE")"

python "$SRC_DIR/gen_response_vllm.py" \
    --model "$MODEL_PATH" \
    --data_file "$DATA_FILE" \
    --output_file "$OUTPUT_FILE" \
    --max_samples $MAX_SAMPLES \
    --temperature 0.0 \
    --top_p 1.0 \
    --max_tokens 1024 \
    --gpu_memory_utilization 0.875

echo ""
echo "========================================"
echo "预测完成!"
echo "结果保存在: $OUTPUT_FILE"
echo "========================================"