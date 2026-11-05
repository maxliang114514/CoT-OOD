#!/bin/bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/models/llama2-7b}"   # 可通过环境变量覆盖
DATA_FILE="${ROOT_DIR}/datasets/gsm8k/test.jsonl"
OUTPUT_FILE="${SCRIPT_DIR}/baseline_response.jsonl"
USE_VLLM=${USE_VLLM:-1}  # 设为0使用原生HF推理，设为1使用vLLM加速

if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate qwen1
fi

echo "======================================"
echo "T-CotMechanism风格 基线测试 (全量)"
echo "======================================"
echo "Model: $MODEL_PATH"
echo "Data: $DATA_FILE"
echo "Output: $OUTPUT_FILE"
echo "vLLM: $USE_VLLM"
echo "======================================"

if [ "$USE_VLLM" = "1" ]; then
    python "${SCRIPT_DIR}/../scripts/gen_response_vllm.py" \
        --model $MODEL_PATH \
        --data_file $DATA_FILE \
        --output_file $OUTPUT_FILE \
        --max_samples 0 \
        --temperature 0.0 \
        --top_p 1.0 \
        --max_tokens 512
else
    python "${SCRIPT_DIR}/../scripts/gen_response.py" \
        --model $MODEL_PATH \
        --data_file $DATA_FILE \
        --output_file $OUTPUT_FILE \
        --max_samples 0 \
        --temperature 0.0 \
        --top_p 1.0 \
        --max_tokens 512
fi

echo ""
echo "======================================"
echo "Testing Accuracy"
echo "======================================"

python "${SCRIPT_DIR}/test_acc.py" --input_file $OUTPUT_FILE