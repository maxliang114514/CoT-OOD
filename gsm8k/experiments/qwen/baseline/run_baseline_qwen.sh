#!/bin/bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/models/qwen2.5-3b}"   # 可通过环境变量覆盖
DATA_FILE="${ROOT_DIR}/datasets/gsm8k/test.jsonl"
USE_VLLM=${USE_VLLM:-1}  # 设为0使用原生HF推理，设为1使用vLLM加速

if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate qwen1
fi

echo "======================================"
echo "Qwen2.5-3B 基线测试 (全量)"
echo "======================================"

echo ""
echo "=== [1] T-CoT统一格式 (vLLM加速) ==="
if [ "$USE_VLLM" = "1" ]; then
    python "${SCRIPT_DIR}/../gen_response_vllm.py" \
        --model $MODEL_PATH \
        --data_file $DATA_FILE \
        --output_file baseline_qwen_unified.jsonl \
        --temperature 0.0 \
        --top_p 1.0 \
        --max_tokens 512
else
    python "${SCRIPT_DIR}/../gen_response_qwen.py" \
        --model $MODEL_PATH \
        --data_file $DATA_FILE \
        --output_file baseline_qwen_unified.jsonl \
        --temperature 0.0 \
        --top_p 1.0 \
        --max_tokens 512
fi

echo ""
echo "--- Unified格式准确率 ---"
python "${SCRIPT_DIR}/test_acc.py" --input_file baseline_qwen_unified.jsonl