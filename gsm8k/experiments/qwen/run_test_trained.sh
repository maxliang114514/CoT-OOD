#!/bin/bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_FILE="${ROOT_DIR}/datasets/gsm8k/test.jsonl"
USE_VLLM=${USE_VLLM:-1}  # 设为0使用原生HF推理，设为1使用vLLM加速

if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate qwen1
fi

QWEN_MODEL="${QWEN_MODEL:-${ROOT_DIR}/experiments/qwen/output/cot_qwen2.5_3b_metamath}"
QWEN_OUTPUT="${SCRIPT_DIR}/trained_qwen_unified_response.jsonl"
QWEN_ACC="${SCRIPT_DIR}/trained_qwen_unified_acc.json"

echo "======================================"
echo "训练后模型测试 - Qwen2.5-3B"
echo "======================================"

echo "--- 生成回复 (vLLM加速) ---"
if [ "$USE_VLLM" = "1" ]; then
    python "${SCRIPT_DIR}/gen_response_vllm.py" \
        --model $QWEN_MODEL \
        --data_file $DATA_FILE \
        --output_file $QWEN_OUTPUT \
        --temperature 0.0 \
        --top_p 1.0 \
        --max_tokens 512
else
    python "${SCRIPT_DIR}/gen_response_qwen.py" \
        --model $QWEN_MODEL \
        --data_file $DATA_FILE \
        --output_file $QWEN_OUTPUT \
        --temperature 0.0 \
        --top_p 1.0 \
        --max_tokens 512
fi

echo ""
echo "--- 准确率 ---"
python "${SCRIPT_DIR}/baseline/test_acc.py" --input_file $QWEN_OUTPUT --output_file $QWEN_ACC

echo ""
echo "======================================"
echo "测试完成！汇总结果："
echo "======================================"
echo "Qwen2.5-3B:  $(cat $QWEN_ACC | python -c 'import json,sys; d=json.load(sys.stdin); print(d["summary"]["accuracy_percent"])')"