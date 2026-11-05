#!/bin/bash
# ============================================================
# ============================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_FILE="${ROOT_DIR}/datasets/gsm8k/test.jsonl"

if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate qwen1
fi
USE_VLLM=${USE_VLLM:-1}  # 设为0使用原生HF推理，设为1使用vLLM加速

LLAMA_MODEL="${LLAMA_MODEL:-${ROOT_DIR}/experiments/llama/output/cot_llama2_7b_metamath}"
LLAMA_OUTPUT="${SCRIPT_DIR}/trained_llama_response.jsonl"
LLAMA_ACC="${SCRIPT_DIR}/trained_llama_acc.json"

echo "======================================"
echo "训练后模型测试 - Llama2-7B"
echo "======================================"

echo "--- 生成回复 (vLLM加速) ---"
if [ "$USE_VLLM" = "1" ]; then
    python "${SCRIPT_DIR}/../scripts/gen_response_vllm.py" \
        --model $LLAMA_MODEL \
        --data_file $DATA_FILE \
        --output_file $LLAMA_OUTPUT \
        --temperature 0.0 \
        --top_p 1.0 \
        --max_tokens 512
else
    python "${SCRIPT_DIR}/../scripts/gen_response.py" \
        --model $LLAMA_MODEL \
        --data_file $DATA_FILE \
        --output_file $LLAMA_OUTPUT \
        --temperature 0.0 \
        --top_p 1.0 \
        --max_tokens 512
fi

echo ""
echo "--- 准确率 ---"
python "${SCRIPT_DIR}/../baseline/test_acc.py" --input_file $LLAMA_OUTPUT --output_file $LLAMA_ACC

echo ""
echo "======================================"
echo "测试完成！汇总结果："
echo "======================================"
echo "Llama2-7B:   $(cat $LLAMA_ACC | python -c 'import json,sys; d=json.load(sys.stdin); print(d["summary"]["accuracy_percent"])')"