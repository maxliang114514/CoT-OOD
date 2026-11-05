#!/bin/bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${SCRIPT_DIR}"
GSM8K_TEST="${ROOT_DIR}/datasets/gsm8k/test.jsonl"
METAMATH="${ROOT_DIR}/datasets/metamath/MetaMathQA-395K.json"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/models/qwen2.5-3b}"   # 可通过环境变量覆盖
GPU=${1:-0}
MODE=${2:-all}  # domain, complexity, magnitude, all
USE_VLLM=${3:-1}  # 设为0使用原生HF推理，设为1使用vLLM加速

if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate qwen1
fi

echo "=========================================="
echo "分布偏移实验 Qwen (mode=$MODE, GPU=$GPU, vllm=$USE_VLLM)"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=$GPU

echo ""
echo ">>> [1/4] 分析GSM8K测试集分布..."
python "${SCRIPT_DIR}/filter_distribution_shift.py" \
    --metamath_file $METAMATH \
    --gsm8k_file $GSM8K_TEST \
    --analyze_only

echo ""
echo ">>> [2/4] 生成分布偏移数据..."
python "${SCRIPT_DIR}/filter_distribution_shift.py" \
    --metamath_file $METAMATH \
    --gsm8k_file $GSM8K_TEST \
    --output_file $EXP_DIR/data/metamath_ood_${MODE}.json \
    --shift_mode $MODE \
    --shift_ratio 0.3

echo ""
echo ">>> [3/4] 训练模型..."
python "${SCRIPT_DIR}/../train_qwen.py" \
    --model_name_or_path $MODEL_PATH \
    --data_path $EXP_DIR/data/metamath_ood_${MODE}.json \
    --max_train_samples 10000 \
    --output_dir $EXP_DIR/qwen_ood_${MODE} \
    --overwrite_output_dir \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 2e-5 \
    --model_max_length 512 \
    --lora_rank 16 --lora_alpha 16 \
    --bf16

echo ""
echo ">>> [4/4] 测试模型..."
if [ "$USE_VLLM" = "1" ]; then
    python "${SCRIPT_DIR}/../gen_response_vllm.py" \
        --model $EXP_DIR/qwen_ood_${MODE} \
        --data_file $GSM8K_TEST \
        --output_file $EXP_DIR/results/qwen_ood_${MODE}.jsonl
else
    python "${SCRIPT_DIR}/../gen_response_qwen.py" \
        --model $EXP_DIR/qwen_ood_${MODE} \
        --data_file $GSM8K_TEST \
        --output_file $EXP_DIR/results/qwen_ood_${MODE}.jsonl
fi

python "${SCRIPT_DIR}/../baseline/test_acc.py" \
    --input_file $EXP_DIR/results/qwen_ood_${MODE}.jsonl \
    --output_file $EXP_DIR/results/qwen_ood_${MODE}_acc.json

echo ""
echo "=========================================="
echo "分布偏移实验完成 (mode=$MODE)"
echo "=========================================="