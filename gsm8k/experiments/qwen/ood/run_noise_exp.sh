#!/bin/bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${SCRIPT_DIR}"
GPU=${1:-0}
RATE=${2:-0.1}
USE_VLLM=${3:-1}  # 设为0使用原生HF推理，设为1使用vLLM加速
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/models/qwen2.5-3b}"   # 可通过环境变量覆盖
METAMATH="${ROOT_DIR}/datasets/metamath/MetaMathQA-395K.json"
GSM8K_TEST="${ROOT_DIR}/datasets/gsm8k/test.jsonl"

if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate qwen1
fi

echo "=========================================="
echo "噪声注入实验 Qwen (rate=$RATE, GPU=$GPU, vllm=$USE_VLLM)"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=$GPU

echo ""
echo ">>> [1/3] 生成噪声数据..."
if [ "$RATE" == "0" ] || [ "$RATE" == "0.0" ]; then
    cp $METAMATH $EXP_DIR/data/metamath_noise${RATE}.json
else
    python "${SCRIPT_DIR}/add_noise.py" \
        --input_file $METAMATH \
        --output_file $EXP_DIR/data/metamath_noise${RATE}.json \
        --noise_rate $RATE \
        --noise_type both
fi

echo ""
echo ">>> [2/3] 训练模型..."
python "${SCRIPT_DIR}/../train_qwen.py" \
    --model_name_or_path $MODEL_PATH \
    --data_path $EXP_DIR/data/metamath_noise${RATE}.json \
    --max_train_samples 10000 \
    --output_dir $EXP_DIR/qwen_noise${RATE} \
    --overwrite_output_dir \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 2e-5 \
    --model_max_length 512 \
    --lora_rank 16 --lora_alpha 16 \
    --bf16

echo ""
echo ">>> [3/3] 测试模型..."
if [ "$USE_VLLM" = "1" ]; then
    python "${SCRIPT_DIR}/../gen_response_vllm.py" \
        --model $EXP_DIR/qwen_noise${RATE} \
        --data_file $GSM8K_TEST \
        --output_file $EXP_DIR/results/qwen_noise${RATE}.jsonl
else
    python "${SCRIPT_DIR}/../gen_response_qwen.py" \
        --model $EXP_DIR/qwen_noise${RATE} \
        --data_file $GSM8K_TEST \
        --output_file $EXP_DIR/results/qwen_noise${RATE}.jsonl
fi

python "${SCRIPT_DIR}/../baseline/test_acc.py" \
    --input_file $EXP_DIR/results/qwen_noise${RATE}.jsonl \
    --output_file $EXP_DIR/results/qwen_noise${RATE}_acc.json

echo ""
echo "=========================================="
echo "噪声实验完成 (rate=$RATE)"
echo "=========================================="