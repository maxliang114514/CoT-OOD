#!/bin/bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${SCRIPT_DIR}"
GPU=${1:-0}
MIN_STEPS=${2:-2}
USE_VLLM=${3:-1}  # 设为0使用原生HF推理，设为1使用vLLM加速
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/models/llama2-7b}"   # 可通过环境变量覆盖
METAMATH="${ROOT_DIR}/datasets/metamath/MetaMathQA-395K.json"
GSM8K_TEST="${ROOT_DIR}/datasets/gsm8k/test.jsonl"

echo "=========================================="
echo "单步实验 (min_steps=$MIN_STEPS, GPU=$GPU, vllm=$USE_VLLM)"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=$GPU

if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate qwen1
fi

echo ""
echo ">>> [1/3] 生成单步数据..."
python "${SCRIPT_DIR}/create_stepwise.py" \
    --input_file $METAMATH \
    --output_file $EXP_DIR/data/metamath_stepwise.jsonl \
    --filter_min_steps $MIN_STEPS \
    --max_samples 10000

echo ""
echo ">>> [2/3] 训练模型..."
python "${SCRIPT_DIR}/../cot_no_cot/train.py" \
    --model_name_or_path $MODEL_PATH \
    --data_path $EXP_DIR/data/metamath_stepwise.jsonl \
    --max_train_samples 100000 \
    --output_dir $EXP_DIR/llama_stepwise \
    --overwrite_output_dir \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 2e-5 \
    --model_max_length 512 \
    --lora_rank 16 --lora_alpha 16 \
    --lora_target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
    --bf16

echo ""
echo ">>> [3/3] 测试模型..."
if [ "$USE_VLLM" = "1" ]; then
    python "${SCRIPT_DIR}/../scripts/gen_response_vllm.py" \
        --model $EXP_DIR/llama_stepwise \
        --data_file $GSM8K_TEST \
        --output_file $EXP_DIR/results/llama_stepwise.jsonl
else
    python "${SCRIPT_DIR}/../scripts/gen_response.py" \
        --model $EXP_DIR/llama_stepwise \
        --data_file $GSM8K_TEST \
        --output_file $EXP_DIR/results/llama_stepwise.jsonl
fi

python "${SCRIPT_DIR}/../baseline/test_acc.py" \
    --input_file $EXP_DIR/results/llama_stepwise.jsonl \
    --output_file $EXP_DIR/results/llama_stepwise_acc.json

echo ""
echo "=========================================="
echo "单步实验完成"
echo "=========================================="