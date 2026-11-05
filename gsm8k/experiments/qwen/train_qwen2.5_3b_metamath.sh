#!/bin/bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_MODEL="${MODEL_PATH:-${ROOT_DIR}/models/qwen2.5-3b}"   # 可通过环境变量覆盖
OUTPUT_PATH="${ROOT_DIR}/experiments/qwen/output/cot_qwen2.5_3b_metamath"
DATA_PATH="${ROOT_DIR}/datasets/metamath/MetaMathQA-395K.json"
MAX_TRAIN_SAMPLES=100000  # T-CoT使用前10万条训练

# - learning_rate: 2e-5
# - lora_rank: 16, lora_alpha: 16
# - max_length: 512
# - per_device_train_batch_size: 1
# - num_train_epochs: 1

python train_qwen.py \
    --model_name_or_path $BASE_MODEL \
    --data_path $DATA_PATH \
    --max_train_samples $MAX_TRAIN_SAMPLES \
    --output_dir $OUTPUT_PATH \
    --overwrite_output_dir \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --model_max_length 512 \
    --lora_rank 16 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --bf16
