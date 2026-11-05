#!/bin/bash

# K-Scaling Experiment Script
# Studies the effect of reasoning chain length on performance
# 3 experiments: k1(short), k2(standard), k3(long)
# Seed=42, Dataset=type7_v2 (3-step)

# Interrupt mechanism
STOP_FILE=$(mktemp /tmp/k_scaling_stop_XXXXXX)
trap 'rm -f "$STOP_FILE"; exit 0' EXIT

shadow_process() {
    while [ -f "$STOP_FILE" ]; do sleep 0.5; done
    kill -TERM -$$ 2>/dev/null
}
shadow_process &
SHADOW_PID=$!

SHOULD_CONTINUE=true
cleanup() {
    echo "Received termination signal, exiting..."
    SHOULD_CONTINUE=false
    rm -f "$STOP_FILE"
    sleep 1
    kill -9 $SHADOW_PID 2>/dev/null
    exit 1
}
trap 'cleanup' SIGINT SIGTERM

check_stop() { [ -f "$STOP_FILE" ]; }
check_continue() {
    if ! check_stop || [ "$SHOULD_CONTINUE" = "false" ]; then
        echo "Detected stop signal, exiting"
        exit 1
    fi
}

# Configuration
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE_DIR="${ROOT_DIR}/experiments/table1_main_cot"
SRC_DIR="${ROOT_DIR}/src"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/models/qwen3-4b}"   # 可通过环境变量覆盖
ORIGINAL_DATASET_DIR="${ROOT_DIR}/datasets"               # 统一基础数据集
RESULTS_DIR="${BASE_DIR}/results"
DATASET_DIR="${BASE_DIR}/datasets/k_scaling"
NUM_EPOCHS=3
MAX_SAMPLES=500
SEED=42

# K-Scaling configuration
K_LEVELS=("k1" "k2" "k3")
K_DESCRIPTIONS=(
    "k1=Short (abbreviated ops: Rev, Dup, Sub, Dedup, Rot)"
    "k2=Standard (full operation names)"
    "k3=Long (with natural language explanations)"
)
ORIGINAL_DATASET="train_3step"

run_prepare_data() {
    local k_level=$1
    local input_file="${ORIGINAL_DATASET_DIR}/${ORIGINAL_DATASET}.jsonl"
    local output_file="${DATASET_DIR}/${ORIGINAL_DATASET}_${k_level}.jsonl"

    mkdir -p "${DATASET_DIR}"

    if [ -f "${output_file}" ]; then
        echo "Data already exists: ${output_file}"
    else
        echo "Generating ${k_level} format data..."
        python ${BASE_DIR}/scripts/generate_k_scaling_data.py \
            --input "${input_file}" \
            --output "${output_file}" \
            --k ${k_level}
    fi
}

run_train() {
    local train_file=$1 output_dir=$2 name=$3
    echo ">>> Starting training: ${name} (seed=${SEED})"

    export PYTHONHASHSEED=${SEED}

    TOKENIZERS_PARALLELISM=false python ${SRC_DIR}/finetune_qwen_atom.py \
        --model_path ${MODEL_PATH} --train_file ${train_file} \
        --output_dir ${output_dir} --num_train_epochs ${NUM_EPOCHS} \
        --train_mode cot_format --skip_eval &
    TRAIN_PID=$!

    while kill -0 $TRAIN_PID 2>/dev/null; do
        if ! check_stop || [ "$SHOULD_CONTINUE" = "false" ]; then
            kill -9 $TRAIN_PID 2>/dev/null
            wait $TRAIN_PID 2>/dev/null
            echo "!!! Training interrupted: ${name} (seed=${SEED})"
            return 1
        fi
        sleep 1
    done

    wait $TRAIN_PID 2>/dev/null
    [ $? -ne 0 ] && [ $? -lt 128 ] && echo "!!! Training failed: ${name} (seed=${SEED})" && return 1
    echo "<<< Training complete: ${name} (seed=${SEED})"
    return 0
}

run_test() {
    local model_dir=$1 exp_result_dir=$2 name=$3
    mkdir -p ${exp_result_dir}
    echo ">>> Starting testing: ${name} (seed=${SEED})"

    for step in 3 5 7; do
        echo "--- ${step}-step test set ---"
        local predict_file="${exp_result_dir}/predictions_${step}step.jsonl"
        local log_file="${exp_result_dir}/test_${step}step.log"
        local test_data_file="${ORIGINAL_DATASET_DIR}/test_${step}step.jsonl"

        echo "Running vLLM inference..."
        python ${SRC_DIR}/gen_response_vllm.py \
            --model ${model_dir} \
            --data_file ${test_data_file} \
            --output_file ${predict_file} \
            --max_samples ${MAX_SAMPLES} \
            --temperature 0.0 \
            --top_p 1.0 \
            --max_tokens 2048 \
            --gpu_memory_utilization 0.875 2>&1 | tee ${log_file}

        echo "Evaluating predictions..."
        python ${SRC_DIR}/eval_accuracy.py ${predict_file} --verbose 2>&1 | tee -a ${log_file}

        python ${SRC_DIR}/export_samples.py ${predict_file} ${MAX_SAMPLES} ${exp_result_dir}/export_samples_${step}step.json 2>&1 | tee -a ${log_file}
    done
    echo "<<< Testing complete: ${name} (seed=${SEED})"
}

export_results() {
    local exp_result_dir=$1 dataset_name=$2 train_file=$3 model_path=$4 k_level=$5
    {
        echo "Dataset: ${dataset_name}"
        echo "K-Scaling Level: ${k_level}"
        echo "Training Data: ${train_file}"
        echo "Model: qwen3-4b"
        echo "Training Epochs: ${NUM_EPOCHS}"
        echo "Test Set Size: ${MAX_SAMPLES} per step"
        echo "Random Seed: ${SEED}"
        echo "max_new_tokens: 2048"
        echo ""
        echo "Test Results:"
        echo "-----------------------------------"
        for step in 3step 5step 7step; do
            echo "${step}:"
            grep -E "Correct:|Accuracy:" ${exp_result_dir}/test_${step}.log 2>/dev/null | tail -2
            echo ""
        done
    } > ${exp_result_dir}/summary.txt

    python3 << PYEOF > ${exp_result_dir}/detailed_results.json
import json
import re
from pathlib import Path
results_dir = Path("${exp_result_dir}")
all_results = {}
for step in [3, 5, 7]:
    log_file = results_dir / f"test_{step}step.log"
    export_file = results_dir / f"export_samples_{step}step.json"
    if not log_file.exists(): continue
    correct = total = 0
    with open(log_file) as f:
        for line in f:
            if "Correct:" in line:
                parts = line.strip().split()
                for i, p in enumerate(parts):
                    if p == "Correct:":
                        fraction = parts[i+1]
                        correct = int(fraction.split("/")[0])
                        total = int(fraction.split("/")[1])
            if "Accuracy:" in line:
                parts = line.strip().split()
                for i, p in enumerate(parts):
                    if p == "Accuracy:":
                        pct = parts[i+1].replace("%","")
                        total = int(float(pct) * total / 100) if total else 0
    predictions, errors = [], []
    if export_file.exists():
        with open(export_file) as f:
            data = json.load(f)
            for item in data.get("samples", []):
                pred, gold = item.get("predicted", ""), item.get("gold", "")
                if pred != gold:
                    errors.append({"input": item.get("input", ""), "task_chain": item.get("task_chain", ""),
                        "predicted": pred, "gold": gold, "response": item.get("response", "")[:500]})
                predictions.append({"input": item.get("input", ""), "task_chain": item.get("task_chain", ""),
                    "predicted": pred, "gold": gold, "correct": pred == gold})
    accuracy_str = f"{correct/len(predictions)*100:.2f}%" if predictions else "N/A"
    all_results[f"{step}step"] = {"correct": correct, "total": len(predictions) or total,
        "accuracy": accuracy_str,
        "num_errors": len(errors), "sample_predictions": predictions[:20], "errors": errors[:20]}
print(json.dumps(all_results, ensure_ascii=False, indent=2))
PYEOF
    echo ">>> Results exported: ${exp_result_dir}"
}

run_k_scaling() {
    local k_level=$1 mode=${2:-full}
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local seed_name="seed${SEED}"
    local dataset_name="${ORIGINAL_DATASET}_${k_level}"
    local train_file="${DATASET_DIR}/${ORIGINAL_DATASET}_${k_level}.jsonl"
    local output_dir="${BASE_DIR}/output/${dataset_name}_${seed_name}"
    local model_dir="${output_dir}/final_model"
    local exp_result_dir="${RESULTS_DIR}/${dataset_name}_${timestamp}_${seed_name}"

    echo "############################################################"
    echo "K-Scaling Experiment: ${dataset_name}"
    echo "K-Level: ${k_level}"
    echo "Mode: ${mode} | Seed: ${SEED}"
    echo "############################################################"

    # Prepare data
    run_prepare_data ${k_level}

    if [ "${mode}" == "full" ] || [ "${mode}" == "train_only" ]; then
        run_train "${train_file}" "${output_dir}" "${dataset_name}" || return 1
    fi
    if [ "${mode}" == "full" ] || [ "${mode}" == "test_only" ]; then
        [ ! -d "${model_dir}" ] && echo "!!! Model not found: ${model_dir}" && return 1
        run_test "${model_dir}" "${exp_result_dir}" "${dataset_name}"
        export_results "${exp_result_dir}" "${dataset_name}" "${train_file}" "${model_dir}" ${k_level}
    fi
}

cd ${BASE_DIR}
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate qwen1
fi
mkdir -p ${RESULTS_DIR} ${DATASET_DIR}

MODE=${1:-full}

echo "============================================================"
echo "K-Scaling Experiment Configuration:"
echo "  max_new_tokens: 2048"
echo "  Random Seed: ${SEED}"
echo "  Dataset: ${ORIGINAL_DATASET} (3-step)"
echo "  Mode: ${MODE} | GPU: ${CUDA_VISIBLE_DEVICES:-0}"
echo "============================================================"
echo ""
echo "K-Scaling Levels:"
for desc in "${K_DESCRIPTIONS[@]}"; do
    echo "  ${desc}"
done
echo ""

for k_level in "${K_LEVELS[@]}"; do
    echo ""
    echo "****************** Starting ${k_level} experiment ******************"
    run_k_scaling ${k_level} ${MODE} || exit 1
    check_continue
done

echo ""
echo "============================================================"
echo "All K-Scaling experiments complete!"
echo "============================================================"
