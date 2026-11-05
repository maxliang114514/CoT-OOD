#!/bin/bash


STOP_FILE=$(mktemp /tmp/scaling_nocot_stop_XXXXXX)
trap 'rm -f "$STOP_FILE"; exit 0' EXIT

shadow_process() {
    while [ -f "$STOP_FILE" ]; do sleep 0.5; done
    kill -TERM -$$ 2>/dev/null
}
shadow_process &
SHADOW_PID=$!

SHOULD_CONTINUE=true
cleanup() {
    echo "收到终止信号，正在退出..."
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
        echo "检测到终止信号，退出脚本"
        exit 1
    fi
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE_DIR="${ROOT_DIR}/experiments/table1_main_nocot"
SRC_DIR="${ROOT_DIR}/src"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/models/qwen3-4b}"   # 可通过环境变量覆盖
ORIGINAL_DATASET_DIR="${ROOT_DIR}/datasets"               # 统一基础数据集
RESULTS_DIR="${BASE_DIR}/results"
NUM_EPOCHS=3
MAX_SAMPLES=500

MAX_NEW_TOKENS=2048
SEEDS=(42 2026 12345)
SINGLE_STEP_DATASET="single_step"

run_train() {
    local train_file=$1 output_dir=$2 name=$3 seed=$4
    echo ">>> 开始训练: ${name} (seed=${seed})"

    export PYTHONHASHSEED=$seed

    TOKENIZERS_PARALLELISM=false python ${SRC_DIR}/finetune_qwen_atom_nocot.py \
        --model_path ${MODEL_PATH} --train_file ${train_file} \
        --output_dir ${output_dir} --num_train_epochs ${NUM_EPOCHS} \
        --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
        --prompt_file ${BASE_DIR}/prompts/prompt_nocot_atom.txt \
        --seed ${seed} &
    TRAIN_PID=$!

    while kill -0 $TRAIN_PID 2>/dev/null; do
        if ! check_stop || [ "$SHOULD_CONTINUE" = "false" ]; then
            kill -9 $TRAIN_PID 2>/dev/null
            wait $TRAIN_PID 2>/dev/null
            echo "!!! 训练被中断: ${name} (seed=${seed})"
            return 1
        fi
        sleep 1
    done

    wait $TRAIN_PID 2>/dev/null
    [ $? -ne 0 ] && [ $? -lt 128 ] && echo "!!! 训练失败: ${name} (seed=${seed})" && return 1
    echo "<<< 训练完成: ${name} (seed=${seed})"
    return 0
}

run_test() {
    local model_dir=$1 exp_result_dir=$2 name=$3 seed=$4
    mkdir -p ${exp_result_dir}
    echo ">>> 开始测试: ${name} (seed=${seed})"

    for step in 3 5 7 9; do
        echo "--- ${step}步测试集 ---"
        local predict_file="${exp_result_dir}/predictions_${step}step.jsonl"
        local log_file="${exp_result_dir}/test_${step}step.log"
        local test_data_file="${ORIGINAL_DATASET_DIR}/test_${step}step.jsonl"

        echo "使用 vLLM 推理..."
        python ${SRC_DIR}/gen_response_vllm.py \
            --model ${model_dir} \
            --data_file ${test_data_file} \
            --output_file ${predict_file} \
            --max_samples ${MAX_SAMPLES} \
            --temperature 0.0 \
            --top_p 1.0 \
            --max_tokens ${MAX_NEW_TOKENS} \
            --gpu_memory_utilization 0.82 2>&1 | tee ${log_file}

        echo "评估预测结果..."
        python ${SRC_DIR}/eval_accuracy.py ${predict_file} --verbose 2>&1 | tee -a ${log_file}

        python ${SRC_DIR}/export_samples.py ${predict_file} ${MAX_SAMPLES} ${exp_result_dir}/export_samples_${step}步组合.json 2>&1 | tee -a ${log_file}
    done
    echo "<<< 测试完成: ${name} (seed=${seed})"
}

export_results() {
    local exp_result_dir=$1 dataset_name=$2 train_file=$3 model_path=$4 seed=$5
    {
        echo "数据集: ${dataset_name} (No-CoT)"
        echo "训练数据: ${train_file}"
        echo "模型: qwen3-4b (No-CoT)"
        echo "训练轮数: ${NUM_EPOCHS}"
        echo "测试集规模: 各${MAX_SAMPLES}条"
        echo "随机种子: ${seed}"
        echo "max_new_tokens: ${MAX_NEW_TOKENS}"
        echo ""
        echo "测试结果:"
        echo "-----------------------------------"
        for step in 3step 5step 7step 9step; do
            echo "${step}:"
            grep -E "正确:|准确率:|Correct:|Accuracy:" ${exp_result_dir}/test_${step}.log 2>/dev/null | tail -2
            echo ""
        done
    } > ${exp_result_dir}/summary.txt

    python3 << PYEOF > ${exp_result_dir}/detailed_results.json
import json
import re
from pathlib import Path
results_dir = Path("${exp_result_dir}")
all_results = {}
for step in [3, 5, 7, 9]:
    log_file = results_dir / f"test_{step}step.log"
    export_file = results_dir / f"export_samples_{step}步组合.json"
    if not log_file.exists(): continue
    correct = total = 0
    with open(log_file) as f:
        for line in f:
            if "正确:" in line or "Correct:" in line:
                parts = line.strip().split()
                for i, p in enumerate(parts):
                    if p == "正确:" or p == "Correct:":
                        fraction = parts[i+1]
                        if "/" in fraction:
                            correct = int(fraction.split("/")[0])
                            total = int(fraction.split("/")[1])
            if "准确率:" in line or "Accuracy:" in line:
                parts = line.strip().split()
                for i, p in enumerate(parts):
                    if p == "准确率:" or p == "Accuracy:":
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
    accuracy_str = f"{correct/len(predictions)*100:.2f}%" if predictions and len(predictions) > 0 else "N/A"
    all_results[f"{step}step"] = {"correct": correct, "total": len(predictions) or total,
        "accuracy": accuracy_str,
        "num_errors": len(errors), "sample_predictions": predictions[:20], "errors": errors[:20]}
print(json.dumps(all_results, ensure_ascii=False, indent=2))
PYEOF
    echo ">>> 结果已导出: ${exp_result_dir}"
}

run_multi_step() {
    local dataset_name=$1 train_file=$2 mode=${3:-full} seed=$4
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local seed_name="seed${seed}"
    local output_dir="${BASE_DIR}/output/${dataset_name}_${seed_name}_nocot"
    local model_dir="${output_dir}/final_model"
    local exp_result_dir="${RESULTS_DIR}/${dataset_name}_${timestamp}_${seed_name}_nocot"

    echo "############################################################"
    echo "实验: ${dataset_name} (No-CoT) | 模式: ${mode} | 种子: ${seed}"
    echo "############################################################"

    if [ "${mode}" == "full" ] || [ "${mode}" == "train_only" ]; then
        run_train "${train_file}" "${output_dir}" "${dataset_name}" ${seed} || return 1
    fi
    if [ "${mode}" == "full" ] || [ "${mode}" == "test_only" ]; then
        [ ! -d "${model_dir}" ] && echo "!!! 模型不存在: ${model_dir}" && return 1
        run_test "${model_dir}" "${exp_result_dir}" "${dataset_name}" ${seed}
        export_results "${exp_result_dir}" "${dataset_name}" "${train_file}" "${model_dir}" ${seed}
    fi
}

run_single_step() {
    local mode=${1:-full} seed=$2
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local seed_name="seed${seed}"
    local dataset_name="${SINGLE_STEP_DATASET}_${seed_name}"
    local train_file="${ORIGINAL_DATASET_DIR}/train_1step.jsonl"
    local output_dir="${BASE_DIR}/output/${SINGLE_STEP_DATASET}_${seed_name}_nocot"
    local model_dir="${output_dir}/final_model"
    local exp_result_dir="${RESULTS_DIR}/${dataset_name}_${timestamp}_nocot"

    echo "############################################################"
    echo "实验: ${dataset_name} (No-CoT) | 模式: ${mode} | 种子: ${seed}"
    echo "############################################################"

    if [ "${mode}" == "full" ] || [ "${mode}" == "train_only" ]; then
        run_train "${train_file}" "${output_dir}" "${dataset_name}" ${seed} || return 1
    fi
    if [ "${mode}" == "full" ] || [ "${mode}" == "test_only" ]; then
        [ ! -d "${model_dir}" ] && echo "!!! 模型不存在: ${model_dir}" && return 1
        run_test "${model_dir}" "${exp_result_dir}" "${dataset_name}" ${seed}
        export_results "${exp_result_dir}" "${dataset_name}" "${train_file}" "${model_dir}" ${seed}
    fi
}

cd ${BASE_DIR}
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate qwen1
fi
mkdir -p ${RESULTS_DIR}

EXP_TYPE=${1:-full}
MODE=${2:-full}

echo "============================================================"
echo "No-CoT Scaling实验配置:"
echo "  max_new_tokens: ${MAX_NEW_TOKENS}"
echo "  随机种子: ${SEEDS[@]}"
echo "  实验类型: ${EXP_TYPE} | 模式: ${MODE} | GPU: ${CUDA_VISIBLE_DEVICES:-0}"
echo "============================================================"

case ${EXP_TYPE} in
    single_step)
        for seed in "${SEEDS[@]}"; do
            run_single_step ${MODE} ${seed} || exit 1
            check_continue
        done ;;
    type7)
        for seed in "${SEEDS[@]}"; do
            run_multi_step "train_3step" "${ORIGINAL_DATASET_DIR}/train_3step.jsonl" ${MODE} ${seed} || exit 1
            check_continue
        done ;;
    type8)
        for seed in "${SEEDS[@]}"; do
            run_multi_step "train_mixed_3_5step" "${ORIGINAL_DATASET_DIR}/train_mixed_3_5step.jsonl" ${MODE} ${seed} || exit 1
            check_continue
        done ;;
    all_3step)
        for seed in "${SEEDS[@]}"; do
            run_multi_step "train_3step" "${ORIGINAL_DATASET_DIR}/train_3step.jsonl" ${MODE} ${seed} || exit 1
            check_continue
            run_multi_step "train_mixed_3_5step" "${ORIGINAL_DATASET_DIR}/train_mixed_3_5step.jsonl" ${MODE} ${seed} || exit 1
            check_continue
        done ;;
    -h|--help|help)
        echo "用法: $0 <实验类型> [模式]"
        echo ""
        echo "实验类型:"
        echo "  single_step - 只训练和测试 single_step (train_1step, 1步数据)"
        echo "  type7       - 只训练和测试 3步数据 (train_3step)"
        echo "  type8       - 只训练和测试 3+5步混合数据 (train_mixed_3_5step)"
        echo "  all_3step   - 训练和测试 train_3step 与 train_mixed_3_5step"
        echo ""
        echo "模式: train_only, test_only, full (默认)"
        echo ""
        echo "示例 - 分开两张卡并行运行:"
        echo "  卡0: CUDA_VISIBLE_DEVICES=0 bash run_scaling.sh type7 full"
        echo "  卡1: CUDA_VISIBLE_DEVICES=1 bash run_scaling.sh type8 full"
        echo ""
        echo "示例 - 单卡顺序运行:"
        echo "  CUDA_VISIBLE_DEVICES=0 bash run_scaling.sh all_3step full"
        exit 0 ;;
    *)
        echo "错误: 未知实验类型 '${EXP_TYPE}'"
        echo "可用: single_step, type7, type8, all_3step"
        exit 1 ;;
esac

echo ""
echo "============================================================"
echo "No-CoT实验完成!"
echo "============================================================"