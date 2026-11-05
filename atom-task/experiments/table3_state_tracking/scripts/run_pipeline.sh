#!/bin/bash

# ============================================================
#
# ============================================================

STOP_FILE=$(mktemp /tmp/pipeline_ablation_stop_XXXXXX)
trap 'rm -f "$STOP_FILE"; echo "清理停止文件"; exit 0' EXIT

shadow_process() {
    while [ -f "$STOP_FILE" ]; do
        sleep 0.5
    done
    echo "影子进程检测到停止信号"
    kill -TERM -$$ 2>/dev/null
    exit 0
}

shadow_process &
SHADOW_PID=$!

SHOULD_CONTINUE=true

cleanup() {
    echo "收到终止信号，正在退出..."
    SHOULD_CONTINUE=false
    rm -f "$STOP_FILE"
    sleep 1
    if kill -0 $SHADOW_PID 2>/dev/null; then
        kill -9 $SHADOW_PID 2>/dev/null
    fi
    exit 1
}
trap 'cleanup' SIGINT SIGTERM

if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

check_stop() {
    if [ ! -f "$STOP_FILE" ]; then
        echo "检测到停止信号"
        return 1
    fi
    return 0
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE_DIR="${ROOT_DIR}/experiments/table3_state_tracking"
SRC_DIR="${ROOT_DIR}/src"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/models/qwen3-4b}"   # 可通过环境变量覆盖
DATASET_DIR="${ROOT_DIR}/datasets"                       # 统一基础数据集
TEST_DATASET_DIR="${ROOT_DIR}/datasets"
RESULTS_DIR="${BASE_DIR}/results"
NUM_EPOCHS=3
MAX_SAMPLES=500

show_help() {
    cat << EOF
用法: bash run_pipeline.sh <实验类型> [模式]

实验类型:
  ablation_no_reverse    - 消融: 去掉 Reverse 操作 (t1)
  ablation_no_substitute - 消融: 去掉 Substitute 操作 (t2)
  ablation_no_duplicate   - 消融: 去掉 Duplicate 操作 (t3)
  ablation_no_deduplicate - 消融: 去掉 Deduplicate 操作 (t4)
  ablation_no_rotate      - 消融: 去掉 Rotate 操作 (t5)
  full               - 运行全部5种消融实验

模式 (默认 full):
  train_only   - 只训练，不测试
  test_only    - 只测试，需要已有模型
  full         - 训练+测试

示例:
  bash run_pipeline.sh full
  bash run_pipeline.sh ablation_no_duplicate full
  CUDA_VISIBLE_DEVICES=1 bash run_pipeline.sh full

注意:
  - 消融实验：训练时生成的 output 思维链去掉 Task_Chain 记录，且 Input_Sequence 替换为 ......
  - CoT格式: Step N: [Task] Task_Chain: [......] | Input_Sequence: [......] | → Output_Sequence: [xxx] | Task_Chain: [......]
  - 数据集位于 ${DATASET_DIR}/

EOF
}

run_or_exit() {
    local cmd="$@"
    if ! $cmd; then
        echo "命令执行失败: $cmd"
        exit 1
    fi
    if [ "$SHOULD_CONTINUE" = "false" ]; then
        echo "检测到终止信号，退出脚本"
        exit 1
    fi
}

run_train() {
    local train_file=$1
    local output_dir=$2
    local name=$3

    echo ""
    echo ">>> 开始训练: ${name}"
    echo "    数据: ${train_file}"
    echo "    输出: ${output_dir}"
    echo "    消融模式: 去掉 Task_Chain 记录 + 遮罩 Input_Sequence"

    TOKENIZERS_PARALLELISM=false python ${SRC_DIR}/finetune_qwen_atom_ablation.py \
        --seed 42 \
        --model_path ${MODEL_PATH} \
        --train_file ${train_file} \
        --output_dir ${output_dir} \
        --num_train_epochs ${NUM_EPOCHS} \
        --train_mode cot_format \
        --ablation_mode \
        --prompt_file ${BASE_DIR}/prompts/prompt_cot_atom_ablation.txt \
        --skip_eval &
    TRAIN_PID=$!

    while kill -0 $TRAIN_PID 2>/dev/null; do
        if ! check_stop || [ "$SHOULD_CONTINUE" = "false" ]; then
            echo "检测到终止信号，kill训练进程 ${TRAIN_PID}..."
            kill -9 $TRAIN_PID 2>/dev/null
            wait $TRAIN_PID 2>/dev/null
            echo "!!! 训练被中断: ${name}"
            return 1
        fi
        sleep 1
    done

    wait $TRAIN_PID 2>/dev/null
    TRAIN_STATUS=$?

    if [ $TRAIN_STATUS -gt 128 ]; then
        echo "!!! 训练被中断: ${name}"
        return 1
    fi

    if [ $TRAIN_STATUS -ne 0 ]; then
        echo "!!! 训练失败: ${name}"
        return 1
    fi
    echo "<<< 训练完成: ${name}"
    return 0
}

run_test() {
    local model_dir=$1
    local exp_result_dir=$2
    local name=$3

    mkdir -p ${exp_result_dir}

    echo ""
    echo ">>> 开始测试: ${name}"
    echo "    模型: ${model_dir}"

    echo "--- 3步测试集 (vLLM) ---"
    local predict_file="${exp_result_dir}/predictions_3step.jsonl"
    local log_file="${exp_result_dir}/test_3step.log"

    python ${SRC_DIR}/gen_response_vllm.py \
        --model ${model_dir} \
        --data_file ${TEST_DATASET_DIR}/test_3step.jsonl \
        --output_file ${predict_file} \
        --max_samples ${MAX_SAMPLES} \
        --temperature 0.0 \
        --top_p 1.0 \
        --max_tokens 1024 \
        --gpu_memory_utilization 0.8 2>&1 | tee ${log_file}

    python ${SRC_DIR}/eval_accuracy.py ${predict_file} --verbose 2>&1 | tee -a ${log_file}
    python ${SRC_DIR}/export_samples.py ${predict_file} ${MAX_SAMPLES} ${exp_result_dir}/export_samples_${MAX_SAMPLES}条.json 2>&1 | tee -a ${log_file}

    echo "--- 5步测试集 (vLLM) ---"
    predict_file="${exp_result_dir}/predictions_5step.jsonl"
    log_file="${exp_result_dir}/test_5step.log"

    python ${SRC_DIR}/gen_response_vllm.py \
        --model ${model_dir} \
        --data_file ${TEST_DATASET_DIR}/test_5step.jsonl \
        --output_file ${predict_file} \
        --max_samples ${MAX_SAMPLES} \
        --temperature 0.0 \
        --top_p 1.0 \
        --max_tokens 1024 \
        --gpu_memory_utilization 0.8 2>&1 | tee ${log_file}

    python ${SRC_DIR}/eval_accuracy.py ${predict_file} --verbose 2>&1 | tee -a ${log_file}
    python ${SRC_DIR}/export_samples.py ${predict_file} ${MAX_SAMPLES} ${exp_result_dir}/export_samples_${MAX_SAMPLES}条.json 2>&1 | tee -a ${log_file}

    echo "--- 7步测试集 (vLLM) ---"
    predict_file="${exp_result_dir}/predictions_7step.jsonl"
    log_file="${exp_result_dir}/test_7step.log"

    python ${SRC_DIR}/gen_response_vllm.py \
        --model ${model_dir} \
        --data_file ${TEST_DATASET_DIR}/test_7step.jsonl \
        --output_file ${predict_file} \
        --max_samples ${MAX_SAMPLES} \
        --temperature 0.0 \
        --top_p 1.0 \
        --max_tokens 1024 \
        --gpu_memory_utilization 0.8 2>&1 | tee ${log_file}

    python ${SRC_DIR}/eval_accuracy.py ${predict_file} --verbose 2>&1 | tee -a ${log_file}
    python ${SRC_DIR}/export_samples.py ${predict_file} ${MAX_SAMPLES} ${exp_result_dir}/export_samples_${MAX_SAMPLES}条.json 2>&1 | tee -a ${log_file}

    echo "<<< 测试完成: ${name}"
}

export_results() {
    local exp_result_dir=$1
    local dataset_name=$2
    local train_file=$3
    local model_path=$4

    {
        echo "数据集: ${dataset_name}"
        echo "训练数据: ${train_file}"
        echo "模型: qwen3-4b"
        echo "训练轮数: ${NUM_EPOCHS}"
        echo "测试集规模: 各${MAX_SAMPLES}条"
        echo "消融实验: 去掉 Task_Chain 记录 + 遮罩 Input_Sequence"
        echo "CoT格式: Step N: [Task] Task_Chain: [......] | Input_Sequence: [......] | → Output_Sequence: [xxx] | Task_Chain: [......]"
        echo ""
        echo "测试结果:"
        echo "-----------------------------------"
        for step in 3step 5step 7step; do
            echo "${step}:"
            grep -E "正确:|准确率:" ${exp_result_dir}/test_${step}.log 2>/dev/null | tail -2
            echo ""
        done
    } > ${exp_result_dir}/summary.txt

    python3 << PYEOF > ${exp_result_dir}/detailed_results.json
import json
import os
from pathlib import Path

results_dir = Path("${exp_result_dir}")
all_results = {}

for step in [3, 5, 7]:
    log_file = results_dir / f"test_{step}step.log"
    export_file = results_dir / f"export_samples_{step}步组合.json"

    if not log_file.exists():
        continue

    correct = 0
    total = 0
    with open(log_file) as f:
        for line in f:
            if "Correct:" in line:
                parts = line.strip().split()
                for i, p in enumerate(parts):
                    if p == "Correct:":
                        correct = int(parts[i+1].split("/")[0])
            if "Accuracy:" in line:
                parts = line.strip().split()
                for i, p in enumerate(parts):
                    if p == "Accuracy:":
                        total = int(parts[i+1].split("/")[-1].replace("%",""))

    predictions = []
    errors = []
    if export_file.exists():
        with open(export_file) as f:
            data = json.load(f)
            for item in data.get("samples", []):
                pred = item.get("predicted", "")
                gold = item.get("gold", "")
                if pred != gold:
                    errors.append({
                        "input": item.get("input", ""),
                        "task_chain": item.get("task_chain", ""),
                        "predicted": pred,
                        "gold": gold,
                        "response": item.get("response", "")[:500]
                    })
                predictions.append({
                    "input": item.get("input", ""),
                    "task_chain": item.get("task_chain", ""),
                    "predicted": pred,
                    "gold": gold,
                    "correct": pred == gold
                })

    all_results[f"{step}step"] = {
        "correct": correct,
        "total": len(predictions) if predictions else total,
        "accuracy": f"{correct/len(predictions)*100:.2f}%" if predictions else "N/A",
        "num_errors": len(errors),
        "sample_predictions": predictions[:20] if predictions else [],
        "errors": errors[:20]
    }

print(json.dumps(all_results, ensure_ascii=False, indent=2))
PYEOF

    echo ">>> 结果已导出: ${exp_result_dir}"
}

run_experiment() {
    local exp_name=$1
    local dataset_name=$2
    local mode=${3:-full}
    local timestamp=$(date +%Y%m%d_%H%M%S)

    local clean_name=${dataset_name%.jsonl}
    local train_file="${DATASET_DIR}/${clean_name}.jsonl"
    local output_dir="${BASE_DIR}/output/${clean_name}"
    local model_dir="${output_dir}/final_model"
    local exp_result_dir="${RESULTS_DIR}/${clean_name}_${timestamp}"

    echo ""
    echo "############################################################"
    echo "实验: ${exp_name} (消融版本 - 去掉 Task_Chain 记录 + 遮罩 Input_Sequence)"
    echo "模式: ${mode}"
    echo "############################################################"

    if [ "${mode}" == "full" ] || [ "${mode}" == "train_only" ]; then
        run_train "${train_file}" "${output_dir}" "${exp_name}" || return 1
    fi

    if [ "${mode}" == "full" ] || [ "${mode}" == "test_only" ]; then
        if [ ! -d "${model_dir}" ]; then
            echo "!!! 模型不存在: ${model_dir}"
            return 1
        fi
        run_test "${model_dir}" "${exp_result_dir}" "${exp_name}"
        export_results "${exp_result_dir}" "${exp_name}" "${train_file}" "${model_dir}"
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
echo "消融实验脚本 - Qwen3-4B"
echo "实验类型: ${EXP_TYPE}"
echo "运行模式: ${MODE}"
echo "GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "============================================================"
echo "消融实验: 训练时生成的 output 思维链去掉 Task_Chain 记录，且 Input_Sequence 替换为 ......"
echo "CoT格式: Step N: [Task] Task_Chain: [......] | Input_Sequence: [......] | → Output_Sequence: [xxx]"
echo "============================================================"

check_continue() {
    if ! check_stop || [ "$SHOULD_CONTINUE" = "false" ]; then
        echo "检测到终止信号，退出脚本"
        exit 1
    fi
}

run_ablation_full() {
    run_experiment "ablation_no_reverse" "ablation_no_reverse.jsonl" ${MODE} || return 1
    check_continue
    run_experiment "ablation_no_substitute" "ablation_no_substitute.jsonl" ${MODE} || return 1
    check_continue
    run_experiment "ablation_no_duplicate" "ablation_no_duplicate.jsonl" ${MODE} || return 1
    check_continue
    run_experiment "ablation_no_deduplicate" "ablation_no_deduplicate.jsonl" ${MODE} || return 1
    check_continue
    run_experiment "ablation_no_rotate" "ablation_no_rotate.jsonl" ${MODE} || return 1
}

case ${EXP_TYPE} in
    ablation_no_reverse)
        run_experiment "ablation_no_reverse" "ablation_no_reverse.jsonl" ${MODE} || exit 1
        ;;
    ablation_no_substitute)
        run_experiment "ablation_no_substitute" "ablation_no_substitute.jsonl" ${MODE} || exit 1
        ;;
    ablation_no_duplicate)
        run_experiment "ablation_no_duplicate" "ablation_no_duplicate.jsonl" ${MODE} || exit 1
        ;;
    ablation_no_deduplicate)
        run_experiment "ablation_no_deduplicate" "ablation_no_deduplicate.jsonl" ${MODE} || exit 1
        ;;
    ablation_no_rotate)
        run_experiment "ablation_no_rotate" "ablation_no_rotate.jsonl" ${MODE} || exit 1
        ;;
    full|ablation_full)
        run_ablation_full || exit 1
        ;;
    -h|--help|help)
        show_help
        exit 0
        ;;
    *)
        echo "错误: 未知实验类型 '${EXP_TYPE}'"
        show_help
        exit 1
        ;;
esac

echo ""
echo "############################################################"
echo "所有实验完成!"
echo "############################################################"