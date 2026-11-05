#!/bin/bash


STOP_FILE=$(mktemp /tmp/nocot_ablation_stop_XXXXXX)
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
BASE_DIR="${ROOT_DIR}/experiments/table2_ablation_nocot"
SRC_DIR="${ROOT_DIR}/src"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/models/qwen3-4b}"   # 可通过环境变量覆盖
DATASET_DIR="${ROOT_DIR}/datasets"                        # 统一基础数据集
RESULTS_DIR="${BASE_DIR}/results"
NUM_EPOCHS=3
MAX_SAMPLES=500
SEED=42
MAX_NEW_TOKENS=2048

run_train() {
    local train_file=$1 output_dir=$2 name=$3
    echo ">>> 开始训练: ${name} (seed=${SEED})"

    export PYTHONHASHSEED=${SEED}

    TOKENIZERS_PARALLELISM=false python ${SRC_DIR}/finetune_qwen_atom_nocot.py \
        --model_path ${MODEL_PATH} --train_file ${train_file} \
        --output_dir ${output_dir} --num_train_epochs ${NUM_EPOCHS} \
        --prompt_file ${BASE_DIR}/prompts/prompt_nocot_atom.txt \
        --seed ${SEED} &
    TRAIN_PID=$!

    while kill -0 $TRAIN_PID 2>/dev/null; do
        if ! check_stop || [ "$SHOULD_CONTINUE" = "false" ]; then
            kill -9 $TRAIN_PID 2>/dev/null
            wait $TRAIN_PID 2>/dev/null
            echo "!!! 训练被中断: ${name}"
            return 1
        fi
        sleep 1
    done

    wait $TRAIN_PID 2>/dev/null
    [ $? -ne 0 ] && [ $? -lt 128 ] && echo "!!! 训练失败: ${name}" && return 1
    echo "<<< 训练完成: ${name}"
    return 0
}

run_test() {
    local model_dir=$1 exp_result_dir=$2 name=$3
    mkdir -p ${exp_result_dir}
    echo ">>> 开始测试: ${name}"

    for step in 3 5 7 9; do
        echo "--- ${step}步测试集 ---"
        local predict_file="${exp_result_dir}/predictions_${step}step.jsonl"
        local log_file="${exp_result_dir}/test_${step}step.log"
        local test_data_file="${DATASET_DIR}/test_${step}step.jsonl"

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
    echo "<<< 测试完成: ${name}"
}

export_results() {
    local exp_result_dir=$1 dataset_name=$2 train_file=$3 model_path=$4
    {
        echo "数据集: ${dataset_name} (No-CoT)"
        echo "训练数据: ${train_file}"
        echo "模型: qwen3-4b (No-CoT)"
        echo "训练轮数: ${NUM_EPOCHS}"
        echo "测试集规模: 各${MAX_SAMPLES}条"
        echo "随机种子: ${SEED}"
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

    python3 << 'PYEOF' > ${exp_result_dir}/detailed_results.json
import json
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

run_ablation() {
    local ablation_name=$1
    local mode=${2:-full}
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local seed_name="seed${SEED}"
    local train_file="${DATASET_DIR}/${ablation_name}.jsonl"
    local output_dir="${BASE_DIR}/output/${ablation_name}_${seed_name}_nocot"
    local model_dir="${output_dir}/final_model"
    local exp_result_dir="${RESULTS_DIR}/${ablation_name}_${timestamp}_${seed_name}_nocot"

    echo "############################################################"
    echo "实验: ${ablation_name} (No-CoT) | 模式: ${mode} | 种子: ${SEED}"
    echo "############################################################"

    if [ "${mode}" == "full" ] || [ "${mode}" == "train_only" ]; then
        run_train "${train_file}" "${output_dir}" "${ablation_name}" || return 1
    fi
    if [ "${mode}" == "full" ] || [ "${mode}" == "test_only" ]; then
        [ ! -d "${model_dir}" ] && echo "!!! 模型不存在: ${model_dir}" && return 1
        run_test "${model_dir}" "${exp_result_dir}" "${ablation_name}"
        export_results "${exp_result_dir}" "${ablation_name}" "${train_file}" "${model_dir}"
    fi
}

cd ${BASE_DIR}
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate qwen1
fi
mkdir -p ${RESULTS_DIR}

MODE=${1:-full}

echo "============================================================"
echo "No-CoT Ablation实验配置:"
echo "  max_new_tokens: ${MAX_NEW_TOKENS}"
echo "  随机种子: ${SEED}"
echo "  模式: ${MODE} | GPU: ${CUDA_VISIBLE_DEVICES:-0}"
echo "============================================================"

for ablation in no_reverse no_duplicate no_deduplicate no_substitute no_rotate; do
    run_ablation "ablation_${ablation}" ${MODE} || exit 1
    check_continue
done

echo ""
echo "============================================================"
echo "No-CoT Ablation实验完成!"
echo "============================================================"