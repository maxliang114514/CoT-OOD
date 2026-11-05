#!/bin/bash


STOP_FILE=$(mktemp /tmp/nm_scaling_stop_XXXXXX)
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
BASE_DIR="${ROOT_DIR}/experiments/table6_7_scaling"
SRC_DIR="${ROOT_DIR}/src"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/models/qwen3-4b}"   # 可通过环境变量覆盖
DATASET_DIR="${BASE_DIR}/datasets"                        # n_scaling 本地子集
ORIGINAL_DATASET_DIR="${ROOT_DIR}/datasets"               # 统一基础数据集
RESULTS_DIR="${BASE_DIR}/results"
NUM_EPOCHS=3
MAX_SAMPLES=500
SEED=42
MAX_NEW_TOKENS=2048

run_train() {
    local train_file=$1 output_dir=$2 name=$3 prompt_file=$4
    echo ">>> 开始训练: ${name}"
    export PYTHONHASHSEED=${SEED}

    export TOKENIZERS_PARALLELISM=false
    python ${SRC_DIR}/finetune_qwen_atom.py \
        --model_path ${MODEL_PATH} --train_file ${train_file} \
        --output_dir ${output_dir} --num_train_epochs ${NUM_EPOCHS} \
        --train_mode cot_format --skip_eval \
        ${prompt_file:+--prompt_file} ${prompt_file} &
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
    local model_dir=$1 exp_result_dir=$2 name=$3 n_shot=$4 prompt_file=$5
    mkdir -p ${exp_result_dir}
    echo ">>> 开始测试: ${name} (m=${n_shot})"

    for step in 3 5 7; do
        echo "--- ${step}步测试集 ---"
        local predict_file="${exp_result_dir}/predictions_${step}step.jsonl"
        local log_file="${exp_result_dir}/test_${step}step.log"
        local test_data_file="${ORIGINAL_DATASET_DIR}/test_${step}step.jsonl"

        local vllm_cmd="python ${SRC_DIR}/gen_response_vllm.py \
            --model ${model_dir} \
            --data_file ${test_data_file} \
            --output_file ${predict_file} \
            --max_samples ${MAX_SAMPLES} \
            --temperature 0.0 \
            --top_p 1.0 \
            --max_tokens ${MAX_NEW_TOKENS} \
            --gpu_memory_utilization 0.875"

        if [ -n "${prompt_file}" ]; then
            vllm_cmd="${vllm_cmd} --prompt_file ${prompt_file}"
        fi

        eval "${vllm_cmd}" 2>&1 | tee ${log_file}

        echo "评估预测结果..."
        python ${SRC_DIR}/eval_accuracy.py ${predict_file} --verbose 2>&1 | tee -a ${log_file}

        python ${SRC_DIR}/export_samples.py ${predict_file} ${MAX_SAMPLES} ${exp_result_dir}/export_samples_${step}步组合.json 2>&1 | tee -a ${log_file}
    done
    echo "<<< 测试完成: ${name}"
}

export_results() {
    local exp_result_dir=$1 name=$2
    {
        echo "实验: ${name}"
        echo "随机种子: ${SEED}"
        echo ""
        echo "测试结果:"
        echo "-----------------------------------"
        for step in 3step 5step 7step 9step; do
            echo "${step}:"
            grep -E "正确:|准确率:" ${exp_result_dir}/test_${step}.log 2>/dev/null | tail -2
            echo ""
        done
    } > ${exp_result_dir}/summary.txt

    python3 << PYEOF > ${exp_result_dir}/detailed_results.json
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
            if "正确:" in line:
                parts = line.strip().split()
                for i, p in enumerate(parts):
                    if p == "正确:":
                        fraction = parts[i+1]
                        correct = int(fraction.split("/")[0])
                        total = int(fraction.split("/")[1])
            if "准确率:" in line:
                parts = line.strip().split()
                for i, p in enumerate(parts):
                    if p == "准确率:":
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
    echo ">>> 结果已导出: ${exp_result_dir}"
}

run_n_scaling() {
    local frac=$1
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local name="n_scaling_${frac}pct"
    local train_file="${DATASET_DIR}/n_scaling_${frac}pct.jsonl"
    local output_dir="${BASE_DIR}/output/${name}"
    local model_dir="${output_dir}/final_model"
    local exp_result_dir="${RESULTS_DIR}/${name}_${timestamp}"
    local prompt_file="${BASE_DIR}/prompts/prompt_cot_atom.txt"

    echo "############################################################"
    echo "N-Scaling 实验: ${name} (训练集 ${frac}%)"
    echo "############################################################"

    run_train "${train_file}" "${output_dir}" "${name}" "${prompt_file}" || return 1

    run_test "${model_dir}" "${exp_result_dir}" "${name}" 1 "${prompt_file}"
    export_results "${exp_result_dir}" "${name}"
}

run_m_scaling() {
    local n_shot=$1
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local name="m_scaling_${n_shot}shot"
    local train_file="${ORIGINAL_DATASET_DIR}/train_3step.jsonl"
    local output_dir="${BASE_DIR}/output/m_scaling_${n_shot}shot"
    local model_dir="${output_dir}/final_model"
    local exp_result_dir="${RESULTS_DIR}/${name}_${timestamp}"
    local prompt_file="${BASE_DIR}/prompts/prompt_nshot_${n_shot}.txt"

    echo "############################################################"
    echo "M-Scaling 实验: ${name} (m=${n_shot})"
    echo "############################################################"

    echo "模型训练中..."
    echo "  数据集: ${train_file}"
    echo "  Prompt: ${prompt_file}"
    run_train "${train_file}" "${output_dir}" "${name}" "${prompt_file}" || return 1

    run_test "${model_dir}" "${exp_result_dir}" "${name}" ${n_shot} "${prompt_file}"
    export_results "${exp_result_dir}" "${name}"
}

cd ${BASE_DIR}
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate qwen1
fi
mkdir -p ${RESULTS_DIR} ${BASE_DIR}/output

EXP_TYPE=${1:-all}

echo "============================================================"
echo "N-M Scaling 实验配置:"
echo "  随机种子: ${SEED}"
echo "  实验类型: ${EXP_TYPE}"
echo "  GPU: ${CUDA_VISIBLE_DEVICES:-0}"
echo "============================================================"

case ${EXP_TYPE} in
    n_scaling)
        for frac in 10 20 50; do
            run_n_scaling ${frac} || exit 1
            check_continue
        done ;;
    m_scaling)
        for n_shot in 0 2 3; do
            run_m_scaling ${n_shot} || exit 1
            check_continue
        done ;;
    m_scaling_3shot)
        run_m_scaling 3 || exit 1
        ;;
    all|n_scaling+m_scaling)
        for frac in 10 20 50; do
            run_n_scaling ${frac} || exit 1
            check_continue
        done
        for n_shot in 0 2 3; do
            run_m_scaling ${n_shot} || exit 1
            check_continue
        done ;;
    -h|--help|help)
        echo "用法: $0 <实验类型>"
        echo ""
        echo "实验类型:"
        echo "  n_scaling    - 训练集规模变化 (10%, 20%, 50%)"
        echo "  m_scaling    - Few-shot数量变化 (0-shot, 2-shot, 3-shot)"
        echo "  m_scaling_3shot - 仅运行 3-shot 实验"
        echo "  all          - 运行全部实验 (默认)"
        echo ""
        echo "注意: 1-shot 为标准组，不需要运行"
        echo ""
        echo "示例:"
        echo "  CUDA_VISIBLE_DEVICES=0 bash run_nm_scaling.sh n_scaling"
        echo "  CUDA_VISIBLE_DEVICES=0 bash run_nm_scaling.sh m_scaling"
        echo "  CUDA_VISIBLE_DEVICES=0 bash run_nm_scaling.sh m_scaling_3shot"
        echo "  CUDA_VISIBLE_DEVICES=0 bash run_nm_scaling.sh all"
        exit 0 ;;
    *)
        echo "错误: 未知实验类型 '${EXP_TYPE}'"
        echo "可用: n_scaling, m_scaling, m_scaling_3shot, all"
        exit 1 ;;
esac

echo ""
echo "============================================================"
echo "实验完成!"
echo "============================================================"