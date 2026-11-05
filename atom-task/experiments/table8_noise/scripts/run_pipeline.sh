#!/bin/bash

# ============================================================
#
#   - train_3step + noise (p=0, 0.05, 0.10, 0.20)
#   - train_mixed_3_5step + noise (p=0, 0.05, 0.10, 0.20)
#   - ablation_no_substitute + noise (p=0, 0.05, 0.10, 0.20)
#
# ============================================================

STOP_FILE=$(mktemp /tmp/pipeline_noise_robust_stop_XXXXXX)
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
BASE_DIR="${ROOT_DIR}/experiments/table8_noise"
SRC_DIR="${ROOT_DIR}/src"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/models/qwen3-4b}"   # 可通过环境变量覆盖
DATASET_DIR="${BASE_DIR}/datasets"
TEST_DATASET_DIR="${ROOT_DIR}/datasets"                  # 统一测试集目录
RESULTS_DIR="${BASE_DIR}/results"
NUM_EPOCHS=3
MAX_SAMPLES=500

show_help() {
    cat << EOF
用法: bash run_pipeline.sh <实验类型> [模式]

实验类型:
  type7_noise_0      - train_3step 无噪声
  type7_noise_5      - train_3step + 5%噪声
  type7_noise_10     - train_3step + 10%噪声
  type7_noise_20     - train_3step + 20%噪声

  type8_noise_0      - train_mixed_3_5step 无噪声
  type8_noise_5      - train_mixed_3_5step + 5%噪声
  type8_noise_10     - train_mixed_3_5step + 10%噪声
  type8_noise_20     - train_mixed_3_5step + 20%噪声

  nosub_noise_0      - ablation_no_substitute 无噪声
  nosub_noise_5      - ablation_no_substitute + 5%噪声
  nosub_noise_10     - ablation_no_substitute + 10%噪声
  nosub_noise_20     - ablation_no_substitute + 20%噪声

  all_type7          - train_3step 的全部噪声比例实验
  all_type8          - train_mixed_3_5step 的全部噪声比例实验
  all_nosub          - ablation_no_substitute 的全部噪声比例实验
  all                - 全部12个实验

模式 (默认 full):
  train_only   - 只训练，不测试
  test_only    - 只测试，需要已有模型
  full         - 训练+测试

示例:
  bash run_pipeline.sh all
  bash run_pipeline.sh type7_noise_10 full
  CUDA_VISIBLE_DEVICES=1 bash run_pipeline.sh all_type7

EOF
}

check_continue() {
    if ! check_stop || [ "$SHOULD_CONTINUE" = "false" ]; then
        echo "检测到终止信号，退出脚本"
        exit 1
    fi
}

generate_datasets() {
    echo ""
    echo ">>> 生成数据集..."
    python ${BASE_DIR}/scripts/generate_datasets.py --noise_ratios 0.05,0.10,0.20
    echo "<<< 数据集生成完成"
}

run_train() {
    local train_file=$1
    local output_dir=$2
    local name=$3

    echo ""
    echo ">>> 开始训练: ${name}"
    echo "    数据: ${train_file}"
    echo "    输出: ${output_dir}"

    TOKENIZERS_PARALLELISM=false python ${SRC_DIR}/finetune_qwen_atom_ablation.py \
        --model_path ${MODEL_PATH} \
        --train_file ${train_file} \
        --output_dir ${output_dir} \
        --num_train_epochs ${NUM_EPOCHS} \
        --train_mode cot_format \
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

    for step in 3 5 7; do
        echo "--- ${step}步测试集 (vLLM) ---"
        local predict_file="${exp_result_dir}/predictions_${step}step.jsonl"
        local log_file="${exp_result_dir}/test_${step}step.log"

        python ${SRC_DIR}/gen_response_vllm.py \
            --model ${model_dir} \
            --data_file ${TEST_DATASET_DIR}/test_${step}step.jsonl \
            --output_file ${predict_file} \
            --max_samples ${MAX_SAMPLES} \
            --temperature 0.0 \
            --top_p 1.0 \
            --max_tokens 1024 \
            --gpu_memory_utilization 0.8 2>&1 | tee ${log_file}

        python ${SRC_DIR}/eval_accuracy.py ${predict_file} --verbose 2>&1 | tee -a ${log_file}
        python ${SRC_DIR}/export_samples.py ${predict_file} ${MAX_SAMPLES} ${exp_result_dir}/export_samples_${MAX_SAMPLES}条.json 2>&1 | tee -a ${log_file}
    done

    echo "<<< 测试完成: ${name}"
}

export_results() {
    local exp_result_dir=$1
    local dataset_name=$2
    local train_file=$3
    local noise_ratio=$4

    {
        echo "数据集: ${dataset_name}"
        echo "训练数据: ${train_file}"
        echo "模型: qwen3-4b"
        echo "训练轮数: ${NUM_EPOCHS}"
        echo "测试集规模: 各${MAX_SAMPLES}条"
        echo "噪声比例: ${noise_ratio}"
        echo ""
        echo "测试结果:"
        echo "-----------------------------------"
        for step in 3step 5step 7step; do
            echo "${step}:"
            grep -E "正确:|准确率:" ${exp_result_dir}/test_${step}.log 2>/dev/null | tail -2
            echo ""
        done
    } > ${exp_result_dir}/summary.txt

    echo ">>> 结果已导出: ${exp_result_dir}"
}

run_experiment() {
    local exp_name=$1
    local dataset_file=$2
    local noise_ratio=$3
    local mode=${4:-full}
    local timestamp=$(date +%Y%m%d_%H%M%S)

    local train_file="${DATASET_DIR}/${dataset_file}.jsonl"
    local output_dir="${BASE_DIR}/output/${exp_name}"
    local model_dir="${output_dir}/final_model"
    local exp_result_dir="${RESULTS_DIR}/${exp_name}_${timestamp}"

    echo ""
    echo "############################################################"
    echo "实验: ${exp_name}"
    echo "噪声比例: ${noise_ratio}"
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
        export_results "${exp_result_dir}" "${exp_name}" "${train_file}" "${noise_ratio}"
    fi
}

cd ${BASE_DIR}

if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate qwen1
fi

mkdir -p ${RESULTS_DIR}

EXP_TYPE=${1:-all}
MODE=${2:-full}

echo "============================================================"
echo "噪声鲁棒性实验脚本 - Qwen3-4B"
echo "实验类型: ${EXP_TYPE}"
echo "运行模式: ${MODE}"
echo "GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "============================================================"

declare -A EXPERIMENTS
EXPERIMENTS[type7_noise_0]="train_3step_noise_0|p=0 (无噪声)"
EXPERIMENTS[type7_noise_5]="train_3step_noise_5|p=0.05"
EXPERIMENTS[type7_noise_10]="train_3step_noise_10|p=0.10"
EXPERIMENTS[type7_noise_20]="train_3step_noise_20|p=0.20"
EXPERIMENTS[type8_noise_0]="train_mixed_3_5step_noise_0|p=0 (无噪声)"
EXPERIMENTS[type8_noise_5]="train_mixed_3_5step_noise_5|p=0.05"
EXPERIMENTS[type8_noise_10]="train_mixed_3_5step_noise_10|p=0.10"
EXPERIMENTS[type8_noise_20]="train_mixed_3_5step_noise_20|p=0.20"
EXPERIMENTS[nosub_noise_0]="ablation_no_substitute_noise_0|p=0 (无噪声)"
EXPERIMENTS[nosub_noise_5]="ablation_no_substitute_noise_5|p=0.05"
EXPERIMENTS[nosub_noise_10]="ablation_no_substitute_noise_10|p=0.10"
EXPERIMENTS[nosub_noise_20]="ablation_no_substitute_noise_20|p=0.20"

run_single() {
    local exp_name=$1
    local config=${EXPERIMENTS[$exp_name]}
    if [ -z "$config" ]; then
        echo "错误: 未知实验 '$exp_name'"
        return 1
    fi
    local dataset_file=$(echo "$config" | cut -d'|' -f1)
    local noise_ratio=$(echo "$config" | cut -d'|' -f2)
    run_experiment "$exp_name" "$dataset_file" "$noise_ratio" "$MODE"
}

run_all_type7() {
    run_single type7_noise_0 || return 1; check_continue
    run_single type7_noise_5 || return 1; check_continue
    run_single type7_noise_10 || return 1; check_continue
    run_single type7_noise_20 || return 1
}

run_all_type8() {
    run_single type8_noise_0 || return 1; check_continue
    run_single type8_noise_5 || return 1; check_continue
    run_single type8_noise_10 || return 1; check_continue
    run_single type8_noise_20 || return 1
}

run_all_nosub() {
    run_single nosub_noise_0 || return 1; check_continue
    run_single nosub_noise_5 || return 1; check_continue
    run_single nosub_noise_10 || return 1; check_continue
    run_single nosub_noise_20 || return 1
}

run_all() {
    generate_datasets
    run_all_type7 || return 1; check_continue
    run_all_type8 || return 1; check_continue
    run_all_nosub || return 1
}

case ${EXP_TYPE} in
    type7_noise_0|type7_noise_5|type7_noise_10|type7_noise_20)
        run_single ${EXP_TYPE} || exit 1
        ;;
    type8_noise_0|type8_noise_5|type8_noise_10|type8_noise_20)
        run_single ${EXP_TYPE} || exit 1
        ;;
    nosub_noise_0|nosub_noise_5|nosub_noise_10|nosub_noise_20)
        run_single ${EXP_TYPE} || exit 1
        ;;
    all_type7)
        generate_datasets
        run_all_type7 || exit 1
        ;;
    all_type8)
        generate_datasets
        run_all_type8 || exit 1
        ;;
    all_nosub)
        generate_datasets
        run_all_nosub || exit 1
        ;;
    all)
        run_all || exit 1
        ;;
    generate_data)
        generate_datasets
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