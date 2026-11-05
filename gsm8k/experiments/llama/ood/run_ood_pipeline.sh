#!/bin/bash
# ============================================================
# ============================================================

STOP_FILE=$(mktemp /tmp/ood_pipeline_stop_XXXXXX)
trap 'rm -f "$STOP_FILE"; echo "清理停止文件"; exit 0' EXIT

SHOULD_CONTINUE=true

cleanup() {
    echo ""
    echo "!!! 收到终止信号，正在退出..."
    SHOULD_CONTINUE=false
    rm -f "$STOP_FILE"
    sleep 1
    exit 1
}
trap 'cleanup' SIGINT SIGTERM

check_stop() {
    if [ ! -f "$STOP_FILE" ]; then
        echo "检测到停止信号"
        return 1
    fi
    return 0
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${SCRIPT_DIR}"
GSM8K_TEST="${ROOT_DIR}/datasets/gsm8k/test.jsonl"
METAMATH="${ROOT_DIR}/datasets/metamath/MetaMathQA-395K.json"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/models/llama2-7b}"   # 可通过环境变量覆盖
USE_VLLM=${USE_VLLM:-1}  # 设为0使用原生HF推理，设为1使用vLLM加速

show_help() {
    cat << EOF
用法: bash run_ood_pipeline.sh <实验类型> [模式]

实验类型:
  baseline      - 基线实验（预训练模型直接测试）
  nocot         - 无CoT对照实验
  noise         - 噪声注入实验 (0.05 / 0.1 / 0.2)
  stepwise      - 单步实验
  ood           - 分布偏移实验 (domain/complexity/magnitude/all)
  all           - 运行全部实验

模式 (默认 full):
  data_only     - 仅生成数据
  train_only    - 仅训练
  test_only     - 仅测试
  full          - 完整流程 (数据+训练+测试)

GPU:
  通过环境变量 CUDA_VISIBLE_DEVICES 指定

示例:
  CUDA_VISIBLE_DEVICES=0 bash run_ood_pipeline.sh all full      # 完整流程，使用GPU 0
  CUDA_VISIBLE_DEVICES=1 bash run_ood_pipeline.sh noise train_only  # 仅训练噪声实验
  CUDA_VISIBLE_DEVICES=2 bash run_ood_pipeline.sh ood test_only    # 仅测试OOD实验

中断:
  Ctrl+C 或 kill -TERM 会安全退出
  rm /tmp/ood_pipeline_stop_* 可远程停止

EOF
}

declare -A EXPERIMENTS
EXPERIMENTS=(
    ["baseline"]="基线实验（预训练模型）"
    ["nocot"]="无CoT对照实验"
    ["noise_0"]="噪声实验-0%（对照）"
    ["noise_05"]="噪声实验-5%"
    ["noise_10"]="噪声实验-10%"
    ["noise_20"]="噪声实验-20%"
    ["noise_50"]="噪声实验-50%"
    ["stepwise"]="单步实验"
    ["ood_domain"]="OOD-主题偏移"
    ["ood_complexity"]="OOD-复杂度偏移"
    ["ood_magnitude"]="OOD-答案规模偏移"
    ["ood_all"]="OOD-综合偏移"
)

TOTAL_EXP=12
CURRENT_EXP=0

get_progress() {
    echo "[${CURRENT_EXP}/${TOTAL_EXP}]"
}

step_data_prepare() {
    local exp_type=$1
    echo ""
    echo ">>> $(get_progress) 数据准备: ${exp_type}"

    if [ "$exp_type" = "baseline" ]; then
        echo "    [跳过] 基线实验无需数据准备"

    elif [ "$exp_type" = "nocot" ]; then
        output="$EXP_DIR/data/metamath_nocot.json"
        if [ -f "$output" ]; then
            echo "    [跳过] ${output} 已存在"
        else
            python $SCRIPT_DIR/../cot_no_cot/remove_cot.py \
                --input_file $METAMATH \
                --output_file $output \
                --max_samples 30000
        fi

    elif [ "$exp_type" = "noise" ]; then
        for rate in 0 5 10 20 50; do
            check_stop || return 1
            output="$EXP_DIR/data/metamath_noise${rate}.json"
            [ -f "$output" ] && echo "    [跳过] ${output} 已存在" && continue
            if [ "$rate" == "0" ]; then
                cp $METAMATH $output
            else
                noise_rate=$(awk "BEGIN {print $rate/100}")
                python $SCRIPT_DIR/add_noise.py \
                    --input_file $METAMATH \
                    --output_file $output \
                    --noise_rate $noise_rate \
                    --noise_type both
            fi
        done

    elif [ "$exp_type" = "stepwise" ]; then
        output="$EXP_DIR/data/metamath_stepwise.jsonl"
        if [ -f "$output" ]; then
            echo "    [跳过] ${output} 已存在"
        else
            python $SCRIPT_DIR/create_stepwise.py \
                --input_file $METAMATH \
                --output_file $output \
                --filter_min_steps 3 \
                --max_samples 30000
        fi

    elif [ "$exp_type" = "ood" ]; then
        for mode in domain complexity magnitude all; do
            check_stop || return 1
            output="$EXP_DIR/data/metamath_ood_${mode}.json"
            [ -f "$output" ] && echo "    [跳过] ${output} 已存在" && continue
            python $SCRIPT_DIR/filter_distribution_shift.py \
                --metamath_file $METAMATH \
                --gsm8k_file $GSM8K_TEST \
                --output_file $output \
                --shift_mode $mode \
                --shift_ratio 0.3
        done
    fi

    echo "<<< 数据准备完成"
    return 0
}

step_train() {
    local exp_name=$1
    local data_file=$2
    local output_dir=$3
    local max_samples=${4:-30000}

    echo ""
    echo ">>> $(get_progress) 训练: ${exp_name}"
    echo "    数据: ${data_file}"
    echo "    输出: ${output_dir}"

    # if [ -f "$output_dir/model-00001-of-00003.safetensors" ] || \
    #    [ -f "$output_dir/adapter_model.safetensors" ]; then
    #     return 0
    # fi

    python $SCRIPT_DIR/../cot_no_cot/train.py \
        --model_name_or_path $MODEL_PATH \
        --data_path "$data_file" \
        --max_train_samples $max_samples \
        --output_dir "$output_dir" \
        --overwrite_output_dir \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --learning_rate 2e-5 \
        --model_max_length 512 \
        --lora_rank 16 --lora_alpha 16 \
        --lora_target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
        --bf16

    echo "<<< 训练完成"
    return 0
}

step_test() {
    local exp_name=$1
    local model_path=$2
    local result_prefix=$3

    echo ""
    echo ">>> $(get_progress) 测试: ${exp_name}"
    echo "    模型: ${model_path}"

    if [ ! -d "$model_path" ]; then
        echo "    [跳过] 模型不存在: ${model_path}"
        return 0
    fi

    has_model=false

    if [ -f "$model_path/model-00001-of-00002.safetensors" ] || \
       [ -f "$model_path/model-00001-of-00003.safetensors" ] || \
       [ -f "$model_path/model.safetensors" ]; then
        has_model=true
    elif [ -f "$model_path/adapter_model.safetensors" ]; then
        # LoRA adapter directly in model path (LlamaFactory default)
        has_model=true
    elif [ -d "$model_path/checkpoint-100000" ]; then
        has_model=true
        model_path="$model_path/checkpoint-100000"
    elif [ -d "$model_path/checkpoint-210000" ]; then
        has_model=true
        model_path="$model_path/checkpoint-210000"
    elif compgen -G "$model_path/checkpoint-*/adapter_model.safetensors" > /dev/null; then
        last_ckpt=$(ls -d $model_path/checkpoint-* 2>/dev/null | sort -V | tail -1)
        if [ -n "$last_ckpt" ]; then
            model_path="$last_ckpt"
            has_model=true
        fi
    fi

    if [ "$has_model" = "false" ]; then
        echo "    [跳过] 无可用模型"
        return 0
    fi

    output_file="$EXP_DIR/results/${result_prefix}.jsonl"
    acc_file="$EXP_DIR/results/${result_prefix}_acc.json"

    mkdir -p "$EXP_DIR/results"

    if [ "$USE_VLLM" = "1" ]; then
        python $SCRIPT_DIR/../scripts/gen_response_vllm.py \
            --model "$model_path" \
            --data_file $GSM8K_TEST \
            --output_file "$output_file"
    else
        python $SCRIPT_DIR/../scripts/gen_response.py \
            --model "$model_path" \
            --data_file $GSM8K_TEST \
            --output_file "$output_file"
    fi

    python $SCRIPT_DIR/../baseline/test_acc.py \
        --input_file "$output_file" \
        --output_file "$acc_file"

    if [ -f "$acc_file" ]; then
        acc=$(python3 -c "import json; d=json.load(open('$acc_file')); print(d['summary']['accuracy_percent'])")
        echo "    准确率: $acc"
    fi

    echo "<<< 测试完成"
    return 0
}

run_baseline_exp() {
    local mode=${1:-full}
    ((CURRENT_EXP++))
    check_stop || return 1

    if [ "$mode" = "data_only" ]; then
        step_data_prepare "baseline"
    elif [ "$mode" = "train_only" ]; then
        echo "    [跳过] 基线实验无需训练"
    elif [ "$mode" = "test_only" ]; then
        step_test "基线" \
            "$MODEL_PATH" \
            "llama_baseline"
    else
        step_test "基线" \
            "$MODEL_PATH" \
            "llama_baseline"
    fi
}

run_nocot_exp() {
    local mode=${1:-full}
    ((CURRENT_EXP++))
    check_stop || return 1

    if [ "$mode" = "data_only" ]; then
        step_data_prepare "nocot"
    elif [ "$mode" = "train_only" ]; then
        step_train "无CoT" \
            "$EXP_DIR/data/metamath_nocot.json" \
            "$EXP_DIR/llama_nocot" \
            30000
    elif [ "$mode" = "test_only" ]; then
        step_test "无CoT" \
            "$EXP_DIR/llama_nocot" \
            "llama_nocot"
    else
        # full
        step_train "无CoT" \
            "$EXP_DIR/data/metamath_nocot.json" \
            "$EXP_DIR/llama_nocot" \
            30000
        check_stop || return 1
        step_test "无CoT" \
            "$EXP_DIR/llama_nocot" \
            "llama_nocot"
    fi
}

run_noise_exp() {
    local mode=${1:-full}

    for rate in 0 5 10 20 50; do
        ((CURRENT_EXP++))
        check_stop || return 1

        if [ "$mode" = "data_only" ]; then
            step_data_prepare "noise"
        elif [ "$mode" = "train_only" ]; then
            step_train "噪声-${rate}%" \
                "$EXP_DIR/data/metamath_noise${rate}.json" \
                "$EXP_DIR/llama_noise${rate}"
        elif [ "$mode" = "test_only" ]; then
            step_test "噪声-${rate}%" \
                "$EXP_DIR/llama_noise${rate}" \
                "llama_noise${rate}"
        else
            # full
            step_train "噪声-${rate}%" \
                "$EXP_DIR/data/metamath_noise${rate}.json" \
                "$EXP_DIR/llama_noise${rate}"
            check_stop || return 1
            step_test "噪声-${rate}%" \
                "$EXP_DIR/llama_noise${rate}" \
                "llama_noise${rate}"
        fi
    done
}

run_stepwise_exp() {
    local mode=${1:-full}
    ((CURRENT_EXP++))
    check_stop || return 1

    if [ "$mode" = "data_only" ]; then
        step_data_prepare "stepwise"
    elif [ "$mode" = "train_only" ]; then
        step_train "单步" \
            "$EXP_DIR/data/metamath_stepwise.jsonl" \
            "$EXP_DIR/llama_stepwise" \
            210000
    elif [ "$mode" = "test_only" ]; then
        step_test "单步" \
            "$EXP_DIR/llama_stepwise" \
            "llama_stepwise"
    else
        step_train "单步" \
            "$EXP_DIR/data/metamath_stepwise.jsonl" \
            "$EXP_DIR/llama_stepwise" \
            210000
        check_stop || return 1
        step_test "单步" \
            "$EXP_DIR/llama_stepwise" \
            "llama_stepwise"
    fi
}

run_ood_exp() {
    local mode=${1:-full}

    for ood_type in domain complexity magnitude all; do
        ((CURRENT_EXP++))
        check_stop || return 1

        ood_name=""
        case $ood_type in
            domain) ood_name="主题偏移" ;;
            complexity) ood_name="复杂度偏移" ;;
            magnitude) ood_name="答案规模偏移" ;;
            all) ood_name="综合偏移" ;;
        esac

        if [ "$mode" = "data_only" ]; then
            step_data_prepare "ood"
        elif [ "$mode" = "train_only" ]; then
            step_train "OOD-${ood_name}" \
                "$EXP_DIR/data/metamath_ood_${ood_type}.json" \
                "$EXP_DIR/llama_ood_${ood_type}"
        elif [ "$mode" = "test_only" ]; then
            step_test "OOD-${ood_name}" \
                "$EXP_DIR/llama_ood_${ood_type}" \
                "llama_ood_${ood_type}"
        else
            step_train "OOD-${ood_name}" \
                "$EXP_DIR/data/metamath_ood_${ood_type}.json" \
                "$EXP_DIR/llama_ood_${ood_type}"
            check_stop || return 1
            step_test "OOD-${ood_name}" \
                "$EXP_DIR/llama_ood_${ood_type}" \
                "llama_ood_${ood_type}"
        fi
    done
}

print_summary() {
    echo ""
    echo "############################################################"
    echo "结果汇总"
    echo "############################################################"
    echo ""

    for acc_file in $EXP_DIR/results/*_acc.json; do
        if [ -f "$acc_file" ]; then
            name=$(basename $acc_file _acc.json)
            acc=$(python3 -c "import json; d=json.load(open('$acc_file')); print(d['summary']['accuracy_percent'])")
            correct=$(python3 -c "import json; d=json.load(open('$acc_file')); print(d['summary']['correct'])")
            total=$(python3 -c "import json; d=json.load(open('$acc_file')); print(d['summary']['total'])")
            echo "  $name: $acc ($correct/$total)"
        fi
    done

    echo ""
}

EXP_TYPE=${1:-all}
MODE=${2:-full}

echo "============================================================"
echo "OOD泛化实验流水线"
echo "============================================================"
echo "实验类型: ${EXP_TYPE}"
echo "运行模式: ${MODE}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "停止文件: ${STOP_FILE}"
echo "============================================================"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen1

cd $SCRIPT_DIR

mkdir -p $EXP_DIR/data
mkdir -p $EXP_DIR/results

case $EXP_TYPE in
    baseline)
        step_data_prepare "baseline"
        run_baseline_exp "$MODE"
        ;;
    nocot)
        step_data_prepare "nocot"
        run_nocot_exp "$MODE"
        ;;
    noise)
        step_data_prepare "noise"
        run_noise_exp "$MODE"
        ;;
    stepwise)
        step_data_prepare "stepwise"
        run_stepwise_exp "$MODE"
        ;;
    ood)
        step_data_prepare "ood"
        run_ood_exp "$MODE"
        ;;
    all)
        CURRENT_EXP=0
        echo ""
        echo "############################################################"
        echo "[${CURRENT_EXP}/${TOTAL_EXP}] 数据准备阶段"
        echo "############################################################"
        step_data_prepare "baseline"
        step_data_prepare "nocot"
        step_data_prepare "noise"
        step_data_prepare "stepwise"
        step_data_prepare "ood"

        run_baseline_exp "$MODE"
        check_stop || { print_summary; exit 1; }
        run_nocot_exp "$MODE"
        check_stop || { print_summary; exit 1; }
        run_noise_exp "$MODE"
        check_stop || { print_summary; exit 1; }
        run_stepwise_exp "$MODE"
        check_stop || { print_summary; exit 1; }
        run_ood_exp "$MODE"
        ;;
    -h|--help|help)
        show_help
        exit 0
        ;;
    *)
        echo "错误: 未知实验类型 '$EXP_TYPE'"
        show_help
        exit 1
        ;;
esac

print_summary

echo ""
echo "############################################################"
echo "所有实验完成!"
echo "############################################################"