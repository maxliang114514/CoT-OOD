#!/usr/bin/env python3
"""
OOD泛化实验一站式流水线

支持三种实验的完整流程:
1. 噪声注入实验 (noise)
2. 单步实验 (stepwise)
3. 分布偏移实验 (ood)

使用方法:
    python run_ood_experiments.py --exp noise --noise_rate 0.1
    python run_ood_experiments.py --exp all
    python run_ood_experiments.py --exp noise --noise_rate 0.05 --train --test

停止机制:
    - 脚本运行时会在 /tmp 创建停止文件 (pipeline_stop_<pid>)
    - kill 链条中的进程后，删除停止文件即可终止后续步骤
    - 例: rm /tmp/pipeline_stop_*
"""

import argparse
import json
import os
import subprocess
import sys
import signal
import atexit
from datetime import datetime
from pathlib import Path



ROOT_DIR = Path(__file__).resolve().parents[3]
BASE_MODEL_LLAMA = os.path.join(ROOT_DIR, "models", "llama2-7b")
BASE_MODEL_QWEN = os.path.join(ROOT_DIR, "models", "qwen2.5-3b")
DATA_DIR = os.path.join(ROOT_DIR, "datasets", "metamath")
OUTPUT_DIR = os.path.join(ROOT_DIR, "experiments", "llama", "ood")
GSM8K_TEST = os.path.join(ROOT_DIR, "datasets", "gsm8k", "test.jsonl")

SCRIPTS_DIR = Path(__file__).parent.absolute()
COT_NO_COT_DIR = SCRIPTS_DIR.parent / "cot_no_cot"


STOP_FILE = f"/tmp/pipeline_stop_{os.getpid()}"
SHOULD_CONTINUE = True

def cleanup_stop_file():
    """清理停止文件"""
    if os.path.exists(STOP_FILE):
        os.remove(STOP_FILE)

def signal_handler(signum, frame):
    """收到信号时设置停止标志"""
    global SHOULD_CONTINUE
    print(f"\n收到信号 {signum}，正在退出...")
    SHOULD_CONTINUE = False
    cleanup_stop_file()


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

atexit.register(cleanup_stop_file)


with open(STOP_FILE, 'w') as f:
    f.write(str(os.getpid()))

def check_stop():
    """检查是否收到停止信号"""
    global SHOULD_CONTINUE
    if not SHOULD_CONTINUE:
        return False
    if not os.path.exists(STOP_FILE):
        print("检测到停止信号")
        SHOULD_CONTINUE = False
        return False
    return True


def run_cmd(cmd, desc="", check=True):
    """执行命令"""
    if not check_stop():
        print(f"步骤已停止: {desc}")
        return False
    print(f"\n{'='*60}")
    print(f"{desc}")
    print(f"{'='*60}")
    print(f"CMD: {cmd}\n")

    pass
    result = subprocess.run(cmd, shell=True, check=check)
    if result.returncode != 0:
        print(f"命令执行失败: {cmd}")
    return result.returncode == 0


def step_data_prepare(exp_type, args):
    """生成实验数据"""
    print(f"\n{'#'*60}")
    print(f"# 数据准备: {exp_type}")
    print(f"{'#'*60}")

    if exp_type == "noise":

        for rate in [0.0, 0.05, 0.1, 0.2]:
            output = f"{OUTPUT_DIR}/data/metamath_noise{int(rate*100)}.json"
            os.makedirs(os.path.dirname(output), exist_ok=True)
            if rate == 0.0:

                cmd = f"cp {DATA_DIR}/MetaMathQA-395K.json {output}"
                run_cmd(cmd, f"生成0噪声对照数据 (rate=0)", check=False)
            else:
                cmd = f"""python {SCRIPTS_DIR}/add_noise.py \
                    --input_file {DATA_DIR}/MetaMathQA-395K.json \
                    --output_file {output} \
                    --noise_rate {rate} \
                    --noise_type {args.noise_type}"""
                run_cmd(cmd, f"生成噪声数据 (rate={rate})", check=False)
        return True

    elif exp_type == "stepwise":
        output = f"{OUTPUT_DIR}/data/metamath_stepwise.jsonl"
        os.makedirs(os.path.dirname(output), exist_ok=True)
        cmd = f"""python {SCRIPTS_DIR}/create_stepwise.py \
            --input_file {DATA_DIR}/MetaMathQA-395K.json \
            --output_file {output} \
            --filter_min_steps {args.min_steps} \
            --max_samples 10000"""
        run_cmd(cmd, "生成单步数据", check=False)
        return True

    elif exp_type == "ood":

        for mode in ["domain", "complexity", "magnitude", "all"]:
            output = f"{OUTPUT_DIR}/data/metamath_ood_{mode}.json"
            os.makedirs(os.path.dirname(output), exist_ok=True)
            cmd = f"""python {SCRIPTS_DIR}/filter_distribution_shift.py \
                --metamath_file {DATA_DIR}/MetaMathQA-395K.json \
                --gsm8k_file {GSM8K_TEST} \
                --output_file {output} \
                --shift_mode {mode} \
                --shift_ratio 0.3"""
            run_cmd(cmd, f"生成OOD数据 (mode={mode})", check=False)
        return True

    return False


def step_train(exp_type, model, gpu, args):
    """训练模型"""
    print(f"\n{'#'*60}")
    print(f"# 训练模型: {exp_type} ({model})")
    print(f"{'#'*60}")

    if model == "llama":
        train_script = f"{COT_NO_COT_DIR}/train.py"
        base_model = BASE_MODEL_LLAMA
    else:
        train_script = f"{COT_NO_COT_DIR}/train_qwen.py"
        base_model = BASE_MODEL_QWEN

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    if exp_type == "noise":
        for rate in [0.0, 0.05, 0.1, 0.2]:
            data_file = f"{OUTPUT_DIR}/data/metamath_noise{int(rate*100)}.json"
            output_dir = f"{OUTPUT_DIR}/llama_noise{int(rate*100)}" if model == "llama" else f"{OUTPUT_DIR}/qwen_noise{int(rate*100)}"

            cmd = f"""python {train_script} \
                --model_name_or_path {base_model} \
                --data_path {data_file} \
                --max_train_samples 10000 \
                --output_dir {output_dir} \
                --overwrite_output_dir \
                --num_train_epochs 1 \
                --per_device_train_batch_size 1 \
                --learning_rate 2e-5 \
                --model_max_length 512 \
                --lora_rank 16 --lora_alpha 16 \
                --bf16"""

            if model == "llama":
                cmd += """ --lora_target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" """

            run_cmd(cmd, f"训练 ({model}, noise={rate})", check=False)

    elif exp_type == "stepwise":
        data_file = f"{OUTPUT_DIR}/data/metamath_stepwise.jsonl"
        output_dir = f"{OUTPUT_DIR}/llama_stepwise" if model == "llama" else f"{OUTPUT_DIR}/qwen_stepwise"

        cmd = f"""python {train_script} \
            --model_name_or_path {base_model} \
            --data_path {data_file} \
            --max_train_samples 100000 \
            --output_dir {output_dir} \
            --overwrite_output_dir \
            --num_train_epochs 1 \
            --per_device_train_batch_size 1 \
            --learning_rate 2e-5 \
            --model_max_length 512 \
            --lora_rank 16 --lora_alpha 16 \
            --bf16"""

        if model == "llama":
            cmd += """ --lora_target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" """

        run_cmd(cmd, f"训练 ({model}, stepwise)", check=False)

    elif exp_type == "ood":
        for mode in ["domain", "complexity", "magnitude", "all"]:
            data_file = f"{OUTPUT_DIR}/data/metamath_ood_{mode}.json"
            output_dir = f"{OUTPUT_DIR}/llama_ood_{mode}" if model == "llama" else f"{OUTPUT_DIR}/qwen_ood_{mode}"

            cmd = f"""python {train_script} \
                --model_name_or_path {base_model} \
                --data_path {data_file} \
                --max_train_samples 10000 \
                --output_dir {output_dir} \
                --overwrite_output_dir \
                --num_train_epochs 1 \
                --per_device_train_batch_size 1 \
                --learning_rate 2e-5 \
                --model_max_length 512 \
                --lora_rank 16 --lora_alpha 16 \
                --bf16"""

            if model == "llama":
                cmd += """ --lora_target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" """

            run_cmd(cmd, f"训练 ({model}, ood={mode})", check=False)

    return True


def step_test(exp_type, model, gpu, args):
    """测试模型"""
    print(f"\n{'#'*60}")
    print(f"# 测试模型: {exp_type} ({model})")
    print(f"{'#'*60}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    USE_VLLM = os.environ.get("USE_VLLM", "1") == "1"

    if model == "llama":
        gen_script = f"{COT_NO_COT_DIR}/gen_response_vllm.py" if USE_VLLM else f"{COT_NO_COT_DIR}/gen_response.py"
    else:
        gen_script = f"{COT_NO_COT_DIR}/gen_response_qwen.py"

    if exp_type == "noise":
        for rate in [0.0, 0.05, 0.1, 0.2]:
            model_path = f"{OUTPUT_DIR}/llama_noise{int(rate*100)}" if model == "llama" else f"{OUTPUT_DIR}/qwen_noise{int(rate*100)}"
            output_file = f"{OUTPUT_DIR}/results/llama_noise{int(rate*100)}.jsonl" if model == "llama" else f"{OUTPUT_DIR}/results/qwen_noise{int(rate*100)}.jsonl"

            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            cmd = f"""python {gen_script} \
                --model {model_path} \
                --data_file {GSM8K_TEST} \
                --output_file {output_file}"""

            run_cmd(cmd, f"生成回复 ({model}, noise={rate})", check=False)


            acc_output = f"{OUTPUT_DIR}/results/llama_noise{int(rate*100)}_acc.json" if model == "llama" else f"{OUTPUT_DIR}/results/qwen_noise{int(rate*100)}_acc.json"
            cmd = f"python {COT_NO_COT_DIR}/test_acc.py --input_file {output_file} --output_file {acc_output}"
            run_cmd(cmd, f"测试准确率 ({model}, noise={rate})", check=False)

    elif exp_type == "stepwise":
        model_path = f"{OUTPUT_DIR}/llama_stepwise" if model == "llama" else f"{OUTPUT_DIR}/qwen_stepwise"
        output_file = f"{OUTPUT_DIR}/results/llama_stepwise.jsonl" if model == "llama" else f"{OUTPUT_DIR}/results/qwen_stepwise.jsonl"

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        cmd = f"""python {gen_script} \
            --model {model_path} \
            --data_file {GSM8K_TEST} \
            --output_file {output_file}"""

        run_cmd(cmd, f"生成回复 ({model}, stepwise)", check=False)

        acc_output = f"{OUTPUT_DIR}/results/llama_stepwise_acc.json" if model == "llama" else f"{OUTPUT_DIR}/results/qwen_stepwise_acc.json"
        cmd = f"python {COT_NO_COT_DIR}/test_acc.py --input_file {output_file} --output_file {acc_output}"
        run_cmd(cmd, f"测试准确率 ({model}, stepwise)", check=False)

    elif exp_type == "ood":
        for mode in ["domain", "complexity", "magnitude", "all"]:
            model_path = f"{OUTPUT_DIR}/llama_ood_{mode}" if model == "llama" else f"{OUTPUT_DIR}/qwen_ood_{mode}"
            output_file = f"{OUTPUT_DIR}/results/llama_ood_{mode}.jsonl" if model == "llama" else f"{OUTPUT_DIR}/results/qwen_ood_{mode}.jsonl"

            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            cmd = f"""python {gen_script} \
                --model {model_path} \
                --data_file {GSM8K_TEST} \
                --output_file {output_file}"""

            run_cmd(cmd, f"生成回复 ({model}, ood={mode})", check=False)

            acc_output = f"{OUTPUT_DIR}/results/llama_ood_{mode}_acc.json" if model == "llama" else f"{OUTPUT_DIR}/results/qwen_ood_{mode}_acc.json"
            cmd = f"python {COT_NO_COT_DIR}/test_acc.py --input_file {output_file} --output_file {acc_output}"
            run_cmd(cmd, f"测试准确率 ({model}, ood={mode})", check=False)

    return True


def summarize_results(exp_type):
    """汇总结果"""
    print(f"\n{'='*60}")
    print(f"实验结果汇总: {exp_type}")
    print(f"{'='*60}")

    results_dir = f"{OUTPUT_DIR}/results"
    if not os.path.exists(results_dir):
        print("结果目录不存在")
        return

    acc_files = [f for f in os.listdir(results_dir) if f.endswith('_acc.json')]

    print(f"\n{'实验配置':<30} {'准确率':<15} {'正确/总数'}")
    print("-" * 60)

    for f in sorted(acc_files):
        try:
            with open(os.path.join(results_dir, f)) as fp:
                data = json.load(fp)
                summary = data.get('summary', {})
                acc = summary.get('accuracy_percent', 'N/A')
                correct = summary.get('correct', 'N/A')
                total = summary.get('total', 'N/A')
                print(f"{f:<30} {acc:<15} {correct}/{total}")
        except Exception as e:
            print(f"{f:<30} Error: {e}")

    print("-" * 60)


def print_usage():
    """打印使用说明"""
    print("""
OOD泛化实验一站式流水线
======================

用法:
    python run_ood_experiments.py --exp <实验类型> [选项]

实验类型:
    noise      噪声注入实验
    stepwise   单步实验
    ood        分布偏移实验
    all        运行全部实验的数据准备

步骤:
    --data     仅生成数据
    --train    仅训练模型
    --test     仅测试模型
    (默认: --data --train --test)

模型:
    --model    llama 或 qwen (默认: llama)

GPU:
    --gpu      GPU编号 (默认: 0)

示例:
    # 噪声实验完整流程
    python run_ood_experiments.py --exp noise --noise_rate 0.1

    # 仅生成数据
    python run_ood_experiments.py --exp all --step data

    # 训练所有实验
    python run_ood_experiments.py --exp all --step train --gpu 1

    # 仅测试
    python run_ood_experiments.py --exp noise --step test

    # 查看结果汇总
    python run_ood_experiments.py --exp noise --summarize
""")


def main():
    parser = argparse.ArgumentParser(description="OOD泛化实验流水线", add_help=False)
    parser.add_argument('--exp', type=str, required=True,
                       choices=['noise', 'stepwise', 'ood', 'all'],
                       help='实验类型')
    parser.add_argument('--step', type=str, default='all',
                       choices=['data', 'train', 'test', 'all'],
                       help='运行步骤 (默认: all)')
    parser.add_argument('--model', type=str, default='llama',
                       choices=['llama', 'qwen'],
                       help='模型类型 (默认: llama)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU编号 (默认: 0)')
    parser.add_argument('--noise_rate', type=float, default=0.1,
                       help='噪声比例 (默认: 0.1)')
    parser.add_argument('--noise_type', type=str, default='both',
                       choices=['delete', 'wrong', 'both'],
                       help='噪声类型 (默认: both)')
    parser.add_argument('--min_steps', type=int, default=2,
                       help='最小步骤数 (默认: 2)')
    parser.add_argument('--summarize', action='store_true',
                       help='汇总现有结果')
    parser.add_argument('--help', action='store_true',
                       help='显示帮助')

    args = parser.parse_args()

    if args.help:
        print_usage()
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/data", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/results", exist_ok=True)

    if args.summarize:
        summarize_results(args.exp)
        return


    exp_types = [args.exp] if args.exp != 'all' else ['noise', 'stepwise', 'ood']


    steps = ['data', 'train', 'test'] if args.step == 'all' else [args.step]

    print(f"\n{'='*60}")
    print(f"OOD泛化实验流水线")
    print(f"{'='*60}")
    print(f"实验类型: {exp_types}")
    print(f"运行步骤: {steps}")
    print(f"模型: {args.model}")
    print(f"GPU: {args.gpu}")
    print(f"{'='*60}")

    for exp_type in exp_types:
        if not check_stop():
            print(f"流水线已停止，退出")
            break
        print(f"\n{'='*60}")
        print(f"开始实验: {exp_type}")
        print(f"{'='*60}")

        if 'data' in steps:
            if not check_stop():
                print(f"流水线已停止，退出")
                break
            step_data_prepare(exp_type, args)

        if not check_stop():
            print(f"流水线已停止，退出")
            break
        if 'train' in steps:
            step_train(exp_type, args.model, args.gpu, args)

        if not check_stop():
            print(f"流水线已停止，退出")
            break
        if 'test' in steps:
            step_test(exp_type, args.model, args.gpu, args)

        summarize_results(exp_type)

    print(f"\n{'='*60}")
    print(f"所有实验完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
