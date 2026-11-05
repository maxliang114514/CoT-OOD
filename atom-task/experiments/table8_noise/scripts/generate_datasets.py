#!/usr/bin/env python3
"""
生成噪声鲁棒性实验数据集

对 train_3step, train_mixed_3_5step, ablation_no_substitute
分别注入不同比例的噪声 (p=0.05, 0.10, 0.20)

注意：每个数据集独立训练，不是混合
"""

import json
import os
import random
import argparse
from typing import List


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))


DATA_SOURCE_DIR = os.path.join(PROJECT_ROOT, 'datasets')


OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'experiments', 'noise_robust', 'datasets')


DATASETS = {
    "train_3step": {
        "source": f"{DATA_SOURCE_DIR}/train_3step.jsonl",
    },
    "train_mixed_3_5step": {
        "source": f"{DATA_SOURCE_DIR}/train_mixed_3_5step.jsonl",
    },
    "ablation_no_substitute": {
        "source": f"{DATA_SOURCE_DIR}/ablation_no_substitute.jsonl",
    },
}


def load_jsonl(file_path: str) -> List[dict]:
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data: List[dict], file_path: str):
    """保存JSONL文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def convert_to_cot_format(sample: dict) -> str:
    """
    将sample的output转换为CoT格式 - 带Task_Chain记录

    CoT格式：
    Step 1: Task_Chain: [Duplicate, Reverse, Substitute] | Input_Sequence: [xxx] | → Output_Sequence: [xxx] | Task_Chain: [Reverse, Substitute]
    Step 2: Task_Chain: [Reverse, Substitute] | Input_Sequence: [xxx] | → Output_Sequence: [xxx] | Task_Chain: [Substitute]
    ...
    Final Answer: [xxx]
    """
    output = sample.get('output', '')
    task_chain_str = sample.get('task_chain_readable', sample.get('task_chain', ''))
    reasoning_steps = sample.get('reasoning_steps', [])


    if '→' in task_chain_str:
        task_chain = [t.strip() for t in task_chain_str.split('→')]
    elif '->' in task_chain_str:
        task_chain = [t.strip() for t in task_chain_str.split('->')]
    else:
        task_chain = [task_chain_str.strip()]

    steps = []
    for i, step_info in enumerate(reasoning_steps):
        step_input = step_info.get('input', '')
        step_output = step_info.get('output', '')


        remaining_tasks = task_chain[i+1:]
        current_task_chain_display = ', '.join(task_chain[i:])
        remaining_task_chain_display = ', '.join(remaining_tasks)


        if remaining_tasks:
            step_text = f"Step {i+1}: Task_Chain: [{current_task_chain_display}] | Input_Sequence: [{step_input}] | → Output_Sequence: [{step_output}] | Task_Chain: [{remaining_task_chain_display}]"
        else:
            step_text = f"Step {i+1}: Task_Chain: [{current_task_chain_display}] | Input_Sequence: [{step_input}] | → Output_Sequence: [{step_output}] | Task_Chain: []"

        steps.append(step_text)


    cot_output = '\n'.join(steps) + f"\nFinal Answer: [{output}]"
    return cot_output


def process_dataset(dataset_name: str, data: List[dict], noise_ratios: List[float], seed: int = 42):
    """
    处理单个数据集：转换为CoT格式，然后注入不同比例的噪声

    Args:
        dataset_name: 数据集名称
        data: 数据列表
        noise_ratios: 噪声比例列表
        seed: 随机种子
    """
    print(f"\n{'='*60}")
    print(f"处理数据集: {dataset_name}, 共 {len(data)} 条")
    print(f"{'='*60}")


    print("转换为CoT格式...")
    for sample in data:
        if 'output' in sample and 'Step' not in sample['output']:
            sample['output'] = convert_to_cot_format(sample)


    clean_path = f"{OUTPUT_DIR}/{dataset_name}_noise_0.jsonl"
    save_jsonl(data, clean_path)
    print(f"保存干净版本: {clean_path}")


    from add_noise_to_cot import inject_noise_to_sample


    for ratio in noise_ratios:
        print(f"\n注入噪声比例 p={ratio}...")

        random.seed(seed + int(ratio * 100))
        noisy_data = []
        for sample in data:
            noisy_sample, _ = inject_noise_to_sample(sample, ratio, 'both')
            noisy_data.append(noisy_sample)

        output_path = f"{OUTPUT_DIR}/{dataset_name}_noise_{int(ratio*100)}.jsonl"
        save_jsonl(noisy_data, output_path)
        print(f"保存: {output_path}, 共 {len(noisy_data)} 条")


def main():
    parser = argparse.ArgumentParser(description='生成噪声鲁棒性实验数据集')
    parser.add_argument('--noise_ratios', type=str, default='0.05,0.10,0.20',
                        help='噪声比例，逗号分隔')
    parser.add_argument('--datasets', type=str, default='train_3step,train_mixed_3_5step,ablation_no_substitute',
                        help='数据集名称，逗号分隔')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    args = parser.parse_args()

    noise_ratios = [float(x) for x in args.noise_ratios.split(',')]
    dataset_names = args.datasets.split(',')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"{'='*60}")
    print("噪声鲁棒性实验数据集生成")
    print(f"数据集: {dataset_names}")
    print(f"噪声比例: {noise_ratios}")
    print(f"{'='*60}")


    for name in dataset_names:
        name = name.strip()
        if name not in DATASETS:
            print(f"警告: 未知数据集 '{name}'，跳过")
            continue

        config = DATASETS[name]
        data = load_jsonl(config["source"])
        process_dataset(name, data, noise_ratios, args.seed)

    print(f"\n{'='*60}")
    print("所有数据集处理完成!")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"{'='*60}")


    print("\n生成的数据集文件:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(fpath) / (1024*1024)
        print(f"  {f} ({size:.1f} MB)")


if __name__ == '__main__':
    main()