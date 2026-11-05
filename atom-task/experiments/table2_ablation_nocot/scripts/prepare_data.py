#!/usr/bin/env python3
"""
将 JSONL 格式的数据集转换为 No-CoT 训练所需的 JSONL 格式
用于 LlamaFactory 训练
"""

import json
import os
import sys
from pathlib import Path



NOCOT_PROMPT_TEMPLATE = """You are given a sequence of operations to perform on a symbol list.
Definitions of atomic operations:
1. Reverse: Reverse the sequence order.
2. Substitute: Apply pairwise swaps {α↔β, γ↔δ, ε↔ω}.
3. Duplicate: Copy each symbol consecutively (e.g., αβγ → ααββγγ).
4. Deduplicate: Remove consecutive duplicate elements.
5. Rotate: Move the first element to the end.

Now solve this problem:
Input Sequence: [{input_sequence}]
Task Chain: [{task_chain}]

Provide the final answer directly without showing reasoning steps."""


def convert_sample(sample: dict, use_cot: bool = True) -> dict:
    """将样本转换为 No-CoT 格式

    Args:
        sample: 原始样本 dict
        use_cot: 是否使用 CoT 格式 (False 表示直接输出最终答案)
    """
    input_text = sample.get('input', '')
    task_chain_readable = sample.get('task_chain_readable', sample.get('task_chain', ''))
    output_text = sample.get('output', '')


    instruction = NOCOT_PROMPT_TEMPLATE.replace('{input_sequence}', input_text)
    instruction = instruction.replace('{task_chain}', task_chain_readable)


    output = f"Final Answer: [{output_text}]"

    return {
        'instruction': instruction,
        'input': '',
        'output': output
    }


def convert_jsonl_to_alpaca(input_file, output_file, use_cot: bool = True):
    """将 JSONL 转换为 Alpaca JSON 格式 (No-CoT)"""
    samples = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            converted = convert_sample(data, use_cot=use_cot)
            samples.append(converted)

    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    return len(samples)


def main():
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / 'data'



    source_dir = project_dir.parent.parent / 'datasets'


    data_dir.mkdir(exist_ok=True)


    datasets = {
        'train_mixed_3_5step.jsonl': ('train_mixed_3_5step.jsonl',),
        'train_3step.jsonl': ('train_3step.jsonl',),
        'train_1step.jsonl': ('train_1step.jsonl',),
        'ablation_no_reverse.jsonl': ('ablation_no_reverse.jsonl',),
        'ablation_no_duplicate.jsonl': ('ablation_no_duplicate.jsonl',),
        'ablation_no_deduplicate.jsonl': ('ablation_no_deduplicate.jsonl',),
        'ablation_no_substitute.jsonl': ('ablation_no_substitute.jsonl',),
        'ablation_no_rotate.jsonl': ('ablation_no_rotate.jsonl',),
        'test_3step.jsonl': ('test_3step.jsonl',),
        'test_5step.jsonl': ('test_5step.jsonl',),
        'test_7step.jsonl': ('test_7step.jsonl',),
    }

    print("=" * 60)
    print("转换数据集格式: JSONL -> Alpaca JSON (No-CoT 格式)")
    print("=" * 60)
    print(f"数据源: {source_dir}")
    print(f"目标目录: {data_dir}")
    print("=" * 60)

    for source_name, (target_name,) in datasets.items():
        source_path = source_dir / source_name
        target_path = data_dir / target_name

        if not source_path.exists():
            print(f"警告: 源文件不存在 {source_path}")
            continue


        print(f"\n转换: {source_name} -> {target_name}")
        print(f"  源: {source_path}")
        count = convert_jsonl_to_alpaca(source_path, target_path, use_cot=False)
        print(f"  转换完成: {count} 条样本")

    print("\n" + "=" * 60)
    print("数据集转换完成!")

    for source_name, (target_name, source_dir) in datasets.items():
        source_path = source_dir / source_name
        target_path = data_dir / target_name

        if not source_path.exists():
            print(f"警告: 源文件不存在 {source_path}")
            continue


        print(f"\n转换: {source_name} -> {target_name}")
        print(f"  源: {source_path}")
        count = convert_jsonl_to_alpaca(source_path, target_path, use_cot=False)
        print(f"  转换完成: {count} 条样本")

    print("\n" + "=" * 60)
    print("数据集转换完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()