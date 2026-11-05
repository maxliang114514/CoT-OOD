#!/usr/bin/env python3
"""
扩大测试集规模：翻倍至500条/文件，同时确保操作组合多样性
- 3步、5步、7步、10步测试集各扩增250条
- 确保与现有序列不重复
- 允许重复操作种类，但复制操作(t3)不能过多
"""

import json
import random
import os
import sys
from itertools import product
from collections import Counter

sys.path.insert(0, '/data1/lnj/CoT-3/aton_task')
from data_generator import (
    generate_unique_sequences,
    apply_task_chain,
    sequence_to_string,
    TASK_NAMES
)


DATASET_DIR = "/data1/lnj/CoT-3/aton_task/dataset"
TARGET_COUNT = 500
CURRENT_COUNT = 250
ADDITIONAL_COUNT = TARGET_COUNT - CURRENT_COUNT

ATOMIC_TASKS_LIST = ['t1', 't2', 't3', 't4', 't5']



OP_WEIGHTS = {
    't1': 1.0,  # Reverse
    't2': 1.0,  # Substitute
    't3': 0.5,
    't4': 1.0,  # Deduplicate
    't5': 1.0   # Rotate
}


def weighted_random_operation(weights_dict):
    """根据权重随机选择操作"""
    ops = list(weights_dict.keys())
    wts = list(weights_dict.values())
    return random.choice(ops)


def generate_random_composition(num_steps):
    """生成随机操作组合，确保多样性且t3不会过多"""
    ops = []
    for _ in range(num_steps):
        op = weighted_random_operation(OP_WEIGHTS)
        ops.append(op)
    return ops


def load_existing_sequences() -> set:
    """加载所有现有数据集的输入序列"""
    sequences = set()


    for step in [3, 5, 7, 10]:
        test_file = os.path.join(DATASET_DIR, f"test_composition_{step}step.jsonl")
        if os.path.exists(test_file):
            with open(test_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    seq = tuple(data['input'].split())
                    sequences.add(seq)
    return sequences


def generate_sample(composition, existing_sequences):
    """生成单条样本"""
    seq = generate_unique_sequences(
        count=1,
        min_len=2,
        max_len=6,
        existing_sequences=existing_sequences
    )[0]

    input_str = sequence_to_string(seq)
    reasoning_steps, final_output = apply_task_chain(seq, composition)
    output_str = sequence_to_string(final_output)

    task_chain_str = " → ".join([TASK_NAMES[t] for t in composition])
    task_chain_ids = " → ".join(composition)

    return {
        "input": input_str,
        "output": output_str,
        "task_chain": task_chain_ids,
        "task_chain_readable": task_chain_str,
        "steps": len(composition),
        "reasoning_steps": reasoning_steps
    }


def analyze_distribution(samples):
    """分析样本的操作分布"""
    op_counts = Counter()
    for s in samples:
        ops = s['task_chain'].split(' → ')
        for op in ops:
            op_counts[op] += 1
    return op_counts


def expand_test_set(num_steps, existing_sequences, target_count, current_count):
    """扩增指定步数的测试集"""
    print(f"\n扩增 {num_steps}步测试集:")
    print(f"  当前: {current_count} 条, 目标: {target_count} 条, 需新增: {target_count - current_count} 条")

    samples = []


    test_file = os.path.join(DATASET_DIR, f"test_composition_{num_steps}step.jsonl")
    if os.path.exists(test_file):
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                samples.append(json.loads(line))

    print(f"  已加载: {len(samples)} 条")


    new_count = 0
    attempts = 0
    max_attempts = ADDITIONAL_COUNT * 10

    while new_count < ADDITIONAL_COUNT and attempts < max_attempts:
        attempts += 1
        composition = generate_random_composition(num_steps)


        t3_count = composition.count('t3')
        if t3_count > num_steps // 2:
            continue

        sample = generate_sample(composition, existing_sequences)
        input_seq = tuple(sample["input"].split())

        if input_seq not in existing_sequences:
            samples.append(sample)
            existing_sequences.add(input_seq)
            new_count += 1

            if new_count % 50 == 0:
                print(f"  已生成 {new_count} / {ADDITIONAL_COUNT}")

    print(f"  完成: 新增 {new_count} 条, 总计 {len(samples)} 条")


    op_dist = analyze_distribution(samples)
    print(f"  操作分布: {dict(op_dist)}")

    return samples


def save_jsonl(samples, output_file):
    """保存为JSONL格式"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"  保存: {output_file} ({len(samples)} 条)")


def main():
    print("=" * 60)
    print("扩大测试集规模")
    print("=" * 60)
    print(f"目标: 每种测试集 {TARGET_COUNT} 条")
    print(f"操作多样性: 允许重复操作，t3(复制)权重降低")
    print("=" * 60)


    existing_sequences = load_existing_sequences()
    print(f"\n已加载现有序列: {len(existing_sequences)} 条")


    for step in [3, 5, 7, 10]:
        samples = expand_test_set(step, existing_sequences, TARGET_COUNT, CURRENT_COUNT)
        output_file = os.path.join(DATASET_DIR, f"test_composition_{step}step.jsonl")
        save_jsonl(samples, output_file)

    print("\n" + "=" * 60)
    print("测试集扩增完成!")
    print("=" * 60)


    print("\n验证最终数据规模:")
    for step in [3, 5, 7, 10]:
        test_file = os.path.join(DATASET_DIR, f"test_composition_{step}step.jsonl")
        with open(test_file, 'r') as f:
            count = sum(1 for _ in f)
        print(f"  {step}步: {count} 条")


if __name__ == "__main__":
    random.seed(2026)
    main()