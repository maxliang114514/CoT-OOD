#!/usr/bin/env python3
"""
生成9-step测试集 - 用于scaling实验
"""

import json
import random
import os
import sys
from collections import Counter


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

sys.path.insert(0, SRC_DIR)
from data_generator import (
    generate_unique_sequences,
    apply_task_chain,
    sequence_to_string,
    TASK_NAMES
)


DATASET_DIR = os.path.join(PROJECT_ROOT, 'datasets')
TARGET_COUNT = 500
MAX_SEQ_LENGTH = 50

ATOMIC_TASKS_LIST = ['t1', 't2', 't3', 't4', 't5']

OP_WEIGHTS = {
    't1': 1.0, 't2': 1.0, 't3': 0.5, 't4': 1.0, 't5': 1.0
}


def load_existing_sequences() -> set:
    """加载所有现有数据集的输入序列"""
    sequences = set()


    for f in ["train_1step.jsonl", "train_3step_all_ops.jsonl"]:
        train_file = os.path.join("/data1/lnj/CoT-3/aton_task/dataset", f)
        if os.path.exists(train_file):
            with open(train_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    seq = tuple(data['input'].split())
                    sequences.add(seq)


    for step in [3, 5, 7, 10]:
        test_file = os.path.join(DATASET_DIR, f"test_{step}step.jsonl")
        if os.path.exists(test_file):
            with open(test_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    seq = tuple(data['input'].split())
                    sequences.add(seq)


    for name in ["train_3step.jsonl", "train_mixed_3_5step.jsonl", "train_9step.jsonl"]:
        train_file = os.path.join(DATASET_DIR, name)
        if os.path.exists(train_file):
            with open(train_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    seq = tuple(data['input'].split())
                    sequences.add(seq)

    return sequences


def check_sequence_lengths(reasoning_steps, final_output) -> bool:
    """检查所有中间步骤和最终输出的序列长度是否都<=50"""
    for step in reasoning_steps:
        if len(step['output']) > MAX_SEQ_LENGTH:
            return False
    if len(final_output) > MAX_SEQ_LENGTH:
        return False
    return True


def generate_sample(composition, existing_sequences):
    """生成单条样本"""
    seq = generate_unique_sequences(
        count=1, min_len=2, max_len=6,
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
    }, reasoning_steps, final_output


def generate_random_composition(num_steps):
    """生成随机操作组合"""
    ops = []
    for _ in range(num_steps):
        op = random.choice(ATOMIC_TASKS_LIST)
        ops.append(op)
    return ops


def generate_9step_test_set(existing_sequences, target_count):
    """生成9步测试集"""
    print(f"生成9步测试集: 目标 {target_count} 条")

    samples = []
    rejected = 0
    total_generated = 0

    while len(samples) < target_count:
        composition = generate_random_composition(9)


        if composition.count('t3') > 4:
            continue

        sample, reasoning_steps, final_output = generate_sample(composition, existing_sequences)
        input_seq = tuple(sample["input"].split())
        total_generated += 1


        if not check_sequence_lengths(reasoning_steps, final_output):
            rejected += 1
            continue

        if input_seq not in existing_sequences:
            samples.append(sample)
            existing_sequences.add(input_seq)

            if len(samples) % 100 == 0:
                print(f"  已生成 {len(samples)} / {target_count} (已拒绝 {rejected}, 总生成 {total_generated})")

    print(f"  完成: {len(samples)} 条 (总生成 {total_generated}, 拒绝 {rejected})")
    return samples


def main():
    print("=" * 60)
    print("生成9-step测试集")
    print("=" * 60)

    existing_sequences = load_existing_sequences()
    print(f"已加载现有序列: {len(existing_sequences)} 条")

    samples = generate_9step_test_set(existing_sequences, TARGET_COUNT)

    output_file = os.path.join(DATASET_DIR, "test_9step.jsonl")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"保存: {output_file} ({len(samples)} 条)")

    print("\n" + "=" * 60)
    print("9-step测试集生成完成!")
    print("=" * 60)


if __name__ == "__main__":
    random.seed(2026)
    main()