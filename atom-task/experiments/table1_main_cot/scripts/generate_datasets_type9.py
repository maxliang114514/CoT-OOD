#!/usr/bin/env python3
"""
生成9-step数据集 - 用于scaling实验
过滤条件：中间步骤或最终步骤中序列长度大于50的样本予以筛去

计算:
- 单步样本约2-6个符号，经过9步操作后，序列长度可能膨胀
- 为保证序列长度可控，筛选掉中间/最终长度>50的样本
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


OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'experiments', 't_scaling', 'datasets')
DATASET_DIR = os.path.join(PROJECT_ROOT, 'datasets')



TARGET_TYPE9_SAMPLES = 556
MAX_SEQ_LENGTH = 50

ATOMIC_TASKS_LIST = ['t1', 't2', 't3', 't4', 't5']


def load_existing_sequences() -> set:
    """加载所有现有数据集的输入序列"""
    sequences = set()


    train_files = [
        os.path.join(DATASET_DIR, "train_1step.jsonl"),
        os.path.join(DATASET_DIR, "train_3step_all_ops.jsonl"),
    ]
    for train_file in train_files:
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


    type7_file = os.path.join(DATASET_DIR, "train_3step.jsonl")
    type8_file = os.path.join(DATASET_DIR, "train_mixed_3_5step.jsonl")
    for f in [type7_file, type8_file]:
        if os.path.exists(f):
            with open(f, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    seq = tuple(data['input'].split())
                    sequences.add(seq)

    return sequences


def check_sequence_lengths(reasoning_steps: list, final_output: str) -> bool:
    """检查所有中间步骤和最终输出的序列长度是否都<=50"""

    for step in reasoning_steps:
        step_output = step['output']
        if len(step_output) > MAX_SEQ_LENGTH:
            return False


    if len(final_output) > MAX_SEQ_LENGTH:
        return False

    return True


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
    }, reasoning_steps, final_output


def generate_random_composition(num_steps):
    """生成随机操作组合"""
    return [random.choice(ATOMIC_TASKS_LIST) for _ in range(num_steps)]


def analyze_distribution(samples):
    """分析样本的操作分布"""
    op_counts = Counter()
    for s in samples:
        ops = s['task_chain'].split(' → ')
        for op in ops:
            op_counts[op] += 1
    return op_counts


def save_jsonl(samples, output_file):
    """保存为JSONL格式"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"  保存: {os.path.basename(output_file)} ({len(samples)} 条)")


def generate_type9(existing_sequences):
    """生成type9: 9步数据集 (过滤序列长度>50的样本)"""
    print(f"\n生成 type9 (9步数据集):")
    print(f"  目标: {TARGET_TYPE9_SAMPLES} 样本 (过滤序列长度>{MAX_SEQ_LENGTH})")

    samples = []
    step_count = 0
    rejected = 0
    total_generated = 0

    while len(samples) < TARGET_TYPE9_SAMPLES:
        composition = generate_random_composition(9)
        sample, reasoning_steps, final_output = generate_sample(composition, existing_sequences)
        input_seq = tuple(sample["input"].split())
        total_generated += 1


        if not check_sequence_lengths(reasoning_steps, final_output):
            rejected += 1
            if total_generated % 100 == 0:
                print(f"  已生成 {len(samples)} / {TARGET_TYPE9_SAMPLES} (已拒绝 {rejected} 条, 总生成 {total_generated})")
            continue

        if input_seq not in existing_sequences:
            samples.append(sample)
            existing_sequences.add(input_seq)
            step_count += 9

            if len(samples) % 100 == 0:
                print(f"  已生成 {len(samples)} / {TARGET_TYPE9_SAMPLES} (已拒绝 {rejected} 条, 总生成 {total_generated})")

    print(f"  完成: {len(samples)} 样本, {step_count} 总步数")
    print(f"  总生成: {total_generated}, 拒绝: {rejected} ({rejected/total_generated*100:.1f}%)")


    op_dist = analyze_distribution(samples)
    print(f"  操作分布: {dict(op_dist)}")

    return samples, existing_sequences


def main():
    print("=" * 60)
    print("生成 type9 (9步数据集) - 用于scaling实验")
    print("=" * 60)
    print(f"目标规模:")
    print(f"  type9 (9步): {TARGET_TYPE9_SAMPLES} 样本 → {TARGET_TYPE9_SAMPLES * 9} 步")
    print(f"  过滤条件: 中间/最终步骤序列长度 <= {MAX_SEQ_LENGTH}")
    print("=" * 60)


    existing_sequences = load_existing_sequences()
    print(f"\n已加载现有序列: {len(existing_sequences)} 条")


    os.makedirs(OUTPUT_DIR, exist_ok=True)


    samples_type9, existing_sequences = generate_type9(existing_sequences)
    save_jsonl(samples_type9, os.path.join(OUTPUT_DIR, "train_9step.jsonl"))

    print("\n" + "=" * 60)
    print("数据集生成完成!")
    print("=" * 60)
    print(f"type9: {len(samples_type9)} 样本, {len(samples_type9) * 9} 步")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    random.seed(2026)
    main()