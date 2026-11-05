#!/usr/bin/env python3
"""
生成 multi_step_only v2 数据集 - 重新确定规模
目标: 使type7和type8每个数据集的总步数都约为5000（匹配single_step的5000条单步数据）

计算:
- single_step: 5000样本 × 1步 = 5000总步数
- 目标: type7 ≈ 5000步, type8 ≈ 5000步

新规模:
- type7 (3步): 1667样本 × 3步 = 5001步
- type8 (3步+5步混合): 各50%混合
  - 3步样本: 833 × 3 = 2499步
  - 5步样本: 500 × 5 = 2500步
  - 小计: 4999步
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


OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'datasets')
DATASET_DIR = os.path.join(PROJECT_ROOT, 'datasets')


TARGET_TYPE7_SAMPLES = 1667
TARGET_TYPE8_SAMPLES = 1333
TYPE8_3STEP_RATIO = 0.625
TYPE8_5STEP_RATIO = 0.375

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


def generate_random_composition(num_steps):
    """生成随机操作组合"""
    return [random.choice(ATOMIC_TASKS_LIST) for _ in range(num_steps)]


def generate_composition_with_2_ops(num_steps):
    """生成只包含2种操作的操作链

    从5种操作中随机选择2种，然后交替排列（保证恰好2种）
    策略: 先放两种操作各一个，然后随机填充
    """

    two_ops = random.sample(ATOMIC_TASKS_LIST, 2)

    composition = [two_ops[0], two_ops[1]]
    composition.extend([random.choice(two_ops) for _ in range(num_steps - 2)])

    random.shuffle(composition)
    return composition


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


def generate_type7(existing_sequences):
    """生成type7: 3步数据集"""
    print(f"\n生成 type7 (3步数据集):")
    print(f"  目标: {TARGET_TYPE7_SAMPLES} 样本")

    samples = []
    step_count = 0

    while len(samples) < TARGET_TYPE7_SAMPLES:
        composition = generate_random_composition(3)
        sample = generate_sample(composition, existing_sequences)
        input_seq = tuple(sample["input"].split())

        if input_seq not in existing_sequences:
            samples.append(sample)
            existing_sequences.add(input_seq)
            step_count += 3

            if len(samples) % 200 == 0:
                print(f"  已生成 {len(samples)} / {TARGET_TYPE7_SAMPLES}")

    print(f"  完成: {len(samples)} 样本, {step_count} 总步数")


    op_dist = analyze_distribution(samples)
    print(f"  操作分布: {dict(op_dist)}")

    return samples, existing_sequences


def generate_type8(existing_sequences):
    """生成type8: 3步+5步混合数据集

    重要: type8的3步和5步样本都只包含2种操作类型
    (参考 train_mixed_type8.jsonl 的设计)
    """
    num_3step = int(TARGET_TYPE8_SAMPLES * TYPE8_3STEP_RATIO)
    num_5step = TARGET_TYPE8_SAMPLES - num_3step

    print(f"\n生成 type8 (3步+5步混合数据集):")
    print(f"  目标: {TARGET_TYPE8_SAMPLES} 样本 (3步: {num_3step}, 5步: {num_5step})")
    print(f"  约束: 所有样本只包含2种操作类型")

    samples = []
    step_count = 0


    while sum(1 for s in samples if s['steps'] == 3) < num_3step:
        composition = generate_composition_with_2_ops(3)
        sample = generate_sample(composition, existing_sequences)
        input_seq = tuple(sample["input"].split())

        if input_seq not in existing_sequences:
            samples.append(sample)
            existing_sequences.add(input_seq)
            step_count += 3

            if sum(1 for s in samples if s['steps'] == 3) % 100 == 0:
                print(f"  3步: {sum(1 for s in samples if s['steps'] == 3)} / {num_3step}")

    print(f"  3步完成: {num_3step} 样本")


    while sum(1 for s in samples if s['steps'] == 5) < num_5step:
        composition = generate_composition_with_2_ops(5)
        sample = generate_sample(composition, existing_sequences)
        input_seq = tuple(sample["input"].split())

        if input_seq not in existing_sequences:
            samples.append(sample)
            existing_sequences.add(input_seq)
            step_count += 5

            if sum(1 for s in samples if s['steps'] == 5) % 100 == 0:
                print(f"  5步: {sum(1 for s in samples if s['steps'] == 5)} / {num_5step}")

    print(f"  5步完成: {num_5step} 样本")
    print(f"  总计: {len(samples)} 样本, {step_count} 总步数")


    op_dist = analyze_distribution(samples)
    print(f"  操作分布: {dict(op_dist)}")


    two_op_count = 0
    for s in samples:
        ops = s['task_chain'].split(' → ')
        if len(set(ops)) == 2:
            two_op_count += 1
    print(f"  验证: {two_op_count}/{len(samples)} 样本包含恰好2种操作")

    return samples, existing_sequences


def main():
    print("=" * 60)
    print("生成 multi_step_only v2 数据集")
    print("=" * 60)
    print(f"目标规模 (每个数据集总步数约5000):")
    print(f"  type7 (3步): {TARGET_TYPE7_SAMPLES} 样本 → {TARGET_TYPE7_SAMPLES * 3} 步")
    num_3step = int(TARGET_TYPE8_SAMPLES * TYPE8_3STEP_RATIO)
    num_5step = TARGET_TYPE8_SAMPLES - num_3step
    type8_steps = num_3step * 3 + num_5step * 5
    print(f"  type8 (3+5步混合): {TARGET_TYPE8_SAMPLES} 样本 ({num_3step}×3步 + {num_5step}×5步) → {type8_steps} 步")
    print("=" * 60)


    existing_sequences = load_existing_sequences()
    print(f"\n已加载现有序列: {len(existing_sequences)} 条")


    os.makedirs(OUTPUT_DIR, exist_ok=True)


    samples_type7, existing_sequences = generate_type7(existing_sequences)
    save_jsonl(samples_type7, os.path.join(OUTPUT_DIR, "train_3step.jsonl"))


    samples_type8, existing_sequences = generate_type8(existing_sequences)
    save_jsonl(samples_type8, os.path.join(OUTPUT_DIR, "train_mixed_3_5step.jsonl"))

    print("\n" + "=" * 60)
    print("数据集生成完成!")
    print("=" * 60)
    print(f"type7: {len(samples_type7)} 样本, {len(samples_type7) * 3} 步")
    print(f"type8: {len(samples_type8)} 样本, ~{type8_steps} 步")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    random.seed(2026)
    main()