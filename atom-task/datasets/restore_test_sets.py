#!/usr/bin/env python3
"""
重新扩增测试集，保持与原始250样本相同的操作分布
"""

import json
import random
import os
import sys
from collections import Counter
from itertools import product

sys.path.insert(0, '/data1/lnj/CoT-3/aton_task')
from data_generator import (
    generate_unique_sequences,
    apply_task_chain,
    sequence_to_string,
    TASK_NAMES
)

DATASET_DIR = "/data1/lnj/CoT-3/aton_task/dataset"
TARGET_COUNT = 500

ATOMIC_TASKS_LIST = ['t1', 't2', 't3', 't4', 't5']


def load_samples(filepath):
    """加载JSONL文件"""
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def analyze_distribution(samples):
    """分析样本的操作分布"""
    op_counts = Counter()
    chain_compositions = []

    for s in samples:
        chain = s['task_chain']
        ops = chain.split(' → ')
        op_counts.update(ops)
        chain_compositions.append(tuple(ops))

    return op_counts, chain_compositions


def generate_matching_sample(num_steps, target_dist, existing_sequences):
    """
    根据目标分布生成匹配的样本
    target_dist: 操作的目标分布概率
    """
    max_attempts = 200

    for _ in range(max_attempts):

        ops = tuple(random.choices(ATOMIC_TASKS_LIST, weights=target_dist, k=num_steps))


        seq = generate_unique_sequences(1, 2, 6, existing_sequences)
        if not seq:
            continue
        seq = seq[0]

        input_str = sequence_to_string(seq)
        reasoning_steps, final_output = apply_task_chain(seq, list(ops))

        task_chain_str = " -> ".join([TASK_NAMES[t] for t in ops])
        task_chain_ids = " -> ".join(ops)

        sample = {
            "input": input_str,
            "output": final_output,
            "task_chain": task_chain_ids,
            "task_chain_readable": task_chain_str,
            "steps": len(ops),
            "reasoning_steps": reasoning_steps
        }

        input_seq = tuple(sample["input"].split())
        if input_seq not in existing_sequences:
            return sample, ops

    return None, None


def expand_test_set(num_steps, target_count=500):
    """扩增指定步数的测试集"""
    print(f"\n扩增 {num_steps}步测试集:")

    test_file = os.path.join(DATASET_DIR, f"test_composition_{num_steps}step.jsonl")
    samples = load_samples(test_file)
    print(f"  原始: {len(samples)} 条")


    target_dist, existing_chains = analyze_distribution(samples)
    total_ops = sum(target_dist.values())


    target_prob = {op: count/total_ops for op, count in target_dist.items()}
    weights = [target_prob[op] for op in ATOMIC_TASKS_LIST]

    print(f"  目标操作分布: {target_dist}")
    print(f"  采样权重: {[f'{w:.3f}' for w in weights]}")


    existing_sequences = set()
    existing_chains = []
    for s in samples:
        existing_sequences.add(tuple(s['input'].split()))
        existing_chains.append(tuple(s['task_chain'].split(' → ')))


    new_count = 0
    attempts = 0
    while len(samples) < target_count and attempts < 10000:
        attempts += 1

        sample, ops = generate_matching_sample(
            num_steps,
            weights,
            existing_sequences
        )

        if sample:
            samples.append(sample)
            existing_sequences.add(tuple(sample["input"].split()))
            existing_chains.append(ops)
            new_count += 1

            if new_count % 50 == 0:
                print(f"  已生成 {new_count} 条新样本")

    print(f"  完成: {len(samples)} 条 (新增 {new_count} 条)")


    final_dist, _ = analyze_distribution(samples)
    print(f"  最终操作分布: {final_dist}")


    with open(test_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"  已保存: {test_file}")

    return len(samples)


def main():
    print("=" * 60)
    print("重新扩增测试集 (保持原始分布)")
    print("=" * 60)


    random.seed(2024)


    for step in [3, 5, 7, 10]:
        expand_test_set(step, TARGET_COUNT)

    print("\n" + "=" * 60)
    print("测试集扩增完成!")
    print("=" * 60)


    print("\n验证:")
    for step in [3, 5, 7, 10]:
        test_file = os.path.join(DATASET_DIR, f"test_composition_{step}step.jsonl")
        with open(test_file, 'r') as f:
            count = sum(1 for _ in f)
        print(f"  {step}步: {count} 条")


if __name__ == "__main__":
    main()