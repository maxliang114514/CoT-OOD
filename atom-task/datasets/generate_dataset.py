#!/usr/bin/env python3
"""
数据集生成脚本
生成以下数据集：
1. train_1step.jsonl - 单步数据（5种操作各2000条）
2. train_3step_all_ops.jsonl - 3步数据（使用全部5种操作）
3. train_3step_2ops.jsonl - 3步数据（固定使用 t1+t2）
4. train_mixed.jsonl - 混合数据（单步:3步 = 3:1）
"""

import json
import random
import os
import sys
from itertools import permutations, product


sys.path.insert(0, '/data1/lnj/CoT-3/aton_task')
from data_generator import (
    generate_unique_sequences,
    apply_task_chain,
    sequence_to_string,
    TASK_NAMES,
    ATOMIC_TASKS
)


OUTPUT_DIR = "/data1/lnj/CoT-3/aton_task/dataset"


ATOMIC_TASKS_LIST = ['t1', 't2', 't3', 't4', 't5']


ALL_3STEP = list(product(ATOMIC_TASKS_LIST, repeat=3))
print(f"全部3步组合 (5³): {len(ALL_3STEP)} 种")


TWO_OPS = ['t1', 't2']
ALL_3STEP_2OPS = list(product(TWO_OPS, repeat=3))
print(f"固定2种操作 (t1+t2) 的3步组合: {len(ALL_3STEP_2OPS)} 种")


def load_test_sequences() -> set:
    """加载测试数据集的输入序列"""
    sequences = set()
    test_files = [
        "/data1/lnj/CoT-3/aton_task/test_composition_3step.jsonl",
    ]
    for test_file in test_files:
        if os.path.exists(test_file):
            with open(test_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    seq = tuple(data['input'].split())
                    sequences.add(seq)
    return sequences


def generate_sample(composition: list, existing_sequences: set) -> dict:
    """生成单条样本"""
    test_seq = generate_unique_sequences(
        count=1,
        min_len=2,
        max_len=6,
        existing_sequences=existing_sequences
    )[0]

    input_str = sequence_to_string(test_seq)
    reasoning_steps, final_output = apply_task_chain(test_seq, composition)

    task_chain_str = " → ".join([TASK_NAMES[t] for t in composition])
    task_chain_ids = " → ".join(composition)

    return {
        "input": input_str,
        "output": final_output,
        "task_chain": task_chain_ids,
        "task_chain_readable": task_chain_str,
        "steps": len(composition),
        "reasoning_steps": [
            {
                "step": i + 1,
                "task": step['task'],
                "input": step['input'],
                "output": step['output']
            }
            for i, step in enumerate(reasoning_steps)
        ]
    }


def generate_1step_data(num_per_op: int = 2000) -> list:
    """生成单步数据"""
    print(f"\n生成单步数据（每种操作 {num_per_op} 条）")

    existing_sequences = load_test_sequences()
    samples = []

    for op in ATOMIC_TASKS_LIST:
        print(f"  生成 {TASK_NAMES[op]}...", end=" ", flush=True)
        op_samples = []
        op_sequences = set()

        while len(op_samples) < num_per_op:
            sample = generate_sample([op], existing_sequences | op_sequences)
            input_seq = tuple(sample["input"].split())

            if input_seq not in op_sequences:
                op_samples.append(sample)
                op_sequences.add(input_seq)
                existing_sequences.add(input_seq)

        samples.extend(op_samples)
        print(f"{len(op_samples)} 条")

    random.shuffle(samples)
    return samples


def generate_3step_all_ops(num_samples: int) -> list:
    """生成3步数据（使用全部5种操作）"""
    print(f"\n生成3步数据（全部5种操作）: {num_samples} 条")

    existing_sequences = load_test_sequences()
    samples = []

    while len(samples) < num_samples:
        composition = random.choice(ALL_3STEP)
        sample = generate_sample(list(composition), existing_sequences)

        input_seq = tuple(sample["input"].split())
        if input_seq not in existing_sequences:
            samples.append(sample)
            existing_sequences.add(input_seq)

        if (len(samples) + 1) % 2000 == 0:
            print(f"  已生成 {len(samples)} / {num_samples}")

    random.shuffle(samples)
    return samples


def generate_3step_2ops(num_samples: int) -> list:
    """生成3步数据（固定使用 t1+t2 两种操作）"""
    print(f"\n生成3步数据（固定使用 t1+t2）: {num_samples} 条")

    existing_sequences = load_test_sequences()
    samples = []

    while len(samples) < num_samples:
        composition = random.choice(ALL_3STEP_2OPS)
        sample = generate_sample(list(composition), existing_sequences)

        input_seq = tuple(sample["input"].split())
        if input_seq not in existing_sequences:
            samples.append(sample)
            existing_sequences.add(input_seq)

        if (len(samples) + 1) % 2000 == 0:
            print(f"  已生成 {len(samples)} / {num_samples}")

    random.shuffle(samples)
    return samples


def generate_mixed_data(num_samples: int, ratio: float = 0.75) -> list:
    """生成混合数据（单步:3步 = 3:1）"""
    print(f"\n生成混合数据: {num_samples} 条（单步:{ratio:.0%}, 3步:{1-ratio:.0%}）")

    num_1step = int(num_samples * ratio)
    num_3step = num_samples - num_1step

    existing_sequences = load_test_sequences()
    samples = []


    print(f"  生成单步数据: {num_1step} 条")
    for op in ATOMIC_TASKS_LIST:
        num_per_op = num_1step // 5
        op_samples = []
        op_sequences = set()

        while len(op_samples) < num_per_op:
            sample = generate_sample([op], existing_sequences | op_sequences)
            input_seq = tuple(sample["input"].split())
            if input_seq not in op_sequences:
                op_samples.append(sample)
                op_sequences.add(input_seq)
                existing_sequences.add(input_seq)

        samples.extend(op_samples)


    print(f"  生成3步数据（t1+t2）: {num_3step} 条")
    while len(samples) < num_samples:
        composition = random.choice(ALL_3STEP_2OPS)
        sample = generate_sample(list(composition), existing_sequences)

        input_seq = tuple(sample["input"].split())
        if input_seq not in existing_sequences:
            samples.append(sample)
            existing_sequences.add(input_seq)

        if (len(samples) - num_1step + 1) % 1000 == 0:
            print(f"  已生成 {len(samples) - num_1step} / {num_3step}")

    random.shuffle(samples)
    return samples[:num_samples]


def save_jsonl(samples: list, output_file: str):
    """保存为 JSONL 格式"""
    output_path = os.path.join(OUTPUT_DIR, output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"  保存: {output_path} ({len(samples)} 条)")



if __name__ == "__main__":
    print("="*60)
    print("生成 Atom Task 数据集")
    print("="*60)


    samples_1step = generate_1step_data(2000)
    save_jsonl(samples_1step, "train_1step.jsonl")


    samples_3step_all = generate_3step_all_ops(10000)
    save_jsonl(samples_3step_all, "train_3step_all_ops.jsonl")


    samples_3step_2ops = generate_3step_2ops(10000)
    save_jsonl(samples_3step_2ops, "train_3step_2ops.jsonl")


    samples_mixed = generate_mixed_data(10000, ratio=0.75)
    save_jsonl(samples_mixed, "train_mixed.jsonl")

    print("\n" + "="*60)
    print("数据集生成完成!")
    print("="*60)


    print("\n验证数据集:")
    for fname in ["train_1step.jsonl", "train_3step_all_ops.jsonl", "train_3step_2ops.jsonl", "train_mixed.jsonl"]:
        with open(os.path.join(OUTPUT_DIR, fname)) as f:
            data = [json.loads(l) for l in f]
        print(f"  {fname}: {len(data)} 条")
