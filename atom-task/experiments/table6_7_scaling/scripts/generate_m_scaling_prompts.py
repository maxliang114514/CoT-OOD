#!/usr/bin/env python3
"""
生成 m-scaling 实验所需的不同 n_shot 的 prompt 模板

0-shot: 无 example
2-shot: 2 个 examples
3-shot: 3 个 examples
"""

import json
import random
import argparse
import os
from pathlib import Path


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def convert_to_cot_format(sample):
    """将sample转换为CoT格式"""
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
        current_tasks = task_chain[i:]
        remaining_tasks = task_chain[i + 1:]

        current_display = '[' + ', '.join(current_tasks) + ']'
        remaining_display = '[' + ', '.join(remaining_tasks) + ']'

        step_text = f"Step {i+1}: Task_Chain: {current_display} | Input_Sequence: [{step_input}] | → Output_Sequence: [{step_output}] | Task_Chain: {remaining_display}"
        steps.append(step_text)

    cot_output = '\n'.join(steps) + f"\nFinal Answer: [{sample.get('output', '')}]"
    return cot_output


BASE_TEMPLATE = """You are given a sequence of operations to perform on a symbol list.
Follow the format of the example below step by step.

Definitions of atomic operations:
1. Reverse: Reverse the sequence order.
2. Substitute: Apply pairwise swaps {{α↔β, γ↔δ, ε↔ω}}.
3. Duplicate: Copy each symbol consecutively (e.g., αβγ → ααββγγ).
4. Deduplicate: Remove consecutive duplicate elements.
5. Rotate: Move the first element to the end.

{definitions}
Now solve this problem:
Input Sequence: [{{input_sequence}}]
Task Chain: [{{task_chain}}]"""


def build_0shot_prompt():
    """0-shot: 无 example"""
    definitions = """Your task is to predict the final output sequence after applying all operations step by step."""
    return BASE_TEMPLATE.format(definitions=definitions)


def build_nshot_prompt(examples, n_shot):
    """n-shot: 包含 n 个 examples"""
    definitions = ""
    for i, ex in enumerate(examples[:n_shot]):
        ex_input = ex.get('input', '')
        ex_chain = ex.get('task_chain_readable', ex.get('task_chain', ''))
        ex_cot = convert_to_cot_format(ex)
        definitions += f"""Example {i+1}:
Input Sequence: [{ex_input}]
Task Chain: [{ex_chain}]
{ex_cot}

"""
    return BASE_TEMPLATE.format(definitions=definitions)


def select_examples(data, n_shot, seed=42):
    """从数据集中选择 n 个有代表性的 examples，覆盖不同操作类型"""
    random.seed(seed)


    by_chain = {}
    for sample in data:
        chain = sample.get('task_chain_readable', '')
        if chain not in by_chain:
            by_chain[chain] = []
        by_chain[chain].append(sample)


    by_steps = {}
    for sample in data:
        steps = sample.get('steps', 3)
        if steps != 3:
            continue
        chain = sample.get('task_chain_readable', '')

        ops = set()
        for op in ['Reverse', 'Substitute', 'Duplicate', 'Deduplicate', 'Rotate']:
            if op in chain:
                ops.add(op)

        key = tuple(sorted(ops))
        if key not in by_steps:
            by_steps[key] = []
        by_steps[key].append(sample)


    selected = []
    keys = list(by_steps.keys())


    random.shuffle(keys)
    for key in keys:
        if len(selected) >= n_shot:
            break
        samples = by_steps[key]
        if samples:
            selected.append(random.choice(samples))


    if len(selected) < n_shot:
        all_3step = [s for s in data if s.get('steps', 3) == 3]
        remaining = [s for s in all_3step if s not in selected]
        while len(selected) < n_shot and remaining:
            s = random.choice(remaining)
            selected.append(s)
            remaining.remove(s)

    return selected[:n_shot]


def main():
    parser = argparse.ArgumentParser(description="生成 m-scaling prompt 模板")
    parser.add_argument('--dataset_file', type=str,
                        default=os.path.join(PROJECT_ROOT, 'datasets', 'train_3step.jsonl'),
                        help="用于提取 examples 的数据集")
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(PROJECT_ROOT, 'experiments', 'nm_scaling', 'prompts'),
                        help="输出目录")
    parser.add_argument('--n_shot', type=int, nargs='+', default=[0, 2, 3],
                        help="Few-shot 数量")
    parser.add_argument('--seed', type=int, default=42, help="随机种子")
    args = parser.parse_args()

    print(f"加载数据集: {args.dataset_file}")
    data = load_jsonl(args.dataset_file)
    print(f"  样本数: {len(data)}")


    examples = select_examples(data, max(args.n_shot), args.seed)
    print(f"  选择的 examples 数量: {len(examples)}")

    output_dir = Path(args.output_dir)

    for n_shot in args.n_shot:
        if n_shot == 0:
            prompt = build_0shot_prompt()
        else:
            prompt = build_nshot_prompt(examples, n_shot)

        output_file = output_dir / f"prompt_nshot_{n_shot}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(prompt)
        print(f"\n生成 {n_shot}-shot prompt:")
        print(f"  文件: {output_file}")
        print(f"  长度: {len(prompt)} 字符")
        print(f"  前200字符:\n{prompt[:200]}...")

    print("\n完成!")


if __name__ == "__main__":
    main()