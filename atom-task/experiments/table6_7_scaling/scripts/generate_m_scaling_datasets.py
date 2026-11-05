#!/usr/bin/env python3
"""
生成 m-scaling 实验所需的 few-shot 训练数据集
为每个 m 值 (0-shot, 2-shot, 3-shot) 创建独立的训练数据集
"""

import json
import random
import argparse
from pathlib import Path


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


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


def build_example_prompt(sample, use_thinking=False):
    """为一个样本构建prompt (不包含Final Answer)"""
    task_chain_str = sample.get('task_chain_readable', sample.get('task_chain', ''))
    input_text = sample.get('input', '')

    prompt = f"""You are given a sequence of operations to perform on a symbol list.
Follow the format of the example below step by step.

Definitions of atomic operations:
1. Reverse: Reverse the sequence order.
2. Substitute: Apply pairwise swaps {{α↔β, γ↔δ, ε↔ω}}.
3. Duplicate: Copy each symbol consecutively (e.g., αβγ → ααββγγ).
4. Deduplicate: Remove consecutive duplicate elements.
5. Rotate: Move the first element to the end.

Example:
Input Sequence: [{sample.get('input', '')}]
Task Chain: [{task_chain_str}]
"""
    return prompt


def build_full_prompt_with_examples(prompt, cot_answer):
    """将example的prompt和answer组合"""
    return f"{prompt}Now solve this problem:\nInput Sequence: [{{input_sequence}}]\nTask Chain: [{{task_chain}}]\n{cot_answer}"


def build_few_shot_sample(sample, demonstrations, use_thinking=False):
    """为当前样本构建few-shot prompt，包含 demonstrations

    格式说明：当 instruction 非空且 input 为空时，format_instruction_qwen 会直接使用 instruction
    """
    task_chain_str = sample.get('task_chain_readable', sample.get('task_chain', ''))
    input_text = sample.get('input', '')
    output = sample.get('output', '')


    current_prompt = f"""You are given a sequence of operations to perform on a symbol list.
Follow the format of the example below step by step.

Definitions of atomic operations:
1. Reverse: Reverse the sequence order.
2. Substitute: Apply pairwise swaps {{α↔β, γ↔δ, ε↔ω}}.
3. Duplicate: Copy each symbol consecutively (e.g., αβγ → ααββγγ).
4. Deduplicate: Remove consecutive duplicate elements.
5. Rotate: Move the first element to the end.

"""

    for demo in demonstrations:
        demo_cot = convert_to_cot_format(demo)
        demo_input = demo.get('input', '')
        demo_chain = demo.get('task_chain_readable', demo.get('task_chain', ''))
        current_prompt += f"""Example:
Input Sequence: [{demo_input}]
Task Chain: [{demo_chain}]
{demo_cot}

"""


    current_prompt += f"""Now solve this problem:
Input Sequence: [{input_text}]
Task Chain: [{task_chain_str}]
"""


    cot_answer = convert_to_cot_format(sample)


    return {
        'input': '',
        'output': cot_answer,
        'task_chain': task_chain_str,
        'task_chain_readable': task_chain_str,
        'steps': sample.get('steps', len(demonstrations) + 1) if isinstance(sample.get('steps'), int) else 3,
        'reasoning_steps': sample.get('reasoning_steps', []),
        'instruction': current_prompt
    }


def generate_m_scaling_dataset(base_data, n_shot, seed=42):
    """从基础数据生成 m-scaling 数据集

    0-shot: instruction 为空，依赖训练脚本使用默认模板
    few-shot: instruction 包含完整 prompt，input 为空字符串
    """
    random.seed(seed)


    if n_shot == 0:
        result = []
        for sample in base_data:
            new_sample = sample.copy()

            new_sample['instruction'] = ''
            result.append(new_sample)
        return result


    result = []
    indices = list(range(len(base_data)))

    for i, sample in enumerate(base_data):

        available_indices = [j for j in indices if j != i]
        demo_indices = random.sample(available_indices, min(n_shot, len(available_indices)))
        demonstrations = [base_data[j] for j in demo_indices]


        few_shot_sample = build_few_shot_sample(sample, demonstrations)
        result.append(few_shot_sample)

    return result


def main():
    parser = argparse.ArgumentParser(description="生成 m-scaling 训练数据集")
    parser.add_argument('--input_file', type=str, required=True, help="基础训练数据文件")
    parser.add_argument('--output_dir', type=str, required=True, help="输出目录")
    parser.add_argument('--n_shot', type=int, nargs='+', default=[0, 2, 3], help="Few-shot 数量")
    parser.add_argument('--seed', type=int, default=42, help="随机种子")
    args = parser.parse_args()

    print(f"加载基础数据: {args.input_file}")
    base_data = load_jsonl(args.input_file)
    print(f"  总样本数: {len(base_data)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for n_shot in args.n_shot:
        print(f"\n生成 {n_shot}-shot 数据集...")
        dataset = generate_m_scaling_dataset(base_data, n_shot, args.seed)

        output_file = output_dir / f"m_scaling_{n_shot}shot.jsonl"
        save_jsonl(dataset, output_file)
        print(f"  保存到: {output_file}")
        print(f"  样本数: {len(dataset)}")


        if dataset:
            print(f"  第一条样本 instruction 长度: {len(dataset[0].get('instruction', ''))}")

    print("\n完成!")


if __name__ == "__main__":
    main()