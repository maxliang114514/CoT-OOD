#!/usr/bin/env python3
"""
噪声注入脚本 - 为CoT推理步骤注入噪声

噪声策略:
1. 选择推理步骤中4个元素( Input_Sequence, Output_Sequence, Current_Task_Chain, Updated_Task_Chain )的任意2个
2. 对选中的元素执行 Tampering(替换为随机符号) 或 Deletion(置空)

噪声比例 p ∈ {0.05, 0.10, 0.20}
"""

import json
import random
import re
import argparse
from typing import List, Tuple, Dict, Set
from pathlib import Path



SYMBOLS = ['α', 'β', 'γ', 'δ', 'ε', 'ω', 'ζ', 'η', 'θ']


def load_jsonl(file_path: str) -> List[dict]:
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data: List[dict], file_path: str):
    """保存JSONL文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def generate_random_symbols(n: int) -> str:
    """生成n个随机符号"""
    return ''.join(random.choices(SYMBOLS, k=n))


def inject_noise_to_step(step_text: str, noise_type: str = 'both') -> Tuple[str, List[str]]:
    """
    对单个推理步骤注入噪声

    Args:
        step_text: 原始步骤文本，格式如:
            Step 1: Task_Chain: [Duplicate, Reverse] | Input_Sequence: [αβγ] | → Output_Sequence: [γβα] | Task_Chain: [Reverse]
        noise_type: 'tamper', 'delete', 或 'both'

    Returns:
        (修改后的步骤文本, 被修改的元素列表)
    """




    pattern = r'(Task_Chain:\s*\[([^\]]*)\])|(Input_Sequence:\s*\[([^\]]*)\])|(→\s*Output_Sequence:\s*\[([^\]]*)\])|(Task_Chain:\s*\[([^\]]*)\]\s*$)'

    elements = {
        'current_task_chain': None,
        'input_sequence': None,      # Input_Sequence
        'output_sequence': None,     # → Output_Sequence
        'updated_task_chain': None,
    }


    # Step 1: Task_Chain: [Duplicate, Reverse] | Input_Sequence: [αβγ] | → Output_Sequence: [γβα] | Task_Chain: [Reverse]


    task_chain_matches = list(re.finditer(r'Task_Chain:\s*\[([^\]]*)\]', step_text))
    input_match = re.search(r'Input_Sequence:\s*\[([^\]]*)\]', step_text)
    output_match = re.search(r'→\s*Output_Sequence:\s*\[([^\]]*)\]', step_text)

    if len(task_chain_matches) >= 2:
        elements['current_task_chain'] = (task_chain_matches[0].start(), task_chain_matches[0].group(0), task_chain_matches[0].group(1))
        elements['updated_task_chain'] = (task_chain_matches[1].start(), task_chain_matches[1].group(0), task_chain_matches[1].group(1))
    if input_match:
        elements['input_sequence'] = (input_match.start(), input_match.group(0), input_match.group(1))
    if output_match:
        elements['output_sequence'] = (output_match.start(), output_match.group(0), output_match.group(1))


    available_elements = [k for k, v in elements.items() if v is not None]

    if len(available_elements) < 2:
        return step_text, []

    selected = random.sample(available_elements, 2)
    modified_elements = []

    result = step_text

    for elem_name in selected:
        start_pos, full_match, content = elements[elem_name]


        if noise_type == 'both':
            action = random.choice(['tamper', 'delete'])
        else:
            action = noise_type

        if action == 'tamper':

            if content:
                new_content = generate_random_symbols(len(content))
            else:
                new_content = generate_random_symbols(3)
            modified_elements.append(f"{elem_name}:{content}→{new_content}(tamper)")
        else:  # delete

            if elem_name == 'input_sequence':
                new_content = '***'
            elif elem_name == 'output_sequence':
                new_content = '***'
            elif elem_name == 'current_task_chain':
                new_content = '***'
            elif elem_name == 'updated_task_chain':
                new_content = '***'
            else:
                new_content = '***'
            modified_elements.append(f"{elem_name}:{content}→(delete)")


        old_str = full_match
        if elem_name == 'current_task_chain' or elem_name == 'updated_task_chain':
            new_str = f"Task_Chain: [{new_content}]"
        elif elem_name == 'input_sequence':
            new_str = f"Input_Sequence: [{new_content}]"
        elif elem_name == 'output_sequence':
            new_str = f"→ Output_Sequence: [{new_content}]"

        result = result.replace(old_str, new_str, 1)

    return result, modified_elements


def inject_noise_to_cot_output(cot_output: str, noise_ratio: float, noise_type: str = 'both') -> Tuple[str, int]:
    """
    对CoT输出注入噪声

    Args:
        cot_output: CoT格式的输出，格式如:
            Step 1: Task_Chain: [Duplicate, Reverse] | Input_Sequence: [αβγ] | → Output_Sequence: [γβα] | Task_Chain: [Reverse]
            Step 2: Task_Chain: [Reverse] | Input_Sequence: [γβα] | → Output_Sequence: [αβγ] | Task_Chain: []
            Final Answer: [αβγ]
        noise_ratio: 噪声比例 p
        noise_type: 'tamper', 'delete', 或 'both'

    Returns:
        (注入噪声后的CoT输出, 被噪声污染的步骤数)
    """
    lines = cot_output.strip().split('\n')


    step_lines = []
    final_answer = ''
    for line in lines:
        if line.startswith('Step '):
            step_lines.append(line)
        elif line.startswith('Final Answer:'):
            final_answer = line

    if not step_lines:
        return cot_output, 0


    num_steps = len(step_lines)
    num_to_corrupt = max(1, int(num_steps * noise_ratio))


    indices_to_corrupt = set(random.sample(range(num_steps), min(num_to_corrupt, num_steps)))

    corrupted_steps = 0
    new_step_lines = []

    for i, line in enumerate(step_lines):
        if i in indices_to_corrupt:
            new_line, modified = inject_noise_to_step(line, noise_type)
            if modified:
                new_step_lines.append(new_line)
                corrupted_steps += 1
        else:
            new_step_lines.append(line)


    result_lines = new_step_lines
    if final_answer:
        result_lines.append(final_answer)

    return '\n'.join(result_lines), corrupted_steps


def inject_noise_to_sample(sample: dict, noise_ratio: float, noise_type: str = 'both') -> Tuple[dict, int]:
    """
    对单个样本注入噪声

    Args:
        sample: 包含 'output' 字段的样本，output是CoT格式的字符串
        noise_ratio: 噪声比例 p
        noise_type: 'tamper', 'delete', 或 'both'

    Returns:
        (修改后的样本, 被噪声污染的步骤数)
    """
    sample = json.loads(json.dumps(sample))

    if 'output' not in sample or 'Step' not in sample['output']:
        return sample, 0

    original_output = sample['output']
    new_output, num_corrupted = inject_noise_to_cot_output(original_output, noise_ratio, noise_type)

    sample['output'] = new_output
    sample['noise_corrupted_steps'] = num_corrupted
    sample['noise_ratio'] = noise_ratio

    return sample, num_corrupted


def main():
    parser = argparse.ArgumentParser(description='为CoT数据注入噪声')
    parser.add_argument('--input', type=str, required=True, help='输入JSONL文件')
    parser.add_argument('--output', type=str, required=True, help='输出JSONL文件')
    parser.add_argument('--noise_ratio', type=float, default=0.1,
                        choices=[0.05, 0.10, 0.20], help='噪声比例 p')
    parser.add_argument('--noise_type', type=str, default='both',
                        choices=['tamper', 'delete', 'both'],
                        help='噪声类型: tamper(替换), delete(删除), both(混合)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    args = parser.parse_args()

    random.seed(args.seed)

    print(f"加载数据: {args.input}")
    data = load_jsonl(args.input)
    print(f"加载 {len(data)} 条样本")

    print(f"注入噪声: ratio={args.noise_ratio}, type={args.noise_type}")

    corrupted_count = 0
    total_corrupted_steps = 0

    for i, sample in enumerate(data):
        _, num_corrupted = inject_noise_to_sample(sample, args.noise_ratio, args.noise_type)
        if num_corrupted > 0:
            corrupted_count += 1
            total_corrupted_steps += num_corrupted

        if (i + 1) % 1000 == 0:
            print(f"处理进度: {i+1}/{len(data)}, 已污染样本: {corrupted_count}, 总污染步骤: {total_corrupted_steps}")

    print(f"完成! 污染样本数: {corrupted_count}/{len(data)}, 总污染步骤: {total_corrupted_steps}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(data, args.output)
    print(f"保存到: {args.output}")


if __name__ == '__main__':
    main()