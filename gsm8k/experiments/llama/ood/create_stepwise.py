"""
单步实验数据创建脚本
将包含多个推理步骤的样本拆分为原子样本

拆分逻辑: 假设一个样本有N个推理步骤，则拆分为N个样本：
- (1) instruction=原题目, input=空, output=步骤1+2 (从0开始推理两步)
- (2) instruction=原题目, input=步骤1, output=步骤2+3 (已推理1步，继续推两步[2,3])
- (3) instruction=原题目, input=步骤1+2, output=步骤3+4 (已推理2步，继续推[3,4])
- ...
- (N) instruction=原题目, input=步骤1...N-1, output=步骤N (最后一步，推出答案)
"""

import json
import re
import os
import argparse
from tqdm import tqdm
from typing import List, Tuple, Optional


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def extract_steps_by_newline(text: str) -> List[str]:
    """
    使用换行符来识别步骤
    每行被视为一个推理步骤
    """
    if not text:
        return []


    lines = [l.strip() for l in text.split('\n') if l.strip()]


    steps = []
    for line in lines:

        if re.match(r'^#{2,}\s*\d+(\.\d+)?$', line.strip()):
            continue
        if re.match(r'^The answer is:\s*[\$\\\boxed]?[\d\.\-]+', line.strip()):
            continue
        if len(line) < 10:
            continue
        steps.append(line)

    return steps


def split_into_atomic_samples(query: str, response: str) -> List[dict]:
    """
    将包含多个步骤的样本拆分

    Args:
        query: 原问题
        response: 完整回答（含多个推理步骤）

    Returns:
        原子样本列表，每个样本预测"后两步"
    """
    steps = extract_steps_by_newline(response)

    if len(steps) < 2:

        return [{
            'instruction': query,
            'input': '',
            'output': response,
            'num_original_steps': 1,
            'step_index': 0,
            'is_final': True
        }]

    atomic_samples = []
    n_steps = len(steps)

    for i in range(n_steps):

        if i == 0:
            input_text = ""
        else:
            input_text = " ".join(steps[:i])


        if i + 1 < n_steps:

            output_text = steps[i] + " " + steps[i + 1]
            is_final = False
        else:

            output_text = steps[i]
            is_final = True

        atomic_samples.append({
            'instruction': query,
            'input': input_text,
            'output': output_text,
            'num_original_steps': n_steps,
            'step_index': i,
            'is_final': is_final
        })

    return atomic_samples


def process_dataset(input_file: str, output_file: str,
                   filter_min_steps: int = 2,
                   max_samples: int = 0,
                   max_atomic_samples: int = 0):
    """
    处理数据集，创建单步样本

    Args:
        input_file: 输入JSON文件
        output_file: 输出JSONL文件
        filter_min_steps: 最小步骤数（少于该值不拆分，默认2步起拆）
        max_samples: 最大原始样本数 (0=全部)
        max_atomic_samples: 最大原子样本输出数 (0=不截断)
    """
    print(f"加载数据: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)

    if isinstance(data, list):
        samples = data
    else:
        samples = [data]


    if max_samples > 0:
        samples = samples[:max_samples]
        print(f"裁剪后原始样本数: {len(samples)}")

    print(f"总样本数: {len(samples)}")


    stats = {
        'total_samples': len(samples),
        'single_step': 0,
        'two_step': 0,
        'multi_step': 0,
        'total_atomic': 0,
        'step_distribution': {}
    }

    atomic_data = []

    for item in tqdm(samples, desc="Processing"):
        query = item.get('query', item.get('instruction', ''))
        response = item.get('response', item.get('output', ''))

        steps = extract_steps_by_newline(response)
        n = len(steps)

        if n < filter_min_steps:

            atomic_samples = [{
                'instruction': query,
                'input': '',
                'output': response,
                'num_original_steps': n,
                'step_index': 0,
                'is_final': True
            }]
            stats['single_step'] += 1
        else:
            atomic_samples = split_into_atomic_samples(query, response)
            if n == 2:
                stats['two_step'] += 1
            else:
                stats['multi_step'] += 1

        stats['total_atomic'] += len(atomic_samples)
        stats['step_distribution'][n] = stats['step_distribution'].get(n, 0) + 1

        for sample in atomic_samples:
            sample['original_query'] = query
            sample['original_response'] = response
            atomic_data.append(sample)


    print(f"\n保存到: {output_file}")
    with open(output_file, 'w') as f:
        for item in atomic_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


    print("\n" + "=" * 60)
    print("统计信息")
    print("=" * 60)
    print(f"总原始样本: {stats['total_samples']}")
    print(f"未拆分(1-2步): {stats['single_step'] + stats['two_step']}")
    print(f"多步拆分: {stats['multi_step']}")
    print(f"总原子样本: {stats['total_atomic']}")


    if max_atomic_samples > 0:
        atomic_data = atomic_data[:max_atomic_samples]
        stats['total_atomic'] = len(atomic_data)
        print(f"截断后原子样本: {stats['total_atomic']}")
    print("\n原始步骤数分布:")
    for n, count in sorted(stats['step_distribution'].items())[:10]:
        print(f"  {n}步: {count}个样本")
    print("=" * 60)


    print("\n示例输出:")

    for item in atomic_data:
        if item['num_original_steps'] == 4 and item['step_index'] == 0:
            print(f"\n--- 4步样本的完整拆分 ---")
            for i, atom in enumerate(atomic_data[:4]):
                if atom['num_original_steps'] == 4:
                    print(f"\n样本 {i+1} (step_index={atom['step_index']}, is_final={atom['is_final']}):")
                    print(f"  instruction: {atom['instruction'][:80]}...")
                    print(f"  input: {atom['input'][:80] if atom['input'] else '(空)'}...")
                    print(f"  output: {atom['output'][:80]}...")
            break


def main():
    parser = argparse.ArgumentParser(description="单步实验数据创建")
    parser.add_argument('--input_file', type=str,
                        default=os.path.join(ROOT_DIR, "datasets", "metamath", "MetaMathQA-395K.json"),
                        help="输入数据文件")
    parser.add_argument('--output_file', type=str,
                        default="metamath_stepwise.jsonl",
                        help="输出数据文件(JSONL)")
    parser.add_argument('--filter_min_steps', type=int, default=2,
                        help="最小步骤数，少于该值不拆分")
    parser.add_argument('--max_samples', type=int, default=0,
                        help="最大样本数(0=全部)")
    args = parser.parse_args()

    process_dataset(
        args.input_file,
        args.output_file,
        args.filter_min_steps,
        args.max_samples
    )


if __name__ == "__main__":
    main()
