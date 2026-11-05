"""
去除CoT推理步骤的脚本
将训练数据的output改为仅含最终答案，用于无CoT对照实验

输入: MetaMathQA-395K.json (CoT格式，含推理步骤)
输出: MetaMathQA-395K-nocot.json (仅含最终答案)
"""

import json
import re
import os
import argparse
from tqdm import tqdm


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def extract_final_answer(response: str) -> str:
    """
    从CoT回答中提取最终答案
    优先级:
    1. "The answer is: X" 格式
    2. "#### X" 格式 (GSM8K格式)
    3. 最后一句话
    """

    if "The answer is:" in response:
        parts = response.split("The answer is:")
        last_part = parts[-1].strip()

        match = re.match(r'([^.\n]+(?:[.\n]|$))', last_part)
        if match:
            return match.group(1).strip()
        return last_part.strip()


    if "####" in response:
        parts = response.split("####")
        last_part = parts[-1].strip()

        lines = last_part.split('\n')
        return lines[0].strip()


    sentences = re.split(r'[.!?]+', response)
    for s in reversed(sentences):
        s = s.strip()
        if s and re.search(r'\d', s):
            return s

    return response.strip()


def remove_intermediate_steps(item: dict) -> dict:
    """
    去除中间推理步骤，仅保留问题和最终答案
    """
    query = item.get('query', item.get('instruction', ''))
    response = item.get('response', item.get('output', ''))


    final_answer = extract_final_answer(response)


    new_item = item.copy()
    new_item['response'] = f"The answer is: {final_answer}"
    new_item['original_response'] = response

    return new_item


def process_dataset(input_file: str, output_file: str, max_samples: int = 0):
    """
    处理数据集，去除CoT步骤

    Args:
        input_file: 输入JSON文件
        output_file: 输出JSON文件
        max_samples: 最大样本数 (0=全部)
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

    print(f"总样本数: {len(samples)}")


    stats = {
        'total': len(samples),
        'with_cot_steps': 0,
        'answer_extracted': 0
    }

    nocot_data = []

    for item in tqdm(samples, desc="Processing"):
        response = item.get('response', item.get('output', ''))


        if '<<' in response and '>>' in response:
            stats['with_cot_steps'] += 1

        new_item = remove_intermediate_steps(item)
        nocot_data.append(new_item)


        if new_item['response'] != response:
            stats['answer_extracted'] += 1


    print(f"\n保存到: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(nocot_data, f, ensure_ascii=False, indent=2)


    print("\n" + "=" * 60)
    print("统计信息")
    print("=" * 60)
    print(f"总样本数: {stats['total']}")
    print(f"含CoT步骤的样本: {stats['with_cot_steps']}")
    print(f"答案已提取: {stats['answer_extracted']}")
    print(f"比例: {stats['answer_extracted']/stats['total']*100:.1f}%")
    print("=" * 60)


    print("\n示例:")
    for i in range(min(3, len(nocot_data))):
        orig = nocot_data[i].get('original_response', '')[:150]
        new = nocot_data[i]['response'][:150]
        print(f"\n--- 样本 {i+1} ---")
        print(f"原始: {orig}...")
        print(f"简化: {new}")


def main():
    parser = argparse.ArgumentParser(description="去除CoT步骤")
    parser.add_argument('--input_file', type=str,
                       default=os.path.join(ROOT_DIR, "datasets", "metamath", "MetaMathQA-395K.json"),
                       help="输入数据文件")
    parser.add_argument('--output_file', type=str,
                       default="MetaMathQA-395K-nocot.json",
                       help="输出数据文件")
    parser.add_argument('--max_samples', type=int, default=0,
                       help="最大样本数(0=全部)")
    args = parser.parse_args()

    process_dataset(args.input_file, args.output_file, args.max_samples)


if __name__ == "__main__":
    main()
