"""
噪声注入实验脚本
对MetaMathQA数据集中的推理步骤注入噪声

噪声类型:
1. 删除步骤 - 将计算式子替换为 "..."
2. 修改数字 - 将计算结果替换为错误数字
"""

import json
import random
import re
import os
import argparse
from typing import List, Tuple, Optional
from tqdm import tqdm


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def extract_steps(text: str) -> List[Tuple[str, str]]:
    """
    从文本中提取所有推理步骤（按换行分割）
    返回: List[(完整步骤, 计算部分)]
    对于噪声注入，我们需要找到步骤中的"计算核心"部分
    """
    steps = []
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    for line in lines:

        if re.match(r'^#{2,}\s*\d+(\.\d+)?$', line.strip()):
            continue
        if re.match(r'^The answer is:\s*[\$\\\boxed]?[\d\.\-]+', line.strip()):
            continue
        if len(line) < 10:
            continue
        steps.append((line, line))
    return steps


def get_step_numbers(calc: str) -> List[float]:
    """从计算步骤中提取所有数字"""
    numbers = re.findall(r'[\-+]?\d[\d,]*(?:\.\d+)?', calc)
    return [float(n.replace(',', '')) for n in numbers]


def corrupt_step_delete(step_with_calc: str, calc: str) -> str:
    """删除该步骤（整行替换为...）"""
    return "..."


def corrupt_step_wrong_number(step_with_calc: str, calc: str) -> str:
    """修改步骤中的数字结果为错误数字"""

    numbers = get_step_numbers(calc)
    if not numbers:
        return "..."




    matches = list(re.finditer(r'(=?\s*)([\d,]+\.?\d*)(\s*(?:\n|$))', step_with_calc))

    if not matches:
        return "..."


    last_match = matches[-1]
    result_str = last_match.group(2)

    try:
        result = float(result_str.replace(',', ''))
    except:
        return "..."


    offset = random.choice([1.1, 1.2, 0.8, 0.9, 1.3, 0.7])
    wrong_result = result * offset


    if result == int(result) and wrong_result == int(wrong_result):
        wrong_result = int(wrong_result)
        replacement = f"= {wrong_result}"
    else:
        replacement = f"= {wrong_result:.2f}"

    return step_with_calc[:last_match.start()] + replacement + last_match.group(3)


def add_noise_to_sample(query: str, response: str, noise_rate: float,
                         noise_type: str = "both") -> Tuple[str, List[int]]:
    """
    对单个样本添加噪声（按样本比例，而非步骤比例）

    策略：每个有步骤的样本，以noise_rate的概率被选中；
    被选中的样本，随机选择1个步骤进行噪声

    Args:
        query: 问题
        response: 回答（含推理步骤）
        noise_rate: 噪声比例 (0.05, 0.1, 0.2, 0.5)
        noise_type: "delete" | "wrong" | "both"

    Returns:
        (修改后的response, 被噪声注入的步骤索引列表)
    """
    steps = extract_steps(response)

    if not steps:
        return response, []


    if random.random() >= noise_rate:
        return response, []


    n_steps = len(steps)
    noise_idx = random.randint(0, n_steps - 1)
    full_step, calc = steps[noise_idx]

    if noise_type == "delete":
        corrupted = corrupt_step_delete(full_step, calc)
    elif noise_type == "wrong":
        corrupted = corrupt_step_wrong_number(full_step, calc)
    else:  # both
        if random.random() < 0.5:
            corrupted = corrupt_step_delete(full_step, calc)
        else:
            corrupted = corrupt_step_wrong_number(full_step, calc)

    modified_response = response.replace(full_step, corrupted, 1)

    return modified_response, [noise_idx]


def process_dataset(input_file: str, output_file: str, noise_rate: float,
                    noise_type: str = "both", max_samples: int = 0):
    """
    处理整个数据集

    Args:
        input_file: 输入JSON文件
        output_file: 输出JSON文件
        noise_rate: 噪声比例 (0.05, 0.1, 0.2)
        noise_type: "delete" | "wrong" | "both"
        max_samples: 最大处理样本数 (0=全部)，先随机抽取后处理
    """
    print(f"加载数据: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)

    if isinstance(data, list):
        samples = data
    else:
        samples = [data]


    if max_samples > 0:
        import random
        random.seed(42)
        random.shuffle(samples)
        samples = samples[:max_samples]
        print(f"随机抽取后样本数: {len(samples)}")

    print(f"总样本数: {len(samples)}")
    print(f"噪声比例: {noise_rate}")
    print(f"噪声类型: {noise_type}")


    stats = {
        'total_samples': len(samples),
        'samples_with_steps': 0,
        'samples_with_noise': 0,
        'total_steps': 0,
        'noisy_steps': 0
    }

    noisy_data = []

    for item in tqdm(samples, desc="Processing"):
        query = item.get('query', item.get('instruction', ''))
        response = item.get('response', item.get('output', ''))


        steps = extract_steps(response)
        if steps:
            stats['samples_with_steps'] += 1
            stats['total_steps'] += len(steps)


            noisy_response, corrupted = add_noise_to_sample(
                query, response, noise_rate, noise_type
            )

            if corrupted:
                stats['samples_with_noise'] += 1
                stats['noisy_steps'] += len(corrupted)

            noisy_item = item.copy()
            noisy_item['response'] = noisy_response
            noisy_item['noise_info'] = {
                'noise_rate': noise_rate,
                'noise_type': noise_type,
                'total_steps': len(steps),
                'corrupted_steps': corrupted
            }
        else:
            noisy_item = item.copy()
            noisy_item['noise_info'] = {
                'noise_rate': noise_rate,
                'noise_type': noise_type,
                'total_steps': 0,
                'corrupted_steps': []
            }

        noisy_data.append(noisy_item)


    print(f"\n保存到: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(noisy_data, f, ensure_ascii=False, indent=2)


    print("\n" + "=" * 60)
    print("统计信息")
    print("=" * 60)
    print(f"总样本数: {stats['total_samples']}")
    print(f"含推理步骤的样本: {stats['samples_with_steps']}")
    print(f"被注入噪声的样本: {stats['samples_with_noise']}")
    print(f"总推理步骤数: {stats['total_steps']}")
    print(f"被噪声修改的步骤: {stats['noisy_steps']}")
    if stats['total_steps'] > 0:
        print(f"实际噪声比例: {stats['noisy_steps']/stats['total_steps']*100:.2f}%")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="MetaMathQA噪声注入")
    parser.add_argument('--input_file', type=str,
                        default=os.path.join(ROOT_DIR, "datasets", "metamath", "MetaMathQA-395K.json"),
                        help="输入数据文件")
    parser.add_argument('--output_file', type=str,
                        default="",
                        help="输出数据文件")
    parser.add_argument('--noise_rate', type=float, default=0.1,
                        help="噪声比例: 0.05, 0.1, 0.2")
    parser.add_argument('--noise_type', type=str, default="both",
                        choices=["delete", "wrong", "both"],
                        help="噪声类型: delete=删除步骤, wrong=修改数字, both=混合")
    parser.add_argument('--max_samples', type=int, default=0,
                        help="最大样本数(0=全部)")
    args = parser.parse_args()


    if not args.output_file:
        rate_str = str(args.noise_rate).replace('.', '')
        args.output_file = f"MetaMathQA_395K_noise{rate_str}.json"

    process_dataset(
        args.input_file,
        args.output_file,
        args.noise_rate,
        args.noise_type,
        args.max_samples
    )


if __name__ == "__main__":
    main()
