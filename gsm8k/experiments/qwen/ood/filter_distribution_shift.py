"""
分布偏移实验数据筛选脚本

目标: 从MetaMathQA中筛选出与GSM8K测试集有分布偏移的训练子集
同时确保样本仍然可以学习（不能差异太大）

筛选策略:
1. 主题偏移: 筛选包含GSM8K中少见主题的样本
2. 复杂度偏移: 筛选与GSM8K步骤数/计算量分布不同的样本
3. 答案规模偏移: 筛选答案数值范围明显不同的样本
"""

import json
import re
import os
import argparse
from collections import Counter, defaultdict
from tqdm import tqdm
import numpy as np


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def extract_keywords(text: str) -> set:
    """提取文本中的关键词（名词、实体等）"""

    math_keywords = {
        'fraction', 'decimal', 'percent', 'percentage', 'ratio',
        'probability', 'average', 'mean', 'median', 'mode',
        'area', 'perimeter', 'volume', 'circumference', 'diameter', 'radius',
        'angle', 'degree', 'triangle', 'square', 'rectangle', 'circle',
        'algebra', 'equation', 'variable', 'polynomial', 'quadratic',
        'derivative', 'integral', 'calculus', 'limit', 'function',
        'profit', 'cost', 'revenue', 'interest', 'discount',
        'distance', 'speed', 'time', 'velocity', 'acceleration',
        'population', 'sample', 'survey', 'data', 'statistic'
    }

    text_lower = text.lower()
    found = set()
    for kw in math_keywords:
        if kw in text_lower:
            found.add(kw)
    return found


def count_calculations(text: str) -> int:
    """统计计算步骤数量"""
    return len(re.findall(r'<<[^<>]+>>', text))


def extract_numbers(text: str) -> list:
    """提取文本中的所有数字"""
    return [float(n.replace(',', '')) for n in re.findall(r'[\-+]?\d[\d,]*(?:\.\d+)?', text)]


def analyze_step_count(text: str) -> int:
    """分析步骤数量"""
    calc_steps = count_calculations(text)
    if calc_steps > 0:
        return calc_steps

    sentences = re.split(r'[.!?]\s+', text)
    return max(1, len([s for s in sentences if s.strip()]))


def get_answer_magnitude(text: str) -> str:
    """获取答案数量级: small(<100), medium(100-1000), large(>1000)"""

    match = re.search(r'(?:The answer is:|####)\s*\$?\s*([\d,]+)', text)
    if match:
        num = float(match.group(1).replace(',', ''))
        if num < 100:
            return 'small'
        elif num < 1000:
            return 'medium'
        else:
            return 'large'
    return 'unknown'


def analyze_gsm8k_distribution(gsm8k_file: str) -> dict:
    """分析GSM8K测试集的分布"""
    print(f"分析GSM8K测试集: {gsm8k_file}")

    with open(gsm8k_file, 'r') as f:
        data = [json.loads(line) for line in f]

    stats = {
        'step_counts': [],
        'answer_magnitudes': Counter(),
        'domains': Counter(),
        'calc_counts': [],
        'total': len(data)
    }

    for item in tqdm(data, desc="分析GSM8K"):
        response = item.get('response', '')
        stats['step_counts'].append(analyze_step_count(response))
        stats['answer_magnitudes'][get_answer_magnitude(response)] += 1
        stats['calc_counts'].append(count_calculations(response))

        keywords = extract_keywords(item.get('query', ''))
        if keywords:
            for kw in keywords:
                stats['domains'][kw] += 1


    stats['avg_steps'] = np.mean(stats['step_counts'])
    stats['std_steps'] = np.std(stats['step_counts'])
    stats['avg_calcs'] = np.mean(stats['calc_counts'])

    print(f"  平均步骤数: {stats['avg_steps']:.2f} ± {stats['std_steps']:.2f}")
    print(f"  平均计算数: {stats['avg_calcs']:.2f}")
    print(f"  答案规模分布: {dict(stats['answer_magnitudes'])}")
    print(f"  主要主题: {stats['domains'].most_common(5)}")

    return stats


def filter_by_distribution_shift(metamath_file: str, gsm8k_stats: dict,
                                 output_file: str,
                                 shift_mode: str = "domain",
                                 shift_ratio: float = 0.3,
                                 max_input_samples: int = 0,
                                 max_output_samples: int = 100000):
    """
    根据分布偏移筛选MetaMath样本

    Args:
        metamath_file: MetaMath数据文件
        gsm8k_stats: GSM8K分布统计
        output_file: 输出文件
        shift_mode: "domain" | "complexity" | "magnitude" | "all"
        shift_ratio: 偏移比例 (0.3 = 筛选30%偏移最大的样本)
        max_input_samples: 最大输入样本数 (0=全部)，先裁剪后处理节省内存
        max_output_samples: 最大输出样本数
    """
    print(f"\n加载MetaMath数据: {metamath_file}")
    with open(metamath_file, 'r') as f:
        data = json.load(f)

    if isinstance(data, list):
        samples = data
    else:
        samples = [data]


    if max_input_samples > 0:
        samples = samples[:max_input_samples]
        print(f"裁剪后输入样本数: {len(samples)}")

    print(f"总MetaMath样本: {len(samples)}")


    scored_samples = []


    gsm8k_keywords = set(dict(gsm8k_stats['domains'].most_common(20)).keys())

    for item in tqdm(samples, desc="计算偏移分数"):
        query = item.get('query', '')
        response = item.get('response', '')


        sample_keywords = extract_keywords(query)
        if gsm8k_keywords:
            keyword_overlap = len(sample_keywords & gsm8k_keywords)
            keyword_shift = len(sample_keywords - gsm8k_keywords) / max(1, len(sample_keywords | gsm8k_keywords))
        else:
            keyword_overlap = 0
            keyword_shift = 0


        sample_steps = analyze_step_count(response)
        sample_calcs = count_calculations(response)
        gsm8k_avg = gsm8k_stats['avg_steps']
        gsm8k_std = max(1, gsm8k_stats['std_steps'])
        complexity_shift = abs(sample_steps - gsm8k_avg) / gsm8k_std


        sample_magnitude = get_answer_magnitude(response)
        gsm8k_magnitude_dist = gsm8k_stats['answer_magnitudes']
        total_gsm8k = sum(gsm8k_magnitude_dist.values())
        gsm8k_magnitude_prob = {k: v/total_gsm8k for k, v in gsm8k_magnitude_dist.items()}
        magnitude_shift = 0
        if sample_magnitude != 'unknown' and sample_magnitude in gsm8k_magnitude_prob:
            magnitude_shift = 1 - gsm8k_magnitude_prob[sample_magnitude]


        if shift_mode == "domain":
            shift_score = keyword_shift
        elif shift_mode == "complexity":
            shift_score = complexity_shift
        elif shift_mode == "magnitude":
            shift_score = magnitude_shift
        else:  # all
            shift_score = (keyword_shift + complexity_shift + magnitude_shift) / 3

        scored_samples.append({
            'sample': item,
            'shift_score': shift_score,
            'keyword_shift': keyword_shift,
            'complexity_shift': complexity_shift,
            'magnitude_shift': magnitude_shift,
            'num_steps': sample_steps,
            'keywords': list(sample_keywords)
        })


    scored_samples.sort(key=lambda x: x['shift_score'], reverse=True)


    n_select = min(max_output_samples, int(len(scored_samples) * shift_ratio))
    selected = scored_samples[:n_select]


    output_data = []
    for item in selected:
        sample = item['sample'].copy()
        sample['shift_info'] = {
            'total_shift': item['shift_score'],
            'keyword_shift': item['keyword_shift'],
            'complexity_shift': item['complexity_shift'],
            'magnitude_shift': item['magnitude_shift'],
            'num_steps': item['num_steps'],
            'shift_mode': shift_mode
        }
        output_data.append(sample)

    print(f"\n保存到: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)


    print("\n" + "=" * 60)
    print("分布偏移筛选统计")
    print("=" * 60)
    print(f"选取样本数: {len(output_data)}")
    print(f"平均偏移分数: {np.mean([s['shift_score'] for s in selected]):.3f}")
    print(f"偏移分数范围: {selected[-1]['shift_score']:.3f} - {selected[0]['shift_score']:.3f}")


    print(f"\n主题偏移: {np.mean([s['keyword_shift'] for s in selected]):.3f}")
    print(f"复杂度偏移: {np.mean([s['complexity_shift'] for s in selected]):.3f}")
    print(f"规模偏移: {np.mean([s['magnitude_shift'] for s in selected]):.3f}")


    step_counts = Counter([s['num_steps'] for s in selected])
    print(f"\n选中样本步骤数分布:")
    for n, count in sorted(step_counts.items())[:10]:
        print(f"  {n}步: {count}")


    all_keywords = Counter()
    for s in selected:
        for kw in s['keywords']:
            all_keywords[kw] += 1
    print(f"\n选中样本主题分布:")
    for kw, count in all_keywords.most_common(10):
        print(f"  {kw}: {count}")

    print("=" * 60)

    return output_data


def analyze_metamath(metamath_file: str, max_samples: int = 10000):
    """分析MetaMath数据集的分布"""
    print(f"\n分析MetaMath数据: {metamath_file}")

    with open(metamath_file, 'r') as f:
        data = json.load(f)

    if isinstance(data, list):
        samples = data[:max_samples]
    else:
        samples = [data]

    step_counts = []
    calc_counts = []
    magnitudes = Counter()
    all_keywords = Counter()

    for item in tqdm(samples, desc="分析MetaMath"):
        response = item.get('response', '')
        query = item.get('query', '')

        step_counts.append(analyze_step_count(response))
        calc_counts.append(count_calculations(response))
        magnitudes[get_answer_magnitude(response)] += 1

        keywords = extract_keywords(query)
        for kw in keywords:
            all_keywords[kw] += 1

    print(f"\nMetaMath分布 (样本数: {len(samples)}):")
    print(f"  平均步骤数: {np.mean(step_counts):.2f}")
    print(f"  平均计算数: {np.mean(calc_counts):.2f}")
    print(f"  答案规模: {dict(magnitudes)}")
    print(f"  主题: {all_keywords.most_common(10)}")


def main():
    parser = argparse.ArgumentParser(description="分布偏移筛选")
    parser.add_argument('--metamath_file', type=str,
                       default=os.path.join(ROOT_DIR, "datasets", "metamath", "MetaMathQA-395K.json"),
                       help="MetaMath数据文件")
    parser.add_argument('--gsm8k_file', type=str,
                       default=os.path.join(ROOT_DIR, "datasets", "gsm8k", "test.jsonl"),
                       help="GSM8K测试文件")
    parser.add_argument('--output_file', type=str,
                       default="metamath_ood_subset.json",
                       help="输出文件")
    parser.add_argument('--shift_mode', type=str, default="all",
                       choices=["domain", "complexity", "magnitude", "all"],
                       help="偏移模式")
    parser.add_argument('--shift_ratio', type=float, default=0.3,
                       help="偏移样本比例 (0.3 = 30%%)")
    parser.add_argument('--max_input_samples', type=int, default=0,
                       help="最大输入样本数 (0=全部)，先裁剪后处理节省内存")
    parser.add_argument('--max_output_samples', type=int, default=100000,
                       help="最大输出样本数")
    parser.add_argument('--analyze_only', action='store_true',
                       help="仅分析，不筛选")
    args = parser.parse_args()


    gsm8k_stats = analyze_gsm8k_distribution(args.gsm8k_file)

    if args.analyze_only:
        analyze_metamath(args.metamath_file)
    else:
        filter_by_distribution_shift(
            args.metamath_file,
            gsm8k_stats,
            args.output_file,
            args.shift_mode,
            args.shift_ratio,
            args.max_input_samples,
            args.max_output_samples
        )


if __name__ == "__main__":
    main()
