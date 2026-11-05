#!/usr/bin/env python3
"""
评估预测结果的准确率
比较模型输出的最终序列与标签中的最终序列
"""

import json
import os
import re
import sys
import argparse
from pathlib import Path


def extract_final_answer(text):
    """从文本中提取最终答案序列"""
    if text is None:
        return None

    text = re.sub(r'<BOS>|<EOS>|<\|endoftext\|>|</?s>|<\|eos\|>', '', text)
    text = text.strip()
    marker = "Final Answer:"
    if marker in text:
        answer = text.split(marker)[-1].strip()

        if " Step" in answer:
            answer = answer.split(" Step")[0].strip()

        answer = answer.strip('"').strip("'").strip()

        if answer.startswith('[') and answer.endswith(']'):
            answer = answer[1:-1]
        return answer

    if "Answer:" in text:
        answer = text.split("Answer:")[-1].strip()

        if " Step" in answer:
            answer = answer.split(" Step")[0].strip()
        answer = answer.strip('"').strip("'").strip()
        if answer.startswith('[') and answer.endswith(']'):
            answer = answer[1:-1]
        return answer
    if "Output:" in text:
        parts = text.split("Output:")
        if len(parts) > 1:
            answer = parts[-1].strip()

            if " Step" in answer:
                answer = answer.split(" Step")[0].strip()
            answer = answer.strip('"').strip("'").strip()
            if answer.startswith('[') and answer.endswith(']'):
                answer = answer[1:-1]
            return answer

    return text.strip('"').strip("'").strip('[]').strip()


def evaluate_prediction(predict_file, verbose=False):
    """评估单个预测文件"""
    correct = 0
    total = 0
    errors = []

    if not os.path.exists(predict_file):
        return None

    with open(predict_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                predict_text = data.get('predict', '')
                label_text = data.get('label', '')

                predict_answer = extract_final_answer(predict_text)
                label_answer = extract_final_answer(label_text)

                if predict_answer is None:
                    if verbose:
                        print(f"[WARN] Line {line_num}: 无法从预测中提取 Final Answer")
                    continue
                if label_answer is None:
                    if verbose:
                        print(f"[WARN] Line {line_num}: 无法从标签中提取 Final Answer")
                    continue

                total += 1

                if predict_answer.strip() == label_answer.strip():
                    correct += 1
                else:
                    if len(errors) < 20:
                        errors.append({
                            'line': line_num,
                            'predict': predict_answer,
                            'label': label_answer
                        })
            except json.JSONDecodeError as e:
                if verbose:
                    print(f"[WARN] Line {line_num}: JSON解析错误 - {e}")
                continue

    accuracy = correct / total if total > 0 else 0.0
    return {
        'correct': correct,
        'total': total,
        'accuracy': accuracy,
        'errors': errors
    }


def main():
    parser = argparse.ArgumentParser(description='评估预测准确率')
    parser.add_argument('path', nargs='?', default=None,
                        help='预测文件路径或目录路径')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='显示详细错误信息')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出文件路径 (JSON格式)')
    args = parser.parse_args()

    if args.path is None:
        print("错误: 请指定预测文件路径")
        sys.exit(1)

    predict_file = Path(args.path)
    if not predict_file.exists():
        print(f"错误: 文件不存在: {predict_file}")
        sys.exit(1)

    result = evaluate_prediction(predict_file, args.verbose)

    if result is None:
        print("错误: 无法评估文件")
        sys.exit(1)


    print("=" * 60)
    print("预测准确率评估结果")
    print("=" * 60)
    print(f"正确: {result['correct']}/{result['total']}")
    print(f"准确率: {result['accuracy']*100:.2f}%")

    if args.verbose and result['errors']:
        print(f"\n错误详情 (前{len(result['errors'])}条):")
        for i, err in enumerate(result['errors'][:5]):
            print(f"  [{err['line']}] 预测: {err['predict']}")
            print(f"           标签: {err['label']}")


    if args.output:
        output_data = {
            'correct': result['correct'],
            'total': result['total'],
            'accuracy': f"{result['accuracy']*100:.2f}%",
            'errors': result['errors']
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n结果已导出到: {args.output}")

    return result


if __name__ == '__main__':
    main()