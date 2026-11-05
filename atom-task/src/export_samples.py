#!/usr/bin/env python3
"""
导出预测样本到 JSON 文件
"""

import json
import sys
import argparse


def extract_final_answer(text):
    """从文本中提取最终答案序列"""
    if text is None:
        return None
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
    return text.strip('"').strip("'").strip('[]').strip()


def export_samples(predict_file, max_samples, output_file):
    """导出预测样本"""
    samples = []
    errors = []

    with open(predict_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if line_num >= max_samples:
                break
            try:
                data = json.loads(line.strip())
                predict_text = data.get('predict', '')
                label_text = data.get('label', '')

                predict_answer = extract_final_answer(predict_text)
                label_answer = extract_final_answer(label_text)

                is_correct = predict_answer.replace(' ', '') == label_answer.replace(' ', '') if predict_answer and label_answer else False

                samples.append({
                    'predicted': predict_answer or '',
                    'gold': label_answer or '',
                    'correct': is_correct,
                    'response': predict_text
                })

                if not is_correct and len(errors) < 20:
                    errors.append({
                        'index': line_num,
                        'predicted': predict_answer or '',
                        'gold': label_answer or '',
                        'response': predict_text[:500]
                    })
            except json.JSONDecodeError:
                continue

    result = {
        'total': len(samples),
        'exported': len(samples),
        'samples': samples,
        'errors': errors
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"已导出 {len(samples)} 个样本到: {output_file}")
    return result


def main():
    parser = argparse.ArgumentParser(description='导出预测样本')
    parser.add_argument('predict_file', type=str, help='预测文件路径')
    parser.add_argument('max_samples', type=int, help='最大样本数')
    parser.add_argument('output_file', type=str, help='输出文件路径')
    args = parser.parse_args()

    export_samples(args.predict_file, args.max_samples, args.output_file)


if __name__ == '__main__':
    main()