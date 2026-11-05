"""
T-CotMechanism风格精度测试脚本
计算GSM8K准确率
"""

import argparse
import json
import re
from collections import defaultdict


def is_number(s):
    """判断是否为数字"""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def extract_answer_number(completion):
    """
    从生成文本中提取答案数字
    优先找 "The answer is:" 后的数字 (T-CotMechanism方式)
    其次找 "####" 后的数字 (GSM8K方式)
    """
    if not completion:
        return None


    text = completion.split('The answer is:')
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d[\d,]*', extract_ans)
        if match:
            try:
                return int(match.group().replace(',', ''))
            except:
                pass


    if "####" in completion:
        parts = completion.split("####")
        if len(parts) > 1:
            match = re.search(r'(\d[\d,]*)', parts[-1])
            if match:
                try:
                    return int(match.group().replace(',', ''))
                except:
                    pass


    dollar_matches = re.findall(r'\$\s*(-?\d[\d,]*)', completion)
    if dollar_matches:
        try:
            return int(dollar_matches[-1].replace(',', ''))
        except:
            pass


    all_numbers = re.findall(r'(-?\d[\d,]*)', completion)
    if all_numbers:
        try:
            return int(all_numbers[-1].replace(',', ''))
        except:
            pass

    return None


def extract_gsm8k_answer(response):
    """从标准response中提取GSM8K答案 (#### 格式)"""
    try:
        ans = response.split('#### ')[1]
        ans = int(ans.split('\n')[0].replace(',', ''))
        return ans
    except:
        return None


def test_accuracy(input_file, output_file=None):
    """测试准确率"""
    results = []

    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            data_type = data.get('type', 'unknown')


            y_pred = extract_answer_number(data.get('output', ''))

            y_true = extract_answer_number(data.get('answer', ''))

            if y_pred is None or y_true is None:
                results.append({
                    'correct': False,
                    'predicted': y_pred,
                    'true': y_true,
                    'type': data_type,
                    'output': data.get('output', '')[:200]
                })
            else:
                results.append({
                    'correct': (y_pred == y_true),
                    'predicted': y_pred,
                    'true': y_true,
                    'type': data_type,
                    'output': data.get('output', '')[:200]
                })


    correct = sum(1 for r in results if r['correct'])
    total = len(results)
    accuracy = correct / total if total > 0 else 0


    summary = {
        'input_file': input_file,
        'correct': correct,
        'total': total,
        'accuracy': accuracy,
        'accuracy_percent': f"{accuracy*100:.2f}%"
    }


    print("\n" + "=" * 60)
    print("Accuracy Results")
    print("=" * 60)
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print("=" * 60)


    errors = [r for r in results if not r['correct']][:10]
    if errors:
        print(f"\n前 {len(errors)} 个错误样本:")
        for i, err in enumerate(errors[:5]):
            print(f"  [{i+1}] 预测: {err['predicted']}, 正确答案: {err['true']}")


    if output_file:
        export_data = {'summary': summary, 'results': results}
        with open(output_file, 'w') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {output_file}")

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="GSM8K精度测试")
    parser.add_argument('--input_file', type=str, required=True, help="推理结果文件")
    parser.add_argument('--output_file', type=str, default=None, help="导出结果文件(json)")
    args = parser.parse_args()

    test_accuracy(args.input_file, args.output_file)


if __name__ == "__main__":
    main()