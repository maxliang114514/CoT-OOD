#!/usr/bin/env python3
"""
实验结果汇总脚本
自动收集并比较所有实验的结果
"""

import json
import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = os.path.join(ROOT_DIR, "experiments", "qwen", "ood")


def load_acc_json(filepath):
    """加载准确率JSON"""
    try:
        with open(filepath) as f:
            data = json.load(f)
            return data.get('summary', {})
    except Exception as e:
        return {'error': str(e)}


def main():
    results_dir = Path(OUTPUT_DIR) / "results"

    if not results_dir.exists():
        print(f"结果目录不存在: {results_dir}")
        return

    print("=" * 80)
    print("OOD泛化实验结果汇总")
    print("=" * 80)


    baseline_dir = Path(ROOT_DIR) / "experiments" / "qwen"

    baselines = {
        "Llama2-7B Baseline": baseline_dir / "baseline_response.jsonl",
    }

    print("\n【基线】")
    print("-" * 80)
    print(f"{'实验配置':<35} {'准确率':<12} {'提升':<10} {'正确/总数'}")
    print("-" * 80)


    baseline_acc = None
    baseline_file = baseline_dir / "baseline_response.jsonl"
    if baseline_file.exists():

        acc_file = results_dir / "llama_baseline_acc.json"
        if acc_file.exists():
            summary = load_acc_json(acc_file)
            if summary:
                baseline_acc = summary.get('accuracy', 0)
                acc_pct = summary.get('accuracy_percent', 'N/A')
                correct = summary.get('correct', 0)
                total = summary.get('total', 0)
                print(f"{'Llama2-7B 基线 (无训练)':<35} {acc_pct:<12} {'-':<10} {correct}/{total}")
        else:

            import subprocess
            result = subprocess.run(
                f"cd {baseline_dir} && python test_acc.py --input_file baseline_response.jsonl 2>/dev/null | grep Accuracy",
                shell=True, capture_output=True, text=True
            )
            if result.stdout:
                print(f"{'Llama2-7B 基线 (无训练)':<35} {result.stdout.strip()}")
            else:
                print(f"{'Llama2-7B 基线 (无训练)':<35} {'N/A':<12}")

    print()
    print("【噪声注入实验】")
    print("-" * 80)
    print(f"{'实验配置':<35} {'准确率':<12} {'提升':<10} {'正确/总数'}")
    print("-" * 80)

    noise_rates = [5, 10, 20]
    noise_results = []
    for rate in noise_rates:
        acc_file = results_dir / f"llama_noise{rate}_acc.json"
        if acc_file.exists():
            summary = load_acc_json(acc_file)
            if summary:
                acc = summary.get('accuracy', 0)
                acc_pct = summary.get('accuracy_percent', 'N/A')
                correct = summary.get('correct', 0)
                total = summary.get('total', 0)
                improvement = f"+{(acc - baseline_acc)*100:.1f}%" if baseline_acc else "N/A"
                print(f"{'Llama 噪声{rate}%':<35} {acc_pct:<12} {improvement:<10} {correct}/{total}")
                noise_results.append((rate, acc))
        else:
            print(f"{'Llama 噪声{rate}%':<35} {'N/A':<12}")

    print()
    print("【单步实验】")
    print("-" * 80)
    acc_file = results_dir / "llama_stepwise_acc.json"
    if acc_file.exists():
        summary = load_acc_json(acc_file)
        if summary:
            acc = summary.get('accuracy', 0)
            acc_pct = summary.get('accuracy_percent', 'N/A')
            correct = summary.get('correct', 0)
            total = summary.get('total', 0)
            improvement = f"+{(acc - baseline_acc)*100:.1f}%" if baseline_acc else "N/A"
            print(f"{'Llama 单步':<35} {acc_pct:<12} {improvement:<10} {correct}/{total}")
    else:
        print(f"{'Llama 单步':<35} {'N/A':<12}")

    print()
    print("【分布偏移实验】")
    print("-" * 80)
    print(f"{'实验配置':<35} {'准确率':<12} {'提升':<10} {'正确/总数'}")
    print("-" * 80)

    ood_modes = ["domain", "complexity", "magnitude", "all"]
    for mode in ood_modes:
        acc_file = results_dir / f"llama_ood_{mode}_acc.json"
        if acc_file.exists():
            summary = load_acc_json(acc_file)
            if summary:
                acc = summary.get('accuracy', 0)
                acc_pct = summary.get('accuracy_percent', 'N/A')
                correct = summary.get('correct', 0)
                total = summary.get('total', 0)
                improvement = f"+{(acc - baseline_acc)*100:.1f}%" if baseline_acc else "N/A"
                print(f"{'Llama OOD ({mode})':<35} {acc_pct:<12} {improvement:<10} {correct}/{total}")
        else:
            print(f"{'Llama OOD ({mode})':<35} {'N/A':<12}")

    print()
    print("=" * 80)
    print("分析总结")
    print("=" * 80)


    all_results = []

    for rate, acc in noise_results:
        all_results.append((f"噪声{rate}%", acc))

    stepwise_file = results_dir / "llama_stepwise_acc.json"
    if stepwise_file.exists():
        summary = load_acc_json(stepwise_file)
        if summary:
            all_results.append(("单步", summary.get('accuracy', 0)))

    for mode in ood_modes:
        acc_file = results_dir / f"llama_ood_{mode}_acc.json"
        if acc_file.exists():
            summary = load_acc_json(acc_file)
            if summary:
                all_results.append((f"OOD({mode})", summary.get('accuracy', 0)))

    if all_results and baseline_acc:
        all_results.sort(key=lambda x: x[1], reverse=True)
        best_name, best_acc = all_results[0]
        print(f"最优配置: {best_name} ({best_acc*100:.2f}%)")
        print(f"相比基线提升: +{(best_acc - baseline_acc)*100:.1f}%")
        print(f"相比基线相对提升: +{(best_acc - baseline_acc)/baseline_acc*100:.1f}%")

    print("=" * 80)


if __name__ == "__main__":
    main()
