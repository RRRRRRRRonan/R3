"""简化版Week 1分析脚本 - 不需要numpy/pandas/scipy

只使用Python标准库进行统计分析
"""

import json
import glob
import statistics
from pathlib import Path
from collections import defaultdict

def load_all_results():
    """加载所有实验结果"""
    results = {
        'zero': {'small': [], 'medium': [], 'large': []},
        'uniform': {'small': [], 'medium': [], 'large': []},
        'action_specific': {'small': [], 'medium': [], 'large': []},
        'state_specific': {'small': [], 'medium': [], 'large': []},
    }

    # 加载baseline (zero)
    for scenario in ['small', 'medium', 'large']:
        files = glob.glob(f'results/week1/baseline/baseline_{scenario}_*.json')
        for f in files:
            with open(f) as fp:
                data = json.load(fp)
                results['zero'][scenario].append(data['improvement_ratio'])

    # 加载其他策略
    for strategy in ['uniform', 'action_specific', 'state_specific']:
        for scenario in ['small', 'medium', 'large']:
            files = glob.glob(f'results/week1/init_experiments/init_{strategy}_{scenario}_*.json')
            for f in files:
                with open(f) as fp:
                    data = json.load(fp)
                    results[strategy][scenario].append(data['improvement_ratio'])

    return results

def wilcoxon_test_simple(x, y):
    """简化的Wilcoxon符号秩检验"""
    # 计算差值
    diffs = [a - b for a, b in zip(x, y)]

    # 去除零值
    diffs = [d for d in diffs if d != 0]

    if len(diffs) == 0:
        return None, "相同"

    # 计算秩
    abs_diffs = [(abs(d), i) for i, d in enumerate(diffs)]
    abs_diffs.sort()

    ranks = {}
    for rank, (abs_val, idx) in enumerate(abs_diffs, 1):
        ranks[idx] = rank

    # 计算正负秩和
    pos_sum = sum(ranks[i] for i, d in enumerate(diffs) if d > 0)
    neg_sum = sum(ranks[i] for i, d in enumerate(diffs) if d < 0)

    W = min(pos_sum, neg_sum)

    # 简化判断（n=10的临界值）
    # n=10, α=0.05: W_critical = 8
    # n=10, α=0.01: W_critical = 3
    if W <= 3:
        significance = "*** (p<0.01)"
    elif W <= 8:
        significance = "** (p<0.05)"
    elif W <= 11:
        significance = "* (p<0.10)"
    else:
        significance = "ns (p>0.10)"

    return W, significance

def cohens_d(x, y):
    """计算Cohen's d效应量"""
    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)

    std_x = statistics.stdev(x)
    std_y = statistics.stdev(y)

    pooled_std = ((std_x**2 + std_y**2) / 2) ** 0.5

    if pooled_std == 0:
        return 0

    d = (mean_x - mean_y) / pooled_std
    return d

def interpret_cohens_d(d):
    """解释Cohen's d"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "微小"
    elif abs_d < 0.5:
        return "小"
    elif abs_d < 0.8:
        return "中等"
    else:
        return "大"

def main():
    print("=" * 80)
    print("Week 1: Q-table初始化策略统计分析")
    print("=" * 80)
    print()

    # 加载数据
    results = load_all_results()

    # 验证数据完整性
    print("数据完整性检查:")
    for strategy in ['zero', 'uniform', 'action_specific', 'state_specific']:
        for scenario in ['small', 'medium', 'large']:
            n = len(results[strategy][scenario])
            status = "✓" if n == 10 else "✗"
            print(f"  {strategy:20} {scenario:10}: {n:2}/10 {status}")
    print()

    # 描述性统计
    print("=" * 80)
    print("描述性统计 (改进率)")
    print("=" * 80)
    print()
    print(f"{'策略':<20} {'Small':<20} {'Medium':<20} {'Large':<20}")
    print("-" * 80)

    for strategy in ['zero', 'uniform', 'action_specific', 'state_specific']:
        row = f"{strategy:<20}"
        for scenario in ['small', 'medium', 'large']:
            data = results[strategy][scenario]
            if data:
                mean = statistics.mean(data) * 100
                std = statistics.stdev(data) * 100
                row += f" {mean:5.2f}% ± {std:4.2f}%   "
            else:
                row += " N/A                 "
        print(row)

    print()

    # 与baseline对比
    print("=" * 80)
    print("与ZERO baseline对比")
    print("=" * 80)
    print()

    for scenario in ['small', 'medium', 'large']:
        print(f"\n{scenario.upper()} 场景:")
        print("-" * 80)
        cohens_d_header = "Cohen's d"
        print(f"{'策略':<20} {'平均差异':<15} {'Wilcoxon检验':<20} {cohens_d_header:<15} {'效应':<10}")
        print("-" * 80)

        baseline = results['zero'][scenario]

        for strategy in ['uniform', 'action_specific', 'state_specific']:
            test_data = results[strategy][scenario]

            if len(baseline) == len(test_data) == 10:
                # 计算平均差异
                mean_diff = (statistics.mean(test_data) - statistics.mean(baseline)) * 100

                # Wilcoxon检验
                W, sig = wilcoxon_test_simple(test_data, baseline)

                # Cohen's d
                d = cohens_d(test_data, baseline)
                effect = interpret_cohens_d(d)

                print(f"{strategy:<20} {mean_diff:+6.2f}%         {sig:<20} {d:+6.3f}         {effect:<10}")
            else:
                print(f"{strategy:<20} 数据不完整")

    print()

    # 总结建议
    print("=" * 80)
    print("总结与建议")
    print("=" * 80)
    print()

    # 找出每个场景的最佳策略
    for scenario in ['small', 'medium', 'large']:
        best_strategy = None
        best_mean = 0

        for strategy in ['zero', 'uniform', 'action_specific', 'state_specific']:
            data = results[strategy][scenario]
            if data:
                mean = statistics.mean(data)
                if mean > best_mean:
                    best_mean = mean
                    best_strategy = strategy

        print(f"{scenario.upper()}场景:")
        print(f"  最佳策略: {best_strategy} ({best_mean*100:.2f}%改进率)")

        # 检查是否显著优于baseline
        if best_strategy != 'zero':
            test_data = results[best_strategy][scenario]
            baseline = results['zero'][scenario]
            W, sig = wilcoxon_test_simple(test_data, baseline)
            d = cohens_d(test_data, baseline)

            print(f"  vs ZERO: {sig}, Cohen's d = {d:+.3f}")
        else:
            print(f"  ZERO (baseline) 表现最佳")
        print()

    print("=" * 80)
    print("分析完成!")
    print("=" * 80)

    # 保存总结到文件
    output_file = "results/week1/analysis_summary.txt"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Week 1 统计分析总结\n")
        f.write("=" * 80 + "\n\n")

        for scenario in ['small', 'medium', 'large']:
            f.write(f"\n{scenario.upper()} 场景:\n")
            f.write("-" * 40 + "\n")

            baseline = results['zero'][scenario]
            baseline_mean = statistics.mean(baseline) * 100
            f.write(f"ZERO (baseline): {baseline_mean:.2f}%\n")

            for strategy in ['uniform', 'action_specific', 'state_specific']:
                test_data = results[strategy][scenario]
                test_mean = statistics.mean(test_data) * 100
                mean_diff = test_mean - baseline_mean

                W, sig = wilcoxon_test_simple(test_data, baseline)
                d = cohens_d(test_data, baseline)

                f.write(f"{strategy}: {test_mean:.2f}% ({mean_diff:+.2f}%), {sig}, d={d:+.3f}\n")

    print(f"\n分析总结已保存到: {output_file}")

if __name__ == "__main__":
    main()
