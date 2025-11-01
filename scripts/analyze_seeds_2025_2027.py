#!/usr/bin/env python3
"""
Analysis of Experiment Results for Seeds 2025, 2026, 2027
Comparing: Matheuristic ALNS, Minimal ALNS, and Matheuristic + Q-learning
"""

from typing import Dict, List
import json
import statistics
from collections import defaultdict

# Data from seeds 2025, 2026, 2027
data_2025 = [
    {"Scale": "Small", "Solver": "Matheuristic ALNS", "Baseline Cost": 48349.41, "Optimised Cost": 34676.16, "Improvement": 28.28},
    {"Scale": "Small", "Solver": "Minimal ALNS", "Baseline Cost": 45355.97, "Optimised Cost": 41406.25, "Improvement": 8.71},
    {"Scale": "Small", "Solver": "Matheuristic + Q-learning", "Baseline Cost": 48349.41, "Optimised Cost": 29222.47, "Improvement": 39.56},
    {"Scale": "Medium", "Solver": "Matheuristic ALNS", "Baseline Cost": 35102.80, "Optimised Cost": 20385.78, "Improvement": 41.93},
    {"Scale": "Medium", "Solver": "Minimal ALNS", "Baseline Cost": 39317.52, "Optimised Cost": 34525.02, "Improvement": 12.19},
    {"Scale": "Medium", "Solver": "Matheuristic + Q-learning", "Baseline Cost": 35102.80, "Optimised Cost": 20961.53, "Improvement": 40.29},
    {"Scale": "Large", "Solver": "Matheuristic ALNS", "Baseline Cost": 52400.92, "Optimised Cost": 39669.68, "Improvement": 24.30},
    {"Scale": "Large", "Solver": "Minimal ALNS", "Baseline Cost": 60709.91, "Optimised Cost": 56409.99, "Improvement": 7.08},
    {"Scale": "Large", "Solver": "Matheuristic + Q-learning", "Baseline Cost": 52400.92, "Optimised Cost": 35000.15, "Improvement": 33.21},
]

data_2026 = [
    {"Scale": "Small", "Solver": "Matheuristic ALNS", "Baseline Cost": 48349.41, "Optimised Cost": 37294.00, "Improvement": 22.87},
    {"Scale": "Small", "Solver": "Minimal ALNS", "Baseline Cost": 45355.97, "Optimised Cost": 41640.57, "Improvement": 8.19},
    {"Scale": "Small", "Solver": "Matheuristic + Q-learning", "Baseline Cost": 48349.41, "Optimised Cost": 32029.42, "Improvement": 33.75},
    {"Scale": "Medium", "Solver": "Matheuristic ALNS", "Baseline Cost": 35102.80, "Optimised Cost": 22884.88, "Improvement": 34.81},
    {"Scale": "Medium", "Solver": "Minimal ALNS", "Baseline Cost": 39317.52, "Optimised Cost": 35942.82, "Improvement": 8.58},
    {"Scale": "Medium", "Solver": "Matheuristic + Q-learning", "Baseline Cost": 35102.80, "Optimised Cost": 16300.69, "Improvement": 53.56},
    {"Scale": "Large", "Solver": "Matheuristic ALNS", "Baseline Cost": 52400.92, "Optimised Cost": 36721.85, "Improvement": 29.92},
    {"Scale": "Large", "Solver": "Minimal ALNS", "Baseline Cost": 60709.91, "Optimised Cost": 55277.84, "Improvement": 8.95},
    {"Scale": "Large", "Solver": "Matheuristic + Q-learning", "Baseline Cost": 52400.92, "Optimised Cost": 32327.26, "Improvement": 38.31},
]

data_2027 = [
    {"Scale": "Small", "Solver": "Minimal ALNS", "Baseline Cost": 45355.97, "Optimised Cost": 41496.56, "Improvement": 8.51},
    {"Scale": "Small", "Solver": "Matheuristic ALNS", "Baseline Cost": 48349.41, "Optimised Cost": 30983.74, "Improvement": 35.92},
    {"Scale": "Small", "Solver": "Matheuristic + Q-learning", "Baseline Cost": 48349.41, "Optimised Cost": 28360.92, "Improvement": 41.34},
    {"Scale": "Medium", "Solver": "Minimal ALNS", "Baseline Cost": 39317.52, "Optimised Cost": 35242.23, "Improvement": 10.37},
    {"Scale": "Medium", "Solver": "Matheuristic ALNS", "Baseline Cost": 35102.80, "Optimised Cost": 18144.33, "Improvement": 48.31},
    {"Scale": "Medium", "Solver": "Matheuristic + Q-learning", "Baseline Cost": 35102.80, "Optimised Cost": 29131.78, "Improvement": 17.01},
    {"Scale": "Large", "Solver": "Minimal ALNS", "Baseline Cost": 60709.91, "Optimised Cost": 54939.60, "Improvement": 9.50},
    {"Scale": "Large", "Solver": "Matheuristic ALNS", "Baseline Cost": 52400.92, "Optimised Cost": 33955.62, "Improvement": 35.20},
    # Note: Large + Q-learning missing for seed 2027
]

# Combine all data
all_data = []
for row in data_2025:
    row = row.copy()
    row['Seed'] = 2025
    all_data.append(row)
for row in data_2026:
    row = row.copy()
    row['Seed'] = 2026
    all_data.append(row)
for row in data_2027:
    row = row.copy()
    row['Seed'] = 2027
    all_data.append(row)

def get_stats(values):
    """Calculate mean and std for a list of values"""
    if not values:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0}
    return {
        'mean': statistics.mean(values),
        'std': statistics.stdev(values) if len(values) > 1 else 0,
        'min': min(values),
        'max': max(values),
        'count': len(values)
    }

print("=" * 80)
print("实验结果分析报告: Seeds 2025-2027")
print("=" * 80)
print()

# 1. Overall Statistics by Solver
print("1. 各求解器总体表现统计")
print("-" * 80)
solver_improvements = defaultdict(list)
for row in all_data:
    solver_improvements[row['Solver']].append(row['Improvement'])

solver_stats = {}
for solver, improvements in solver_improvements.items():
    solver_stats[solver] = get_stats(improvements)

# Sort by mean
sorted_solvers = sorted(solver_stats.items(), key=lambda x: x[1]['mean'], reverse=True)

print(f"{'Solver':<40} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Count':<10}")
print("-" * 80)
for solver, stats in sorted_solvers:
    print(f"{solver:<40} {stats['mean']:>8.2f}% {stats['std']:>8.2f}% {stats['min']:>8.2f}% {stats['max']:>8.2f}% {stats['count']:>10}")
print()

# 2. Performance by Scale and Solver
print("2. 不同规模下各求解器的平均改进率")
print("-" * 80)
scale_solver_improvements = defaultdict(lambda: defaultdict(list))
for row in all_data:
    scale_solver_improvements[row['Scale']][row['Solver']].append(row['Improvement'])

for scale in ['Small', 'Medium', 'Large']:
    print(f"\n{scale} 规模:")
    for solver in ['Matheuristic + Q-learning', 'Matheuristic ALNS', 'Minimal ALNS']:
        improvements = scale_solver_improvements[scale][solver]
        if improvements:
            stats = get_stats(improvements)
            print(f"  {solver:<40} 平均={stats['mean']:>6.2f}%, 标准差={stats['std']:>6.2f}%, 数据点={stats['count']}")
print()

# 3. Detailed comparison: Q-learning vs Matheuristic ALNS
print("3. Q-learning vs Matheuristic ALNS 详细对比")
print("-" * 80)
q_learning_data = [row for row in all_data if row['Solver'] == 'Matheuristic + Q-learning']
matheuristic_data = [row for row in all_data if row['Solver'] == 'Matheuristic ALNS']

print("\nA. 按规模对比:")
for scale in ['Small', 'Medium', 'Large']:
    q_scale = [row['Improvement'] for row in q_learning_data if row['Scale'] == scale]
    m_scale = [row['Improvement'] for row in matheuristic_data if row['Scale'] == scale]

    q_stats = get_stats(q_scale)
    m_stats = get_stats(m_scale)

    print(f"\n{scale} 规模:")
    print(f"  Q-learning:        平均={q_stats['mean']:>6.2f}%, 标准差={q_stats['std']:>6.2f}%, 数据点={q_stats['count']}")
    print(f"  Matheuristic ALNS: 平均={m_stats['mean']:>6.2f}%, 标准差={m_stats['std']:>6.2f}%, 数据点={m_stats['count']}")
    if q_stats['count'] > 0 and m_stats['count'] > 0:
        diff = q_stats['mean'] - m_stats['mean']
        winner = '(Q-learning胜)' if diff > 0 else '(Matheuristic胜)' if diff < 0 else '(平)'
        print(f"  差异: {diff:+6.2f}% {winner}")

print("\n\nB. 按Seed对比:")
for seed in [2025, 2026, 2027]:
    q_seed = [row['Improvement'] for row in q_learning_data if row['Seed'] == seed]
    m_seed = [row['Improvement'] for row in matheuristic_data if row['Seed'] == seed]

    q_stats = get_stats(q_seed)
    m_stats = get_stats(m_seed)

    print(f"\nSeed {seed}:")
    print(f"  Q-learning:        平均={q_stats['mean']:>6.2f}%, 标准差={q_stats['std']:>6.2f}%, 数据点={q_stats['count']}")
    print(f"  Matheuristic ALNS: 平均={m_stats['mean']:>6.2f}%, 标准差={m_stats['std']:>6.2f}%, 数据点={m_stats['count']}")
    if q_stats['count'] > 0 and m_stats['count'] > 0:
        diff = q_stats['mean'] - m_stats['mean']
        winner = '(Q-learning胜)' if diff > 0 else '(Matheuristic胜)' if diff < 0 else '(平)'
        print(f"  差异: {diff:+6.2f}% {winner}")

# 4. Win rate analysis
print("\n\n4. 胜率分析 (逐对比较)")
print("-" * 80)

comparisons = []
for seed in [2025, 2026, 2027]:
    for scale in ['Small', 'Medium', 'Large']:
        q_rows = [row for row in q_learning_data if row['Seed'] == seed and row['Scale'] == scale]
        m_rows = [row for row in matheuristic_data if row['Seed'] == seed and row['Scale'] == scale]

        if q_rows and m_rows:
            q_imp = q_rows[0]['Improvement']
            m_imp = m_rows[0]['Improvement']
            winner = 'Q-learning' if q_imp > m_imp else 'Matheuristic ALNS' if m_imp > q_imp else 'Tie'
            diff = q_imp - m_imp
            comparisons.append({
                'Seed': seed,
                'Scale': scale,
                'Q-learning': q_imp,
                'Matheuristic ALNS': m_imp,
                'Winner': winner,
                'Diff': diff
            })

print(f"{'Seed':<6} {'Scale':<8} {'Q-learning':<12} {'Matheuristic':<12} {'Diff':<10} {'Winner':<20}")
print("-" * 80)
for comp in comparisons:
    print(f"{comp['Seed']:<6} {comp['Scale']:<8} {comp['Q-learning']:>10.2f}% {comp['Matheuristic ALNS']:>10.2f}% {comp['Diff']:>+9.2f}% {comp['Winner']:<20}")

q_wins = sum(1 for c in comparisons if c['Winner'] == 'Q-learning')
m_wins = sum(1 for c in comparisons if c['Winner'] == 'Matheuristic ALNS')
total = len(comparisons)

print(f"\n胜率统计:")
print(f"  Q-learning 胜: {q_wins}/{total} ({q_wins/total*100:.1f}%)")
print(f"  Matheuristic ALNS 胜: {m_wins}/{total} ({m_wins/total*100:.1f}%)")

# 5. Variance analysis
print("\n\n5. 方差和稳定性分析")
print("-" * 80)

print("\n各求解器在不同seeds间的标准差:")
for solver in ['Matheuristic + Q-learning', 'Matheuristic ALNS', 'Minimal ALNS']:
    solver_data = [row for row in all_data if row['Solver'] == solver]
    print(f"\n{solver}:")
    for scale in ['Small', 'Medium', 'Large']:
        scale_data = [row['Improvement'] for row in solver_data if row['Scale'] == scale]
        if scale_data:
            stats = get_stats(scale_data)
            cv = (stats['std'] / stats['mean'] * 100) if stats['mean'] > 0 else 0
            print(f"  {scale}: 平均={stats['mean']:>6.2f}%, 标准差={stats['std']:>6.2f}%, 变异系数={cv:>6.2f}%")

# 6. Missing data analysis
print("\n\n6. 缺失数据检查")
print("-" * 80)
print(f"总数据点: {len(all_data)}")
print(f"预期数据点: 3 seeds × 3 scales × 3 solvers = 27")
print(f"缺失: {27 - len(all_data)} 个数据点")

missing = []
for seed in [2025, 2026, 2027]:
    for scale in ['Small', 'Medium', 'Large']:
        for solver in ['Matheuristic ALNS', 'Minimal ALNS', 'Matheuristic + Q-learning']:
            found = any(row['Seed'] == seed and row['Scale'] == scale and row['Solver'] == solver for row in all_data)
            if not found:
                missing.append(f"Seed {seed}, {scale}, {solver}")

if missing:
    print("\n缺失的数据点:")
    for m in missing:
        print(f"  - {m}")

# 7. Recommendation
print("\n\n" + "=" * 80)
print("7. 关于10-seed敏感性测试的建议")
print("=" * 80)

q_improvements = [row['Improvement'] for row in q_learning_data]
m_improvements = [row['Improvement'] for row in matheuristic_data]

q_stats_overall = get_stats(q_improvements)
m_stats_overall = get_stats(m_improvements)

print(f"\n当前数据 (基于{len(q_improvements)}个Q-learning数据点, {len(m_improvements)}个Matheuristic数据点):")
print(f"  Q-learning平均改进率:        {q_stats_overall['mean']:>6.2f}% ± {q_stats_overall['std']:.2f}%")
print(f"  Matheuristic ALNS平均改进率: {m_stats_overall['mean']:>6.2f}% ± {m_stats_overall['std']:.2f}%")
print(f"  平均差异:                     {q_stats_overall['mean'] - m_stats_overall['mean']:>+6.2f}%")
print(f"  Q-learning变异系数:          {q_stats_overall['std']/q_stats_overall['mean']*100:>6.2f}%")
print(f"  Matheuristic变异系数:        {m_stats_overall['std']/m_stats_overall['mean']*100:>6.2f}%")

# Calculate coefficient of variation
cv_threshold = 20  # 20% coefficient of variation
seed_variance_high = False
high_variance_cases = []

for solver in ['Matheuristic + Q-learning', 'Matheuristic ALNS']:
    solver_data = [row for row in all_data if row['Solver'] == solver]
    for scale in ['Small', 'Medium', 'Large']:
        scale_data = [row['Improvement'] for row in solver_data if row['Scale'] == scale]
        if len(scale_data) > 1:
            stats = get_stats(scale_data)
            cv = stats['std'] / stats['mean'] * 100
            if cv > cv_threshold:
                seed_variance_high = True
                high_variance_cases.append(f"{solver} @ {scale} 规模, CV={cv:.2f}%")

if high_variance_cases:
    print("\n⚠️  高方差检测:")
    for case in high_variance_cases:
        print(f"  - {case}")

# Determine recommendation
mean_diff = abs(q_stats_overall['mean'] - m_stats_overall['mean'])
combined_std = (q_stats_overall['std'] + m_stats_overall['std']) / 2

print("\n\n决策分析:")
print("-" * 80)
print(f"1. 平均差异 ({mean_diff:.2f}%) vs 组合标准差 ({combined_std:.2f}%)")
print(f"   → 差异{'明显' if mean_diff > combined_std else '不明显'}")

print(f"\n2. 高方差情况: {'检测到{len(high_variance_cases)}处' if seed_variance_high else '无'}")

print(f"\n3. 胜率分布: Q-learning {q_wins}/{total} vs Matheuristic {m_wins}/{total}")
print(f"   → {'比较均衡' if abs(q_wins - m_wins) <= 2 else 'Q-learning占优' if q_wins > m_wins else 'Matheuristic占优'}")

print(f"\n4. 数据完整性: {len(all_data)}/27 数据点")
print(f"   → {'存在缺失' if len(all_data) < 27 else '完整'}")

# Final recommendation
recommend_10seeds = False
reasons = []

if seed_variance_high:
    recommend_10seeds = True
    reasons.append("存在高方差情况(CV>20%),需要更多样本评估稳定性")

if mean_diff < combined_std:
    recommend_10seeds = True
    reasons.append("两种方法差异不明显,需要更多样本确定统计显著性")

if len(all_data) < 27:
    recommend_10seeds = True
    reasons.append("当前数据存在缺失,需要补充完整")

if abs(q_wins - m_wins) <= 2 and total >= 6:
    recommend_10seeds = True
    reasons.append("胜率接近,需要更多样本判断优劣")

print("\n\n" + "=" * 80)
print("最终建议:")
print("=" * 80)
if recommend_10seeds:
    print("\n✅ 强烈建议执行10-seed敏感性测试\n")
    print("理由:")
    for i, reason in enumerate(reasons, 1):
        print(f"  {i}. {reason}")
    print("\n预期收益:")
    print("  - 获得更可靠的统计结果(样本量从3增至10)")
    print("  - 更准确地评估两种方法的稳定性和鲁棒性")
    print("  - 可进行严格的统计检验(t-test, Wilcoxon等)")
    print("  - 识别异常值和极端情况")
    print("  - 为论文/报告提供更有说服力的数据支撑")
else:
    print("\n🤔 10-seed测试可选,但不是必须\n")
    print("当前3-seed数据已经显示出较为明确的趋势")
    print("如果时间和计算资源允许,仍建议进行以增强结论可信度")

# Save detailed results
results_summary = {
    'overall_stats': {solver: stats for solver, stats in solver_stats.items()},
    'q_learning': {
        'mean': q_stats_overall['mean'],
        'std': q_stats_overall['std'],
        'count': q_stats_overall['count']
    },
    'matheuristic': {
        'mean': m_stats_overall['mean'],
        'std': m_stats_overall['std'],
        'count': m_stats_overall['count']
    },
    'win_rate': {
        'q_learning': f"{q_wins}/{total}",
        'matheuristic': f"{m_wins}/{total}"
    },
    'recommendation': 'RECOMMEND_10SEEDS' if recommend_10seeds else 'OPTIONAL',
    'reasons': reasons,
    'high_variance_cases': high_variance_cases
}

with open('/home/user/R3/seeds_2025_2027_analysis_summary.json', 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, indent=2, ensure_ascii=False)

print("\n\n详细分析结果已保存到: seeds_2025_2027_analysis_summary.json")
print("=" * 80)
