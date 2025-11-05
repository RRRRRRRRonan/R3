#!/usr/bin/env python3
"""
Compare tuning branch results vs original main branch results
"""

import json
from pathlib import Path
import statistics

# Original main branch results (from user's initial data)
original_data = {
    2025: {
        "small": {"q_learning": 0.3955982542270056, "matheuristic": 0.28280080346089326},
        "medium": {"q_learning": 0.40285299572454286, "matheuristic": 0.41925492448738616},
        "large": {"q_learning": 0.33206995218640145, "matheuristic": 0.24295818356760468}
    },
    2026: {
        "small": {"q_learning": 0.33754276540635963, "matheuristic": 0.22865661021711944},
        "medium": {"q_learning": 0.5356299915501123, "matheuristic": 0.3480610974142054},
        "large": {"q_learning": 0.38307825368478793, "matheuristic": 0.2992136250625295}
    },
    2027: {
        "small": {"q_learning": 0.4134175475664488, "matheuristic": 0.35917032670090593},
        "medium": {"q_learning": 0.17010096165604163, "matheuristic": 0.4831086681045907},
        "large": {"q_learning": 0.3611295244665938, "matheuristic": 0.3520032775124901}
    },
    2028: {
        "small": {"q_learning": 0.5726364150154785, "matheuristic": 0.32638437727762803},
        "medium": {"q_learning": 0.4193555365217615, "matheuristic": 0.3935753041245572},
        "large": {"q_learning": 0.32711960630082243, "matheuristic": 0.36056086330344517}
    },
    2029: {
        "small": {"q_learning": 0.32318034526679407, "matheuristic": 0.3790324799183929},
        "medium": {"q_learning": 0.37216421900414015, "matheuristic": 0.4270290046828811},
        "large": {"q_learning": 0.27106918548804515, "matheuristic": 0.3228775825392727}
    },
    2030: {
        "small": {"q_learning": 0.3722966887288654, "matheuristic": 0.2874264513449585},
        "medium": {"q_learning": 0.4286315859648916, "matheuristic": 0.416338525321018},
        "large": {"q_learning": 0.22637613027091974, "matheuristic": 0.028815140283150183}
    },
    2031: {
        "small": {"q_learning": 0.404894063215317, "matheuristic": 0.03343700913897571},
        "medium": {"q_learning": 0.4011910924692934, "matheuristic": 0.538701307082421},
        "large": {"q_learning": 0.08339176778473952, "matheuristic": 0.28340806253391615}
    },
    2032: {
        "small": {"q_learning": 0.4765789905924246, "matheuristic": 0.32549386763187116},
        "medium": {"q_learning": 0.47858429255982093, "matheuristic": 0.37126618350252344},
        "large": {"q_learning": 0.2269993408011763, "matheuristic": 0.2627176722201592}
    },
    2033: {
        "small": {"q_learning": 0.3608806533506343, "matheuristic": 0.33718078984088223},
        "medium": {"q_learning": 0.4696042031218793, "matheuristic": 0.3967652813620645},
        "large": {"q_learning": 0.2584911604326036, "matheuristic": 0.02942348670729533}
    },
    2034: {
        "small": {"q_learning": 0.44150634050650883, "matheuristic": 0.48065762818304714},
        "medium": {"q_learning": 0.352125419657083, "matheuristic": 0.39408363676858893},
        "large": {"q_learning": 0.3035276130209655, "matheuristic": 0.351560123295822}
    }
}

# Load tuned results
project_root = Path(__file__).parent.parent
tuned_data = {}

for seed in range(2025, 2035):
    json_file = project_root / f"docs/data/alns_regression_results {seed}.json"
    if json_file.exists():
        with open(json_file) as f:
            data = json.load(f)
            tuned_data[seed] = {}
            for scale in ['small', 'medium', 'large']:
                tuned_data[seed][scale] = {
                    'q_learning': data[scale]['q_learning']['improvement_ratio'],
                    'matheuristic': data[scale]['matheuristic']['improvement_ratio']
                }

def to_pct(val):
    return val * 100

print("="*100)
print("参数调优前后对比分析：完整10-Seed结果")
print("="*100)
print()

# Collect all comparisons
comparisons = []
for seed in sorted(original_data.keys()):
    if seed not in tuned_data:
        continue

    for scale in ['small', 'medium', 'large']:
        orig_q = to_pct(original_data[seed][scale]['q_learning'])
        orig_m = to_pct(original_data[seed][scale]['matheuristic'])

        tuned_q = to_pct(tuned_data[seed][scale]['q_learning'])
        tuned_m = to_pct(tuned_data[seed][scale]['matheuristic'])

        comparisons.append({
            'seed': seed,
            'scale': scale,
            'orig_q': orig_q,
            'orig_m': orig_m,
            'tuned_q': tuned_q,
            'tuned_m': tuned_m,
            'q_change': tuned_q - orig_q,
            'm_change': tuned_m - orig_m,
            'orig_gap': orig_q - orig_m,
            'tuned_gap': tuned_q - tuned_m,
            'gap_change': (tuned_q - tuned_m) - (orig_q - orig_m)
        })

# Overall statistics
orig_q_all = [c['orig_q'] for c in comparisons]
tuned_q_all = [c['tuned_q'] for c in comparisons]
orig_m_all = [c['orig_m'] for c in comparisons]
tuned_m_all = [c['tuned_m'] for c in comparisons]

print("1. 整体统计对比")
print("-"*100)
print(f"{'方法':<20} {'调优前均值':<15} {'调优后均值':<15} {'变化':<15} {'评价':<15}")
print("-"*100)
print(f"{'Q-learning':<20} {statistics.mean(orig_q_all):>12.2f}% {statistics.mean(tuned_q_all):>12.2f}% "
      f"{statistics.mean(tuned_q_all)-statistics.mean(orig_q_all):>+12.2f}% {'⚠️ 几乎无变化':<15}")
print(f"{'Matheuristic':<20} {statistics.mean(orig_m_all):>12.2f}% {statistics.mean(tuned_m_all):>12.2f}% "
      f"{statistics.mean(tuned_m_all)-statistics.mean(orig_m_all):>+12.2f}% {'⚠️ 几乎无变化':<15}")
print()

orig_gap = statistics.mean(orig_q_all) - statistics.mean(orig_m_all)
tuned_gap = statistics.mean(tuned_q_all) - statistics.mean(tuned_m_all)
print(f"平均差异 (Q - M):")
print(f"  调优前: {orig_gap:+.2f}%")
print(f"  调优后: {tuned_gap:+.2f}%")
print(f"  变化:   {tuned_gap - orig_gap:+.2f}%")
print()

# Check problematic cases
print("\n2. 重点失败案例对比")
print("-"*100)
print(f"{'Seed':<6} {'Scale':<8} {'调优前Q':<12} {'调优后Q':<12} {'变化':<12} {'评价':<15}")
print("-"*100)

problem_cases = [
    (2027, 'medium'),
    (2031, 'large'),
    (2029, 'small'),
    (2029, 'medium'),
    (2029, 'large')
]

for seed, scale in problem_cases:
    comp = next((c for c in comparisons if c['seed']==seed and c['scale']==scale), None)
    if comp:
        change = comp['q_change']
        status = '✅ 改善' if change > 5 else '⚠️ 轻微改善' if change > 0 else '❌ 无改善/恶化'
        print(f"{seed:<6} {scale:<8} {comp['orig_q']:>10.2f}% {comp['tuned_q']:>10.2f}% "
              f"{change:>+10.2f}% {status:<15}")

print()

# By scale analysis
print("\n3. 按规模分析")
print("-"*100)
for scale in ['small', 'medium', 'large']:
    scale_comps = [c for c in comparisons if c['scale'] == scale]

    orig_q_scale = [c['orig_q'] for c in scale_comps]
    tuned_q_scale = [c['tuned_q'] for c in scale_comps]
    orig_m_scale = [c['orig_m'] for c in scale_comps]
    tuned_m_scale = [c['tuned_m'] for c in scale_comps]

    print(f"\n{scale.upper()} 规模:")
    print(f"  Q-learning:  {statistics.mean(orig_q_scale):.2f}% → {statistics.mean(tuned_q_scale):.2f}% "
          f"({statistics.mean(tuned_q_scale)-statistics.mean(orig_q_scale):+.2f}%)")
    print(f"  Matheuristic: {statistics.mean(orig_m_scale):.2f}% → {statistics.mean(tuned_m_scale):.2f}% "
          f"({statistics.mean(tuned_m_scale)-statistics.mean(orig_m_scale):+.2f}%)")
    print(f"  差异: {statistics.mean(orig_q_scale)-statistics.mean(orig_m_scale):+.2f}% → "
          f"{statistics.mean(tuned_q_scale)-statistics.mean(tuned_m_scale):+.2f}%")

print()

# Find biggest changes
print("\n4. 最大变化案例 (绝对值)")
print("-"*100)
comparisons_sorted = sorted(comparisons, key=lambda x: abs(x['q_change']), reverse=True)[:10]
print(f"{'Seed':<6} {'Scale':<8} {'调优前Q':<12} {'调优后Q':<12} {'变化':<12}")
print("-"*100)
for c in comparisons_sorted:
    print(f"{c['seed']:<6} {c['scale']:<8} {c['orig_q']:>10.2f}% {c['tuned_q']:>10.2f}% {c['q_change']:>+10.2f}%")

print()

# Variance analysis
print("\n5. 方差分析")
print("-"*100)
orig_q_std = statistics.stdev(orig_q_all)
tuned_q_std = statistics.stdev(tuned_q_all)
print(f"Q-learning标准差: {orig_q_std:.2f}% → {tuned_q_std:.2f}% ({tuned_q_std-orig_q_std:+.2f}%)")
print(f"变异系数: {orig_q_std/statistics.mean(orig_q_all)*100:.2f}% → "
      f"{tuned_q_std/statistics.mean(tuned_q_all)*100:.2f}%")

print("\n"+"="*100)
print("结论")
print("="*100)
print()

avg_q_change = statistics.mean(tuned_q_all) - statistics.mean(orig_q_all)
if abs(avg_q_change) < 1:
    print("❌ 参数调优几乎无效果")
    print(f"   Q-learning平均变化仅 {avg_q_change:+.2f}%")
    print()
    print("可能的原因：")
    print("  1. 参数调整幅度不够大")
    print("  2. 问题不在参数层面，而在算法设计")
    print("  3. Q-learning的状态空间设计不合理")
    print("  4. 奖励函数需要重新设计")
elif avg_q_change > 3:
    print("✅ 参数调优有显著效果")
    print(f"   Q-learning平均提升 {avg_q_change:+.2f}%")
else:
    print("⚠️ 参数调优有轻微效果")
    print(f"   Q-learning平均变化 {avg_q_change:+.2f}%")
    print("   但不足以达到统计显著性")
