#!/usr/bin/env python3
"""
Comprehensive Analysis of 10-Seed Experiment Results (2025-2034)
Comparing: Matheuristic ALNS, Minimal ALNS, and Matheuristic + Q-learning
"""

import json
import statistics
from collections import defaultdict
from typing import Dict, List

# Complete data from seeds 2025-2034
seeds_data = {
    2025: {
        "small": {"minimal": 0.08708265600428591, "matheuristic": 0.28280080346089326, "q_learning": 0.3955982542270056},
        "medium": {"minimal": 0.12189234296414962, "matheuristic": 0.41925492448738616, "q_learning": 0.40285299572454286},
        "large": {"minimal": 0.07082740616394431, "matheuristic": 0.24295818356760468, "q_learning": 0.33206995218640145}
    },
    2026: {
        "small": {"minimal": 0.08191631999043075, "matheuristic": 0.22865661021711944, "q_learning": 0.33754276540635963},
        "medium": {"minimal": 0.08583207664258699, "matheuristic": 0.3480610974142054, "q_learning": 0.5356299915501123},
        "large": {"minimal": 0.08947587113134817, "matheuristic": 0.2992136250625295, "q_learning": 0.38307825368478793}
    },
    2027: {
        "small": {"minimal": 0.08509143104489574, "matheuristic": 0.35917032670090593, "q_learning": 0.4134175475664488},
        "medium": {"minimal": 0.10365073367535398, "matheuristic": 0.4831086681045907, "q_learning": 0.17010096165604163},
        "large": {"minimal": 0.09504724486044848, "matheuristic": 0.3520032775124901, "q_learning": 0.3611295244665938}
    },
    2028: {
        "small": {"minimal": 0.13429662482521842, "matheuristic": 0.32638437727762803, "q_learning": 0.5726364150154785},
        "medium": {"minimal": 0.08065887143292771, "matheuristic": 0.3935753041245572, "q_learning": 0.4193555365217615},
        "large": {"minimal": 0.08645622432330649, "matheuristic": 0.36056086330344517, "q_learning": 0.32711960630082243}
    },
    2029: {
        "small": {"minimal": 0.07546675140762671, "matheuristic": 0.3790324799183929, "q_learning": 0.32318034526679407},
        "medium": {"minimal": 0.1268064307372114, "matheuristic": 0.4270290046828811, "q_learning": 0.37216421900414015},
        "large": {"minimal": 0.09685760084596205, "matheuristic": 0.3228775825392727, "q_learning": 0.27106918548804515}
    },
    2030: {
        "small": {"minimal": 0.09575727162952878, "matheuristic": 0.2874264513449585, "q_learning": 0.3722966887288654},
        "medium": {"minimal": 0.09050074637370056, "matheuristic": 0.416338525321018, "q_learning": 0.4286315859648916},
        "large": {"minimal": 0.07528422195684081, "matheuristic": 0.028815140283150183, "q_learning": 0.22637613027091974}
    },
    2031: {
        "small": {"minimal": 0.09569494708558336, "matheuristic": 0.03343700913897571, "q_learning": 0.404894063215317},
        "medium": {"minimal": 0.08065887143292771, "matheuristic": 0.538701307082421, "q_learning": 0.4011910924692934},
        "large": {"minimal": 0.09297248063319732, "matheuristic": 0.28340806253391615, "q_learning": 0.08339176778473952}
    },
    2032: {
        "small": {"minimal": 0.11320988409915532, "matheuristic": 0.32549386763187116, "q_learning": 0.4765789905924246},
        "medium": {"minimal": 0.08771577191451198, "matheuristic": 0.37126618350252344, "q_learning": 0.47858429255982093},
        "large": {"minimal": 0.07081540766148316, "matheuristic": 0.2627176722201592, "q_learning": 0.2269993408011763}
    },
    2033: {
        "small": {"minimal": 0.08514652677371919, "matheuristic": 0.33718078984088223, "q_learning": 0.3608806533506343},
        "medium": {"minimal": 0.0859345529402963, "matheuristic": 0.3967652813620645, "q_learning": 0.4696042031218793},
        "large": {"minimal": 0.05735229289990987, "matheuristic": 0.02942348670729533, "q_learning": 0.2584911604326036}
    },
    2034: {
        "small": {"minimal": 0.08253552996420334, "matheuristic": 0.48065762818304714, "q_learning": 0.44150634050650883},
        "medium": {"minimal": 0.09879549001051105, "matheuristic": 0.39408363676858893, "q_learning": 0.352125419657083},
        "large": {"minimal": 0.0794012263160924, "matheuristic": 0.351560123295822, "q_learning": 0.3035276130209655}
    }
}

def to_percentage(ratio):
    """Convert ratio to percentage"""
    return ratio * 100

def get_stats(values):
    """Calculate mean, std, and other statistics"""
    if not values:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0, 'median': 0}
    return {
        'mean': statistics.mean(values),
        'std': statistics.stdev(values) if len(values) > 1 else 0,
        'min': min(values),
        'max': max(values),
        'median': statistics.median(values),
        'count': len(values)
    }

# Organize data by method and scale
data_by_method_scale = defaultdict(lambda: defaultdict(list))
data_by_method = defaultdict(list)

for seed, scales in seeds_data.items():
    for scale, methods in scales.items():
        for method, improvement in methods.items():
            data_by_method_scale[method][scale].append(to_percentage(improvement))
            data_by_method[method].append(to_percentage(improvement))

print("=" * 100)
print("10-SEED EXPERIMENT RESULTS ANALYSIS (Seeds 2025-2034)")
print("=" * 100)
print()

# 1. Overall Performance Summary
print("1. OVERALL PERFORMANCE SUMMARY")
print("-" * 100)
print(f"{'Method':<30} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Median':<10} {'Count':<10}")
print("-" * 100)

overall_stats = {}
for method in ['q_learning', 'matheuristic', 'minimal']:
    stats = get_stats(data_by_method[method])
    overall_stats[method] = stats
    method_name = {
        'q_learning': 'Q-learning',
        'matheuristic': 'Matheuristic ALNS',
        'minimal': 'Minimal ALNS'
    }[method]
    print(f"{method_name:<30} {stats['mean']:>8.2f}% {stats['std']:>8.2f}% {stats['min']:>8.2f}% "
          f"{stats['max']:>8.2f}% {stats['median']:>8.2f}% {stats['count']:>10}")

print()

# 2. Performance by Scale
print("2. PERFORMANCE BY SCALE")
print("-" * 100)

for scale in ['small', 'medium', 'large']:
    print(f"\n{scale.upper()} Scale:")
    print(f"{'Method':<30} {'Mean':<10} {'Std':<10} {'CV':<10} {'Min':<10} {'Max':<10}")
    print("-" * 100)

    scale_stats = {}
    for method in ['q_learning', 'matheuristic', 'minimal']:
        stats = get_stats(data_by_method_scale[method][scale])
        scale_stats[method] = stats
        cv = (stats['std'] / stats['mean'] * 100) if stats['mean'] > 0 else 0
        method_name = {
            'q_learning': 'Q-learning',
            'matheuristic': 'Matheuristic ALNS',
            'minimal': 'Minimal ALNS'
        }[method]
        print(f"{method_name:<30} {stats['mean']:>8.2f}% {stats['std']:>8.2f}% {cv:>8.2f}% "
              f"{stats['min']:>8.2f}% {stats['max']:>8.2f}%")

    # Compare Q-learning vs Matheuristic
    q_mean = scale_stats['q_learning']['mean']
    m_mean = scale_stats['matheuristic']['mean']
    diff = q_mean - m_mean
    winner = "Q-learning" if diff > 0 else "Matheuristic" if diff < 0 else "Tie"
    print(f"\n  → Difference: Q-learning {diff:+.2f}% ({winner} wins)")

print()

# 3. Head-to-head comparison
print("3. HEAD-TO-HEAD COMPARISON (Q-learning vs Matheuristic)")
print("-" * 100)

wins = {'q_learning': 0, 'matheuristic': 0, 'tie': 0}
comparisons = []

for seed in sorted(seeds_data.keys()):
    for scale in ['small', 'medium', 'large']:
        q_val = to_percentage(seeds_data[seed][scale]['q_learning'])
        m_val = to_percentage(seeds_data[seed][scale]['matheuristic'])
        diff = q_val - m_val

        if abs(diff) < 0.01:
            winner = 'tie'
        elif diff > 0:
            winner = 'q_learning'
        else:
            winner = 'matheuristic'

        wins[winner] += 1
        comparisons.append({
            'seed': seed,
            'scale': scale,
            'q': q_val,
            'm': m_val,
            'diff': diff,
            'winner': winner
        })

print(f"{'Seed':<8} {'Scale':<8} {'Q-learning':<12} {'Matheuristic':<12} {'Diff':<12} {'Winner':<15}")
print("-" * 100)

for comp in comparisons:
    winner_display = {'q_learning': '🟢 Q-learning', 'matheuristic': '🔴 Matheuristic', 'tie': '🟡 Tie'}[comp['winner']]
    print(f"{comp['seed']:<8} {comp['scale']:<8} {comp['q']:>10.2f}% {comp['m']:>10.2f}% "
          f"{comp['diff']:>+10.2f}% {winner_display:<15}")

total_comparisons = len(comparisons)
print()
print(f"Win Rate Summary:")
print(f"  Q-learning wins:     {wins['q_learning']}/{total_comparisons} ({wins['q_learning']/total_comparisons*100:.1f}%)")
print(f"  Matheuristic wins:   {wins['matheuristic']}/{total_comparisons} ({wins['matheuristic']/total_comparisons*100:.1f}%)")
print(f"  Ties:                {wins['tie']}/{total_comparisons} ({wins['tie']/total_comparisons*100:.1f}%)")

print()

# 4. Statistical Significance Test (paired t-test approximation)
print("4. STATISTICAL SIGNIFICANCE ANALYSIS")
print("-" * 100)

# Collect paired data
q_values = []
m_values = []
for seed in sorted(seeds_data.keys()):
    for scale in ['small', 'medium', 'large']:
        q_values.append(to_percentage(seeds_data[seed][scale]['q_learning']))
        m_values.append(to_percentage(seeds_data[seed][scale]['matheuristic']))

# Calculate differences
differences = [q - m for q, m in zip(q_values, m_values)]
diff_stats = get_stats(differences)

print(f"Paired differences (Q-learning - Matheuristic):")
print(f"  Mean difference:     {diff_stats['mean']:+.2f}%")
print(f"  Std deviation:       {diff_stats['std']:.2f}%")
print(f"  95% CI (approx):     [{diff_stats['mean'] - 1.96*diff_stats['std']/len(differences)**0.5:.2f}%, "
      f"{diff_stats['mean'] + 1.96*diff_stats['std']/len(differences)**0.5:.2f}%]")

# Calculate t-statistic
if diff_stats['std'] > 0:
    t_stat = diff_stats['mean'] / (diff_stats['std'] / len(differences)**0.5)
    print(f"  t-statistic:         {t_stat:.3f}")
    print(f"  Degrees of freedom:  {len(differences) - 1}")

    # Critical value for α=0.05, df=29 is approximately 2.045
    if abs(t_stat) > 2.045:
        print(f"  ✅ Statistically significant at α=0.05 (|t| = {abs(t_stat):.3f} > 2.045)")
    else:
        print(f"  ❌ NOT statistically significant at α=0.05 (|t| = {abs(t_stat):.3f} < 2.045)")

print()

# 5. Problematic Cases Analysis
print("5. PROBLEMATIC CASES ANALYSIS")
print("-" * 100)

# Find worst Q-learning performances
q_worst_cases = []
for seed in sorted(seeds_data.keys()):
    for scale in ['small', 'medium', 'large']:
        q_val = to_percentage(seeds_data[seed][scale]['q_learning'])
        m_val = to_percentage(seeds_data[seed][scale]['matheuristic'])
        if q_val < m_val - 5:  # Q-learning loses by >5%
            q_worst_cases.append({
                'seed': seed,
                'scale': scale,
                'q': q_val,
                'm': m_val,
                'gap': m_val - q_val
            })

q_worst_cases.sort(key=lambda x: x['gap'], reverse=True)

print(f"Cases where Q-learning underperforms Matheuristic by >5%:")
print(f"{'Seed':<8} {'Scale':<8} {'Q-learning':<12} {'Matheuristic':<12} {'Gap':<12}")
print("-" * 100)
for case in q_worst_cases:
    print(f"{case['seed']:<8} {case['scale']:<8} {case['q']:>10.2f}% {case['m']:>10.2f}% {case['gap']:>10.2f}%")

if not q_worst_cases:
    print("  No cases found ✅")

print()

# Find best Q-learning performances
q_best_cases = []
for seed in sorted(seeds_data.keys()):
    for scale in ['small', 'medium', 'large']:
        q_val = to_percentage(seeds_data[seed][scale]['q_learning'])
        m_val = to_percentage(seeds_data[seed][scale]['matheuristic'])
        if q_val > m_val + 10:  # Q-learning wins by >10%
            q_best_cases.append({
                'seed': seed,
                'scale': scale,
                'q': q_val,
                'm': m_val,
                'advantage': q_val - m_val
            })

q_best_cases.sort(key=lambda x: x['advantage'], reverse=True)

print(f"\nCases where Q-learning outperforms Matheuristic by >10%:")
print(f"{'Seed':<8} {'Scale':<8} {'Q-learning':<12} {'Matheuristic':<12} {'Advantage':<12}")
print("-" * 100)
for case in q_best_cases:
    print(f"{case['seed']:<8} {case['scale']:<8} {case['q']:>10.2f}% {case['m']:>10.2f}% {case['advantage']:>+10.2f}%")

print()

# 6. Variance Analysis
print("6. VARIANCE AND STABILITY ANALYSIS")
print("-" * 100)

print(f"\n{'Method':<30} {'Scale':<10} {'Mean':<10} {'Std':<10} {'CV':<10} {'Stability':<15}")
print("-" * 100)

for method in ['q_learning', 'matheuristic', 'minimal']:
    method_name = {
        'q_learning': 'Q-learning',
        'matheuristic': 'Matheuristic ALNS',
        'minimal': 'Minimal ALNS'
    }[method]

    for scale in ['small', 'medium', 'large']:
        stats = get_stats(data_by_method_scale[method][scale])
        cv = (stats['std'] / stats['mean'] * 100) if stats['mean'] > 0 else 0

        if cv < 15:
            stability = "✅ Stable"
        elif cv < 30:
            stability = "⚠️ Moderate"
        else:
            stability = "❌ Unstable"

        print(f"{method_name:<30} {scale:<10} {stats['mean']:>8.2f}% {stats['std']:>8.2f}% "
              f"{cv:>8.2f}% {stability:<15}")

print()

# 7. Key Findings Summary
print("7. KEY FINDINGS SUMMARY")
print("=" * 100)

q_mean = overall_stats['q_learning']['mean']
m_mean = overall_stats['matheuristic']['mean']
overall_diff = q_mean - m_mean

print(f"\n✅ STRENGTHS:")
if overall_diff > 0:
    print(f"  • Q-learning achieves {overall_diff:.2f}% higher improvement on average")
print(f"  • Q-learning wins {wins['q_learning']}/{total_comparisons} comparisons ({wins['q_learning']/total_comparisons*100:.1f}%)")

# Check which scales Q-learning dominates
for scale in ['small', 'medium', 'large']:
    q_scale_mean = get_stats(data_by_method_scale['q_learning'][scale])['mean']
    m_scale_mean = get_stats(data_by_method_scale['matheuristic'][scale])['mean']
    if q_scale_mean > m_scale_mean:
        print(f"  • Dominates in {scale} scale ({q_scale_mean:.2f}% vs {m_scale_mean:.2f}%)")

print(f"\n❌ WEAKNESSES:")
if len(q_worst_cases) > 0:
    print(f"  • {len(q_worst_cases)} cases where Q-learning loses by >5%")
    worst = max(q_worst_cases, key=lambda x: x['gap'])
    print(f"  • Worst case: Seed {worst['seed']} {worst['scale']} (gap: {worst['gap']:.2f}%)")

# Check variance issues
for scale in ['small', 'medium', 'large']:
    stats = get_stats(data_by_method_scale['q_learning'][scale])
    cv = (stats['std'] / stats['mean'] * 100)
    if cv > 30:
        print(f"  • High variance in {scale} scale (CV={cv:.2f}%)")

print()

# 8. Recommendations
print("8. RECOMMENDATIONS FOR JOURNAL PUBLICATION")
print("=" * 100)

print("\n🎯 PRIORITY ACTIONS:")

# Check if statistically significant
if diff_stats['std'] > 0:
    t_stat = abs(diff_stats['mean'] / (diff_stats['std'] / len(differences)**0.5))
    if t_stat < 2.045:
        print("  1. ⚠️  Results are NOT statistically significant - need to:")
        print("      • Investigate and fix problematic cases")
        print("      • Consider algorithm improvements")
        print("      • Add more diverse instances (Solomon benchmark)")

# Check worst cases
if len(q_worst_cases) > 0:
    print(f"  2. 🔧 Fix {len(q_worst_cases)} underperforming cases:")
    for i, case in enumerate(q_worst_cases[:3], 1):
        print(f"      {i}. Seed {case['seed']} {case['scale']}: Q={case['q']:.1f}% < M={case['m']:.1f}% (gap: {case['gap']:.1f}%)")

# Check variance
high_var_scales = []
for scale in ['small', 'medium', 'large']:
    stats = get_stats(data_by_method_scale['q_learning'][scale])
    cv = (stats['std'] / stats['mean'] * 100)
    if cv > 25:
        high_var_scales.append(f"{scale} (CV={cv:.1f}%)")

if high_var_scales:
    print(f"  3. 📊 Reduce variance in: {', '.join(high_var_scales)}")

print("\n📚 FOR PUBLICATION:")
print("  • Current results show Q-learning wins 60% of cases")
print("  • But performance is inconsistent (see problematic cases)")
print("  • Recommended next steps:")
print("    1. Debug and fix underperforming cases")
print("    2. Run ablation studies (Q-learning vs Random vs Roulette)")
print("    3. Add Solomon benchmark for credibility")
print("    4. Ensure statistical significance (t > 2.045)")

# Save summary
summary = {
    'overall': {k: v for k, v in overall_stats.items()},
    'by_scale': {
        scale: {method: get_stats(data_by_method_scale[method][scale]) for method in ['q_learning', 'matheuristic', 'minimal']}
        for scale in ['small', 'medium', 'large']
    },
    'win_rate': {
        'q_learning': f"{wins['q_learning']}/{total_comparisons}",
        'matheuristic': f"{wins['matheuristic']}/{total_comparisons}"
    },
    'statistical_test': {
        'mean_diff': diff_stats['mean'],
        'std_diff': diff_stats['std'],
        't_statistic': t_stat if diff_stats['std'] > 0 else 0,
        'significant': t_stat > 2.045 if diff_stats['std'] > 0 else False
    },
    'problematic_cases': q_worst_cases,
    'best_cases': q_best_cases
}

with open('/home/user/R3/analysis_10seeds_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=float)

print("\n\n✅ Analysis complete! Summary saved to: analysis_10seeds_summary.json")
print("=" * 100)
