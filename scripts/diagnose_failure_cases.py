#!/usr/bin/env python3
"""
简化诊断脚本：快速对比成功与失败案例

专门分析：
- Seed 2027 Medium (失败: 17.01%)  vs  Seed 2026 Medium (成功: 53.56%)
"""

import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.optimization.common import ScenarioConfig, build_scenario, run_minimal_trial
from tests.optimization.presets import get_scale_preset


def diagnose_case(seed: int, scale: str, iterations: int = 50):
    """运行单个案例并返回结果"""

    print(f"\n{'='*80}")
    print(f"运行案例: Seed {seed}, Scale {scale}")
    print(f"{'='*80}\n")

    # 获取规模预设
    preset = get_scale_preset(scale)

    # 创建场景配置
    config = ScenarioConfig(
        num_tasks=preset['num_tasks'],
        num_charging=preset['num_stations'],
        area_size=(100.0, 100.0),
        vehicle_capacity=150.0,
        battery_capacity=1.5,
        consumption_per_km=0.15,
        charging_rate=1.0,
        seed=seed
    )

    # 构建场景
    scenario = build_scenario(config)

    # 运行实验（use_adaptive=True会启用Q-learning）
    baseline_cost, optimised_cost = run_minimal_trial(
        scenario,
        iterations=iterations,
        seed=seed,
        repair_mode='adaptive'
    )

    improvement = (baseline_cost - optimised_cost) / baseline_cost * 100

    print(f"Baseline cost:  {baseline_cost:.2f}")
    print(f"Optimised cost: {optimised_cost:.2f}")
    print(f"Improvement:    {improvement:.2f}%")

    return {
        'seed': seed,
        'scale': scale,
        'baseline_cost': baseline_cost,
        'optimised_cost': optimised_cost,
        'improvement': improvement
    }


def main():
    """主函数：对比成功和失败案例"""

    print("="*80)
    print("Q-learning失败案例诊断")
    print("="*80)

    # 诊断失败案例
    print("\n\n【第一部分】诊断失败案例: Seed 2027 Medium")
    failure_result = diagnose_case(seed=2027, scale='medium', iterations=50)

    # 诊断成功案例作为参照
    print("\n\n【第二部分】诊断成功案例: Seed 2026 Medium（参照）")
    success_result = diagnose_case(seed=2026, scale='medium', iterations=50)

    # 对比分析
    print("\n\n" + "="*80)
    print("对比分析")
    print("="*80)

    print(f"\n成功案例 (Seed {success_result['seed']}):")
    print(f"  改进率: {success_result['improvement']:.2f}%")

    print(f"\n失败案例 (Seed {failure_result['seed']}):")
    print(f"  改进率: {failure_result['improvement']:.2f}%")

    gap = success_result['improvement'] - failure_result['improvement']
    print(f"\n差距: {gap:.2f}%")

    # 保存报告
    report = {
        'failure_case': failure_result,
        'success_case': success_result,
        'gap': gap
    }

    output_file = project_root / 'diagnostic_report_seed2027.json'
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n\n诊断报告已保存到: {output_file}")

    # 生成修复建议
    print(f"\n\n{'='*80}")
    print(f"诊断结论和修复建议")
    print(f"{'='*80}")

    if gap > 25:
        print(f"\n🔥 严重失败 (差距 {gap:.1f}%)")
        print("\n可能的原因:")
        print("  1. Q-learning参数对该seed不适配")
        print("  2. 算子选择陷入次优策略")
        print("  3. stagnation_threshold设置不当")
        print("  4. 探索率衰减过快")

        print("\n建议的修复方向:")
        print("  【修复A】增加探索率")
        print("    文件: src/planner/q_learning.py")
        print("    修改: epsilon_min = 0.1  (从0.01改为0.1)")

        print("\n  【修复B】放宽stagnation阈值")
        print("    文件: src/planner/alns.py")
        print("    修改: stagnation_threshold = 15  (Medium规模需要更宽松)")

        print("\n  【修复C】降低学习率")
        print("    文件: src/planner/q_learning.py")
        print("    修改: learning_rate = 0.1  (避免过度更新)")

        print("\n  【推荐】组合修复: A + B + C")

    elif gap > 10:
        print(f"\n⚠️  中等失败 (差距 {gap:.1f}%)")
        print("  建议先尝试修复A（增加探索率）")

    else:
        print(f"\n✅ 差距可接受 (差距 {gap:.1f}%)")
        print("  可能是随机性导致，建议重复运行验证")

    print("\n\n下一步操作:")
    print("  1. 根据建议修改参数")
    print("  2. 重新运行此脚本验证修复效果")
    print("  3. 如果修复成功，重新运行完整10-seed测试")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
