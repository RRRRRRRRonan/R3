#!/usr/bin/env python3
"""
诊断脚本：对比成功与失败案例，找出Q-learning失败的根本原因

专门分析：
- Seed 2027 Medium (失败: 17.01%)  vs  Seed 2026 Medium (成功: 53.56%)
- Seed 2031 Large (失败: 8.34%)    vs  Seed 2026 Large (成功: 38.31%)
"""

import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.instance_generator import generate_test_instance
from planner.alns import MinimalALNS
from tests.optimization.common import run_q_learning_trial
import random


def diagnose_single_case(seed: int, scale: str, iterations: int = 50):
    """运行单个案例并收集诊断信息"""

    print(f"\n{'='*80}")
    print(f"诊断案例: Seed {seed}, Scale {scale}")
    print(f"{'='*80}\n")

    # 设置随机种子
    random.seed(seed)

    # 生成问题实例
    scale_config = {
        'small': {'num_tasks': 10, 'num_stations': 1},
        'medium': {'num_tasks': 20, 'num_stations': 2},
        'large': {'num_tasks': 30, 'num_stations': 3}
    }

    config = scale_config[scale]
    distance_matrix, task_pool, charging_stations, vehicle_template = generate_test_instance(
        num_tasks=config['num_tasks'],
        num_charging_stations=config['num_stations'],
        seed=seed
    )

    # 运行baseline
    baseline_alns = MinimalALNS(
        distance_matrix=distance_matrix,
        task_pool=task_pool,
        charging_stations=charging_stations,
        repair_mode='greedy',
        use_adaptive=False
    )
    baseline = baseline_alns.generate_initial_solution(vehicle_template)
    baseline_cost = baseline_alns.evaluate(baseline)

    print(f"Baseline cost: {baseline_cost:.2f}")

    # 创建ALNS实例，启用详细日志
    alns = MinimalALNS(
        distance_matrix=distance_matrix,
        task_pool=task_pool,
        charging_stations=charging_stations,
        repair_mode='adaptive',
        use_adaptive=True,
        verbose=True  # 启用详细输出
    )

    # 收集诊断信息
    diagnostics = {
        'seed': seed,
        'scale': scale,
        'baseline_cost': baseline_cost,
        'iterations': iterations,
        'operator_selections': {},
        'state_transitions': [],
        'q_values_samples': [],
        'improvements': []
    }

    # 运行优化并监控
    print(f"\n开始优化 (迭代次数: {iterations})...")
    print("-" * 80)

    try:
        # 运行Q-learning优化
        optimised = alns.optimize(baseline, max_iterations=iterations)
        optimised_cost = alns.evaluate(optimised)
        improvement = (baseline_cost - optimised_cost) / baseline_cost * 100

        diagnostics['optimised_cost'] = optimised_cost
        diagnostics['improvement'] = improvement

        print(f"\n优化完成!")
        print(f"Optimised cost: {optimised_cost:.2f}")
        print(f"Improvement: {improvement:.2f}%")

        # 如果ALNS有Q-learning agent，获取其统计信息
        if hasattr(alns, '_q_agent') and alns._q_agent is not None:
            q_agent = alns._q_agent

            # 获取算子使用统计
            if hasattr(q_agent, 'action_counts'):
                diagnostics['operator_selections'] = dict(q_agent.action_counts)

                print(f"\n算子使用统计:")
                print("-" * 80)
                total_selections = sum(q_agent.action_counts.values())
                for action, count in sorted(q_agent.action_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = count / total_selections * 100
                    print(f"  {action}: {count} 次 ({percentage:.1f}%)")

            # 获取Q-table样本
            if hasattr(q_agent, 'q_table'):
                q_table_sample = {}
                for state in ['normal', 'stuck', 'deep_stuck']:
                    if state in q_agent.q_table:
                        q_table_sample[state] = dict(q_agent.q_table[state])
                diagnostics['q_values_samples'] = q_table_sample

                print(f"\nQ-values样本 (当前状态):")
                print("-" * 80)
                for state, values in q_table_sample.items():
                    print(f"  State: {state}")
                    for action, q_val in sorted(values.items(), key=lambda x: x[1], reverse=True):
                        print(f"    {action}: {q_val:.3f}")

        # 检查状态转换历史（如果有记录）
        if hasattr(alns, 'state_history'):
            diagnostics['state_transitions'] = alns.state_history

            print(f"\n状态转换历史:")
            print("-" * 80)
            for i, state_change in enumerate(alns.state_history):
                print(f"  {i+1}. Iteration {state_change['iteration']}: {state_change['from']} -> {state_change['to']}")

    except Exception as e:
        print(f"\n❌ 优化过程中出错: {e}")
        import traceback
        traceback.print_exc()
        diagnostics['error'] = str(e)

    return diagnostics


def compare_cases(success_diag, failure_diag):
    """对比成功和失败案例，找出关键差异"""

    print(f"\n\n{'='*80}")
    print(f"对比分析: 成功 vs 失败")
    print(f"{'='*80}\n")

    print(f"成功案例: Seed {success_diag['seed']} {success_diag['scale']}")
    print(f"  改进率: {success_diag.get('improvement', 0):.2f}%")

    print(f"\n失败案例: Seed {failure_diag['seed']} {failure_diag['scale']}")
    print(f"  改进率: {failure_diag.get('improvement', 0):.2f}%")

    print(f"\n差距: {success_diag.get('improvement', 0) - failure_diag.get('improvement', 0):.2f}%")

    # 对比算子使用
    print(f"\n\n1. 算子使用对比")
    print("-" * 80)

    success_ops = success_diag.get('operator_selections', {})
    failure_ops = failure_diag.get('operator_selections', {})

    all_operators = set(success_ops.keys()) | set(failure_ops.keys())

    print(f"{'算子':<30} {'成功案例':<15} {'失败案例':<15} {'差异':<15}")
    print("-" * 80)

    for op in sorted(all_operators):
        success_count = success_ops.get(op, 0)
        failure_count = failure_ops.get(op, 0)

        success_total = sum(success_ops.values()) if success_ops else 1
        failure_total = sum(failure_ops.values()) if failure_ops else 1

        success_pct = success_count / success_total * 100
        failure_pct = failure_count / failure_total * 100

        diff = failure_pct - success_pct

        marker = ""
        if abs(diff) > 15:
            marker = "⚠️ 差异大"

        print(f"{op:<30} {success_pct:>6.1f}% ({success_count:>3}) {failure_pct:>6.1f}% ({failure_count:>3}) {diff:>+6.1f}% {marker}")

    # 对比Q-values
    print(f"\n\n2. Q-values对比")
    print("-" * 80)

    success_q = success_diag.get('q_values_samples', {})
    failure_q = failure_diag.get('q_values_samples', {})

    for state in ['normal', 'stuck', 'deep_stuck']:
        if state in success_q or state in failure_q:
            print(f"\nState: {state}")
            print(f"  {'算子':<30} {'成功案例 Q':<15} {'失败案例 Q':<15} {'差异':<15}")
            print("-" * 80)

            success_state_q = success_q.get(state, {})
            failure_state_q = failure_q.get(state, {})

            all_actions = set(success_state_q.keys()) | set(failure_state_q.keys())

            for action in sorted(all_actions):
                success_val = success_state_q.get(action, 0.0)
                failure_val = failure_state_q.get(action, 0.0)
                diff = failure_val - success_val

                marker = ""
                if abs(diff) > 1.0:
                    marker = "⚠️"

                print(f"  {action:<30} {success_val:>10.3f} {failure_val:>10.3f} {diff:>+10.3f} {marker}")

    # 诊断结论
    print(f"\n\n3. 诊断结论")
    print("=" * 80)

    # 检查算子失衡
    if failure_ops:
        failure_total = sum(failure_ops.values())
        for op, count in failure_ops.items():
            pct = count / failure_total * 100
            if pct > 60:
                print(f"⚠️  算子严重失衡: '{op}' 被使用了 {pct:.1f}%")

    # 检查Q-values异常
    for state, values in failure_q.items():
        for action, q_val in values.items():
            if abs(q_val) > 100:
                print(f"⚠️  Q-value异常: state={state}, action={action}, Q={q_val:.3f}")
            if q_val != q_val:  # NaN check
                print(f"❌ Q-value为NaN: state={state}, action={action}")

    # 对比状态转换
    success_transitions = success_diag.get('state_transitions', [])
    failure_transitions = failure_diag.get('state_transitions', [])

    if success_transitions and failure_transitions:
        print(f"\n状态转换对比:")
        print(f"  成功案例: {len(success_transitions)} 次转换")
        print(f"  失败案例: {len(failure_transitions)} 次转换")

        # 检查是否过早进入stuck
        if failure_transitions:
            first_stuck = next((t for t in failure_transitions if 'stuck' in t.get('to', '')), None)
            if first_stuck and first_stuck.get('iteration', 999) < 10:
                print(f"⚠️  过早进入stuck状态: 第{first_stuck['iteration']}次迭代")


def main():
    """主函数：运行诊断流程"""

    print("="*80)
    print("Q-learning失败案例诊断工具")
    print("="*80)

    # 诊断案例1: Seed 2027 Medium (失败)
    print("\n\n第一部分: 诊断失败案例")
    failure_diag_2027 = diagnose_single_case(seed=2027, scale='medium', iterations=50)

    # 诊断案例2: Seed 2026 Medium (成功)
    print("\n\n第二部分: 诊断成功案例（参照）")
    success_diag_2026 = diagnose_single_case(seed=2026, scale='medium', iterations=50)

    # 对比分析
    compare_cases(success_diag_2026, failure_diag_2027)

    # 保存诊断报告
    report = {
        'failure_case': failure_diag_2027,
        'success_case': success_diag_2026,
        'timestamp': str(__import__('datetime').datetime.now())
    }

    output_file = Path(__file__).parent.parent / 'diagnostic_report_seed2027.json'
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n\n诊断报告已保存到: {output_file}")

    # 生成修复建议
    print(f"\n\n{'='*80}")
    print(f"修复建议")
    print(f"{'='*80}")

    improvement_diff = success_diag_2026.get('improvement', 0) - failure_diag_2027.get('improvement', 0)

    if improvement_diff > 25:
        print(f"\n🔥 严重失败 (差距 {improvement_diff:.1f}%)")
        print("\n建议的修复方向:")
        print("  1. 检查Q-learning参数设置（learning_rate, epsilon_decay）")
        print("  2. 调整stagnation_threshold（可能对Medium规模不适配）")
        print("  3. 增加epsilon_min保持探索")
        print("  4. 检查reward_scaling是否合理")
        print("\n具体操作:")
        print("  在 src/planner/alns.py 中:")
        print("    - stagnation_threshold: 从当前值改为 15-20")
        print("  在 src/planner/q_learning.py 中:")
        print("    - epsilon_min: 从 0.01 改为 0.1")
        print("    - learning_rate: 尝试 0.1-0.3")

    print("\n\n下一步:")
    print("  1. 查看诊断报告: diagnostic_report_seed2027.json")
    print("  2. 根据建议调整参数")
    print("  3. 重新运行: python scripts/generate_alns_visualization.py --seed 2027")
    print("  4. 验证改进效果")


if __name__ == '__main__':
    main()
