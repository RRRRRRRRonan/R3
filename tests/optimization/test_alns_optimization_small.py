"""
小规模场景ALNS优化效果测试
================================

场景：10任务 + 1充电站
目标：对比不同充电策略下的ALNS优化效果

输出：
- 初始解 vs 优化解的详细对比
- 不同充电策略的性能对比
- 优化过程的改进曲线
"""

import sys
sys.path.append('src')

from core.node import DepotNode, create_task_node_pair, ChargingNode, NodeType
from core.task import Task, TaskPool
from core.route import create_empty_route
from core.vehicle import create_vehicle
from physics.distance import DistanceMatrix
from physics.energy import EnergyConfig
from physics.time import TimeWindow, TimeWindowType
from planner.alns import MinimalALNS, CostParameters
from strategy.charging_strategies import (
    FullRechargeStrategy,
    PartialRechargeFixedStrategy,
    PartialRechargeMinimalStrategy
)
import random
import time as time_module

random.seed(42)


def create_small_scenario():
    """
    创建小规模测试场景

    规模：
    - 10个任务
    - 1个充电站
    - 仓库范围：1000m × 1000m (1km × 1km，更贴近test_strategy_comparison.py)
    """
    depot = DepotNode(coordinates=(0.0, 0.0))

    # 创建充电站 (ID必须遵循NodeIDHelper约定: 2*num_tasks + 1 = 21)
    charging_station = ChargingNode(
        node_id=21,  # 10任务 → 充电站ID从21开始
        coordinates=(500.0, 500.0),  # 中心位置
        node_type=NodeType.CHARGING
    )

    tasks = []
    coordinates = {0: depot.coordinates, 21: charging_station.coordinates}

    # 创建10个任务（带软时间窗）
    for i in range(1, 11):
        pickup_x = random.uniform(50, 950)  # 1km范围
        pickup_y = random.uniform(50, 950)
        delivery_x = random.uniform(50, 950)
        delivery_y = random.uniform(50, 950)

        # 软时间窗
        pickup_tw_start = i * 80
        pickup_tw_end = pickup_tw_start + 150
        delivery_tw_start = pickup_tw_end + 30
        delivery_tw_end = delivery_tw_start + 150

        pickup, delivery = create_task_node_pair(
            task_id=i,
            pickup_id=i*2-1,
            delivery_id=i*2,
            pickup_coords=(pickup_x, pickup_y),
            delivery_coords=(delivery_x, delivery_y),
            demand=15.0,
            pickup_time_window=TimeWindow(pickup_tw_start, pickup_tw_end, TimeWindowType.SOFT),
            delivery_time_window=TimeWindow(delivery_tw_start, delivery_tw_end, TimeWindowType.SOFT)
        )

        task = Task(task_id=i, pickup_node=pickup, delivery_node=delivery, demand=15.0)
        tasks.append(task)
        coordinates[pickup.node_id] = pickup.coordinates
        coordinates[delivery.node_id] = delivery.coordinates

    distance_matrix = DistanceMatrix(
        coordinates=coordinates,
        num_tasks=10,
        num_charging_stations=1
    )

    vehicle = create_vehicle(
        vehicle_id=1,
        capacity=150.0,
        battery_capacity=1.5,  # 1.5 kWh - 确保需要充电
        initial_battery=1.5
    )

    energy_config = EnergyConfig(
        consumption_rate=0.5,  # 0.5 kWh/km (实际使用单位，与test_strategy_comparison.py一致)
>>>>>>> f789f02c58034df850218a35783ed08ce0b29233
        charging_rate=3.0/3600
    )

    return depot, tasks, charging_station, distance_matrix, vehicle, energy_config


def analyze_route(route, alns, label=""):
    """分析路径的详细指标"""
    from core.node import NodeType

    # 计算总距离
    total_distance = 0.0
    for i in range(len(route.nodes) - 1):
        node1 = route.nodes[i]
        node2 = route.nodes[i + 1]
        dist = alns.distance.get_distance(node1.node_id, node2.node_id)
        total_distance += dist

    # 计算总充电量和充电站访问次数
    total_charging = 0.0
    num_charging_stops = 0
    for i in range(len(route.nodes)):
        node = route.nodes[i]
        if node.node_type == NodeType.CHARGING:
            num_charging_stops += 1
            # 充电量存储在节点的charge_amount属性中
            if hasattr(node, 'charge_amount'):
                total_charging += node.charge_amount

    # 计算总时间和延迟
    total_time = 0.0
    total_delay = 0.0
    current_time = 0.0
    vehicle_speed = 1.5  # m/s

    for i in range(len(route.nodes)):
        node = route.nodes[i]

        # 检查时间窗
        if hasattr(node, 'time_window') and node.time_window:
            tw = node.time_window
            if current_time > tw.latest:
                delay = current_time - tw.latest
                if not tw.is_hard():  # 只计算软时间窗的延迟
                    total_delay += delay

        # 移动到下一个节点
        if i < len(route.nodes) - 1:
            next_node = route.nodes[i + 1]
            dist = alns.distance.get_distance(node.node_id, next_node.node_id)
            travel_time = dist / vehicle_speed
            current_time += travel_time

            # 充电时间
            if next_node.node_type == NodeType.CHARGING:
                if hasattr(next_node, 'charge_amount'):
                    charging_amount = next_node.charge_amount
                    charging_time = charging_amount / alns.energy_config.charging_rate
                    current_time += charging_time

    total_time = current_time

    # 计算总成本
    total_cost = alns.evaluate_cost(route)

    metrics = {
        'total_distance': total_distance,
        'total_charging': total_charging,
        'total_time': total_time,
        'total_delay': total_delay,
        'num_charging_stops': num_charging_stops,
        'total_cost': total_cost
    }

    print(f"\n{label}:")
    print(f"  服务任务数: {len(route.get_served_tasks())}/10")
    print(f"  总距离: {metrics['total_distance']:.2f}m")
    print(f"  总充电量: {metrics['total_charging']:.2f}kWh")
    print(f"  总时间: {metrics['total_time']:.2f}s ({metrics['total_time']/60:.1f}分钟)")
    print(f"  总延迟: {metrics['total_delay']:.2f}s")
    print(f"  充电站访问: {metrics['num_charging_stops']}次")
    print(f"  总成本: {metrics['total_cost']:.2f}")

    return metrics


def test_optimization_with_strategy(strategy_name, strategy, depot, tasks, distance_matrix, vehicle, energy_config):
    """测试单个充电策略的优化效果"""

    print(f"\n{'='*70}")
    print(f"充电策略: {strategy_name}")
    print(f"{'='*70}")

    # 创建task pool
    task_pool = TaskPool()
    for task in tasks:
        task_pool.add_task(task)

    # 创建ALNS优化器
    alns = MinimalALNS(
        distance_matrix=distance_matrix,
        task_pool=task_pool,
        repair_mode='mixed',  # 使用混合模式
        cost_params=CostParameters(
            C_tr=1.0,
            C_ch=0.6,
            C_delay=2.0,
            C_time=0.1
        ),
        charging_strategy=strategy
    )
    alns.vehicle = vehicle
    alns.energy_config = energy_config

    # 步骤1: 创建初始解（使用greedy）
    print(f"\n步骤1: 创建初始解（Greedy插入）")
    route = create_empty_route(1, depot)
    removed_task_ids = [t.task_id for t in tasks]

    start_time = time_module.time()
    initial_route = alns.greedy_insertion(route, removed_task_ids)
    initial_time = time_module.time() - start_time

    print(f"  构建时间: {initial_time:.2f}秒")
    initial_metrics = analyze_route(initial_route, alns, "初始解")

    # 步骤2: ALNS优化
    print(f"\n步骤2: ALNS优化（50次迭代）")
    start_time = time_module.time()
    optimized_route = alns.optimize(initial_route, max_iterations=50)
    optimization_time = time_module.time() - start_time

    print(f"  优化时间: {optimization_time:.2f}秒")
    optimized_metrics = analyze_route(optimized_route, alns, "优化后")

    # 步骤3: 改进分析
    print(f"\n步骤3: 优化改进分析")
    distance_improvement = initial_metrics['total_distance'] - optimized_metrics['total_distance']
    cost_improvement = initial_metrics['total_cost'] - optimized_metrics['total_cost']

    distance_pct = (distance_improvement/initial_metrics['total_distance']*100) if initial_metrics['total_distance'] > 0 else 0
    cost_pct = (cost_improvement/initial_metrics['total_cost']*100) if initial_metrics['total_cost'] > 0 else 0

    print(f"  距离改进: {distance_improvement:.2f}m ({distance_pct:.1f}%)")
    print(f"  成本改进: {cost_improvement:.2f} ({cost_pct:.1f}%)")
    print(f"  总耗时: {initial_time + optimization_time:.2f}秒")

    return {
        'strategy': strategy_name,
        'initial_cost': initial_metrics['total_cost'],
        'optimized_cost': optimized_metrics['total_cost'],
        'improvement': cost_improvement,
        'improvement_pct': cost_improvement/initial_metrics['total_cost']*100,
        'total_time': initial_time + optimization_time,
        'initial_distance': initial_metrics['total_distance'],
        'optimized_distance': optimized_metrics['total_distance'],
        'initial_charging': initial_metrics['total_charging'],
        'optimized_charging': optimized_metrics['total_charging']
    }


def main():
    """主测试流程"""

    print("="*70)
    print("小规模场景ALNS优化效果测试")
    print("="*70)

    # 创建场景
    depot, tasks, charging_station, distance_matrix, vehicle, energy_config = create_small_scenario()

    print("\n场景配置:")
    print("  任务数: 10个")
    print("  充电站: 1个")
    print("  仓库范围: 1000m × 1000m")
    print(f"  车辆容量: {vehicle.capacity}kg")
    print(f"  电池容量: {vehicle.battery_capacity}kWh")
    print(f"  能耗率: {energy_config.consumption_rate}kWh/km")

    # 测试三种充电策略
    strategies = [
        ("完全充电 (FR)", FullRechargeStrategy()),
        ("固定50% (PR-Fixed)", PartialRechargeFixedStrategy(charge_ratio=0.5)),
        ("最小充电 (PR-Minimal)", PartialRechargeMinimalStrategy(safety_margin=0.1))
    ]

    results = []

    for strategy_name, strategy in strategies:
        result = test_optimization_with_strategy(
            strategy_name, strategy, depot, tasks,
            distance_matrix, vehicle, energy_config
        )
        results.append(result)

    # 最终对比
    print(f"\n{'='*70}")
    print("最终对比总结")
    print(f"{'='*70}")

    print(f"\n{'策略':<20} {'初始成本':<12} {'优化成本':<12} {'改进':<12} {'改进率':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['strategy']:<20} {r['initial_cost']:<12.2f} {r['optimized_cost']:<12.2f} "
              f"{r['improvement']:<12.2f} {r['improvement_pct']:<10.1f}%")

    print(f"\n{'='*70}")
    print("详细指标对比")
    print(f"{'='*70}")

    print(f"\n距离对比:")
    for r in results:
        distance_change = r['initial_distance'] - r['optimized_distance']
        print(f"  {r['strategy']:<20} {r['initial_distance']:>8.1f}m → {r['optimized_distance']:>8.1f}m "
              f"({distance_change:+.1f}m)")

    print(f"\n充电量对比:")
    for r in results:
        charging_change = r['initial_charging'] - r['optimized_charging']
        print(f"  {r['strategy']:<20} {r['initial_charging']:>8.2f}kWh → {r['optimized_charging']:>8.2f}kWh "
              f"({charging_change:+.2f}kWh)")

    # 推荐策略
    best_result = min(results, key=lambda x: x['optimized_cost'])
    print(f"\n推荐策略: {best_result['strategy']}")
    print(f"  最优成本: {best_result['optimized_cost']:.2f}")
    print(f"  总改进: {best_result['improvement']:.2f} ({best_result['improvement_pct']:.1f}%)")

    print(f"\n{'='*70}")
    print("✓ 小规模优化测试完成")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
