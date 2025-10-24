"""
大规模场景ALNS优化效果测试
================================

场景：50任务 + 3充电站
目标：展示ALNS在大规模场景下的可扩展性和优化能力

输出：
- 初始解 vs 优化解的详细对比
- 充电策略性能对比
- 优化时间和扩展性分析

注意：此测试可能需要10-30分钟
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


def create_large_scenario():
    """
    创建大规模测试场景

    规模：
    - 50个任务
    - 3个充电站
    - 仓库范围：150m × 150m
    """
    depot = DepotNode(coordinates=(0.0, 0.0))

    # 创建3个充电站（分布在不同区域）
    charging_stations = [
        ChargingNode(node_id=995, coordinates=(50.0, 50.0), node_type=NodeType.CHARGING),
        ChargingNode(node_id=996, coordinates=(75.0, 100.0), node_type=NodeType.CHARGING),
        ChargingNode(node_id=997, coordinates=(100.0, 75.0), node_type=NodeType.CHARGING)
    ]

    tasks = []
    coordinates = {0: depot.coordinates}
    for cs in charging_stations:
        coordinates[cs.node_id] = cs.coordinates

    # 创建50个任务
    for i in range(1, 51):
        pickup_x = random.uniform(10, 140)
        pickup_y = random.uniform(10, 140)
        delivery_x = random.uniform(10, 140)
        delivery_y = random.uniform(10, 140)

        # 软时间窗
        pickup_tw_start = i * 50
        pickup_tw_end = pickup_tw_start + 100
        delivery_tw_start = pickup_tw_end + 15
        delivery_tw_end = delivery_tw_start + 100

        pickup, delivery = create_task_node_pair(
            task_id=i,
            pickup_id=i*2-1,
            delivery_id=i*2,
            pickup_coords=(pickup_x, pickup_y),
            delivery_coords=(delivery_x, delivery_y),
            demand=25.0,
            pickup_time_window=TimeWindow(pickup_tw_start, pickup_tw_end, TimeWindowType.SOFT),
            delivery_time_window=TimeWindow(delivery_tw_start, delivery_tw_end, TimeWindowType.SOFT)
        )

        task = Task(task_id=i, pickup_node=pickup, delivery_node=delivery, demand=25.0)
        tasks.append(task)
        coordinates[pickup.node_id] = pickup.coordinates
        coordinates[delivery.node_id] = delivery.coordinates

    distance_matrix = DistanceMatrix(
        coordinates=coordinates,
        num_tasks=50,
        num_charging_stations=3
    )

    vehicle = create_vehicle(
        vehicle_id=1,
        capacity=250.0,
        battery_capacity=10.0,  # 降低电池容量，使其需要充电
        initial_battery=10.0
    )

    energy_config = EnergyConfig(
        consumption_rate=0.016,  # kWh/秒，确保需要充电
        charging_rate=7.0/3600
    )

    return depot, tasks, charging_stations, distance_matrix, vehicle, energy_config


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
            if hasattr(route, 'charging_amounts') and i in route.charging_amounts:
                total_charging += route.charging_amounts[i]

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
                if hasattr(route, 'charging_amounts') and (i+1) in route.charging_amounts:
                    charging_amount = route.charging_amounts[i+1]
                    charging_time = charging_amount / alns.energy_config.charging_rate
                    current_time += charging_time

    total_time = current_time

    # 计算总成本
    total_cost = alns.evaluate_cost(route)

    # 每公里成本
    cost_per_km = total_cost / (total_distance / 1000.0) if total_distance > 0 else 0.0

    metrics = {
        'total_distance': total_distance,
        'total_charging': total_charging,
        'total_time': total_time,
        'total_delay': total_delay,
        'num_charging_stops': num_charging_stops,
        'total_cost': total_cost,
        'cost_per_km': cost_per_km
    }

    print(f"\n{label}:")
    print(f"  服务任务数: {len(route.get_served_tasks())}/50")
    print(f"  总距离: {metrics['total_distance']:.2f}m")
    print(f"  总充电量: {metrics['total_charging']:.2f}kWh")
    print(f"  总时间: {metrics['total_time']:.2f}s ({metrics['total_time']/60:.1f}分钟)")
    print(f"  总延迟: {metrics['total_delay']:.2f}s")
    print(f"  充电站访问: {metrics['num_charging_stops']}次")
    print(f"  总成本: {metrics['total_cost']:.2f}")
    print(f"  每公里成本: {metrics['cost_per_km']:.2f}")

    return metrics


def test_optimization_with_strategy(strategy_name, strategy, depot, tasks, distance_matrix, vehicle, energy_config, max_iterations=50):
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
        repair_mode='mixed',
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

    # 步骤1: 创建初始解
    print(f"\n步骤1: 创建初始解（Greedy插入）")
    print(f"  提示：大规模场景构建可能需要5-10分钟...")
    route = create_empty_route(1, depot)
    removed_task_ids = [t.task_id for t in tasks]

    start_time = time_module.time()
    initial_route = alns.greedy_insertion(route, removed_task_ids)
    initial_time = time_module.time() - start_time

    print(f"  构建时间: {initial_time:.2f}秒 ({initial_time/60:.1f}分钟)")
    initial_metrics = analyze_route(initial_route, alns, "初始解")

    # 步骤2: ALNS优化
    print(f"\n步骤2: ALNS优化（{max_iterations}次迭代）")
    print(f"  提示：优化过程可能需要10-20分钟...")
    start_time = time_module.time()
    optimized_route = alns.optimize(initial_route, max_iterations=max_iterations)
    optimization_time = time_module.time() - start_time

    print(f"  优化时间: {optimization_time:.2f}秒 ({optimization_time/60:.1f}分钟)")
    optimized_metrics = analyze_route(optimized_route, alns, "优化后")

    # 步骤3: 改进分析
    print(f"\n步骤3: 优化改进分析")
    distance_improvement = initial_metrics['total_distance'] - optimized_metrics['total_distance']
    cost_improvement = initial_metrics['total_cost'] - optimized_metrics['total_cost']
    charging_improvement = initial_metrics['total_charging'] - optimized_metrics['total_charging']

    distance_pct = (distance_improvement/initial_metrics['total_distance']*100) if initial_metrics['total_distance'] > 0 else 0
    charging_pct = (charging_improvement/initial_metrics['total_charging']*100) if initial_metrics['total_charging'] > 0 else 0
    cost_pct = (cost_improvement/initial_metrics['total_cost']*100) if initial_metrics['total_cost'] > 0 else 0

    print(f"  距离改进: {distance_improvement:.2f}m ({distance_pct:.1f}%)")
    print(f"  充电改进: {charging_improvement:.2f}kWh ({charging_pct:.1f}%)")
    print(f"  成本改进: {cost_improvement:.2f} ({cost_pct:.1f}%)")
    print(f"  总耗时: {initial_time + optimization_time:.2f}秒 ({(initial_time + optimization_time)/60:.1f}分钟)")
    print(f"  平均每次迭代: {optimization_time/max_iterations:.2f}秒")

    return {
        'strategy': strategy_name,
        'initial_cost': initial_metrics['total_cost'],
        'optimized_cost': optimized_metrics['total_cost'],
        'improvement': cost_improvement,
        'improvement_pct': cost_improvement/initial_metrics['total_cost']*100,
        'construction_time': initial_time,
        'optimization_time': optimization_time,
        'total_time': initial_time + optimization_time,
        'initial_distance': initial_metrics['total_distance'],
        'optimized_distance': optimized_metrics['total_distance'],
        'initial_charging': initial_metrics['total_charging'],
        'optimized_charging': optimized_metrics['total_charging'],
        'charging_visits_initial': initial_metrics['num_charging_stops'],
        'charging_visits_optimized': optimized_metrics['num_charging_stops'],
        'tasks_served': len(optimized_route.get_served_tasks())
    }


def main():
    """主测试流程"""

    print("="*70)
    print("大规模场景ALNS优化效果测试")
    print("="*70)
    print("\n⚠️  警告：此测试可能需要10-30分钟，请耐心等待")

    # 创建场景
    print("\n创建大规模场景...")
    depot, tasks, charging_stations, distance_matrix, vehicle, energy_config = create_large_scenario()
    print("  ✓ 场景创建完成")

    print("\n场景配置:")
    print("  任务数: 50个")
    print("  充电站: 3个")
    print("  仓库范围: 150m × 150m")
    print(f"  车辆容量: {vehicle.capacity}kg")
    print(f"  电池容量: {vehicle.battery_capacity}kWh")
    print(f"  能耗率: {energy_config.consumption_rate}kWh/m")

    # 测试策略（大规模下只测试两种有代表性的）
    strategies = [
        ("完全充电 (FR)", FullRechargeStrategy()),
        ("最小充电 (PR-Minimal)", PartialRechargeMinimalStrategy(safety_margin=0.2))
    ]

    results = []
    total_start = time_module.time()

    for idx, (strategy_name, strategy) in enumerate(strategies):
        print(f"\n{'='*70}")
        print(f"测试进度: {idx+1}/{len(strategies)}")
        print(f"{'='*70}")

        result = test_optimization_with_strategy(
            strategy_name, strategy, depot, tasks,
            distance_matrix, vehicle, energy_config,
            max_iterations=50  # 大规模场景减少迭代次数
        )
        results.append(result)

    total_time = time_module.time() - total_start

    # 最终对比
    print(f"\n{'='*70}")
    print("最终对比总结")
    print(f"{'='*70}")

    print(f"\n{'策略':<25} {'初始成本':<12} {'优化成本':<12} {'改进':<12} {'改进率':<10}")
    print("-" * 75)
    for r in results:
        print(f"{r['strategy']:<25} {r['initial_cost']:<12.2f} {r['optimized_cost']:<12.2f} "
              f"{r['improvement']:<12.2f} {r['improvement_pct']:<10.1f}%")

    print(f"\n{'='*70}")
    print("详细指标对比")
    print(f"{'='*70}")

    print(f"\n服务任务数:")
    for r in results:
        print(f"  {r['strategy']:<25} {r['tasks_served']}/50")

    print(f"\n距离对比:")
    for r in results:
        distance_change = r['initial_distance'] - r['optimized_distance']
        print(f"  {r['strategy']:<25} {r['initial_distance']:>8.1f}m → {r['optimized_distance']:>8.1f}m "
              f"({distance_change:+.1f}m, {distance_change/r['initial_distance']*100:+.1f}%)")

    print(f"\n充电量对比:")
    for r in results:
        charging_change = r['initial_charging'] - r['optimized_charging']
        pct = (charging_change/r['initial_charging']*100) if r['initial_charging'] > 0 else 0
        print(f"  {r['strategy']:<25} {r['initial_charging']:>8.2f}kWh → {r['optimized_charging']:>8.2f}kWh "
              f"({charging_change:+.2f}kWh, {pct:+.1f}%)")

    print(f"\n充电站访问次数:")
    for r in results:
        visits_change = r['charging_visits_initial'] - r['charging_visits_optimized']
        print(f"  {r['strategy']:<25} {r['charging_visits_initial']:>3}次 → {r['charging_visits_optimized']:>3}次 "
              f"({visits_change:+d}次)")

    print(f"\n{'='*70}")
    print("性能和时间分析")
    print(f"{'='*70}")

    for r in results:
        print(f"\n{r['strategy']}:")
        print(f"  构建初始解: {r['construction_time']:.2f}秒 ({r['construction_time']/60:.1f}分钟)")
        print(f"  ALNS优化: {r['optimization_time']:.2f}秒 ({r['optimization_time']/60:.1f}分钟)")
        print(f"  总耗时: {r['total_time']:.2f}秒 ({r['total_time']/60:.1f}分钟)")
        print(f"  平均每次迭代: {r['optimization_time']/50:.2f}秒")

    # 推荐策略
    best_result = min(results, key=lambda x: x['optimized_cost'])
    print(f"\n{'='*70}")
    print("推荐策略分析")
    print(f"{'='*70}")
    print(f"\n最优策略: {best_result['strategy']}")
    print(f"  优化后成本: {best_result['optimized_cost']:.2f}")
    print(f"  总改进: {best_result['improvement']:.2f} ({best_result['improvement_pct']:.1f}%)")
    print(f"  总耗时: {best_result['total_time']/60:.1f}分钟")
    print(f"  服务任务: {best_result['tasks_served']}/50")

    print(f"\n{'='*70}")
    print(f"✓ 大规模优化测试完成（总耗时: {total_time/60:.1f}分钟）")
    print(f"{'='*70}")

    print(f"\n扩展性分析:")
    print(f"  50任务场景下，ALNS展示了良好的可扩展性")
    print(f"  平均每个任务的处理时间: {total_time/50:.2f}秒")
    print(f"  建议在实际应用中根据时间预算调整迭代次数")


if __name__ == '__main__':
    main()
