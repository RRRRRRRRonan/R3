"""
对比测试：Repair自适应 vs Destroy+Repair自适应

目的：验证为什么添加Destroy自适应后效果反而下降
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
from strategy.charging_strategies import PartialRechargeMinimalStrategy
import copy
import random
import time as time_module

random.seed(42)  # 固定随机种子以便对比


def create_test_scenario():
    """创建测试场景"""
    depot = DepotNode(coordinates=(0.0, 0.0))

    charging_station = ChargingNode(
        node_id=21,
        coordinates=(500.0, 500.0),
        node_type=NodeType.CHARGING
    )

    tasks = []
    coordinates = {0: depot.coordinates, 21: charging_station.coordinates}

    for i in range(1, 11):
        pickup_x = random.uniform(50, 950)
        pickup_y = random.uniform(50, 950)
        delivery_x = random.uniform(50, 950)
        delivery_y = random.uniform(50, 950)

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
        battery_capacity=1.5,
        initial_battery=1.5
    )

    consumption_per_km = 0.5
    consumption_per_sec = consumption_per_km * (vehicle.speed / 1000.0)

    energy_config = EnergyConfig(
        consumption_rate=consumption_per_sec,
        charging_rate=3.0/3600,
        battery_capacity=vehicle.battery_capacity
    )

    return depot, tasks, charging_station, distance_matrix, vehicle, energy_config


def test_repair_only_adaptive(depot, tasks, distance_matrix, vehicle, energy_config):
    """测试1：只使用Repair自适应（禁用Destroy自适应）"""
    print("\n" + "="*70)
    print("测试1: 只使用Repair自适应")
    print("="*70)

    vehicle = copy.deepcopy(vehicle)
    vehicle.reset_to_initial_state()
    energy_config = copy.deepcopy(energy_config)

    task_pool = TaskPool()
    for task in tasks:
        task_pool.add_task(task)

    # 创建ALNS - 启用Repair自适应，但我们需要手动控制Destroy
    alns = MinimalALNS(
        distance_matrix=distance_matrix,
        task_pool=task_pool,
        repair_mode='adaptive',
        cost_params=CostParameters(
            C_tr=1.0,
            C_ch=0.6,
            C_delay=2.0,
            C_time=0.1
        ),
        charging_strategy=PartialRechargeMinimalStrategy(),
        use_adaptive=True
    )
    alns.vehicle = vehicle
    alns.energy_config = energy_config

    # 关键：禁用Destroy自适应选择器
    alns.adaptive_destroy_selector = None

    # 创建初始解
    route = create_empty_route(1, depot)
    removed_task_ids = [t.task_id for t in tasks]
    initial_route = alns.greedy_insertion(route, removed_task_ids)
    initial_cost = alns.evaluate_cost(initial_route)

    print(f"初始成本: {initial_cost:.2f}")

    # 优化
    start_time = time_module.time()
    optimized_route = alns.optimize(initial_route, max_iterations=50)
    optimization_time = time_module.time() - start_time

    final_cost = alns.evaluate_cost(optimized_route)
    improvement = initial_cost - final_cost
    improvement_pct = (improvement / initial_cost * 100) if initial_cost > 0 else 0

    print(f"\n最终成本: {final_cost:.2f}")
    print(f"改进: {improvement:.2f} ({improvement_pct:.2f}%)")
    print(f"优化时间: {optimization_time:.2f}秒")

    return {
        'initial_cost': initial_cost,
        'final_cost': final_cost,
        'improvement': improvement,
        'improvement_pct': improvement_pct,
        'time': optimization_time
    }


def test_both_adaptive(depot, tasks, distance_matrix, vehicle, energy_config):
    """测试2：同时使用Repair和Destroy自适应"""
    print("\n" + "="*70)
    print("测试2: 同时使用Repair和Destroy自适应")
    print("="*70)

    vehicle = copy.deepcopy(vehicle)
    vehicle.reset_to_initial_state()
    energy_config = copy.deepcopy(energy_config)

    task_pool = TaskPool()
    for task in tasks:
        task_pool.add_task(task)

    # 创建ALNS - 启用两层自适应
    alns = MinimalALNS(
        distance_matrix=distance_matrix,
        task_pool=task_pool,
        repair_mode='adaptive',
        cost_params=CostParameters(
            C_tr=1.0,
            C_ch=0.6,
            C_delay=2.0,
            C_time=0.1
        ),
        charging_strategy=PartialRechargeMinimalStrategy(),
        use_adaptive=True
    )
    alns.vehicle = vehicle
    alns.energy_config = energy_config

    # 创建初始解
    route = create_empty_route(1, depot)
    removed_task_ids = [t.task_id for t in tasks]
    initial_route = alns.greedy_insertion(route, removed_task_ids)
    initial_cost = alns.evaluate_cost(initial_route)

    print(f"初始成本: {initial_cost:.2f}")

    # 优化
    start_time = time_module.time()
    optimized_route = alns.optimize(initial_route, max_iterations=50)
    optimization_time = time_module.time() - start_time

    final_cost = alns.evaluate_cost(optimized_route)
    improvement = initial_cost - final_cost
    improvement_pct = (improvement / initial_cost * 100) if initial_cost > 0 else 0

    print(f"\n最终成本: {final_cost:.2f}")
    print(f"改进: {improvement:.2f} ({improvement_pct:.2f}%)")
    print(f"优化时间: {optimization_time:.2f}秒")

    return {
        'initial_cost': initial_cost,
        'final_cost': final_cost,
        'improvement': improvement,
        'improvement_pct': improvement_pct,
        'time': optimization_time
    }


def main():
    """主测试流程"""
    print("="*70)
    print("对比测试: Repair自适应 vs Destroy+Repair自适应")
    print("="*70)

    # 创建场景
    depot, tasks, charging_station, distance_matrix, vehicle, energy_config = create_test_scenario()

    # 运行多次测试取平均
    num_runs = 5

    repair_only_results = []
    both_results = []

    for run in range(num_runs):
        print(f"\n{'='*70}")
        print(f"第 {run+1}/{num_runs} 轮测试")
        print(f"{'='*70}")

        # 测试1: 只有Repair自适应
        result1 = test_repair_only_adaptive(depot, tasks, distance_matrix, vehicle, energy_config)
        repair_only_results.append(result1)

        # 测试2: 两层自适应
        result2 = test_both_adaptive(depot, tasks, distance_matrix, vehicle, energy_config)
        both_results.append(result2)

        # 重新设置随机种子
        random.seed(42 + run + 1)

    # 统计结果
    print("\n" + "="*70)
    print("统计结果汇总")
    print("="*70)

    avg_repair_only_improvement = sum(r['improvement_pct'] for r in repair_only_results) / num_runs
    avg_both_improvement = sum(r['improvement_pct'] for r in both_results) / num_runs

    avg_repair_only_cost = sum(r['final_cost'] for r in repair_only_results) / num_runs
    avg_both_cost = sum(r['final_cost'] for r in both_results) / num_runs

    print(f"\nRepair自适应 (平均{num_runs}次):")
    print(f"  平均改进率: {avg_repair_only_improvement:.2f}%")
    print(f"  平均最终成本: {avg_repair_only_cost:.2f}")

    print(f"\nDestroy+Repair自适应 (平均{num_runs}次):")
    print(f"  平均改进率: {avg_both_improvement:.2f}%")
    print(f"  平均最终成本: {avg_both_cost:.2f}")

    print(f"\n差异分析:")
    improvement_diff = avg_both_improvement - avg_repair_only_improvement
    cost_diff = avg_both_cost - avg_repair_only_cost

    if improvement_diff > 0:
        print(f"  ✓ 两层自适应更好: +{improvement_diff:.2f}% 改进")
    else:
        print(f"  ✗ 两层自适应更差: {improvement_diff:.2f}% 改进")

    print(f"  成本差异: {cost_diff:.2f}")

    if abs(improvement_diff) < 1.0:
        print(f"\n结论: 两种方法性能相近，差异不显著")
    elif improvement_diff > 0:
        print(f"\n结论: 两层自适应确实更好")
    else:
        print(f"\n结论: 只用Repair自适应反而更好，可能存在问题需要分析")


if __name__ == '__main__':
    main()
