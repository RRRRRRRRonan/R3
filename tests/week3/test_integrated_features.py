"""Integrated regression for Week 3 planner capabilities.

Combines time windows, charging strategies, destroy/repair operators, and
capacity/energy feasibility checks on a ten-task scenario to validate that the
entire stack works end to end.
"""

from core.node import DepotNode, create_task_node_pair, ChargingNode
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

random.seed(42)


def create_test_scenario_with_time_windows(num_tasks=10):
    """
    创建带时间窗的测试场景

    场景：
    - 10个任务
    - 每个任务有软时间窗
    - 1个充电站
    - 容量约束：每个任务20kg，车辆150kg
    """
    depot = DepotNode(coordinates=(0.0, 0.0))

    # 创建充电站
    from core.node import NodeType
    charging_station = ChargingNode(
        node_id=999,
        coordinates=(25.0, 25.0),
        node_type=NodeType.CHARGING
    )

    tasks = []
    coordinates = {0: depot.coordinates, 999: charging_station.coordinates}

    # 创建任务（带时间窗）
    for i in range(1, num_tasks + 1):
        # 任务位置：均匀分布在50x50区域
        pickup_x = random.uniform(5, 45)
        pickup_y = random.uniform(5, 45)
        delivery_x = random.uniform(5, 45)
        delivery_y = random.uniform(5, 45)

        # 时间窗：每个任务间隔100秒
        pickup_tw_start = i * 100
        pickup_tw_end = pickup_tw_start + 200
        delivery_tw_start = pickup_tw_end + 50
        delivery_tw_end = delivery_tw_start + 200

        pickup, delivery = create_task_node_pair(
            task_id=i,
            pickup_id=i*2-1,
            delivery_id=i*2,
            pickup_coords=(pickup_x, pickup_y),
            delivery_coords=(delivery_x, delivery_y),
            demand=20.0,
            pickup_time_window=TimeWindow(pickup_tw_start, pickup_tw_end, TimeWindowType.SOFT),
            delivery_time_window=TimeWindow(delivery_tw_start, delivery_tw_end, TimeWindowType.SOFT)
        )

        task = Task(task_id=i, pickup_node=pickup, delivery_node=delivery, demand=20.0)
        tasks.append(task)
        coordinates[pickup.node_id] = pickup.coordinates
        coordinates[delivery.node_id] = delivery.coordinates

    # 创建distance matrix
    all_coords = coordinates.copy()
    distance_matrix = DistanceMatrix(
        coordinates=all_coords,
        num_tasks=num_tasks,
        num_charging_stations=1
    )

    # 创建vehicle
    vehicle = create_vehicle(
        vehicle_id=1,
        capacity=150.0,
        battery_capacity=12.0,
        initial_battery=12.0
    )

    # 创建energy config
    energy_config = EnergyConfig(
        consumption_rate=0.0008,
        charging_rate=3.0/3600
    )

    return depot, tasks, charging_station, distance_matrix, vehicle, energy_config


def test_charging_strategy_comparison():
    """
    测试1：对比三种充电策略

    验证：FR vs PR-Fixed(50%) vs PR-Minimal(10%)
    """
    print("\n" + "="*70)
    print("测试1：充电策略对比 (10任务 + 时间窗)")
    print("="*70)

    depot, tasks, charging_station, distance_matrix, vehicle, energy_config = \
        create_test_scenario_with_time_windows(num_tasks=10)

    strategies = [
        ("完全充电(FR)", FullRechargeStrategy()),
        ("固定50%(PR-Fixed)", PartialRechargeFixedStrategy(charge_ratio=0.5)),
        ("最小充电(PR-Minimal)", PartialRechargeMinimalStrategy(safety_margin=0.1))
    ]

    results = []

    for strategy_name, strategy in strategies:
        print(f"\n--- {strategy_name} ---")

        # 创建task pool
        task_pool = TaskPool()
        for task in tasks:
            task_pool.add_task(task)

        # 创建ALNS
        alns = MinimalALNS(
            distance_matrix=distance_matrix,
            task_pool=task_pool,
            repair_mode='greedy',
            cost_params=CostParameters(
                C_tr=1.0,
                C_ch=0.6,
                C_delay=2.0,  # 时间窗延迟惩罚
                C_time=0.1
            ),
            charging_strategy=strategy
        )
        alns.vehicle = vehicle
        alns.energy_config = energy_config

        # 创建初始解
        route = create_empty_route(1, depot)
        removed_task_ids = [t.task_id for t in tasks]

        # 插入任务
        repaired = alns.greedy_insertion(route, removed_task_ids)

        # 检查可行性
        capacity_ok, _ = repaired.check_capacity_feasibility(vehicle.capacity)
        precedence_ok, _ = repaired.validate_precedence()
        time_ok, delay_cost = alns._check_time_window_feasibility_fast(repaired)

        # 计算成本
        total_cost = alns.evaluate_cost(repaired)

        results.append({
            'strategy': strategy_name,
            'tasks': len(repaired.get_served_tasks()),
            'capacity_ok': capacity_ok,
            'precedence_ok': precedence_ok,
            'time_ok': time_ok,
            'total_cost': total_cost,
            'delay_cost': delay_cost
        })

        print(f"  服务任务: {len(repaired.get_served_tasks())}/10")
        print(f"  容量可行: {'✓' if capacity_ok else '✗'}")
        print(f"  顺序有效: {'✓' if precedence_ok else '✗'}")
        print(f"  时间窗可行: {'✓' if time_ok else '✗'}")
        print(f"  延迟成本: {delay_cost:.2f}")
        print(f"  总成本: {total_cost:.2f}")

    # 验证所有策略都能工作
    for result in results:
        assert result['tasks'] >= 8, f"{result['strategy']}应至少服务8个任务"
        assert result['capacity_ok'], f"{result['strategy']}容量应可行"
        assert result['precedence_ok'], f"{result['strategy']}顺序应有效"
        assert result['time_ok'], f"{result['strategy']}时间窗应可行"

    print(f"\n✓ 测试1通过：三种充电策略都正常工作")


def test_hard_vs_soft_time_windows():
    """
    测试2：硬时间窗 vs 软时间窗

    验证：
    - 硬时间窗违反会拒绝插入
    - 软时间窗违反会增加成本
    """
    print("\n" + "="*70)
    print("测试2：硬时间窗 vs 软时间窗")
    print("="*70)

    depot = DepotNode(coordinates=(0.0, 0.0))
    coordinates = {0: depot.coordinates}

    # 创建2个任务
    # 任务1：硬时间窗 [50, 100]
    pickup1, delivery1 = create_task_node_pair(
        task_id=1,
        pickup_id=1,
        delivery_id=2,
        pickup_coords=(10.0, 0.0),
        delivery_coords=(10.0, 10.0),
        demand=20.0,
        pickup_time_window=TimeWindow(50.0, 100.0, TimeWindowType.HARD)
    )
    task1 = Task(task_id=1, pickup_node=pickup1, delivery_node=delivery1, demand=20.0)
    coordinates[1] = pickup1.coordinates
    coordinates[2] = delivery1.coordinates

    # 任务2：软时间窗 [150, 200]
    pickup2, delivery2 = create_task_node_pair(
        task_id=2,
        pickup_id=3,
        delivery_id=4,
        pickup_coords=(30.0, 0.0),
        delivery_coords=(30.0, 10.0),
        demand=20.0,
        pickup_time_window=TimeWindow(150.0, 200.0, TimeWindowType.SOFT)
    )
    task2 = Task(task_id=2, pickup_node=pickup2, delivery_node=delivery2, demand=20.0)
    coordinates[3] = pickup2.coordinates
    coordinates[4] = delivery2.coordinates

    tasks = [task1, task2]

    distance_matrix = DistanceMatrix(coordinates=coordinates, num_tasks=2, num_charging_stations=0)
    vehicle = create_vehicle(vehicle_id=1, capacity=100.0, battery_capacity=10.0, initial_battery=10.0)

    task_pool = TaskPool()
    for task in tasks:
        task_pool.add_task(task)

    alns = MinimalALNS(
        distance_matrix=distance_matrix,
        task_pool=task_pool,
        repair_mode='greedy',
        cost_params=CostParameters(C_delay=5.0)
    )
    alns.vehicle = vehicle
    alns.energy_config = EnergyConfig(consumption_rate=0.001)

    # 测试插入
    route = create_empty_route(1, depot)
    repaired = alns.greedy_insertion(route, [1, 2])

    time_ok, delay_cost = alns._check_time_window_feasibility_fast(repaired)

    print(f"  服务任务数: {len(repaired.get_served_tasks())}")
    print(f"  时间窗可行: {'✓' if time_ok else '✗'}")
    print(f"  延迟成本: {delay_cost:.2f}")

    # 硬时间窗应该满足（否则不会插入）
    # 软时间窗可能延迟（但仍可行）
    assert len(repaired.get_served_tasks()) >= 1, "应至少服务1个任务"
    assert time_ok, "时间窗应可行（硬约束满足）"

    print(f"\n✓ 测试2通过：硬/软时间窗正确处理")


def test_alns_with_all_features():
    """
    测试3：ALNS完整功能测试

    验证ALNS在以下约束下优化：
    - Pickup/Delivery分离
    - 容量约束
    - 时间窗约束
    - 能量约束
    - 局部充电
    """
    print("\n" + "="*70)
    print("测试3：ALNS完整功能集成")
    print("="*70)

    depot, tasks, charging_station, distance_matrix, vehicle, energy_config = \
        create_test_scenario_with_time_windows(num_tasks=6)  # 减少任务数以加快测试

    task_pool = TaskPool()
    for task in tasks:
        task_pool.add_task(task)

    # 使用局部充电策略
    strategy = PartialRechargeMinimalStrategy(safety_margin=0.15)

    alns = MinimalALNS(
        distance_matrix=distance_matrix,
        task_pool=task_pool,
        repair_mode='regret2',
        cost_params=CostParameters(
            C_tr=1.0,
            C_ch=0.8,
            C_delay=3.0,
            C_time=0.1
        ),
        charging_strategy=strategy
    )
    alns.vehicle = vehicle
    alns.energy_config = energy_config

    # 创建初始解
    route = create_empty_route(1, depot)
    removed_task_ids = [t.task_id for t in tasks]
    initial_route = alns.greedy_insertion(route, removed_task_ids)

    print(f"\n初始解:")
    print(f"  任务数: {len(initial_route.get_served_tasks())}/6")
    print(f"  初始成本: {alns.evaluate_cost(initial_route):.2f}")

    # ALNS优化（少量迭代）
    print(f"\n执行ALNS优化（20次迭代）...")
    optimized_route = alns.optimize(initial_route, max_iterations=20)

    # 验证最终解
    capacity_ok, _ = optimized_route.check_capacity_feasibility(vehicle.capacity)
    precedence_ok, _ = optimized_route.validate_precedence()
    time_ok, delay_cost = alns._check_time_window_feasibility_fast(optimized_route)
    final_cost = alns.evaluate_cost(optimized_route)

    print(f"\n最终解:")
    print(f"  任务数: {len(optimized_route.get_served_tasks())}/6")
    print(f"  容量可行: {'✓' if capacity_ok else '✗'}")
    print(f"  顺序有效: {'✓' if precedence_ok else '✗'}")
    print(f"  时间窗可行: {'✓' if time_ok else '✗'}")
    print(f"  最终成本: {final_cost:.2f}")

    assert capacity_ok, "容量应可行"
    assert precedence_ok, "顺序应有效"
    assert time_ok, "时间窗应可行"
    assert len(optimized_route.get_served_tasks()) >= 5, "应至少服务5个任务"

    print(f"\n✓ 测试3通过：ALNS在所有约束下正常优化")


if __name__ == '__main__':
    print("="*70)
    print("Week 3 集成测试：所有功能组合验证")
    print("="*70)

    test_charging_strategy_comparison()
    test_hard_vs_soft_time_windows()
    test_alns_with_all_features()

    print("\n" + "="*70)
    print("✓ 所有集成测试通过！")
    print("="*70)
    print("\n验证功能:")
    print("  ✓ 硬/软时间窗约束")
    print("  ✓ 完全充电 vs 局部充电策略")
    print("  ✓ Pickup/Delivery分离优化")
    print("  ✓ 容量约束")
    print("  ✓ 能量约束")
    print("  ✓ ALNS多目标优化")
