"""
ALNS + 充电策略集成测试 (Week 2)
====================================
对比不同充电策略和insertion模式在ALNS优化中的表现

测试矩阵:
  充电策略: FR, PR-Fixed 30%, PR-Minimal 10%
  Insertion模式: greedy, regret2, random
  总计: 3 × 3 = 9种组合

测试场景:
  - 12个任务 (大规模场景)
  - 4个充电站
  - 70kWh电池容量
  - zigzag分布，必须充电
"""

import sys
sys.path.append('src')

from core.node import create_depot, create_task_node_pair, create_charging_node
from core.route import Route, create_empty_route
from core.task import Task
from core.vehicle import create_vehicle
from physics.distance import DistanceMatrix
from physics.energy import EnergyConfig
from physics.time import TimeConfig
from strategy.charging_strategies import (
    FullRechargeStrategy,
    PartialRechargeFixedStrategy,
    PartialRechargeMinimalStrategy
)
from planner.alns import MinimalALNS, CostParameters
from core.task import TaskPool


def create_large_scenario():
    """
    创建大规模测试场景（12任务）

    场景设计:
        - 12个任务分布在50km×50km区域
        - 4个充电站战略分布
        - 70kWh电池容量
        - 总路程预计~200km，能耗~100kWh
        - 必须充电才能完成
    """
    print("=" * 60)
    print("创建大规模测试场景 (12任务)")
    print("=" * 60)

    # 1. 创建depot
    depot = create_depot((0, 0))

    # 2. 创建12个任务 (zigzag分布，跨越80km范围，必须充电)
    task_locations = [
        # 第一圈（东南区）- 远距离
        ((15000, 10000), (20000, 12000)),   # Task 1
        ((28000, 8000), (35000, 10000)),    # Task 2
        ((42000, 15000), (48000, 18000)),   # Task 3

        # 第二圈（东北区）- 远端
        ((55000, 22000), (60000, 25000)),   # Task 4
        ((65000, 30000), (70000, 35000)),   # Task 5
        ((75000, 40000), (78000, 45000)),   # Task 6

        # 第三圈（北区）- 横跨
        ((70000, 50000), (65000, 55000)),   # Task 7
        ((58000, 58000), (52000, 60000)),   # Task 8
        ((45000, 62000), (38000, 65000)),   # Task 9

        # 第四圈（西区）- 返回
        ((30000, 60000), (22000, 58000)),   # Task 10
        ((15000, 52000), (10000, 48000)),   # Task 11
        ((8000, 40000), (5000, 32000)),     # Task 12
    ]

    tasks = []
    all_task_objs = []
    node_id_counter = 1

    for i, (pickup_loc, delivery_loc) in enumerate(task_locations):
        pickup, delivery = create_task_node_pair(
            task_id=i+1,
            pickup_id=node_id_counter,
            delivery_id=node_id_counter + 1,
            pickup_coords=pickup_loc,
            delivery_coords=delivery_loc,
            demand=20.0
        )
        task_obj = Task(
            task_id=i+1,
            pickup_node=pickup,
            delivery_node=delivery,
            demand=20.0
        )
        tasks.append(task_obj)
        all_task_objs.append(task_obj)
        node_id_counter += 2

    # 3. 创建4个充电站（战略分布，覆盖80km范围）
    charging_stations = [
        create_charging_node(100, (35000, 15000)),  # CS1: 东南-东区中间
        create_charging_node(101, (65000, 30000)),  # CS2: 东北区
        create_charging_node(102, (50000, 55000)),  # CS3: 北区中心
        create_charging_node(103, (15000, 45000)),  # CS4: 西区
    ]

    # 4. 创建距离矩阵
    coordinates = {depot.node_id: depot.coordinates}

    for task in tasks:
        coordinates[task.pickup_node.node_id] = task.pickup_node.coordinates
        coordinates[task.delivery_node.node_id] = task.delivery_node.coordinates

    for station in charging_stations:
        coordinates[station.node_id] = station.coordinates

    distance_matrix = DistanceMatrix(
        coordinates=coordinates,
        num_tasks=len(tasks),
        num_charging_stations=len(charging_stations)
    )

    # 5. 创建车辆（70kWh电池）
    vehicle = create_vehicle(
        vehicle_id=1,
        capacity=200.0,
        battery_capacity=70.0  # 限制电池容量，必须充电
    )

    # 6. 创建TaskPool
    task_pool = TaskPool()
    for task in all_task_objs:
        task_pool.add_task(task)

    print(f"✓ 仓库: {depot.coordinates}")
    print(f"✓ 任务数: {len(tasks)}")
    print(f"✓ 充电站: {len(charging_stations)}个")
    print(f"✓ 车辆电池: {vehicle.battery_capacity} kWh")
    print(f"✓ 预计总路程: ~250km (需要~125kWh) - 必须充电！")

    return depot, tasks, charging_stations, distance_matrix, vehicle, task_pool


def create_initial_solution(depot, tasks, charging_stations, vehicle, distance_matrix):
    """
    创建初始解（简单顺序插入 + 充电站）

    策略：
    - 每3个任务后插入一个充电站
    - 确保初始解是可行的
    """
    route = create_empty_route(1, depot)

    # 按顺序插入所有任务，每3个任务后插入充电站
    for i, task in enumerate(tasks):
        route.insert_task(task, (len(route.nodes)-1, len(route.nodes)))

        # 每3个任务后插入一个充电站
        if (i + 1) % 3 == 0 and (i + 1) // 3 <= len(charging_stations):
            cs_idx = ((i + 1) // 3) - 1
            if cs_idx < len(charging_stations):
                route.nodes.insert(len(route.nodes)-1, charging_stations[cs_idx])

    print(f"\n初始解创建完成: {len(route.get_served_tasks())}个任务")
    print(f"初始路径包含: {len([n for n in route.nodes if n.is_charging_station()])}个充电站")
    print(f"初始路径长度: {route.calculate_total_distance(distance_matrix)/1000:.1f} km")

    return route


def test_alns_with_strategies():
    """
    主测试: 对比不同充电策略和insertion模式

    测试矩阵:
      - 充电策略: FR, PR-Fixed 30%, PR-Minimal 10%
      - Insertion: greedy, regret2, random
      - 总计: 9种组合
    """
    print("\n" + "=" * 60)
    print("ALNS + 充电策略集成测试")
    print("=" * 60)

    # 1. 创建场景
    depot, tasks, charging_stations, distance_matrix, vehicle, task_pool = create_large_scenario()

    # 2. 配置
    energy_config = EnergyConfig(
        consumption_rate=0.5,
        charging_rate=50.0/3600,
        charging_efficiency=0.9
    )

    cost_params = CostParameters(
        C_tr=1.0,
        C_ch=0.6,
        C_time=0.1,
        C_delay=2.0
    )

    # 3. 定义测试组合
    strategies = [
        (FullRechargeStrategy(), "FR-完全充电"),
        (PartialRechargeFixedStrategy(charge_ratio=0.3), "PR-Fixed-30%"),
        (PartialRechargeMinimalStrategy(safety_margin=0.1), "PR-Minimal-10%"),
    ]

    insertion_modes = ['greedy', 'regret2', 'random']

    # 4. 运行所有组合
    results = []

    for strategy, strategy_name in strategies:
        for insertion_mode in insertion_modes:
            print(f"\n{'='*60}")
            print(f"测试组合: {strategy_name} + {insertion_mode.upper()}")
            print(f"{'='*60}")

            # 创建初始解
            initial_route = create_initial_solution(depot, tasks, charging_stations, vehicle, distance_matrix)

            # 配置ALNS
            alns = MinimalALNS(
                distance_matrix=distance_matrix,
                task_pool=task_pool,
                repair_mode=insertion_mode,
                cost_params=cost_params,
                charging_strategy=strategy
            )
            alns.vehicle = vehicle
            alns.energy_config = energy_config

            # 运行优化（增加迭代次数以获得更好的解）
            best_route = alns.optimize(initial_route, max_iterations=50)

            # 收集统计
            breakdown = alns.get_cost_breakdown(best_route)
            final_cost = alns.evaluate_cost(best_route)

            # 检查路径中的充电站数量
            num_cs_in_route = len([n for n in best_route.nodes if n.is_charging_station()])

            # 检查可行性
            battery_feasible = alns._check_battery_feasibility(best_route)

            result = {
                'strategy': strategy_name,
                'insertion': insertion_mode,
                'final_cost': final_cost,
                'distance_km': breakdown['total_distance'] / 1000,
                'charging_kwh': breakdown['total_charging'],
                'num_charging': breakdown['num_charging_stops'],
                'time_min': breakdown['total_time'] / 60,
                'num_cs': num_cs_in_route,
                'battery_feasible': battery_feasible,
            }
            results.append(result)

            print(f"\n最终结果:")
            print(f"  总成本: {final_cost:.2f}")
            print(f"  总距离: {breakdown['total_distance']/1000:.2f} km")
            print(f"  充电站数: {num_cs_in_route}")
            print(f"  充电量: {breakdown['total_charging']:.2f} kWh")
            print(f"  充电次数: {breakdown['num_charging_stops']}")
            print(f"  总时间: {breakdown['total_time']/60:.1f} min")
            print(f"  电池可行: {'✓' if battery_feasible else '✗不可行!'}")

    # 5. 对比分析
    print(f"\n{'='*60}")
    print("综合对比结果")
    print(f"{'='*60}")

    print(f"\n{'策略':<20} {'Insertion':<10} {'总成本':<12} {'距离(km)':<10} {'充电(kWh)':<12} {'充电次数':<10}")
    print("-" * 90)

    for r in results:
        print(f"{r['strategy']:<20} {r['insertion']:<10} {r['final_cost']:<12.2f} "
              f"{r['distance_km']:<10.2f} {r['charging_kwh']:<12.2f} {r['num_charging']:<10}")

    # 6. 找出最优组合
    best_result = min(results, key=lambda x: x['final_cost'])
    print(f"\n{'='*60}")
    print("最优组合")
    print(f"{'='*60}")
    print(f"策略: {best_result['strategy']}")
    print(f"Insertion: {best_result['insertion']}")
    print(f"总成本: {best_result['final_cost']:.2f}")
    print(f"距离: {best_result['distance_km']:.2f} km")
    print(f"充电: {best_result['charging_kwh']:.2f} kWh")

    return results


if __name__ == "__main__":
    try:
        results = test_alns_with_strategies()

        print(f"\n{'='*60}")
        print("🎉 测试成功完成!")
        print(f"{'='*60}")
        print("\n关键发现:")
        print("  1. 充电策略对总成本的影响")
        print("  2. Insertion模式对优化质量的影响")
        print("  3. 最优组合的识别")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
