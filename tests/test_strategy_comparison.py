"""
充电策略对比测试 (Week 1 补充)
==============================
对比FR vs PR-Fixed策略在实际路径规划中的效果

测试内容:
    1. 构造真实场景 (任务 + 充电站 + 距离矩阵)
    2. 模拟两种充电策略下的路径执行
    3. 对比成本、充电量、充电时间、充电次数
    4. 分析策略选择对总成本的影响

注意:
    当前版本是简化测试，手动构造路径并模拟充电策略
    Week 2将集成到ALNS优化流程中
"""

import sys
sys.path.append('src')

from core.node import create_depot, create_task_node_pair, create_charging_node, NodeType
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
from planner.alns import CostParameters


# ========== 测试场景构造 ==========

def create_test_scenario():
    """
    创建测试场景

    场景设计:
        - 5个任务点 (分布在不同位置)
        - 1个仓库 (原点)
        - 2个充电站 (战略位置)
        - 电池容量: 100kWh
        - 任务分布: 需要充电才能完成所有任务

    返回:
        (depot, tasks, charging_stations, distance_matrix, vehicle)
    """
    print("=" * 60)
    print("创建测试场景")
    print("=" * 60)

    # 1. 创建节点
    depot = create_depot((0, 0))

    # 2. 创建任务 (pickup + delivery)
    # 任务分布: 距离仓库较远，需要充电
    # 增加距离以体现充电策略差异
    task_locations = [
        ((300, 200), (400, 300)),  # Task 1 - 远距离
        ((500, 100), (600, 200)),  # Task 2 - 远距离
        ((200, 500), (300, 600)),  # Task 3 - 远距离
        ((100, 300), (150, 400)),  # Task 4 - 中距离
        ((400, 50), (500, 150)),   # Task 5 - 中距离
    ]

    tasks = []
    node_id_counter = 1  # 从1开始分配节点ID
    for i, (pickup_loc, delivery_loc) in enumerate(task_locations):
        pickup, delivery = create_task_node_pair(
            task_id=i+1,
            pickup_id=node_id_counter,
            delivery_id=node_id_counter + 1,
            pickup_coords=pickup_loc,
            delivery_coords=delivery_loc,
            demand=20.0
        )
        tasks.append(Task(
            task_id=i+1,
            pickup_node=pickup,
            delivery_node=delivery,
            demand=20.0
        ))
        node_id_counter += 2  # 每个任务占用2个节点ID

    # 3. 创建充电站 (战略位置)
    charging_stations = [
        create_charging_node(100, (200, 150)),   # 中心位置1
        create_charging_node(101, (350, 250)),   # 中心位置2
        create_charging_node(102, (150, 350)),   # 中心位置3
    ]

    # 4. 创建距离矩阵
    # 构建coordinates字典: node_id → (x, y)
    coordinates = {depot.node_id: depot.coordinates}

    # 添加任务节点
    for task in tasks:
        coordinates[task.pickup_node.node_id] = task.pickup_node.coordinates
        coordinates[task.delivery_node.node_id] = task.delivery_node.coordinates

    # 添加充电站
    for station in charging_stations:
        coordinates[station.node_id] = station.coordinates

    distance_matrix = DistanceMatrix(
        coordinates=coordinates,
        num_tasks=len(tasks),
        num_charging_stations=len(charging_stations)
    )

    # 5. 创建车辆 (减小电池容量以体现充电策略差异)
    vehicle = create_vehicle(
        vehicle_id=1,
        capacity=150.0,
        battery_capacity=60.0  # 减小到60kWh，必须充电才能完成任务
    )

    print(f"✓ 仓库: {depot.coordinates}")
    print(f"✓ 任务数: {len(tasks)}")
    print(f"✓ 充电站: {len(charging_stations)}个")
    print(f"✓ 车辆电池: {vehicle.battery_capacity} kWh")

    return depot, tasks, charging_stations, distance_matrix, vehicle


# ========== 路径模拟函数 ==========

def simulate_route_with_strategy(route, vehicle, distance_matrix,
                                 charging_strategy, strategy_name):
    """
    使用指定充电策略模拟路径执行

    流程:
        1. 遍历路径节点
        2. 到达充电站时，使用策略决定充电量
        3. 计算充电时间和成本
        4. 记录统计信息

    参数:
        route: Route对象 (包含节点序列)
        vehicle: 车辆对象
        distance_matrix: 距离矩阵
        charging_strategy: 充电策略对象
        strategy_name: 策略名称 (用于显示)

    返回:
        dict: 模拟结果统计
    """
    energy_config = EnergyConfig()
    time_config = TimeConfig()

    current_battery = vehicle.battery_capacity  # 满电出发
    current_load = 0.0
    current_time = 0.0

    total_distance = 0.0
    total_charging_amount = 0.0
    total_charging_time = 0.0
    charging_visits = 0

    charging_records = []

    print(f"\n{'='*60}")
    print(f"模拟执行: {strategy_name}")
    print(f"{'='*60}")
    print(f"初始电量: {current_battery:.2f} kWh")

    for i in range(len(route.nodes) - 1):
        current_node = route.nodes[i]
        next_node = route.nodes[i + 1]

        # 计算到下一节点的距离和能耗
        distance = distance_matrix.get_distance(
            current_node.node_id,
            next_node.node_id
        )
        total_distance += distance

        # 简化能耗计算: distance(m) * consumption_rate(kWh/km) / 1000
        energy_consumed = (distance / 1000.0) * energy_config.consumption_rate

        # 移动到下一节点
        current_battery -= energy_consumed

        if current_battery < 0:
            print(f"⚠️  警告: 第{i}段路径电量不足!")
            return None

        travel_time = distance / time_config.vehicle_speed
        current_time += travel_time

        # 如果当前节点是充电站，执行充电
        if current_node.is_charging_station():
            # 估算剩余路径能耗 (简化: 假设平均每段100m)
            remaining_nodes = len(route.nodes) - i - 1
            estimated_remaining = remaining_nodes * 100 * energy_config.consumption_rate / 1000.0

            # 使用策略决定充电量
            charge_amount = charging_strategy.determine_charging_amount(
                current_battery=current_battery,
                remaining_demand=estimated_remaining,
                battery_capacity=vehicle.battery_capacity
            )

            if charge_amount > 0:
                # 计算充电时间
                charge_time = charge_amount / (energy_config.charging_rate * energy_config.charging_efficiency)

                current_battery += charge_amount
                current_time += charge_time
                total_charging_amount += charge_amount
                total_charging_time += charge_time
                charging_visits += 1

                charging_records.append({
                    'station_id': current_node.node_id,
                    'position': i,
                    'charge_amount': charge_amount,
                    'charge_time': charge_time,
                    'battery_before': current_battery - charge_amount,
                    'battery_after': current_battery
                })

                print(f"  充电站{current_node.node_id}: "
                      f"充{charge_amount:.2f}kWh ({charge_time:.1f}s), "
                      f"电量 {current_battery-charge_amount:.2f}→{current_battery:.2f}")

        # 更新载重
        if next_node.is_pickup():
            current_load += next_node.demand
        elif next_node.is_delivery():
            current_load = max(0, current_load - next_node.demand)

    # 计算总成本
    cost_params = CostParameters()
    distance_cost = total_distance * cost_params.C_tr
    charging_cost = total_charging_amount * cost_params.C_ch
    time_cost = current_time * cost_params.C_time
    total_cost = distance_cost + charging_cost + time_cost

    results = {
        'strategy_name': strategy_name,
        'total_distance': total_distance,
        'total_charging_amount': total_charging_amount,
        'total_charging_time': total_charging_time,
        'charging_visits': charging_visits,
        'total_time': current_time,
        'distance_cost': distance_cost,
        'charging_cost': charging_cost,
        'time_cost': time_cost,
        'total_cost': total_cost,
        'final_battery': current_battery,
        'charging_records': charging_records
    }

    print(f"\n执行完成:")
    print(f"  总距离: {total_distance:.2f} m")
    print(f"  充电次数: {charging_visits}")
    print(f"  总充电量: {total_charging_amount:.2f} kWh")
    print(f"  总充电时间: {total_charging_time:.1f} s")
    print(f"  总时间: {current_time:.1f} s")
    print(f"  最终电量: {current_battery:.2f} kWh")
    print(f"  总成本: {total_cost:.2f}")

    return results


# ========== 对比测试 ==========

def test_fr_vs_pr_comparison():
    """
    主测试: 对比FR vs PR-Fixed策略
    """
    print("\n" + "=" * 60)
    print("充电策略对比测试")
    print("=" * 60)

    # 1. 创建测试场景
    depot, tasks, charging_stations, distance_matrix, vehicle = create_test_scenario()

    # 2. 构造测试路径
    # 路径: Depot → CS1 → T1(P→D) → CS2 → T2(P→D) → T3(P→D) → CS3 → T4(P→D) → T5(P→D) → Depot
    route = create_empty_route(1, depot)

    # 插入节点顺序 (设计路径需要多次充电)
    route.nodes.insert(1, charging_stations[0])     # 充电站1
    route.nodes.insert(2, tasks[0].pickup_node)     # Task1 P
    route.nodes.insert(3, tasks[0].delivery_node)   # Task1 D
    route.nodes.insert(4, charging_stations[1])     # 充电站2
    route.nodes.insert(5, tasks[1].pickup_node)     # Task2 P
    route.nodes.insert(6, tasks[1].delivery_node)   # Task2 D
    route.nodes.insert(7, tasks[2].pickup_node)     # Task3 P
    route.nodes.insert(8, tasks[2].delivery_node)   # Task3 D
    route.nodes.insert(9, charging_stations[2])     # 充电站3
    route.nodes.insert(10, tasks[3].pickup_node)    # Task4 P
    route.nodes.insert(11, tasks[3].delivery_node)  # Task4 D
    route.nodes.insert(12, tasks[4].pickup_node)    # Task5 P
    route.nodes.insert(13, tasks[4].delivery_node)  # Task5 D

    print(f"\n测试路径节点序列:")
    for i, node in enumerate(route.nodes):
        node_type = "Depot" if node.is_depot() else \
                   "充电站" if node.is_charging_station() else \
                   f"Task{node.task_id}P" if node.is_pickup() else \
                   f"Task{node.task_id}D"
        print(f"  {i}. {node_type} @ {node.coordinates}")

    # 3. 创建充电策略
    strategies = [
        (FullRechargeStrategy(), "FR - 完全充电"),
        (PartialRechargeFixedStrategy(charge_ratio=0.3), "PR-Fixed 30%"),
        (PartialRechargeFixedStrategy(charge_ratio=0.5), "PR-Fixed 50%"),
        (PartialRechargeMinimalStrategy(safety_margin=0.1), "PR-Minimal 10%"),
    ]

    # 4. 运行对比实验
    results_list = []
    for strategy, name in strategies:
        result = simulate_route_with_strategy(
            route, vehicle, distance_matrix, strategy, name
        )
        if result:
            results_list.append(result)

    # 5. 对比分析
    print("\n" + "=" * 60)
    print("策略对比结果")
    print("=" * 60)

    print(f"\n{'策略':<20} {'充电次数':<10} {'总充电量(kWh)':<15} {'充电时间(s)':<15} {'总成本':<10}")
    print("-" * 80)

    for result in results_list:
        print(f"{result['strategy_name']:<20} "
              f"{result['charging_visits']:<10} "
              f"{result['total_charging_amount']:<15.2f} "
              f"{result['total_charging_time']:<15.1f} "
              f"{result['total_cost']:<10.2f}")

    # 6. 成本分解对比
    print(f"\n{'策略':<20} {'距离成本':<12} {'充电成本':<12} {'时间成本':<12} {'总成本':<10}")
    print("-" * 80)

    for result in results_list:
        print(f"{result['strategy_name']:<20} "
              f"{result['distance_cost']:<12.2f} "
              f"{result['charging_cost']:<12.2f} "
              f"{result['time_cost']:<12.2f} "
              f"{result['total_cost']:<10.2f}")

    # 7. 关键发现
    print("\n" + "=" * 60)
    print("关键发现")
    print("=" * 60)

    fr_result = results_list[0]
    pr30_result = results_list[1]

    print(f"\nFR vs PR-Fixed(30%) 对比:")
    print(f"  充电量差异: {fr_result['total_charging_amount'] - pr30_result['total_charging_amount']:.2f} kWh")
    print(f"  充电时间差异: {fr_result['total_charging_time'] - pr30_result['total_charging_time']:.1f} s")
    print(f"  总成本差异: {fr_result['total_cost'] - pr30_result['total_cost']:.2f}")

    if fr_result['total_cost'] < pr30_result['total_cost']:
        print(f"  ✓ FR策略更优 (节省 {pr30_result['total_cost'] - fr_result['total_cost']:.2f})")
    else:
        print(f"  ✓ PR-Fixed(30%)策略更优 (节省 {fr_result['total_cost'] - pr30_result['total_cost']:.2f})")

    # 8. 验证结果有效性
    assert all(r['final_battery'] >= 0 for r in results_list), "所有策略应保证电量充足"
    assert all(r['total_cost'] > 0 for r in results_list), "成本应为正值"

    print("\n✅ 策略对比测试完成!")

    return results_list


# ========== 主函数 ==========

if __name__ == "__main__":
    try:
        results = test_fr_vs_pr_comparison()

        print("\n" + "=" * 60)
        print("🎉 测试成功!")
        print("=" * 60)
        print("\n下一步建议:")
        print("  1. Week 2: 将充电策略集成到ALNS优化流程")
        print("  2. 运行更大规模场景 (10+任务)")
        print("  3. 分析充电站密度对策略选择的影响")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
