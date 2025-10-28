"""Integration-style comparison of charging strategies.

The script simulates a handcrafted scenario for the FR and PR-Fixed strategies
to illustrate the impact on charging frequency, energy usage, and cost.  It is
primarily used for exploratory analysis and documentation examples rather than
formal assertions.
"""

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

    场景设计（充电必需场景）:
        - 8个任务点 (分布在40km范围)
        - 1个仓库 (原点)
        - 3个充电站 (战略位置)
        - 电池容量: 70kWh (关键：小于总能耗，必须充电！)
        - 能耗率: 0.5 kWh/km (正常能耗)
        - 车速: 10 m/s = 36 km/h (AMR合理速度)
        - 充电功率: 50 kW (现实的快充功率)
        - 预计总路程: ~160km，总能耗: ~80kWh > 70kWh (必须充电!)

    返回:
        (depot, tasks, charging_stations, distance_matrix, vehicle)
    """
    print("=" * 60)
    print("创建测试场景")
    print("=" * 60)

    # 1. 创建节点
    depot = create_depot((0, 0))

    # 2. 创建任务 (pickup + delivery)
    # 挑战性场景设计（更大范围）：
    # - 坐标单位：米
    # - 任务分布在40km范围内（模拟跨区域配送）
    # - 总路程约200km，能耗约100kWh
    # - 100kWh电池无法一次完成，必须充电2-3次
    # - 设计原则：任务间距离更大，zigzag路径增加总里程
    task_locations = [
        ((10000, 8000), (15000, 10000)),    # Task 1 - 东南区
        ((25000, 5000), (30000, 8000)),     # Task 2 - 东区
        ((35000, 15000), (38000, 18000)),   # Task 3 - 东北区
        ((32000, 25000), (30000, 28000)),   # Task 4 - 北区
        ((20000, 32000), (18000, 35000)),   # Task 5 - 西北区
        ((8000, 30000), (5000, 28000)),     # Task 6 - 西区
        ((3000, 20000), (2000, 15000)),     # Task 7 - 西南区
        ((8000, 10000), (5000, 8000)),      # Task 8 - 南区
    ]
    # 预期能耗计算（zigzag路径）：
    # 实际总路程: ~160km, 能耗: ~80kWh
    # 电池容量: 70kWh < 80kWh
    # → 不充电无法完成！所有策略都必须充电
    # → 可以真正对比不同充电策略的差异

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

    # 3. 创建充电站 (战略位置 - 覆盖更大任务区域)
    charging_stations = [
        create_charging_node(100, (25000, 12000)),  # CS1: 东部区域（T1-T3后）
        create_charging_node(101, (15000, 28000)),  # CS2: 北部区域（T4-T6后）
        create_charging_node(102, (5000, 18000)),   # CS3: 西部区域（T7-T8后）
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

    # 5. 创建车辆 (限制电池容量以强制充电)
    vehicle = create_vehicle(
        vehicle_id=1,
        capacity=150.0,
        battery_capacity=70.0  # 70kWh电池（<80kWh总能耗，必须充电！）
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
    使用指定充电策略模拟路径执行（动态优化版本）

    流程:
        1. 预先判断每个充电站是否需要访问
        2. 如果策略决定不充电，则跳过该充电站
        3. 基于实际访问的节点计算距离和成本
        4. 记录统计信息

    改进点:
        - 如果不需要充电，不访问充电站，节省距离和时间成本
        - 不同策略会产生不同的实际路径和距离成本

    参数:
        route: Route对象 (包含节点序列，可能包含充电站)
        vehicle: 车辆对象
        distance_matrix: 距离矩阵
        charging_strategy: 充电策略对象
        strategy_name: 策略名称 (用于显示)

    返回:
        dict: 模拟结果统计
    """
    # 现实场景配置
    # 注意: EnergyConfig中charging_rate单位是 kWh/s (能量/秒)
    # 50 kW = 50 kWh/hour = 50/3600 kWh/s ≈ 0.0139 kWh/s
    energy_config = EnergyConfig(
        consumption_rate=0.5,     # 0.5 kWh/km (正常能耗)
        charging_rate=50.0/3600,  # 50 kW = 50/3600 kWh/s (现实快充)
        charging_efficiency=0.9
    )
    time_config = TimeConfig(
        vehicle_speed=10.0  # 10 m/s = 36 km/h (AMR合理速度)
    )

    # 第一阶段：构建实际访问的路径（跳过不需要的充电站）
    print(f"\n{'='*60}")
    print(f"模拟执行: {strategy_name}")
    print(f"{'='*60}")
    print(f"初始电量: {vehicle.battery_capacity:.2f} kWh")
    print(f"\n第一阶段: 规划实际路径")

    actual_path = []
    simulated_battery = vehicle.battery_capacity

    i = 0
    while i < len(route.nodes):
        node = route.nodes[i]

        # 非充电站节点直接加入
        if not node.is_charging_station():
            actual_path.append(node)
            # 更新模拟电量（前进到下一个节点）
            if i + 1 < len(route.nodes):
                next_node = route.nodes[i + 1]
                dist = distance_matrix.get_distance(node.node_id, next_node.node_id)
                energy = (dist / 1000.0) * energy_config.consumption_rate
                simulated_battery -= energy
            i += 1
        else:
            # 充电站：判断是否需要访问
            cs_node = node

            # 计算剩余路径（跳过此充电站）的能量需求
            remaining_energy_demand = 0.0
            if i + 1 < len(route.nodes):
                # 从充电站后的第一个节点开始
                prev_node = actual_path[-1]  # 上一个访问的节点

                # 直接到充电站后节点的距离
                next_node = route.nodes[i + 1]
                dist_skip = distance_matrix.get_distance(prev_node.node_id, next_node.node_id)
                remaining_energy_demand += (dist_skip / 1000.0) * energy_config.consumption_rate

                # 后续路径的能量需求
                for j in range(i + 1, len(route.nodes) - 1):
                    dist_seg = distance_matrix.get_distance(
                        route.nodes[j].node_id,
                        route.nodes[j + 1].node_id
                    )
                    remaining_energy_demand += (dist_seg / 1000.0) * energy_config.consumption_rate

            # 使用策略判断是否需要充电
            charge_amount = charging_strategy.determine_charging_amount(
                current_battery=simulated_battery,
                remaining_demand=remaining_energy_demand,
                battery_capacity=vehicle.battery_capacity
            )

            if charge_amount > 0:
                # 需要充电，访问此充电站
                actual_path.append(cs_node)
                simulated_battery += charge_amount
                print(f"  → 访问充电站{cs_node.node_id} (需充电{charge_amount:.2f}kWh, 电量{simulated_battery-charge_amount:.1f}→{simulated_battery:.1f}kWh)")
            else:
                # 不需要充电，跳过此充电站
                print(f"  → 跳过充电站{cs_node.node_id} (当前{simulated_battery:.1f}kWh足够)")

            i += 1

    print(f"\n第二阶段: 执行实际路径")
    print(f"实际访问: {len(actual_path)}个节点 (原计划: {len(route.nodes)}个)")

    skipped_cs = len(route.nodes) - len(actual_path)
    if skipped_cs > 0:
        print(f"跳过了 {skipped_cs} 个充电站，节省距离成本")

    # 第二阶段：基于实际路径执行模拟
    current_battery = vehicle.battery_capacity
    current_load = 0.0
    current_time = 0.0

    total_distance = 0.0
    total_charging_amount = 0.0
    total_charging_time = 0.0
    charging_visits = 0
    charging_records = []

    for i in range(len(actual_path) - 1):
        current_node = actual_path[i]
        next_node = actual_path[i + 1]

        # 计算到下一节点的距离和能耗
        distance = distance_matrix.get_distance(
            current_node.node_id,
            next_node.node_id
        )
        total_distance += distance

        energy_consumed = (distance / 1000.0) * energy_config.consumption_rate
        current_battery -= energy_consumed

        if current_battery < 0:
            print(f"⚠️  路径不可行: 第{i}段电量不足")
            return {
                'strategy_name': strategy_name,
                'feasible': False,
                'failure_reason': f'第{i}段电量不足',
                'total_distance': 0,
                'total_cost': float('inf')
            }

        travel_time = distance / time_config.vehicle_speed
        current_time += travel_time

        # 如果到达充电站，执行充电
        if next_node.is_charging_station():
            # 计算剩余路径能耗
            remaining_distance = 0.0
            for j in range(i + 1, len(actual_path) - 1):
                seg_distance = distance_matrix.get_distance(
                    actual_path[j].node_id,
                    actual_path[j + 1].node_id
                )
                remaining_distance += seg_distance

            estimated_remaining = (remaining_distance / 1000.0) * energy_config.consumption_rate

            # 使用策略决定充电量
            charge_amount = charging_strategy.determine_charging_amount(
                current_battery=current_battery,
                remaining_demand=estimated_remaining,
                battery_capacity=vehicle.battery_capacity
            )

            if charge_amount > 0:
                charge_time = charge_amount / (energy_config.charging_rate * energy_config.charging_efficiency)

                current_battery += charge_amount
                current_time += charge_time
                total_charging_amount += charge_amount
                total_charging_time += charge_time
                charging_visits += 1

                charging_records.append({
                    'station_id': next_node.node_id,
                    'charge_amount': charge_amount,
                    'charge_time': charge_time,
                    'battery_before': current_battery - charge_amount,
                    'battery_after': current_battery
                })

                print(f"  充电站{next_node.node_id}: "
                      f"充{charge_amount:.2f}kWh ({charge_time/60:.1f}min), "
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
        'feasible': True,  # 路径可行
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
    print(f"  总距离: {total_distance/1000:.2f} km")
    print(f"  充电次数: {charging_visits}")
    print(f"  总充电量: {total_charging_amount:.2f} kWh")
    print(f"  总充电时间: {total_charging_time/60:.1f} min")
    print(f"  总时间: {current_time/60:.1f} min")
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

    # 2. 构造测试路径 (强制充电场景：电池容量<总能耗)
    # 路径设计（zigzag跨区域，70kWh电池无法一次完成）:
    #   Depot(70kWh) → T1-3 (东南→东→东北)
    #   → CS1(约77km, 38.5kWh, 剩余31.5kWh) ← 必须充电！
    #   → T4-6 (北→西北→西)
    #   → CS2(约50km, 25kWh, 剩余?) ← 取决于策略
    #   → T7-8 → Depot (约33km, 16.5kWh)
    # 总计: ~160km, ~80kWh > 70kWh（所有策略都必须充电）
    route = create_empty_route(1, depot)

    # 插入节点顺序
    route.nodes.insert(1, tasks[0].pickup_node)     # Task1 P
    route.nodes.insert(2, tasks[0].delivery_node)   # Task1 D
    route.nodes.insert(3, tasks[1].pickup_node)     # Task2 P
    route.nodes.insert(4, tasks[1].delivery_node)   # Task2 D
    route.nodes.insert(5, tasks[2].pickup_node)     # Task3 P
    route.nodes.insert(6, tasks[2].delivery_node)   # Task3 D
    route.nodes.insert(7, charging_stations[0])     # CS1 (完成3任务后)
    route.nodes.insert(8, tasks[3].pickup_node)     # Task4 P
    route.nodes.insert(9, tasks[3].delivery_node)   # Task4 D
    route.nodes.insert(10, tasks[4].pickup_node)    # Task5 P
    route.nodes.insert(11, tasks[4].delivery_node)  # Task5 D
    route.nodes.insert(12, tasks[5].pickup_node)    # Task6 P
    route.nodes.insert(13, tasks[5].delivery_node)  # Task6 D
    route.nodes.insert(14, charging_stations[1])    # CS2 (完成6任务后)
    route.nodes.insert(15, tasks[6].pickup_node)    # Task7 P
    route.nodes.insert(16, tasks[6].delivery_node)  # Task7 D
    route.nodes.insert(17, tasks[7].pickup_node)    # Task8 P
    route.nodes.insert(18, tasks[7].delivery_node)  # Task8 D

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

    # 筛选可行的策略
    feasible_results = [r for r in results_list if r.get('feasible', True)]

    print(f"\n{'策略':<20} {'可行性':<8} {'充电次数':<10} {'总充电量(kWh)':<15} {'总成本':<10}")
    print("-" * 80)

    for result in results_list:
        feasible_str = "✓可行" if result.get('feasible', True) else "✗不可行"
        visits = result.get('charging_visits', 0)
        amount = result.get('total_charging_amount', 0)
        cost = result.get('total_cost', float('inf'))
        cost_str = f"{cost:.2f}" if cost < 1e6 else "∞ (不可行)"

        print(f"{result['strategy_name']:<20} "
              f"{feasible_str:<8} "
              f"{visits:<10} "
              f"{amount:<15.2f} "
              f"{cost_str:<10}")

    # 6. 成本分解对比 (只显示可行的策略)
    if feasible_results:
        print(f"\n{'策略':<20} {'距离成本':<12} {'充电成本':<12} {'时间成本':<12} {'总成本':<10}")
        print("-" * 80)

        for result in feasible_results:
            print(f"{result['strategy_name']:<20} "
                  f"{result.get('distance_cost', 0):<12.2f} "
                  f"{result.get('charging_cost', 0):<12.2f} "
                  f"{result.get('time_cost', 0):<12.2f} "
                  f"{result['total_cost']:<10.2f}")

    # 7. 关键发现
    print("\n" + "=" * 60)
    print("关键发现")
    print("=" * 60)

    if len(feasible_results) == 0:
        print("\n⚠️  所有策略均不可行！需要重新设计场景或增加充电站")
    elif len(feasible_results) == 1:
        print(f"\n只有 {feasible_results[0]['strategy_name']} 策略可行")
        print(f"  总成本: {feasible_results[0]['total_cost']:.2f}")
        print(f"  充电量: {feasible_results[0]['total_charging_amount']:.2f} kWh")
    else:
        fr_result = results_list[0]  # FR总是第一个

        print(f"\n可行策略数量: {len(feasible_results)}/{len(results_list)}")
        print(f"\n各策略可行性:")
        for r in results_list:
            status = "✓ 可行" if r.get('feasible', True) else "✗ 不可行"
            reason = f" - {r.get('failure_reason', '')}" if not r.get('feasible', True) else ""
            print(f"  {r['strategy_name']:<25} {status}{reason}")

        if fr_result.get('feasible', True) and len(feasible_results) > 1:
            # 找到第一个可行的PR策略
            pr_result = next((r for r in results_list[1:] if r.get('feasible', True)), None)
            if pr_result:
                print(f"\nFR vs {pr_result['strategy_name']} 对比:")
                print(f"  充电量差异: {fr_result['total_charging_amount'] - pr_result['total_charging_amount']:.2f} kWh")
                print(f"  充电时间差异: {(fr_result['total_charging_time'] - pr_result['total_charging_time'])/60:.1f} min")
                print(f"  总成本差异: {fr_result['total_cost'] - pr_result['total_cost']:.2f}")

                if fr_result['total_cost'] < pr_result['total_cost']:
                    print(f"  ✓ FR策略更优 (节省 {pr_result['total_cost'] - fr_result['total_cost']:.2f})")
                else:
                    print(f"  ✓ {pr_result['strategy_name']}策略更优 (节省 {fr_result['total_cost'] - pr_result['total_cost']:.2f})")

    # 8. 验证结果有效性
    feasible_final_battery = [r.get('final_battery', 0) for r in feasible_results]
    if feasible_final_battery:
        assert all(b >= 0 for b in feasible_final_battery), "可行策略应保证电量非负"

    print("\n✅ 策略对比测试完成!")
    print(f"   {len(feasible_results)}/{len(results_list)} 策略可行")

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
