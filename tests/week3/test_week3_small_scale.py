"""
Week 3 小规模测试：仓库机器人场景（5-10任务）
====================================

基于真实仓库机器人参数的小规模测试。
验证Week 3功能在实际场景下的正确性。

场景：
- 仓库大小：50-60m × 50-60m
- 任务数量：5-10个
- 机器人：MiR100类型（100-150kg载重，8-12kWh电池）
"""

import sys
sys.path.append('src')

import random
from typing import List, Tuple
from core.node import DepotNode, create_task_node_pair, create_charging_node
from core.task import Task, TaskPool
from core.route import Route, create_empty_route
from core.vehicle import create_vehicle
from physics.distance import DistanceMatrix
from physics.energy import EnergyConfig
from planner.alns import MinimalALNS, CostParameters

from warehouse_test_config import (
    SMALL_WAREHOUSE_5_TASKS,
    SMALL_WAREHOUSE_10_TASKS,
    print_config_summary
)


def create_grid_layout(warehouse_size: Tuple[float, float],
                       num_tasks: int,
                       num_charging_stations: int = 0) -> Tuple[DepotNode, List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    创建仓库网格布局

    布局策略：
    - Depot在仓库中心
    - Pickup点在左半区域均匀分布
    - Delivery点在右半区域均匀分布
    - 充电站在仓库战略位置
    """
    width, height = warehouse_size

    # Depot在中心
    depot = DepotNode(coordinates=(width/2, height/2))

    # Pickup点：左半区域，网格分布
    pickup_coords_list = []
    grid_size = int(num_tasks ** 0.5) + 1
    for i in range(num_tasks):
        row = i // grid_size
        col = i % grid_size
        x = width * 0.15 + (col * width * 0.3 / max(1, grid_size-1)) if grid_size > 1 else width * 0.25
        y = height * 0.15 + (row * height * 0.7 / max(1, grid_size-1)) if grid_size > 1 else height * 0.5
        # 添加随机扰动（±10%），模拟实际货架位置
        x += random.uniform(-width*0.05, width*0.05)
        y += random.uniform(-height*0.05, height*0.05)
        pickup_coords_list.append((x, y))

    # Delivery点：右半区域，网格分布
    delivery_coords_list = []
    for i in range(num_tasks):
        row = i // grid_size
        col = i % grid_size
        x = width * 0.55 + (col * width * 0.3 / max(1, grid_size-1)) if grid_size > 1 else width * 0.75
        y = height * 0.15 + (row * height * 0.7 / max(1, grid_size-1)) if grid_size > 1 else height * 0.5
        x += random.uniform(-width*0.05, width*0.05)
        y += random.uniform(-height*0.05, height*0.05)
        delivery_coords_list.append((x, y))

    # 充电站位置（如果需要）
    charging_coords_list = []
    if num_charging_stations > 0:
        # 充电站放置在仓库四个角落
        positions = [
            (width * 0.15, height * 0.15),
            (width * 0.85, height * 0.15),
            (width * 0.15, height * 0.85),
            (width * 0.85, height * 0.85),
        ]
        charging_coords_list = positions[:num_charging_stations]

    return depot, pickup_coords_list, delivery_coords_list, charging_coords_list


def create_warehouse_scenario(config):
    """
    根据配置创建仓库场景

    返回：
        depot, tasks, distance_matrix, vehicle, energy_config, charging_nodes
    """
    print(f"\n创建场景：{config.name}")
    print(f"仓库：{config.warehouse_size}, 任务：{config.num_tasks}")

    # 1. 创建布局
    depot, pickup_coords_list, delivery_coords_list, charging_coords_list = create_grid_layout(
        config.warehouse_size,
        config.num_tasks,
        config.num_charging_stations
    )

    # 2. 创建任务
    tasks = []
    node_id_counter = 1
    coordinates = {0: depot.coordinates}

    for i in range(config.num_tasks):
        task_id = i + 1
        pickup_coords = pickup_coords_list[i]
        delivery_coords = delivery_coords_list[i]

        # 随机任务需求
        demand = random.uniform(*config.task_demand_range)

        pickup, delivery = create_task_node_pair(
            task_id=task_id,
            pickup_id=node_id_counter,
            delivery_id=node_id_counter + 1,
            pickup_coords=pickup_coords,
            delivery_coords=delivery_coords,
            demand=demand
        )

        task = Task(
            task_id=task_id,
            pickup_node=pickup,
            delivery_node=delivery,
            demand=demand
        )
        tasks.append(task)

        coordinates[pickup.node_id] = pickup_coords
        coordinates[delivery.node_id] = delivery_coords
        node_id_counter += 2

    # 3. 创建充电站
    charging_nodes = []
    for i, charging_coords in enumerate(charging_coords_list):
        charging_node_id = 100 + i  # 充电站ID从100开始
        charging_node = create_charging_node(
            node_id=charging_node_id,
            coordinates=charging_coords
        )
        charging_nodes.append(charging_node)
        coordinates[charging_node_id] = charging_coords

    # 4. 创建距离矩阵
    distance_matrix = DistanceMatrix(
        coordinates=coordinates,
        num_tasks=config.num_tasks,
        num_charging_stations=config.num_charging_stations
    )

    # 5. 创建车辆
    vehicle = create_vehicle(
        vehicle_id=1,
        capacity=config.robot_capacity,
        battery_capacity=config.robot_battery,
        initial_battery=config.robot_battery
    )

    # 6. 创建能量配置
    energy_config = EnergyConfig(
        consumption_rate=config.consumption_rate,
        charging_rate=config.charging_rate,
        battery_capacity=config.robot_battery
    )

    print(f"✓ 场景创建完成")
    print(f"  任务：{len(tasks)}个")
    print(f"  充电站：{len(charging_nodes)}个")
    print(f"  坐标点：{len(coordinates)}个")

    return depot, tasks, distance_matrix, vehicle, energy_config, charging_nodes


def test_small_5_tasks_basic():
    """测试1：5任务基础场景（无充电站）"""
    print("\n" + "="*70)
    print("测试1：5任务基础场景")
    print("="*70)

    config = SMALL_WAREHOUSE_5_TASKS
    depot, tasks, distance_matrix, vehicle, energy_config, charging_nodes = create_warehouse_scenario(config)

    # 创建TaskPool
    task_pool = TaskPool()
    for task in tasks:
        task_pool.add_task(task)

    # 使用greedy插入创建初始解
    alns = MinimalALNS(
        distance_matrix=distance_matrix,
        task_pool=task_pool,
        repair_mode='greedy'
    )
    alns.vehicle = vehicle
    alns.energy_config = energy_config

    route = create_empty_route(1, depot)
    removed_task_ids = [t.task_id for t in tasks]

    print(f"\n执行Greedy插入...")
    route = alns.greedy_insertion(route, removed_task_ids)

    # 验证结果
    served_tasks = route.get_served_tasks()
    capacity_feasible, capacity_error = route.check_capacity_feasibility(vehicle.capacity, debug=False)
    precedence_valid, precedence_error = route.validate_precedence()

    print(f"\n结果：")
    print(f"  服务任务数：{len(served_tasks)}/{len(tasks)}")
    print(f"  路径节点数：{len(route.nodes)}")
    print(f"  容量可行：{'✓' if capacity_feasible else '✗ ' + (capacity_error or '')}")
    print(f"  顺序有效：{'✓' if precedence_valid else '✗ ' + (precedence_error or '')}")

    # 断言
    assert len(served_tasks) == len(tasks), f"应服务{len(tasks)}个任务，实际{len(served_tasks)}个"
    assert capacity_feasible, "容量应可行"
    assert precedence_valid, "顺序应有效"

    print(f"\n✓ 测试1通过：5任务基础场景正常工作")


def test_small_10_tasks_with_charging():
    """测试2：10任务 + 充电站场景"""
    print("\n" + "="*70)
    print("测试2：10任务 + 充电站场景")
    print("="*70)

    config = SMALL_WAREHOUSE_10_TASKS
    depot, tasks, distance_matrix, vehicle, energy_config, charging_nodes = create_warehouse_scenario(config)

    # 创建TaskPool
    task_pool = TaskPool()
    for task in tasks:
        task_pool.add_task(task)

    # 使用regret2插入
    alns = MinimalALNS(
        distance_matrix=distance_matrix,
        task_pool=task_pool,
        repair_mode='regret2',
        cost_params=CostParameters(C_tr=1.0, C_ch=5.0)
    )
    alns.vehicle = vehicle
    alns.energy_config = energy_config

    # 充电站已经通过DistanceMatrix注册，无需额外添加
    # ALNS会在插入过程中自动考虑充电站

    route = create_empty_route(1, depot)
    removed_task_ids = [t.task_id for t in tasks]

    print(f"\n执行Regret-2插入（含充电站）...")
    route = alns.regret2_insertion(route, removed_task_ids)

    # 验证结果
    served_tasks = route.get_served_tasks()
    capacity_feasible, capacity_error = route.check_capacity_feasibility(vehicle.capacity, debug=False)
    precedence_valid, precedence_error = route.validate_precedence()

    # 统计充电站使用
    charging_count = sum(1 for node in route.nodes if hasattr(node, 'is_charging_station') and node.is_charging_station())

    print(f"\n结果：")
    print(f"  服务任务数：{len(served_tasks)}/{len(tasks)}")
    print(f"  路径节点数：{len(route.nodes)}")
    print(f"  充电站访问：{charging_count}次")
    print(f"  容量可行：{'✓' if capacity_feasible else '✗ ' + (capacity_error or '')}")
    print(f"  顺序有效：{'✓' if precedence_valid else '✗ ' + (precedence_error or '')}")

    # 断言
    assert len(served_tasks) >= len(tasks) * 0.8, f"至少应服务80%任务（{len(tasks)*0.8:.0f}个），实际{len(served_tasks)}个"
    assert capacity_feasible, "容量应可行"
    assert precedence_valid, "顺序应有效"

    print(f"\n✓ 测试2通过：10任务+充电站场景正常工作")


def test_small_week3_operators():
    """测试3：Week 3算子在小规模场景下的表现"""
    print("\n" + "="*70)
    print("测试3：Week 3算子测试（Partial Removal + Pair Exchange）")
    print("="*70)

    config = SMALL_WAREHOUSE_10_TASKS
    depot, tasks, distance_matrix, vehicle, energy_config, charging_nodes = create_warehouse_scenario(config)

    # 创建TaskPool
    task_pool = TaskPool()
    for task in tasks:
        task_pool.add_task(task)

    alns = MinimalALNS(
        distance_matrix=distance_matrix,
        task_pool=task_pool,
        repair_mode='greedy'
    )
    alns.vehicle = vehicle
    alns.energy_config = energy_config

    # 创建初始路径（所有任务连续插入）
    route = create_empty_route(1, depot)
    for task in tasks:
        route.nodes.insert(-1, task.pickup_node)
        route.nodes.insert(-1, task.delivery_node)

    print(f"\n初始路径：{len(route.nodes)}个节点，{len(tasks)}个任务")

    # 测试Partial Removal
    print(f"\n--- 测试Partial Removal ---")
    destroyed_route, removed_task_ids = alns.partial_removal(route, q=3)

    print(f"移除的任务：{removed_task_ids}")
    for task_id in removed_task_ids:
        task = task_pool.get_task(task_id)
        has_pickup = any(n.node_id == task.pickup_node.node_id for n in destroyed_route.nodes)
        has_delivery = any(n.node_id == task.delivery_node.node_id for n in destroyed_route.nodes)
        print(f"  任务{task_id}: pickup={has_pickup}, delivery={has_delivery}")
        assert has_pickup, f"任务{task_id}的pickup应该保留"
        assert not has_delivery, f"任务{task_id}的delivery应该移除"

    print(f"✓ Partial Removal正确")

    # 测试Pair Exchange
    print(f"\n--- 测试Pair Exchange ---")
    exchanged_route = alns.pair_exchange(route)

    original_tasks = set(route.get_served_tasks())
    exchanged_tasks = set(exchanged_route.get_served_tasks())

    print(f"原始任务：{sorted(original_tasks)}")
    print(f"交换后任务：{sorted(exchanged_tasks)}")

    assert original_tasks == exchanged_tasks, "任务集合应保持不变"

    precedence_valid, _ = exchanged_route.validate_precedence()
    assert precedence_valid, "Precedence约束应保持"

    print(f"✓ Pair Exchange正确")

    print(f"\n✓ 测试3通过：Week 3算子在小规模场景正常工作")


def test_small_end_to_end():
    """测试4：端到端工作流程"""
    print("\n" + "="*70)
    print("测试4：端到端工作流程（Destroy → Repair循环）")
    print("="*70)

    config = SMALL_WAREHOUSE_10_TASKS
    depot, tasks, distance_matrix, vehicle, energy_config, charging_nodes = create_warehouse_scenario(config)

    # 创建TaskPool
    task_pool = TaskPool()
    for task in tasks:
        task_pool.add_task(task)

    alns = MinimalALNS(
        distance_matrix=distance_matrix,
        task_pool=task_pool,
        repair_mode='mixed'
    )
    alns.vehicle = vehicle
    alns.energy_config = energy_config

    # 创建初始解
    print(f"\n步骤1: 创建初始解")
    route = create_empty_route(1, depot)
    removed_task_ids = [t.task_id for t in tasks]
    route = alns.greedy_insertion(route, removed_task_ids)

    initial_tasks = len(route.get_served_tasks())
    print(f"  初始任务数：{initial_tasks}")

    # 执行多轮destroy-repair
    print(f"\n步骤2: 执行3轮Destroy-Repair")
    for iteration in range(3):
        # Destroy
        if random.random() < 0.5:
            destroyed_route, removed = alns.partial_removal(route, q=3)
            method = "Partial Removal"
        else:
            destroyed_route, removed = alns.random_removal(route, q=3)
            method = "Random Removal"

        # Repair
        if random.random() < 0.5:
            repaired_route = alns.greedy_insertion(destroyed_route, removed)
            repair_method = "Greedy"
        else:
            repaired_route = alns.regret2_insertion(destroyed_route, removed)
            repair_method = "Regret-2"

        tasks_after = len(repaired_route.get_served_tasks())
        capacity_ok, _ = repaired_route.check_capacity_feasibility(vehicle.capacity)
        precedence_ok, _ = repaired_route.validate_precedence()

        print(f"  轮次{iteration+1}: {method} + {repair_method}")
        print(f"    任务数：{tasks_after}, 容量：{'✓' if capacity_ok else '✗'}, 顺序：{'✓' if precedence_ok else '✗'}")

        assert capacity_ok, f"轮次{iteration+1}容量应可行"
        assert precedence_ok, f"轮次{iteration+1}顺序应有效"

        route = repaired_route

    final_tasks = len(route.get_served_tasks())
    print(f"\n最终任务数：{final_tasks}/{len(tasks)}")

    assert final_tasks >= len(tasks) * 0.8, f"至少应保留80%任务"

    print(f"\n✓ 测试4通过：端到端工作流程正常")


# ============================================================================
# 主测试流程
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Week 3 小规模测试：仓库机器人场景")
    print("="*70)

    # 显示配置信息
    print("\n" + "="*70)
    print("测试场景配置")
    print("="*70)
    print_config_summary(SMALL_WAREHOUSE_5_TASKS)
    print_config_summary(SMALL_WAREHOUSE_10_TASKS)

    try:
        # 运行所有测试
        test_small_5_tasks_basic()
        test_small_10_tasks_with_charging()
        test_small_week3_operators()
        test_small_end_to_end()

        print("\n" + "="*70)
        print("✓ 所有小规模测试通过！")
        print("="*70)
        print("\n总结:")
        print("1. ✓ 5任务基础场景正常工作")
        print("2. ✓ 10任务+充电站场景正常工作")
        print("3. ✓ Week 3算子（Partial Removal, Pair Exchange）正常工作")
        print("4. ✓ 端到端Destroy-Repair工作流程正常")
        print("\nWeek 3在仓库机器人小规模场景下验证成功！")

    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n✗ 运行错误: {e}")
        import traceback
        traceback.print_exc()
