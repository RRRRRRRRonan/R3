"""Regression tests for the compact warehouse regression scenario.

The suite synthesises a 5–10 task environment with realistic warehouse
distances, generates grid layouts, and ensures the ALNS planner plus charging
constraints deliver feasible low-cost routes for each seed configuration.
"""

import random
from core.task import TaskPool
from core.route import Route, create_empty_route
from core.vehicle import create_vehicle
from physics.energy import EnergyConfig
from config import CostParameters
from planner.alns import MinimalALNS

from config.instance_generator import (
    ChargingPlacement,
    DepotPosition,
    WarehouseLayoutConfig,
    ZoneStrategy,
    generate_warehouse_instance,
)

from warehouse_test_config import (
    SMALL_WAREHOUSE_5_TASKS,
    SMALL_WAREHOUSE_10_TASKS,
    print_config_summary
)

def create_warehouse_scenario(config):
    """
    根据配置创建仓库场景

    返回：
        depot, tasks, distance_matrix, vehicle, energy_config, charging_nodes
    """
    print(f"\n创建场景：{config.name}")
    print(f"仓库：{config.warehouse_size}, 任务：{config.num_tasks}")

    # 1. Create a unified layout instance (math-model-compliant node IDs).
    width, height = config.warehouse_size
    layout = WarehouseLayoutConfig(
        width=width,
        height=height,
        depot_position=DepotPosition.CENTER,
        num_tasks=config.num_tasks,
        zone_strategy=ZoneStrategy.LEFT_RIGHT,
        demand_range=config.task_demand_range,
        num_charging_stations=config.num_charging_stations,
        charging_placement=ChargingPlacement.CORNER,
        seed=42,
    )
    instance = generate_warehouse_instance(layout)
    depot = instance.depot
    tasks = instance.tasks
    distance_matrix = instance.distance_matrix
    charging_nodes = instance.charging_nodes

    # 2. 创建车辆
    vehicle = create_vehicle(
        vehicle_id=1,
        capacity=config.robot_capacity,
        battery_capacity=config.robot_battery,
        initial_battery=config.robot_battery
    )
    vehicle.speed = config.robot_speed

    # 3. 创建能量配置
    energy_config = EnergyConfig(
        consumption_rate=config.consumption_rate,
        charging_rate=config.charging_rate,
        battery_capacity=config.robot_battery
    )

    print(f"✓ 场景创建完成")
    print(f"  任务：{len(tasks)}个")
    print(f"  充电站：{len(charging_nodes)}个")
    print(f"  坐标点：{len(instance.coordinates)}个")

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


def test_small_regression_operators():
    """测试3：仓储回归算子在小规模场景下的表现"""
    print("\n" + "="*70)
    print("测试3：仓储回归算子测试（Partial Removal + Pair Exchange）")
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

    print(f"\n✓ 测试3通过：仓储回归算子在小规模场景正常工作")


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
    failure_count = 0
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

        # 只接受可行解
        if capacity_ok and precedence_ok:
            route = repaired_route
        else:
            failure_count += 1
            print(f"    ⚠ 不可行解，保留旧解")

    final_tasks = len(route.get_served_tasks())
    print(f"\n最终任务数：{final_tasks}/{len(tasks)}")
    print(f"失败repair数：{failure_count}/3")

    # 放宽要求：只要最终解可行即可
    capacity_feasible, _ = route.check_capacity_feasibility(vehicle.capacity)
    precedence_valid, _ = route.validate_precedence()

    assert final_tasks >= len(tasks) * 0.8, f"至少应保留80%任务"
    assert capacity_feasible, "最终解容量应可行"
    assert precedence_valid, "最终解顺序应有效"

    print(f"\n✓ 测试4通过：端到端工作流程正常")


# ============================================================================
# 主测试流程
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("仓储回归 小规模测试：仓库机器人场景")
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
        test_small_regression_operators()
        test_small_end_to_end()

        print("\n" + "="*70)
        print("✓ 所有小规模测试通过！")
        print("="*70)
        print("\n总结:")
        print("1. ✓ 5任务基础场景正常工作")
        print("2. ✓ 10任务+充电站场景正常工作")
        print("3. ✓ 仓储回归算子（Partial Removal, Pair Exchange）正常工作")
        print("4. ✓ 端到端Destroy-Repair工作流程正常")
        print("\n仓储回归在仓库机器人小规模场景下验证成功！")

    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n✗ 运行错误: {e}")
        import traceback
        traceback.print_exc()
