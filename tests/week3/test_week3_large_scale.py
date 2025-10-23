"""
Week 3 大规模测试：仓库机器人场景（50-100任务）
====================================

验证Week 3功能在大规模实际场景下的可扩展性和性能。

场景：
- 仓库大小：150-200m × 150-200m
- 任务数量：50-100个
- 机器人：Kiva类型（300-350kg载重，25-30kWh电池）
- 充电站：3-4个

测试重点：
- 算法可扩展性
- 执行时间
- 解的质量
"""

import sys
sys.path.append('src')

import random
import time
from typing import List, Tuple
from core.node import DepotNode, create_task_node_pair, create_charging_node
from core.task import Task, TaskPool
from core.route import Route, create_empty_route
from core.vehicle import create_vehicle
from physics.distance import DistanceMatrix
from physics.energy import EnergyConfig
from planner.alns import MinimalALNS, CostParameters

from warehouse_test_config import (
    LARGE_WAREHOUSE_50_TASKS,
    LARGE_WAREHOUSE_100_TASKS,
    print_config_summary
)


def create_grid_layout(warehouse_size: Tuple[float, float],
                       num_tasks: int,
                       num_charging_stations: int = 0):
    """创建仓库网格布局"""
    width, height = warehouse_size
    depot = DepotNode(coordinates=(width/2, height/2))

    # Pickup点：左半区域
    pickup_coords_list = []
    grid_size = int(num_tasks ** 0.5) + 1
    for i in range(num_tasks):
        row = i // grid_size
        col = i % grid_size
        x = width * 0.15 + (col * width * 0.3 / max(1, grid_size-1)) if grid_size > 1 else width * 0.25
        y = height * 0.15 + (row * height * 0.7 / max(1, grid_size-1)) if grid_size > 1 else height * 0.5
        x += random.uniform(-width*0.05, width*0.05)
        y += random.uniform(-height*0.05, height*0.05)
        pickup_coords_list.append((x, y))

    # Delivery点：右半区域
    delivery_coords_list = []
    for i in range(num_tasks):
        row = i // grid_size
        col = i % grid_size
        x = width * 0.55 + (col * width * 0.3 / max(1, grid_size-1)) if grid_size > 1 else width * 0.75
        y = height * 0.15 + (row * height * 0.7 / max(1, grid_size-1)) if grid_size > 1 else height * 0.5
        x += random.uniform(-width*0.05, width*0.05)
        y += random.uniform(-height*0.05, height*0.05)
        delivery_coords_list.append((x, y))

    # 充电站位置（均匀分布）
    charging_coords_list = []
    if num_charging_stations > 0:
        positions = [
            (width * 0.15, height * 0.15),
            (width * 0.85, height * 0.15),
            (width * 0.15, height * 0.85),
            (width * 0.85, height * 0.85),
        ]
        charging_coords_list = positions[:num_charging_stations]

    return depot, pickup_coords_list, delivery_coords_list, charging_coords_list


def create_warehouse_scenario(config):
    """根据配置创建仓库场景"""
    print(f"\n创建场景：{config.name}")
    print(f"  仓库：{config.warehouse_size[0]}m × {config.warehouse_size[1]}m")
    print(f"  任务数：{config.num_tasks}")
    print(f"  充电站：{config.num_charging_stations}个")

    depot, pickup_coords_list, delivery_coords_list, charging_coords_list = create_grid_layout(
        config.warehouse_size,
        config.num_tasks,
        config.num_charging_stations
    )

    tasks = []
    node_id_counter = 1
    coordinates = {0: depot.coordinates}

    for i in range(config.num_tasks):
        task_id = i + 1
        pickup_coords = pickup_coords_list[i]
        delivery_coords = delivery_coords_list[i]
        demand = random.uniform(*config.task_demand_range)

        pickup, delivery = create_task_node_pair(
            task_id=task_id,
            pickup_id=node_id_counter,
            delivery_id=node_id_counter + 1,
            pickup_coords=pickup_coords,
            delivery_coords=delivery_coords,
            demand=demand
        )

        task = Task(task_id=task_id, pickup_node=pickup, delivery_node=delivery, demand=demand)
        tasks.append(task)
        coordinates[pickup.node_id] = pickup_coords
        coordinates[delivery.node_id] = delivery_coords
        node_id_counter += 2

    charging_nodes = []
    for i, charging_coords in enumerate(charging_coords_list):
        charging_node_id = 100 + i
        charging_node = create_charging_node(node_id=charging_node_id, coordinates=charging_coords)
        charging_nodes.append(charging_node)
        coordinates[charging_node_id] = charging_coords

    print(f"  创建距离矩阵中...（{len(coordinates)}个坐标点）")
    distance_matrix = DistanceMatrix(
        coordinates=coordinates,
        num_tasks=config.num_tasks,
        num_charging_stations=config.num_charging_stations
    )

    vehicle = create_vehicle(
        vehicle_id=1,
        capacity=config.robot_capacity,
        battery_capacity=config.robot_battery,
        initial_battery=config.robot_battery
    )

    energy_config = EnergyConfig(
        consumption_rate=config.consumption_rate,
        charging_rate=config.charging_rate,
        battery_capacity=config.robot_battery
    )

    print(f"✓ 场景创建完成")

    return depot, tasks, distance_matrix, vehicle, energy_config, charging_nodes


def test_large_50_tasks_greedy():
    """测试1：50任务 - Greedy插入可行性测试"""
    print("\n" + "="*70)
    print("测试1：50任务 - Greedy插入可行性测试")
    print("="*70)

    config = LARGE_WAREHOUSE_50_TASKS
    depot, tasks, distance_matrix, vehicle, energy_config, charging_nodes = create_warehouse_scenario(config)

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

    route = create_empty_route(1, depot)
    removed_task_ids = [t.task_id for t in tasks]

    print(f"\n执行Greedy插入（{len(tasks)}任务）...")
    start_time = time.time()
    route = alns.greedy_insertion(route, removed_task_ids)
    elapsed_time = time.time() - start_time

    served_tasks = route.get_served_tasks()
    capacity_feasible, _ = route.check_capacity_feasibility(vehicle.capacity)
    precedence_valid, _ = route.validate_precedence()

    print(f"\n结果：")
    print(f"  服务任务数：{len(served_tasks)}/{len(tasks)} ({len(served_tasks)/len(tasks)*100:.1f}%)")
    print(f"  路径节点数：{len(route.nodes)}")
    print(f"  执行时间：{elapsed_time:.2f}秒")
    print(f"  容量可行：{'✓' if capacity_feasible else '✗'}")
    print(f"  顺序有效：{'✓' if precedence_valid else '✗'}")

    # 大规模场景放宽要求：服务≥70%任务即可
    assert len(served_tasks) >= len(tasks) * 0.7, f"至少应服务70%任务（{len(tasks)*0.7:.0f}个）"
    assert capacity_feasible, "容量应可行"
    assert precedence_valid, "顺序应有效"
    assert elapsed_time < 120.0, f"执行时间应<120秒"

    print(f"\n✓ 测试1通过：50任务Greedy插入可行")


def test_large_50_tasks_regret2():
    """测试2：50任务 - Regret-2插入质量测试"""
    print("\n" + "="*70)
    print("测试2：50任务 - Regret-2插入质量测试")
    print("="*70)

    config = LARGE_WAREHOUSE_50_TASKS
    depot, tasks, distance_matrix, vehicle, energy_config, charging_nodes = create_warehouse_scenario(config)

    task_pool = TaskPool()
    for task in tasks:
        task_pool.add_task(task)

    alns = MinimalALNS(
        distance_matrix=distance_matrix,
        task_pool=task_pool,
        repair_mode='regret2'
    )
    alns.vehicle = vehicle
    alns.energy_config = energy_config

    route = create_empty_route(1, depot)
    removed_task_ids = [t.task_id for t in tasks]

    print(f"\n执行Regret-2插入（{len(tasks)}任务）...")
    print(f"  (这可能需要较长时间...)")
    start_time = time.time()
    route = alns.regret2_insertion(route, removed_task_ids)
    elapsed_time = time.time() - start_time

    served_tasks = route.get_served_tasks()
    capacity_feasible, _ = route.check_capacity_feasibility(vehicle.capacity)
    precedence_valid, _ = route.validate_precedence()

    print(f"\n结果：")
    print(f"  服务任务数：{len(served_tasks)}/{len(tasks)} ({len(served_tasks)/len(tasks)*100:.1f}%)")
    print(f"  路径节点数：{len(route.nodes)}")
    print(f"  执行时间：{elapsed_time:.2f}秒")
    print(f"  容量可行：{'✓' if capacity_feasible else '✗'}")
    print(f"  顺序有效：{'✓' if precedence_valid else '✗'}")

    assert len(served_tasks) >= len(tasks) * 0.7, f"至少应服务70%任务"
    assert capacity_feasible, "容量应可行"
    assert precedence_valid, "顺序应有效"
    assert elapsed_time < 300.0, f"执行时间应<300秒（5分钟）"

    print(f"\n✓ 测试2通过：50任务Regret-2插入可行")


def test_large_100_tasks_stress():
    """测试3：100任务 - 压力测试"""
    print("\n" + "="*70)
    print("测试3：100任务 - 压力测试")
    print("="*70)

    config = LARGE_WAREHOUSE_100_TASKS
    depot, tasks, distance_matrix, vehicle, energy_config, charging_nodes = create_warehouse_scenario(config)

    task_pool = TaskPool()
    for task in tasks:
        task_pool.add_task(task)

    alns = MinimalALNS(
        distance_matrix=distance_matrix,
        task_pool=task_pool,
        repair_mode='greedy'  # 使用faster的greedy
    )
    alns.vehicle = vehicle
    alns.energy_config = energy_config

    route = create_empty_route(1, depot)
    removed_task_ids = [t.task_id for t in tasks]

    print(f"\n执行Greedy插入（{len(tasks)}任务）...")
    print(f"  (这是压力测试，可能需要较长时间...)")
    start_time = time.time()
    route = alns.greedy_insertion(route, removed_task_ids)
    elapsed_time = time.time() - start_time

    served_tasks = route.get_served_tasks()
    capacity_feasible, _ = route.check_capacity_feasibility(vehicle.capacity)
    precedence_valid, _ = route.validate_precedence()

    print(f"\n结果：")
    print(f"  服务任务数：{len(served_tasks)}/{len(tasks)} ({len(served_tasks)/len(tasks)*100:.1f}%)")
    print(f"  路径节点数：{len(route.nodes)}")
    print(f"  执行时间：{elapsed_time:.2f}秒 ({elapsed_time/60:.1f}分钟)")
    print(f"  容量可行：{'✓' if capacity_feasible else '✗'}")
    print(f"  顺序有效：{'✓' if precedence_valid else '✗'}")

    # 100任务场景：更宽松的要求
    assert len(served_tasks) >= len(tasks) * 0.6, f"至少应服务60%任务（{len(tasks)*0.6:.0f}个）"
    assert capacity_feasible, "容量应可行"
    assert precedence_valid, "顺序应有效"
    assert elapsed_time < 600.0, f"执行时间应<600秒（10分钟）"

    print(f"\n✓ 测试3通过：100任务压力测试通过")


def test_large_scalability_analysis():
    """测试4：可扩展性分析"""
    print("\n" + "="*70)
    print("测试4：可扩展性分析（不同任务规模的性能对比）")
    print("="*70)

    task_sizes = [10, 20, 30, 50]
    results = []

    for num_tasks in task_sizes:
        print(f"\n--- {num_tasks}任务测试 ---")

        # 创建临时配置
        from warehouse_test_config import WarehouseConfig
        config = WarehouseConfig(
            name=f"可扩展性测试_{num_tasks}任务",
            warehouse_size=(100.0, 100.0),
            num_tasks=num_tasks,
            task_demand_range=(20.0, 40.0),
            robot_capacity=200.0,
            robot_battery=20.0,
            robot_speed=1.8,
            consumption_rate=0.0012,
            num_charging_stations=2,
            charging_rate=5.0/3600,
        )

        depot, tasks, distance_matrix, vehicle, energy_config, _ = create_warehouse_scenario(config)

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

        route = create_empty_route(1, depot)
        removed_task_ids = [t.task_id for t in tasks]

        start_time = time.time()
        route = alns.greedy_insertion(route, removed_task_ids)
        elapsed_time = time.time() - start_time

        served = len(route.get_served_tasks())
        results.append({
            'tasks': num_tasks,
            'served': served,
            'time': elapsed_time
        })

        print(f"  服务：{served}/{num_tasks}，时间：{elapsed_time:.2f}秒")

    # 分析结果
    print(f"\n可扩展性分析结果：")
    print(f"{'任务数':<10} {'完成率':<12} {'时间(秒)':<12} {'平均耗时/任务(ms)':<20}")
    print("-" * 55)
    for r in results:
        completion = r['served'] / r['tasks'] * 100
        avg_time = r['time'] / r['tasks'] * 1000
        print(f"{r['tasks']:<10} {completion:<12.1f} {r['time']:<12.2f} {avg_time:<20.1f}")

    print(f"\n✓ 测试4通过：可扩展性分析完成")


# ============================================================================
# 主测试流程
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Week 3 大规模测试：仓库机器人场景")
    print("="*70)

    # 显示配置信息
    print("\n场景配置信息：")
    print_config_summary(LARGE_WAREHOUSE_50_TASKS)

    try:
        test_large_50_tasks_greedy()
        test_large_50_tasks_regret2()
        test_large_100_tasks_stress()
        test_large_scalability_analysis()

        print("\n" + "="*70)
        print("✓ 所有大规模测试通过！")
        print("="*70)
        print("\n总结:")
        print("1. ✓ 50任务Greedy插入可行")
        print("2. ✓ 50任务Regret-2插入可行")
        print("3. ✓ 100任务压力测试通过")
        print("4. ✓ 可扩展性分析完成")
        print("\nWeek 3在仓库机器人大规模场景下验证成功！")
        print("\n🎉 Week 3所有规模测试（小、中、大）全部通过！")

    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n✗ 运行错误: {e}")
        import traceback
        traceback.print_exc()
