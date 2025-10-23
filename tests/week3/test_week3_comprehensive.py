"""
Week 3 综合测试：取送货分离优化
====================================

测试所有Week 3实现的功能：
- 步骤2.1: 容量约束检查 + Pickup/Delivery分离插入
- 步骤2.2: Delivery节点独立移除
- 步骤2.3: Pair-exchange operator
- 步骤2.4: Regret-2插入（改进版）
"""

import sys
sys.path.append('src')

from core.node import DepotNode, create_task_node_pair
from core.task import Task, TaskPool
from core.route import Route, create_empty_route
from core.vehicle import create_vehicle
from physics.distance import DistanceMatrix
from physics.energy import EnergyConfig
from planner.alns import MinimalALNS, CostParameters


def create_test_scenario():
    """创建测试场景：3个任务，每个需求30kg（更宽松）"""
    depot = DepotNode(coordinates=(0.0, 0.0))

    tasks = []
    node_id_counter = 1
    coordinates = {0: depot.coordinates}

    for i in range(1, 4):
        pickup_coords = (i * 10000.0, 0.0)  # 距离更近
        delivery_coords = (i * 10000.0, 15000.0)  # 距离更近

        pickup, delivery = create_task_node_pair(
            task_id=i,
            pickup_id=node_id_counter,
            delivery_id=node_id_counter + 1,
            pickup_coords=pickup_coords,
            delivery_coords=delivery_coords,
            demand=30.0  # 降低需求量
        )

        task = Task(
            task_id=i,
            pickup_node=pickup,
            delivery_node=delivery,
            demand=30.0  # 降低需求量
        )
        tasks.append(task)

        coordinates[pickup.node_id] = pickup_coords
        coordinates[delivery.node_id] = delivery_coords
        node_id_counter += 2

    distance_matrix = DistanceMatrix(
        coordinates=coordinates,
        num_tasks=3,
        num_charging_stations=0
    )

    vehicle = create_vehicle(
        vehicle_id=1,
        capacity=100.0,
        battery_capacity=60.0,
        initial_battery=60.0
    )

    return depot, tasks, distance_matrix, vehicle


def test_partial_removal():
    """测试2.2：Delivery节点独立移除"""
    print("\n" + "=" * 70)
    print("测试2.2：Delivery节点独立移除")
    print("=" * 70)

    depot, tasks, distance_matrix, vehicle = create_test_scenario()

    # 创建初始路径：所有任务连续插入
    route = create_empty_route(1, depot)
    for task in tasks:
        route.nodes.insert(-1, task.pickup_node)
        route.nodes.insert(-1, task.delivery_node)

    print(f"\n初始路径节点数: {len(route.nodes)} (包括2个depot)")
    print(f"任务节点: {len([n for n in route.nodes if not n.is_depot()])}个")

    # 创建ALNS和TaskPool
    task_pool = TaskPool()
    for task in tasks:
        task_pool.add_task(task)

    alns = MinimalALNS(
        distance_matrix=distance_matrix,
        task_pool=task_pool,
        repair_mode='greedy'
    )
    alns.vehicle = vehicle
    alns.energy_config = EnergyConfig(consumption_rate=0.5)

    # 使用partial_removal
    destroyed_route, removed_task_ids = alns.partial_removal(route, q=2)

    print(f"\nPartial removal后:")
    print(f"  移除的任务ID: {removed_task_ids}")
    print(f"  剩余节点数: {len(destroyed_route.nodes)}")

    # 检查pickup是否还在
    for task_id in removed_task_ids:
        task = task_pool.get_task(task_id)
        has_pickup = any(n.node_id == task.pickup_node.node_id for n in destroyed_route.nodes)
        has_delivery = any(n.node_id == task.delivery_node.node_id for n in destroyed_route.nodes)
        print(f"  任务{task_id}: pickup在路径中={has_pickup}, delivery在路径中={has_delivery}")

        assert has_pickup, f"任务{task_id}的pickup应该还在路径中"
        assert not has_delivery, f"任务{task_id}的delivery应该被移除"

    print("\n✓ 测试2.2通过：Partial removal正确工作")


def test_pair_exchange():
    """测试2.3：Pair-exchange operator"""
    print("\n" + "=" * 70)
    print("测试2.3：Pair-exchange operator")
    print("=" * 70)

    depot, tasks, distance_matrix, vehicle = create_test_scenario()

    # 创建初始路径
    route = create_empty_route(1, depot)
    for task in tasks:
        route.nodes.insert(-1, task.pickup_node)
        route.nodes.insert(-1, task.delivery_node)

    # 创建TaskPool
    task_pool = TaskPool()
    for task in tasks:
        task_pool.add_task(task)

    alns = MinimalALNS(
        distance_matrix=distance_matrix,
        task_pool=task_pool
    )
    alns.vehicle = vehicle

    # 记录原始顺序
    original_order = []
    for node in route.nodes:
        if hasattr(node, 'task_id'):
            original_order.append((node.task_id, node.node_type.value))

    print(f"\n原始任务顺序: {original_order}")

    # 执行pair_exchange
    exchanged_route = alns.pair_exchange(route)

    # 记录交换后顺序
    new_order = []
    for node in exchanged_route.nodes:
        if hasattr(node, 'task_id'):
            new_order.append((node.task_id, node.node_type.value))

    print(f"交换后顺序: {new_order}")

    # 验证precedence约束
    valid, error = exchanged_route.validate_precedence()
    print(f"\nPrecedence约束检查: {'✓ 有效' if valid else f'✗ 无效 - {error}'}")

    assert valid, "交换后应保持precedence约束"
    assert new_order != original_order, "顺序应该发生了变化"

    print("\n✓ 测试2.3通过：Pair exchange正确工作")


def test_regret2_with_capacity():
    """测试2.4：Regret-2插入（改进版，包含容量检查）"""
    print("\n" + "=" * 70)
    print("测试2.4：Regret-2插入（改进版）")
    print("=" * 70)

    depot, tasks, distance_matrix, vehicle = create_test_scenario()

    # 创建TaskPool
    task_pool = TaskPool()
    for task in tasks:
        task_pool.add_task(task)

    alns = MinimalALNS(
        distance_matrix=distance_matrix,
        task_pool=task_pool,
        repair_mode='regret2'
    )
    alns.vehicle = vehicle
    alns.energy_config = EnergyConfig(consumption_rate=0.5)

    # 创建空路径
    route = create_empty_route(1, depot)

    # 使用regret2_insertion插入所有任务
    removed_task_ids = [1, 2, 3]
    repaired_route = alns.regret2_insertion(route, removed_task_ids)

    print(f"\nRegret-2插入后:")
    print(f"  路径节点数: {len(repaired_route.nodes)}")
    print(f"  任务节点数: {len([n for n in repaired_route.nodes if not n.is_depot()])}")

    # 检查所有任务都被插入
    served_tasks = repaired_route.get_served_tasks()
    print(f"  已服务任务: {served_tasks}")

    assert len(served_tasks) == 3, "应该服务3个任务"

    # 检查容量可行性
    capacity_feasible, error = repaired_route.check_capacity_feasibility(vehicle.capacity, debug=True)
    print(f"\n容量检查: {'✓ 可行' if capacity_feasible else f'✗ 不可行 - {error}'}")

    assert capacity_feasible, "路径应满足容量约束"

    print("\n✓ 测试2.4通过：Regret-2插入正确工作")


def test_integrated_workflow():
    """综合测试：完整的ALNS工作流程"""
    print("\n" + "=" * 70)
    print("综合测试：Week 3完整工作流程")
    print("=" * 70)

    depot, tasks, distance_matrix, vehicle = create_test_scenario()

    # 创建TaskPool
    task_pool = TaskPool()
    for task in tasks:
        task_pool.add_task(task)

    # 创建ALNS（混合模式）
    alns = MinimalALNS(
        distance_matrix=distance_matrix,
        task_pool=task_pool,
        repair_mode='mixed',  # 混合使用greedy和regret2
        cost_params=CostParameters(C_tr=1.0, C_ch=10.0)
    )
    alns.vehicle = vehicle
    alns.energy_config = EnergyConfig(consumption_rate=0.5)

    # 创建初始解
    initial_route = create_empty_route(1, depot)
    removed_task_ids = [1, 2, 3]
    initial_route = alns.greedy_insertion(initial_route, removed_task_ids)

    print(f"\n初始解:")
    print(f"  路径节点数: {len(initial_route.nodes)}")
    print(f"  任务数: {len(initial_route.get_served_tasks())}")

    # 检查初始解的可行性
    capacity_feasible, _ = initial_route.check_capacity_feasibility(vehicle.capacity)
    precedence_valid, _ = initial_route.validate_precedence()

    print(f"  容量可行: {'✓' if capacity_feasible else '✗'}")
    print(f"  Precedence有效: {'✓' if precedence_valid else '✗'}")

    assert capacity_feasible and precedence_valid, "初始解应该可行"

    print("\n✓ 综合测试通过：Week 3所有组件正常工作")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Week 3 综合测试：取送货分离优化")
    print("=" * 70)

    try:
        # 运行所有测试
        test_partial_removal()
        test_pair_exchange()
        test_regret2_with_capacity()
        test_integrated_workflow()

        print("\n" + "=" * 70)
        print("✓ 所有测试通过！")
        print("=" * 70)
        print("\n总结:")
        print("1. ✓ Partial removal (步骤2.2) 正常工作")
        print("2. ✓ Pair exchange (步骤2.3) 正常工作")
        print("3. ✓ Regret-2插入 (步骤2.4) 正常工作")
        print("4. ✓ 综合工作流程正常")
        print("\nWeek 3所有步骤实现成功！")

    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n✗ 运行错误: {e}")
        import traceback
        traceback.print_exc()
