"""
测试ALNS接口方法
==================
验证Route类的新方法能正常工作
"""

import sys
sys.path.append('src')

from core.node import DepotNode, create_task_node_pair, create_depot
from core.task import create_task
from core.route import Route
from copy import deepcopy
from physics.time import TimeWindow


def test_route_copy():
    """测试1：copy()方法"""
    print("=" * 60)
    print("测试1：Route.copy()")
    print("=" * 60)
    
    # 创建depot（注意：只传coordinates）
    depot = DepotNode(coordinates=(0, 0))
    
    # 创建任务节点
    p1, d1 = create_task_node_pair(
        task_id=1,
        pickup_id=1,
        delivery_id=2,
        pickup_coords=(10, 10),
        delivery_coords=(20, 20),
        pickup_time_window=TimeWindow(0, 100),
        delivery_time_window=TimeWindow(50, 150)
    )
    
    # 创建路径（注意：终点depot需要复制）
    route1 = Route(vehicle_id=1, nodes=[depot, p1, d1, deepcopy(depot)])
    
    # 复制
    route2 = route1.copy()
    
    # 修改route2
    route2.nodes.append(DepotNode(coordinates=(99, 99)))
    
    # 验证route1没有被影响
    print(f"✓ 原路径节点数: {len(route1.nodes)}")
    print(f"✓ 复制路径节点数: {len(route2.nodes)}")
    assert len(route1.nodes) == 4
    assert len(route2.nodes) == 5
    print("✓ 深拷贝工作正常！\n")

def test_insert_and_remove_task():
    """测试2：insert_task()和remove_task()"""
    print("=" * 60)
    print("测试2：insert_task() 和 remove_task()")
    print("=" * 60)
    
    # 创建初始路径：Depot → B → Depot
    depot = create_depot((0, 0))
    
    pB, dB = create_task_node_pair(
        task_id=2,
        pickup_id=3,
        delivery_id=4,
        pickup_coords=(30, 30),
        delivery_coords=(40, 40),
        pickup_time_window=TimeWindow(0, 200),
        delivery_time_window=TimeWindow(50, 250)
    )
    taskB = create_task(2, pB, dB)
    
    route = Route(vehicle_id=1, nodes=[depot, pB, dB, deepcopy(depot)])
    print(f"初始路径: {[n.node_id for n in route.nodes]}")
    
    # 插入任务A
    pA, dA = create_task_node_pair(
        task_id=1,
        pickup_id=1,
        delivery_id=2,
        pickup_coords=(10, 10),
        delivery_coords=(20, 20),
        pickup_time_window=TimeWindow(0, 100),
        delivery_time_window=TimeWindow(50, 150)
    )
    taskA = create_task(1, pA, dA)
    
    route.insert_task(taskA, (1, 2))
    print(f"插入任务A后: {[n.node_id for n in route.nodes]}")
    assert len(route.nodes) == 6  # Depot, A取, A送, B取, B送, Depot
    print("✓ 插入成功")
    
    # 移除任务A
    route.remove_task(taskA)
    print(f"移除任务A后: {[n.node_id for n in route.nodes]}")
    assert len(route.nodes) == 4  # Depot, B取, B送, Depot
    print("✓ 移除成功\n")

def test_get_served_tasks():
    """测试3：get_served_tasks()"""
    print("=" * 60)
    print("测试3：get_served_tasks()")
    print("=" * 60)
    
    # 创建包含3个任务的路径
    depot = create_depot((0, 0))
    
    p1, d1 = create_task_node_pair(
        task_id=1, pickup_id=1, delivery_id=4,
        pickup_coords=(10, 10), delivery_coords=(20, 20),
        pickup_time_window=TimeWindow(0, 100),
        delivery_time_window=TimeWindow(50, 150)
    )
    
    p2, d2 = create_task_node_pair(
        task_id=2, pickup_id=2, delivery_id=5,
        pickup_coords=(30, 30), delivery_coords=(40, 40),
        pickup_time_window=TimeWindow(0, 100),
        delivery_time_window=TimeWindow(50, 150)
    )
    
    p3, d3 = create_task_node_pair(
        task_id=3, pickup_id=3, delivery_id=6,
        pickup_coords=(50, 50), delivery_coords=(60, 60),
        pickup_time_window=TimeWindow(0, 100),
        delivery_time_window=TimeWindow(50, 150)
    )
    
    route = Route(
        vehicle_id=1,
        nodes=[depot, p1, d1, p2, d2, p3, d3, deepcopy(depot)]
    )
    
    task_ids = route.get_served_tasks()
    print(f"路径中的任务: {task_ids}")
    assert len(task_ids) == 3
    assert 1 in task_ids
    assert 2 in task_ids
    assert 3 in task_ids
    print("✓ get_served_tasks()工作正常\n")

def test_combined_operations():
    """测试4：组合操作（模拟ALNS的一次destroy-repair）"""
    print("=" * 60)
    print("测试4：模拟ALNS操作")
    print("=" * 60)
    
    # 创建初始路径：3个任务
    depot = create_depot((0, 0))
    
    p1, d1 = create_task_node_pair(
        task_id=1, pickup_id=1, delivery_id=4,
        pickup_coords=(10, 10), delivery_coords=(20, 20),
        pickup_time_window=TimeWindow(0, 100),
        delivery_time_window=TimeWindow(50, 150)
    )
    
    p2, d2 = create_task_node_pair(
        task_id=2, pickup_id=2, delivery_id=5,
        pickup_coords=(30, 30), delivery_coords=(40, 40),
        pickup_time_window=TimeWindow(0, 100),
        delivery_time_window=TimeWindow(50, 150)
    )
    
    p3, d3 = create_task_node_pair(
        task_id=3, pickup_id=3, delivery_id=6,
        pickup_coords=(50, 50), delivery_coords=(60, 60),
        pickup_time_window=TimeWindow(0, 100),
        delivery_time_window=TimeWindow(50, 150)
    )
    
    task1 = create_task(1, p1, d1)
    task2 = create_task(2, p2, d2)
    task3 = create_task(3, p3, d3)
    
    route = Route(
        vehicle_id=1,
        nodes=[depot, p1, d1, p2, d2, p3, d3, deepcopy(depot)]
    )
    print(f"初始路径任务: {route.get_served_tasks()}")
    
    # ALNS Destroy: 移除任务1和3
    print("\n[Destroy] 移除任务1和3...")
    route_destroyed = route.copy()
    route_destroyed.remove_task(task1)
    route_destroyed.remove_task(task3)
    print(f"Destroy后任务: {route_destroyed.get_served_tasks()}")
    assert route_destroyed.get_served_tasks() == [2]
    
    # ALNS Repair: 重新插入任务1和3
    print("\n[Repair] 重新插入任务1和3...")
    route_destroyed.insert_task(task1, (1, 2))  # 插在前面
    route_destroyed.insert_task(task3, (5, 6))  # 插在后面
    print(f"Repair后任务: {route_destroyed.get_served_tasks()}")
    assert len(route_destroyed.get_served_tasks()) == 3
    
    print("✓ ALNS操作流程验证成功！\n")


if __name__ == "__main__":
    test_route_copy()
    test_insert_and_remove_task()
    test_get_served_tasks()
    test_combined_operations()
    
    print("=" * 60)
    print("🎉 所有ALNS接口测试通过！")
    print("=" * 60)
