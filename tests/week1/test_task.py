"""测试Task模块"""
import sys
sys.path.append('src')

from core.node import create_task_node_pair
from core.task import create_task, TaskPool, TaskStatus
from physics.time import TimeWindow

def test_task_creation():
    """测试任务创建"""
    print("=" * 50)
    print("测试1: 任务创建")
    print("=" * 50)
    
    # 创建节点对
    pickup, delivery = create_task_node_pair(
        task_id=1,
        pickup_id=1,
        delivery_id=4,
        pickup_coords=(10, 20),
        delivery_coords=(15, 25),
        pickup_time_window=TimeWindow(0, 100),
        delivery_time_window=TimeWindow(50, 150)
    )
    
    print(f"✓ Pickup节点: {pickup}")
    print(f"✓ Delivery节点: {delivery}")
    
    # 创建任务
    task = create_task(1, pickup, delivery)
    print(f"\n✓ 任务创建成功: {task}")
    print(f"  - Pickup ID: {task.pickup_id}")
    print(f"  - Delivery ID: {task.delivery_id}")
    print(f"  - 需求量: {task.demand}")
    print(f"  - 总服务时间: {task.get_total_service_time()}秒")

def test_task_pool():
    """测试任务池"""
    print("\n" + "=" * 50)
    print("测试2: 任务池管理")
    print("=" * 50)
    
    # 创建多个任务
    pool = TaskPool()
    
    for i in range(1, 4):  # 创建3个任务
        pickup, delivery = create_task_node_pair(
            task_id=i,
            pickup_id=i,
            delivery_id=i + 3,
            pickup_coords=(i*10, i*20),
            delivery_coords=(i*10+5, i*20+5),
            pickup_time_window=TimeWindow(0, 100),
            delivery_time_window=TimeWindow(50, 150)
        )
        task = create_task(i, pickup, delivery, priority=i)
        pool.add_task(task)
    
    print(f"✓ 创建了3个任务")
    print(f"  任务池: {pool}")
    
    # 获取统计信息
    stats = pool.get_statistics()
    print(f"\n✓ 统计信息:")
    print(f"  - 总任务数: {stats['total']}")
    print(f"  - 待分配: {stats['pending']}")
    print(f"  - 已分配: {stats['assigned']}")
    print(f"  - 已完成: {stats['completed']}")
    
    # 分配任务
    print(f"\n✓ 分配任务1给AMR-1")
    pool.assign_task(1, vehicle_id=1)
    
    print(f"✓ 分配任务2给AMR-2")
    pool.assign_task(2, vehicle_id=2)
    
    stats = pool.get_statistics()
    print(f"\n  更新后的统计:")
    print(f"  - 待分配: {stats['pending']}")
    print(f"  - 已分配: {stats['assigned']}")
    
    # 查询特定AMR的任务
    vehicle_1_tasks = pool.get_tasks_for_vehicle(1)
    print(f"\n✓ AMR-1的任务: {[str(t) for t in vehicle_1_tasks]}")

def test_task_state_transitions():
    """测试任务状态转换"""
    print("\n" + "=" * 50)
    print("测试3: 任务状态转换")
    print("=" * 50)
    
    # 创建任务
    pickup, delivery = create_task_node_pair(
        task_id=1,
        pickup_id=1,
        delivery_id=4,
        pickup_coords=(10, 20),
        delivery_coords=(15, 25),
        pickup_time_window=TimeWindow(0, 100),
        delivery_time_window=TimeWindow(50, 150)
    )
    task = create_task(1, pickup, delivery)
    
    pool = TaskPool()
    pool.add_task(task)
    
    tracker = pool.get_tracker(1)
    print(f"初始状态: {tracker.status}")
    
    # 状态转换
    print(f"\n✓ 分配任务给AMR-1")
    pool.assign_task(1, vehicle_id=1)
    print(f"  状态: {tracker.status}")
    
    print(f"\n✓ 开始执行（pickup完成）")
    tracker.start_execution(pickup_time=50.0)
    print(f"  状态: {tracker.status}")
    print(f"  Pickup时间: {tracker.pickup_time}")
    
    print(f"\n✓ 完成任务（delivery完成）")
    tracker.complete(delivery_time=150.0)
    print(f"  状态: {tracker.status}")
    print(f"  Delivery时间: {tracker.delivery_time}")
    print(f"  执行耗时: {tracker.get_execution_time()}秒")

if __name__ == "__main__":
    test_task_creation()
    test_task_pool()
    test_task_state_transitions()
    
    print("\n" + "=" * 50)
    print("🎉 所有测试通过！")
    print("=" * 50)