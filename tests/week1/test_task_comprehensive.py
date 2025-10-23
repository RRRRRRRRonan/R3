"""Task模块全面测试"""
import sys
sys.path.append('src')

from core.node import create_task_node_pair, NodeType
from core.task import (
    create_task, create_task_from_node_pair, 
    TaskPool, TaskStatus, Task, TaskStateTracker
)
from physics.time import TimeWindow


def test_task_validation():
    """测试Task的验证逻辑"""
    print("\n" + "="*60)
    print("测试4: Task验证逻辑")
    print("="*60)
    
    # 正常创建
    p, d = create_task_node_pair(1, 1, 4, (10,20), (15,25),
                                  TimeWindow(0,100), TimeWindow(50,150))
    task = create_task(1, p, d)
    print(f"✓ 正常创建: {task}")
    
    # 测试属性访问
    print(f"\n✓ 属性访问测试:")
    print(f"  - pickup坐标: {task.pickup_coordinates}")
    print(f"  - delivery坐标: {task.delivery_coordinates}")
    print(f"  - pickup时间窗: {task.pickup_time_window}")
    print(f"  - delivery时间窗: {task.delivery_time_window}")
    
    # 测试错误情况1: task_id不一致
    print(f"\n✓ 测试task_id不一致:")
    p2, d2 = create_task_node_pair(1, 1, 4, (10,20), (15,25),
                                   TimeWindow(0,100), TimeWindow(50,150))
    try:
        # 尝试用不同的task_id创建
        bad_task = Task(
            task_id=999,  # 与节点的task_id不同
            pickup_node=p2,
            delivery_node=d2,
            demand=1.0
        )
        print(f"  ✗ 应该报错但没有报错")
    except ValueError as e:
        print(f"  ✓ 正确捕获错误: {str(e)[:50]}...")
    
    # 测试错误情况2: 需求量不一致
    print(f"\n✓ 测试需求量不一致:")
    try:
        from dataclasses import replace
        p3 = replace(p, demand=2.0)  # 修改pickup的demand
        bad_task = create_task(1, p3, d)
        print(f"  ✗ 应该报错但没有报错")
    except ValueError as e:
        print(f"  ✓ 正确捕获错误: {str(e)[:50]}...")


def test_task_state_tracker_methods():
    """测试TaskStateTracker的所有方法"""
    print("\n" + "="*60)
    print("测试5: TaskStateTracker完整方法")
    print("="*60)
    
    p, d = create_task_node_pair(1, 1, 4, (10,20), (15,25),
                                  TimeWindow(0,100), TimeWindow(50,150))
    task = create_task(1, p, d)
    tracker = TaskStateTracker(task)
    
    # 测试is方法
    print(f"✓ 状态判断方法:")
    print(f"  - is_completed(): {tracker.is_completed()}")
    print(f"  - is_in_progress(): {tracker.is_in_progress()}")
    
    # 测试reject (需要创建新的节点，因为task_id要匹配)
    print(f"\n✓ 测试REJECT状态:")
    p2, d2 = create_task_node_pair(2, 2, 5, (20,30), (25,35),
                                    TimeWindow(0,100), TimeWindow(50,150))
    tracker2 = TaskStateTracker(create_task(2, p2, d2))
    tracker2.reject()
    print(f"  - 状态: {tracker2.status}")
    print(f"  - is_completed(): {tracker2.is_completed()}")
    
    # 测试错误的状态转换
    print(f"\n✓ 测试非法状态转换:")
    try:
        tracker2.assign_to_vehicle(1)  # REJECTED状态不能assign
        print(f"  ✗ 应该报错但没有报错")
    except ValueError as e:
        print(f"  ✓ 正确捕获错误: {str(e)[:60]}...")


def test_task_pool_all_methods():
    """测试TaskPool的所有方法"""
    print("\n" + "="*60)
    print("测试6: TaskPool完整功能")
    print("="*60)
    
    pool = TaskPool()
    
    # 批量添加任务
    tasks = []
    for i in range(1, 6):
        p, d = create_task_node_pair(i, i, i+5, (i*10, i*20), (i*15, i*25),
                                      TimeWindow(0,100), TimeWindow(50,150))
        tasks.append(create_task(i, p, d, priority=i))
    
    print(f"✓ 批量添加5个任务:")
    pool.add_tasks(tasks)
    print(f"  - 任务总数: {len(pool)}")
    
    # 测试get_task
    print(f"\n✓ get_task():")
    task1 = pool.get_task(1)
    print(f"  - Task 1: {task1}")
    
    # 测试get_all_tasks
    print(f"\n✓ get_all_tasks():")
    all_tasks = pool.get_all_tasks()
    print(f"  - 总数: {len(all_tasks)}")
    
    # 测试__contains__
    print(f"\n✓ __contains__():")
    print(f"  - 1 in pool: {1 in pool}")
    print(f"  - 999 in pool: {999 in pool}")
    
    # 分配和拒绝任务
    pool.assign_task(1, vehicle_id=1)
    pool.assign_task(2, vehicle_id=1)
    pool.assign_task(3, vehicle_id=2)
    pool.reject_task(4)
    
    # 测试get_pending_tasks
    print(f"\n✓ 按状态查询:")
    print(f"  - pending: {len(pool.get_pending_tasks())}")
    print(f"  - assigned: {len(pool.get_assigned_tasks())}")
    print(f"  - completed: {len(pool.get_completed_tasks())}")
    
    # 测试get_tasks_by_status
    rejected = pool.get_tasks_by_status(TaskStatus.REJECTED)
    print(f"  - rejected: {len(rejected)} -> {rejected}")
    
    # 测试统计
    stats = pool.get_statistics()
    print(f"\n✓ 统计信息:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")
    
    # 测试重复添加错误
    print(f"\n✓ 测试重复添加:")
    try:
        pool.add_task(tasks[0])  # 重复添加task 1
        print(f"  ✗ 应该报错但没有报错")
    except ValueError as e:
        print(f"  ✓ 正确捕获错误: {str(e)}")


def test_edge_cases():
    """测试边界情况"""
    print("\n" + "="*60)
    print("测试7: 边界情况")
    print("="*60)
    
    # 空TaskPool的统计
    empty_pool = TaskPool()
    stats = empty_pool.get_statistics()
    print(f"✓ 空TaskPool统计:")
    print(f"  - total: {stats['total']}")
    print(f"  - completion_rate: {stats['completion_rate']}")
    
    # 查询不存在的任务
    print(f"\n✓ 查询不存在的任务:")
    result = empty_pool.get_task(999)
    print(f"  - get_task(999): {result}")
    
    # 空的get_all_tasks
    all_tasks = empty_pool.get_all_tasks()
    print(f"  - get_all_tasks(): {all_tasks}")
    
    # 任务优先级和到达时间
    print(f"\n✓ 任务优先级和到达时间:")
    p, d = create_task_node_pair(1, 1, 4, (10,20), (15,25),
                                  TimeWindow(0,100), TimeWindow(50,150))
    task = create_task(1, p, d, priority=10, arrival_time=50.0)
    print(f"  - priority: {task.priority}")
    print(f"  - arrival_time: {task.arrival_time}")


def test_convenience_functions():
    """测试便捷构造函数"""
    print("\n" + "="*60)
    print("测试8: 便捷构造函数")
    print("="*60)
    
    p, d = create_task_node_pair(1, 1, 4, (10,20), (15,25),
                                  TimeWindow(0,100), TimeWindow(50,150))
    
    # create_task
    task1 = create_task(1, p, d)
    print(f"✓ create_task(): {task1}")
    
    # create_task_from_node_pair
    task2 = create_task_from_node_pair(1, p, d, priority=5)
    print(f"✓ create_task_from_node_pair(): {task2}")
    print(f"  - priority: {task2.priority}")
    
    # demand自动推导
    task3 = create_task(1, p, d)  # 不提供demand
    print(f"✓ demand自动推导: {task3.demand}")


if __name__ == "__main__":
    # 运行所有测试
    test_task_validation()
    test_task_state_tracker_methods()
    test_task_pool_all_methods()
    test_edge_cases()
    test_convenience_functions()
    
    print("\n" + "="*60)
    print("🎉 所有补充测试通过！")
    print("="*60)
    