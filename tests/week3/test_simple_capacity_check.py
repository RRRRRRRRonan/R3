"""
Week 3 步骤2.1 简单测试：容量可行性检查
"""

import sys
sys.path.append('src')

from core.node import DepotNode, create_task_node_pair
from core.task import Task
from core.route import Route, create_empty_route

def test_basic_capacity_check():
    """测试基本容量检查"""
    print("测试容量可行性检查方法...")

    # 创建depot
    depot = DepotNode(coordinates=(0.0, 0.0))

    # 创建1个任务，需求40kg
    pickup, delivery = create_task_node_pair(
        task_id=1,
        pickup_id=1,
        delivery_id=2,
        pickup_coords=(10000.0, 0.0),
        delivery_coords=(10000.0, 20000.0),
        demand=40.0
    )

    task = Task(
        task_id=1,
        pickup_node=pickup,
        delivery_node=delivery,
        demand=40.0
    )

    # 创建路径
    route = create_empty_route(1, depot)
    route.nodes.insert(-1, pickup)
    route.nodes.insert(-1, delivery)

    # 检查容量（容量100kg，需求40kg）
    feasible, error = route.check_capacity_feasibility(100.0, debug=True)

    print(f"\n结果: {'✓ 可行' if feasible else f'✗ 不可行 - {error}'}")

    assert feasible, "应该可行"
    print("\n✓ 测试通过")

if __name__ == "__main__":
    test_basic_capacity_check()
