"""
Debug: 测试insert_task的行为
"""
import sys
sys.path.append('src')

from core.node import create_depot, create_task_node_pair
from core.task import create_task
from core.route import Route


def test_insert_task_basic():
    """测试insert_task的基本行为"""
    print("=" * 60)
    print("测试：insert_task 方法")
    print("=" * 60)
    
    depot = create_depot((0, 0))
    
    p1, d1 = create_task_node_pair(
        task_id=1,
        pickup_id=1,
        delivery_id=2,
        pickup_coords=(10, 10),
        delivery_coords=(20, 20)
    )
    task1 = create_task(1, p1, d1)
    
    # 初始路径
    route = Route(vehicle_id=1, nodes=[depot, depot])
    print(f"初始路径: {[str(n) for n in route.nodes]}")
    print(f"  索引0: {route.nodes[0]}")
    print(f"  索引1: {route.nodes[1]}")
    
    # 插入任务
    print(f"\n插入任务 (pickup_pos=1, delivery_pos=2):")
    route.insert_task(task1, (1, 2))
    
    print(f"插入后路径: {[str(n) for n in route.nodes]}")
    for i, node in enumerate(route.nodes):
        print(f"  索引{i}: {node}")
    
    # 检查顺序
    print("\n✅ 期望顺序: Depot → Pickup → Delivery → Depot")
    if (route.nodes[0].is_depot() and 
        route.nodes[1].is_pickup() and 
        route.nodes[2].is_delivery() and 
        route.nodes[3].is_depot()):
        print("✅✅✅ 顺序正确！")
    else:
        print("❌❌❌ 顺序错误！")


if __name__ == "__main__":
    test_insert_task_basic()