"""
测试最简ALNS
============
验证ALNS能优化路径距离
"""

import sys
sys.path.append('src')

from core.node import create_depot, create_task_node_pair
from core.task import create_task, TaskPool
from core.route import Route
from core.vehicle import create_vehicle
from physics.distance import create_distance_matrix_from_layout
from physics.time import TimeWindow
from planner.alns import MinimalALNS


def create_test_scenario():
    """创建测试场景：1个AMR，5个任务"""
    
    # 创建节点
    depot = create_depot((0, 0))
    
    tasks = []
    task_pool = TaskPool()
    
    # 创建5个任务（故意设计成初始顺序不优）
    locations = [
        ((10, 10), (15, 15)),   # 任务1：近
        ((50, 50), (55, 55)),   # 任务2：远
        ((20, 20), (25, 25)),   # 任务3：中
        ((45, 45), (48, 48)),   # 任务4：远
        ((12, 12), (17, 17)),   # 任务5：近
    ]
    
    nodes_list = []
    for i, (pickup_loc, delivery_loc) in enumerate(locations, start=1):
        p, d = create_task_node_pair(
            task_id=i,
            pickup_id=i,
            delivery_id=i + 10,
            pickup_coords=pickup_loc,
            delivery_coords=delivery_loc,
            pickup_time_window=TimeWindow(0, 1000),
            delivery_time_window=TimeWindow(0, 1500)
        )
        task = create_task(i, p, d)
        tasks.append(task)
        task_pool.add_task(task)
        nodes_list.extend([p, d])
    
    # 创建初始路径（按任务顺序：1,2,3,4,5）
    initial_route = Route(
        vehicle_id=1,
        nodes=[depot] + nodes_list + [depot]
    )
    
    return initial_route, task_pool


def test_alns_optimization():
    """测试ALNS能否改进路径"""
    print("=" * 60)
    print("测试：ALNS路径优化")
    print("=" * 60)
    
    # 创建场景
    initial_route, task_pool = create_test_scenario()
    
    # 创建距离矩阵
    # 先收集所有节点
    all_nodes = initial_route.nodes
    coords = {node.node_id: node.coordinates for node in all_nodes}
    distance_matrix = create_distance_matrix_from_layout(coords)
    
    # 初始成本
    initial_cost = initial_route.calculate_total_distance(distance_matrix)
    print(f"\n初始路径成本: {initial_cost:.2f}m")
    print(f"初始任务顺序: {initial_route.get_served_tasks()}")
    
    # 运行ALNS
    alns = MinimalALNS(distance_matrix, task_pool)
    optimized_route = alns.optimize(initial_route, max_iterations=100)
    
    # 优化后成本
    final_cost = optimized_route.calculate_total_distance(distance_matrix)
    improvement = initial_cost - final_cost
    improvement_pct = (improvement / initial_cost) * 100
    
    print(f"\n优化后路径成本: {final_cost:.2f}m")
    print(f"优化后任务顺序: {optimized_route.get_served_tasks()}")
    print(f"改进: {improvement:.2f}m ({improvement_pct:.1f}%)")
    
    # 验证
    assert final_cost <= initial_cost, "ALNS应该改进或保持成本不变"
    print("\n✓ ALNS优化测试通过！")


if __name__ == "__main__":
    test_alns_optimization()
