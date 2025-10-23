"""
对照实验：Greedy vs Regret-2
"""

import sys
sys.path.append('src')
import random
import numpy as np

from planner.alns import MinimalALNS
from physics.distance import DistanceMatrix
from core.node import create_depot, create_task_node_pair
from core.task import create_task, TaskPool
from core.route import Route
from physics.time import TimeWindow


def create_test_scenario():
    """创建测试场景：1个AMR，5个任务"""
    depot = create_depot((0, 0))
    
    tasks = []
    task_pool = TaskPool()
    
    locations = [
        ((10, 10), (15, 15)),
        ((50, 50), (55, 55)),
        ((20, 20), (25, 25)),
        ((45, 45), (48, 48)),
        ((12, 12), (17, 17)),
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
    
    initial_route = Route(
        vehicle_id=1,
        nodes=[depot] + nodes_list + [depot]
    )
    
    return initial_route, task_pool


def test_single_mode(mode_name, repair_mode):
    """测试单一算子模式"""
    print("=" * 60)
    print(f"实验：{mode_name}（10次运行）")
    print("=" * 60)
    
    results = []
    
    for run in range(10):
        random.seed(run)
        
        initial_route, task_pool = create_test_scenario()
        distance_matrix = DistanceMatrix(
            coordinates={node.node_id: node.coordinates for node in initial_route.nodes},
            num_tasks=5,
            num_charging_stations=0
        )
        
        initial_cost = initial_route.calculate_total_distance(distance_matrix)
        
        # ✅ 关键：这里传入repair_mode参数
        alns = MinimalALNS(distance_matrix, task_pool, repair_mode=repair_mode)
        
        optimized_route = alns.optimize(initial_route, max_iterations=300)
        
        final_cost = optimized_route.calculate_total_distance(distance_matrix)
        improvement = (initial_cost - final_cost) / initial_cost * 100
        
        results.append(improvement)
        print(f"Run {run+1}: {improvement:.1f}%")
    
    print(f"\n{mode_name}统计:")
    print(f"  平均: {np.mean(results):.1f}%")
    print(f"  标准差: {np.std(results):.1f}%")
    print(f"  最好: {np.max(results):.1f}%")
    print(f"  最差: {np.min(results):.1f}%")
    
    return results


if __name__ == "__main__":
    print("\n" + "🔬" * 30)
    print("ALNS算子对照实验")
    print("🔬" * 30 + "\n")
    
    # 实验A：纯Greedy
    greedy_results = test_single_mode("纯Greedy插入", repair_mode='greedy')
    
    print("\n")
    
    # 实验B：纯Regret-2
    regret_results = test_single_mode("纯Regret-2插入", repair_mode='regret2')
    
    print("\n" + "=" * 60)
    print("📊 对比结论:")
    print("=" * 60)
    print(f"Greedy:   平均 {np.mean(greedy_results):.1f}% (标准差 {np.std(greedy_results):.1f}%)")
    print(f"Regret-2: 平均 {np.mean(regret_results):.1f}% (标准差 {np.std(regret_results):.1f}%)")
    
    if np.mean(greedy_results) > np.mean(regret_results):
        print("\n✅ 结论：Greedy更好")
    else:
        print("\n✅ 结论：Regret-2更好")
