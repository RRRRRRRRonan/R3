import sys
sys.path.append('src')

from planner.alns import MinimalALNS
from physics.distance import DistanceMatrix
from test_minimal_alns import create_test_scenario

def test_regret2_only():
    """单独测试Regret-2是否正常工作"""
    print("=" * 60)
    print("测试：纯Regret-2优化")
    print("=" * 60)
    
    initial_route, task_pool = create_test_scenario()
    
    distance_matrix = DistanceMatrix(
        coordinates={node.node_id: node.coordinates for node in initial_route.nodes},
        num_tasks=5,
        num_charging_stations=0
    )
    
    initial_cost = initial_route.calculate_total_distance(distance_matrix)
    print(f"初始成本: {initial_cost:.2f}m")
    
    # 只测试一次Regret-2 repair
    alns = MinimalALNS(distance_matrix, task_pool)
    
    # 破坏路径
    destroyed_route, removed_tasks = alns.random_removal(initial_route, q=2)
    print(f"移除任务: {removed_tasks}")
    print(f"破坏后成本: {destroyed_route.calculate_total_distance(distance_matrix):.2f}m")
    
    # 用Regret-2修复
    try:
        repaired_route = alns.regret2_insertion(destroyed_route, removed_tasks)
        repaired_cost = repaired_route.calculate_total_distance(distance_matrix)
        print(f"Regret-2修复后成本: {repaired_cost:.2f}m")
        print(f"修复后任务: {repaired_route.get_served_tasks()}")
    except Exception as e:
        print(f"❌ Regret-2失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_regret2_only()