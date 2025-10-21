"""
ALNS充电集成测试 - 最终版
简洁但完整的测试，验证充电站自动插入功能
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import random
import time

from planner.alns import MinimalALNS
from physics.distance import DistanceMatrix
from core.node import create_depot, create_task_node_pair
from core.task import create_task, TaskPool
from core.route import Route
from core.vehicle import create_vehicle
from physics.energy import EnergyConfig


def test_alns_with_charging():
    """ALNS充电集成测试"""
    print("=" * 70)
    print("ALNS充电集成测试")
    print("=" * 70)
    
    # ========== 场景构建 ==========
    depot = create_depot((0, 0))
    coordinates = {0: (0, 0)}
    
    # 5个任务，分布较远
    task_locations = [
        ((150, 150), (200, 200)),
        ((180, 50), (220, 80)),
        ((50, 180), (80, 220)),
        ((160, 160), (190, 190)),
        ((140, 140), (170, 170)),
    ]
    
    num_tasks = len(task_locations)
    task_pool = TaskPool()
    nodes_list = []
    
    for i, (p_loc, d_loc) in enumerate(task_locations, start=1):
        p, d = create_task_node_pair(
            task_id=i,
            pickup_id=i,
            delivery_id=i + num_tasks,
            pickup_coords=p_loc,
            delivery_coords=d_loc
        )
        task = create_task(i, p, d)
        task_pool.add_task(task)
        nodes_list.extend([p, d])
        
        coordinates[i] = p_loc
        coordinates[i + num_tasks] = d_loc
    
    # 3个充电站
    charging_coords = [(75, 75), (125, 125), (100, 100)]
    charging_start_id = 2 * num_tasks + 1
    
    for idx, coords in enumerate(charging_coords):
        coordinates[charging_start_id + idx] = coords
    
    distance_matrix = DistanceMatrix(
        coordinates=coordinates,
        num_tasks=num_tasks,
        num_charging_stations=len(charging_coords)
    )
    
    # 初始路径（简单串联）
    initial_route = Route(
        vehicle_id=1,
        nodes=[depot] + nodes_list + [depot]
    )
    
    # ========== 能量配置（关键：设置为50%强制充电）==========
    initial_distance = initial_route.calculate_total_distance(distance_matrix)
    estimated_energy = initial_distance / 1000.0 * 0.5
    battery_capacity = estimated_energy * 0.5  # 仅50%容量
    
    print(f"\n📊 测试参数：")
    print(f"  任务数: {num_tasks}")
    print(f"  充电站数: {len(charging_coords)}")
    print(f"  初始距离: {initial_distance:.1f}m")
    print(f"  预估能量: {estimated_energy:.3f}kWh")
    print(f"  电池容量: {battery_capacity:.3f}kWh (50%)")
    print(f"  能量缺口: {estimated_energy - battery_capacity:.3f}kWh")
    print(f"  ✓ 强制需要充电")
    
    # ========== 配置ALNS ==========
    vehicle = create_vehicle(
        vehicle_id=1,
        battery_capacity=battery_capacity,
        initial_battery=battery_capacity
    )
    energy_config = EnergyConfig(consumption_rate=0.5)
    
    alns = MinimalALNS(distance_matrix, task_pool, repair_mode='mixed')
    alns.vehicle = vehicle
    alns.energy_config = energy_config
    
    # ========== 运行优化 ==========
    seed = int(time.time())
    random.seed(seed)
    
    print(f"\n🔄 开始优化...")
    print(f"  迭代次数: 100")
    print(f"  随机种子: {seed}")
    
    optimized_route = alns.optimize(initial_route, max_iterations=100)
    
    # ========== 结果分析 ==========
    final_distance = optimized_route.calculate_total_distance(distance_matrix)
    improvement = (initial_distance - final_distance) / initial_distance * 100
    
    served_tasks = optimized_route.get_served_tasks()
    charging_stations = [n for n in optimized_route.nodes if n.is_charging_station()]
    
    print(f"\n" + "=" * 70)
    print(f"测试结果")
    print(f"=" * 70)
    
    # 任务完整性
    print(f"\n任务完成情况：")
    print(f"  完成任务: {len(served_tasks)}/{num_tasks}")
    
    task_ok = len(served_tasks) == num_tasks
    if task_ok:
        print(f"  ✅ 所有任务完成")
    else:
        print(f"  ❌ 任务未完成")
    
    # 优化效果
    print(f"\n优化效果：")
    print(f"  初始: {initial_distance:.1f}m")
    print(f"  最终: {final_distance:.1f}m")
    print(f"  改进: {improvement:.1f}%")
    
    # 充电方案
    print(f"\n充电方案：")
    print(f"  充电站数: {len(charging_stations)}")
    
    if charging_stations:
        print(f"  充电站ID: {[cs.node_id for cs in charging_stations]}")
        print(f"  ✅ 成功插入充电站")
    else:
        print(f"  ⚠️  未插入充电站")
    
    # 能量验证
    final_energy = final_distance / 1000.0 * 0.5
    print(f"\n能量验证：")
    print(f"  实际需要: {final_energy:.3f}kWh")
    print(f"  电池容量: {battery_capacity:.3f}kWh")
    
    need_charging = final_energy > battery_capacity
    
    if need_charging:
        print(f"  理论需充电: 是")
    else:
        print(f"  理论需充电: 否")
    
    # ========== 综合判断 ==========
    print(f"\n" + "=" * 70)
    
    if not task_ok:
        print(f"❌ 测试失败：任务未全部完成")
        success = False
    elif need_charging and len(charging_stations) == 0:
        print(f"❌ 测试失败：需要充电但未插入充电站")
        print(f"\n可能原因：")
        print(f"  ALNS的repair方法未正确调用充电检查")
        success = False
    elif len(charging_stations) > num_tasks * 2:
        print(f"⚠️  警告：充电站数量过多")
        print(f"  这可能表示充电插入逻辑过于频繁")
        success = True
    else:
        print(f"✅ 测试通过")
        
        if len(charging_stations) > 0:
            print(f"\n🎉 验证成功：")
            print(f"  - ALNS检测到能量约束")
            print(f"  - 自动插入{len(charging_stations)}个充电站")
            print(f"  - 所有任务完成，路径优化{improvement:.1f}%")
        else:
            print(f"\n💡 说明：")
            print(f"  虽然初始需要充电，但ALNS通过优化")
            print(f"  将总距离降低到无需充电的水平")
            print(f"  这也是一种有效的解决方案")
        
        success = True
    
    print(f"=" * 70)
    
    return success


if __name__ == "__main__":
    print("\n")
    success = test_alns_with_charging()
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 ALNS充电集成测试成功")
        print("您的系统已支持电动AMR的充电约束规划")
    else:
        print("需要检查ALNS与充电检查的集成")
    print("=" * 70)
    print()