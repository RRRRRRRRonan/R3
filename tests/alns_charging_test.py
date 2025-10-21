"""
完整的ALNS充电集成测试
确保充电站正确配置并被插入

测试步骤：
1. 创建场景（depot + 任务 + 充电站）
2. 配置小电池（强制需要充电）
3. 验证ALNS是否插入充电站
"""
import sys
sys.path.append('src')

from core.node import create_depot, create_task_node_pair, create_charging_node, NodeType
from core.task import create_task, TaskPool
from core.vehicle import create_vehicle
from core.route import Route
from physics.distance import DistanceMatrix
from physics.energy import EnergyConfig, calculate_energy_consumption
from planner.alns import MinimalALNS
import random


def test_alns_with_charging_comprehensive():
    """完整的ALNS充电测试"""
    print("=" * 70)
    print("ALNS充电功能完整测试")
    print("=" * 70)
    
    # ========== 第1步：创建场景 ==========
    print("\n【第1步】创建测试场景")
    print("-" * 70)
    
    depot = create_depot((0, 0))
    
    # 创建5个任务（确保距离足够远，需要充电）
    task_locations = [
        ((50, 50), (60, 60)),    # Task 1
        ((70, 70), (80, 80)),    # Task 2
        ((90, 10), (100, 20)),   # Task 3
        ((30, 90), (40, 100)),   # Task 4
        ((110, 110), (120, 120)) # Task 5
    ]
    
    task_pool = TaskPool()
    nodes_list = []
    coordinates = {0: (0, 0)}  # depot坐标
    
    for i, (pickup_loc, delivery_loc) in enumerate(task_locations, start=1):
        pickup_id = 2 * i - 1
        delivery_id = 2 * i
        
        p, d = create_task_node_pair(
            task_id=i,
            pickup_id=pickup_id,
            delivery_id=delivery_id,
            pickup_coords=pickup_loc,
            delivery_coords=delivery_loc
        )
        task = create_task(i, p, d)
        task_pool.add_task(task)
        nodes_list.extend([p, d])
        
        coordinates[pickup_id] = pickup_loc
        coordinates[delivery_id] = delivery_loc
    
    print(f"✓ 创建了 {len(task_locations)} 个任务")
    
    # ========== 关键：创建多个充电站 ==========
    charging_coords = [
        (50, 0),    # 充电站1 - 靠近起点
        (75, 75),   # 充电站2 - 中心位置
        (100, 50),  # 充电站3 - 右侧
        (50, 100),  # 充电站4 - 上方
    ]
    
    charging_start_id = 2 * len(task_locations) + 1

    for idx, coords in enumerate(charging_coords):
        coordinates[charging_start_id + idx] = coords
    
    print(f"✓ 创建了 {len(charging_coords)} 个充电站")
    print(f"  充电站位置: {charging_coords}")
    
    # ========== 第2步：创建距离矩阵（包含充电站） ==========
    print("\n【第2步】创建距离矩阵")
    print("-" * 70)
    
    distance_matrix = DistanceMatrix(
        coordinates=coordinates,
        num_tasks=len(task_locations),
        num_charging_stations=len(charging_coords)
    )
    
    print(f"✓ 距离矩阵创建成功")
    print(f"  总节点数: {len(coordinates)}")
    print(f"  任务节点: {len(task_locations) * 2}")
    print(f"  充电站数: {len(charging_coords)}")
    
    # 验证充电站是否在距离矩阵中
    try:
        test_station_id, test_dist = distance_matrix.get_nearest_charging_station(1)
        print(f"✓ 充电站查询测试成功: 最近充电站ID={test_station_id}, 距离={test_dist:.2f}m")
    except Exception as e:
        print(f"❌ 充电站查询失败: {e}")
        print(f"  这是问题所在！距离矩阵中没有正确配置充电站")
        return False
    
    # ========== 第3步：创建初始路径 ==========
    print("\n【第3步】创建初始路径")
    print("-" * 70)
    
    initial_route = Route(
        vehicle_id=1,
        nodes=[depot] + nodes_list + [depot]
    )
    
    initial_distance = initial_route.calculate_total_distance(distance_matrix)
    print(f"✓ 初始路径创建成功")
    print(f"  总距离: {initial_distance:.2f}m")
    print(f"  节点数: {len(initial_route.nodes)}")
    
    # ========== 第4步：配置小电池（强制充电） ==========
    print("\n【第4步】配置AMR参数（强制充电场景）")
    print("-" * 70)
    
    # 计算预估能量需求
    estimated_energy = initial_distance / 1000.0 * 0.5  # kWh
    
    # 🔧 关键：设置电池容量为预估能量的70%，强制需要充电
    battery_capacity = estimated_energy * 0.7
    
    vehicle = create_vehicle(
        vehicle_id=1,
        battery_capacity=battery_capacity,
        initial_battery=battery_capacity
    )
    energy_config = EnergyConfig(consumption_rate=0.5)
    
    print(f"✓ AMR配置完成（小电池）")
    print(f"  电池容量: {battery_capacity:.3f}kWh")
    print(f"  预估需要: {estimated_energy:.3f}kWh")
    print(f"  能量缺口: {estimated_energy - battery_capacity:.3f}kWh")
    print(f"  📌 容量不足，必须充电！")
    
    # ========== 第5步：手动测试充电检查功能 ==========
    print("\n【第5步】手动测试充电检查功能")
    print("-" * 70)
    
    # 创建一个简单的测试路径
    test_route = Route(vehicle_id=1, nodes=[depot, depot])
    task1 = task_pool.get_task(1)
    
    # 测试插入第一个任务是否会触发充电
    feasible, charging_plan = test_route.check_energy_feasibility_for_insertion(
        task=task1,
        insert_position=(1, 2),
        vehicle=vehicle,
        distance_matrix=distance_matrix,
        energy_config=energy_config
    )
    
    print(f"充电检查结果:")
    print(f"  可行性: {feasible}")
    print(f"  充电计划: {charging_plan}")
    
    if charging_plan:
        print(f"✓ 充电检查正常工作！")
        print(f"  需要插入 {len(charging_plan)} 个充电站:")
        for i, plan in enumerate(charging_plan, 1):
            print(f"    {i}. 充电站 {plan['station_node'].node_id}")
            print(f"       位置: {plan['position']}")
            print(f"       充电量: {plan['amount']:.2f}kWh")
    else:
        print(f"⚠️ 警告：充电检查没有返回充电计划")
        print(f"  可能原因：")
        print(f"  1. 电池容量足够（不需要充电）")
        print(f"  2. 充电站配置有问题")
        print(f"  3. check_energy_feasibility_for_insertion 逻辑有bug")
        
        if not feasible and not charging_plan:
            print(f"\n❌ 严重问题：不可行但没有充电计划！")
            return False
    
    # ========== 第6步：配置ALNS并运行优化 ==========
    print("\n【第6步】运行ALNS优化")
    print("-" * 70)
    
    random.seed(42)  # 固定随机种子便于复现
    
    alns = MinimalALNS(distance_matrix, task_pool, repair_mode='greedy') #设置repair_mode的格式
    
    # 🔧 关键：必须设置这两个属性！
    alns.vehicle = vehicle
    alns.energy_config = energy_config
    
    print(f"✓ ALNS配置完成")
    print(f"  Repair模式: greedy")
    print(f"  迭代次数: 100")
    
    # 运行优化
    optimized_route = alns.optimize(initial_route, max_iterations=100)
    
    # ========== 第7步：验证结果 ==========
    print("\n【第7步】验证优化结果")
    print("=" * 70)
    
    final_distance = optimized_route.calculate_total_distance(distance_matrix)
    improvement = (initial_distance - final_distance) / initial_distance * 100
    
    # 统计充电站数量
    charging_stations_count = sum(1 for node in optimized_route.nodes 
                                  if node.node_type == NodeType.CHARGING)
    
    # 验证任务完成情况
    served_tasks = optimized_route.get_served_tasks()
    all_tasks = set(range(1, len(task_locations) + 1))
    missing_tasks = all_tasks - set(served_tasks)
    
    print(f"\n任务完成情况：")
    print(f"  完成任务: {len(served_tasks)}/{len(all_tasks)}")
    if missing_tasks:
        print(f"  ❌ 未完成任务: {missing_tasks}")
    else:
        print(f"  ✅ 所有任务完成")
    
    print(f"\n优化效果：")
    print(f"  初始距离: {initial_distance:.1f}m")
    print(f"  最终距离: {final_distance:.1f}m")
    print(f"  改进率: {improvement:.1f}%")
    
    print(f"\n充电方案：")
    print(f"  充电站数: {charging_stations_count}")
    
    if charging_stations_count > 0:
        print(f"  ✅ 成功插入充电站！")
        # 显示充电站位置
        for i, node in enumerate(optimized_route.nodes):
            if node.node_type == NodeType.CHARGING:
                print(f"    位置{i}: 充电站 {node.node_id}")
    else:
        print(f"  ⚠️ 未插入充电站")
    
    print(f"\n能量验证：")
    final_energy = final_distance / 1000.0 * 0.5
    print(f"  实际需要: {final_energy:.3f}kWh")
    print(f"  电池容量: {battery_capacity:.3f}kWh")
    print(f"  理论需充电: {'是' if final_energy > battery_capacity else '否'}")
    
    # ========== 最终判定 ==========
    print("\n" + "=" * 70)
    
    # 判定标准
    tasks_ok = len(missing_tasks) == 0
    charging_ok = charging_stations_count > 0 if final_energy > battery_capacity else True
    
    if tasks_ok and charging_ok:
        print("🎉 测试通过！ALNS充电功能正常工作")
        print("=" * 70)
        return True
    else:
        print("❌ 测试失败")
        if not tasks_ok:
            print(f"  原因：未完成所有任务")
        if not charging_ok:
            print(f"  原因：需要充电但未插入充电站")
        print("=" * 70)
        return False


if __name__ == "__main__":
    print("\n")
    success = test_alns_with_charging_comprehensive()
    print()
    
    if not success:
        print("\n💡 调试建议：")
        print("1. 检查 route.py 中的 check_energy_feasibility_for_insertion 方法")
        print("2. 确认 distance_matrix.get_nearest_charging_station 能正常工作")
        print("3. 验证 ChargingNode 的创建是否正确")
        print("4. 检查 insert_charging_visit 方法是否正确执行")