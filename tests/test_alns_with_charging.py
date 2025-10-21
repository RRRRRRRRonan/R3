import sys
sys.path.append('src')
import random
from planner.alns import MinimalALNS
from physics.distance import DistanceMatrix
from core.node import create_depot, create_task_node_pair
from core.task import create_task, TaskPool
from core.route import Route
from core.vehicle import create_vehicle
from physics.energy import EnergyConfig


def verify_route_completeness(route: Route, expected_task_count: int) -> bool:
    """验证路径是否完成了所有任务"""
    served_tasks = route.get_served_tasks()
    pickup_nodes = route.get_pickup_nodes()
    delivery_nodes = route.get_delivery_nodes()
    
    print(f"\n📋 任务完整性验证：")
    print(f"  预期任务数: {expected_task_count}")
    print(f"  实际完成: {len(served_tasks)}")
    print(f"  Pickup/Delivery节点: {len(pickup_nodes)}/{len(delivery_nodes)}")
    
    if len(served_tasks) != expected_task_count:
        print(f"  ❌ 任务不完整！")
        return False
    
    if len(pickup_nodes) != len(delivery_nodes):
        print(f"  ❌ Pickup和Delivery数量不匹配！")
        return False
    
    is_valid, error_msg = route.validate_precedence()
    if not is_valid:
        print(f"  ❌ 约束违反: {error_msg}")
        return False
    
    print(f"  ✅ 所有任务完成且满足约束")
    return True


def test_alns_with_charging():
    """测试ALNS + 充电集成"""
    print("=" * 70)
    print("测试：ALNS优化 + 充电集成（修复版）")
    print("=" * 70)
    
    # ========== 场景设计 ==========
    depot = create_depot((0, 0))
    coordinates = {0: (0, 0)}
    
    # 5个远距离任务（扩大范围强制需要充电）
    locations = [
        ((150, 150), (200, 200)),
        ((180, 50), (220, 80)),
        ((50, 180), (80, 220)),
        ((160, 160), (190, 190)),
        ((140, 140), (170, 170)),
    ]
    
    num_tasks = len(locations)
    task_pool = TaskPool()
    nodes_list = []
    
    # 创建任务
    for i, (pickup_loc, delivery_loc) in enumerate(locations, start=1):
        pickup_id = i
        delivery_id = i + num_tasks
        
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
    
    # 创建充电站（增加数量和分布）
    charging_coords = [(75, 75), (100, 100), (125, 125), (150, 50), (50, 150)]
    charging_start_id = 2 * num_tasks + 1
    
    for idx, coords in enumerate(charging_coords):
        coordinates[charging_start_id + idx] = coords
    
    distance_matrix = DistanceMatrix(
        coordinates=coordinates,
        num_tasks=num_tasks,
        num_charging_stations=len(charging_coords)
    )
    
    # 创建初始路径
    initial_route = Route(
        vehicle_id=1,
        nodes=[depot] + nodes_list + [depot]
    )
    
    # ========== 关键修复：降低电池容量 ==========
    initial_distance = initial_route.calculate_total_distance(distance_matrix)
    estimated_energy = initial_distance / 1000.0 * 0.5  # 假设0.5 kWh/km
    
    # 🔧 修复1：设置为80%预估能量，强制需要充电
    battery_capacity = estimated_energy * 0.8
    
    print(f"\n📊 场景参数：")
    print(f"  任务数: {num_tasks}")
    print(f"  充电站数: {len(charging_coords)}")
    print(f"  初始距离: {initial_distance:.2f}m")
    print(f"  预估能量: {estimated_energy:.2f}kWh")
    print(f"  电池容量: {battery_capacity:.2f}kWh ({battery_capacity/estimated_energy*100:.0f}%预估)")
    print(f"  📌 容量设置为不足，必须充电才能完成任务")
    
    # ========== 配置ALNS ==========
    test_vehicle = create_vehicle(
        vehicle_id=1,
        battery_capacity=battery_capacity,
        initial_battery=battery_capacity
    )
    test_energy_config = EnergyConfig(consumption_rate=0.5)
    
    alns = MinimalALNS(distance_matrix, task_pool, repair_mode='mixed')
    alns.vehicle = test_vehicle
    alns.energy_config = test_energy_config
    
    # ========== 运行优化 ==========
    import time
    seed = int(time.time())
    random.seed(seed)
    print(f"  随机种子: {seed}")
    
    optimized_route = alns.optimize(initial_route, max_iterations=100)
    
    # ========== 结果验证 ==========
    final_cost = optimized_route.calculate_total_distance(distance_matrix)
    improvement = (initial_distance - final_cost) / initial_distance * 100
    charging_stations = [n for n in optimized_route.nodes if n.is_charging_station()]
    
    print(f"\n" + "=" * 70)
    print(f"优化结果")
    print(f"=" * 70)
    
    # 验证任务完整性
    is_complete = verify_route_completeness(optimized_route, num_tasks)
    
    print(f"\n📈 性能指标：")
    print(f"  初始成本: {initial_distance:.2f}m")
    print(f"  优化成本: {final_cost:.2f}m")
    print(f"  改进幅度: {improvement:.1f}%")
    print(f"  节点数: {len(optimized_route.nodes)}")
    print(f"  充电站数: {len(charging_stations)}")
    
    if charging_stations:
        print(f"  充电站位置: {[cs.node_id for cs in charging_stations]}")
    
    # ========== 能量验证 ==========
    print(f"\n⚡ 能量分析：")
    final_distance = optimized_route.calculate_total_distance(distance_matrix)
    actual_energy_needed = final_distance / 1000.0 * 0.5
    print(f"  实际能量需求: {actual_energy_needed:.2f}kWh")
    print(f"  电池容量: {battery_capacity:.2f}kWh")
    print(f"  能量缺口: {actual_energy_needed - battery_capacity:.2f}kWh")
    
    if actual_energy_needed > battery_capacity:
        print(f"  ✅ 确认需要充电（缺口 {actual_energy_needed - battery_capacity:.2f}kWh）")
    else:
        print(f"  ⚠️  理论上可不充电（富余 {battery_capacity - actual_energy_needed:.2f}kWh）")
    
    # ========== 判断测试是否通过 ==========
    if not is_complete:
        print(f"\n❌ 测试失败：任务未全部完成")
        return False
    
    # 如果确实需要充电但没有充电站
    if actual_energy_needed > battery_capacity and len(charging_stations) == 0:
        print(f"\n❌ 测试失败：需要充电但未插入充电站")
        print(f"   原因排查：")
        print(f"   1. ALNS repair时未调用check_energy_feasibility_for_insertion")
        print(f"   2. 充电检查逻辑未触发")
        print(f"   3. 充电站无法解决能量问题")
        return False
    
    # 充电站数量合理性检查
    if len(charging_stations) > num_tasks:
        print(f"\n⚠️  警告：充电站数量({len(charging_stations)})超过任务数({num_tasks})")
        print(f"   可能原因：充电插入逻辑过于激进")
    
    print(f"\n✅ 测试通过！")
    return True


if __name__ == "__main__":
    success = test_alns_with_charging()
    
    if success:
        print("\n" + "=" * 70)
        print("🎉 ALNS充电集成测试成功！")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("❌ 测试未达到预期 - 需要检查ALNS充电逻辑")
        print("=" * 70)