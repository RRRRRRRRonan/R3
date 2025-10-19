"""
强制充电测试：确保充电功能真正起作用
"""
import sys
sys.path.append('src')

from core.node import create_depot, create_task_node_pair, NodeType
from core.task import create_task
from core.vehicle import create_vehicle
from core.route import Route
from physics.distance import create_distance_matrix_from_layout
from physics.energy import EnergyConfig, calculate_energy_consumption


def test_scenario_1_must_charge():
    """场景1：小电池 + 中等距离 = 必须充电"""
    print("=" * 70)
    print("测试场景1：强制充电（小电池）")
    print("=" * 70)
    
    depot = create_depot((0, 0))
    
    # 中等距离的任务
    p1, d1 = create_task_node_pair(
        task_id=1,
        pickup_id=1,
        delivery_id=2,
        pickup_coords=(60, 60),
        delivery_coords=(80, 80)
    )
    task1 = create_task(1, p1, d1)
    
    # 创建距离矩阵（包含充电站）
    distance_matrix = create_distance_matrix_from_layout(
        depot=(0, 0),
        task_locations=[((60, 60), (80, 80))],
        charging_stations=[(40, 40)]
    )
    
    # ⭐ 关键：使用小电池（50kWh）
    vehicle = create_vehicle(1, battery_capacity=50.0, initial_battery=50.0)
    energy_config = EnergyConfig(consumption_rate=0.5)
    
    print(f"\n🔋 AMR参数：")
    print(f"  电池容量: {vehicle.battery_capacity}kWh （小电池）")
    print(f"  初始电量: {vehicle.current_battery}kWh")
    
    # 预测能量需求
    dist_to_pickup = 84.9
    dist_pickup_to_delivery = 28.3
    dist_delivery_to_depot = 113.1
    total_dist = dist_to_pickup + dist_pickup_to_delivery + dist_delivery_to_depot
    
    # 简化计算：假设空载能耗
    estimated_energy = total_dist / 2.0 * 0.5  # 距离/速度 * 能耗率
    print(f"\n📊 预估能量需求: ~{estimated_energy:.1f}kWh")
    print(f"  当前电量: {vehicle.current_battery}kWh")
    
    if estimated_energy > vehicle.current_battery:
        print(f"  ⚡ 预计需要充电（缺口: {estimated_energy - vehicle.current_battery:.1f}kWh）")
    else:
        print(f"  ⚠️  警告：场景可能不会触发充电")
    
    # 初始路径
    initial_route = Route(vehicle_id=1, nodes=[depot, depot])
    insert_pos = (1, 2)
    
    # 检查电量
    feasible, charging_plan = initial_route.check_energy_feasibility_for_insertion(
        task1, insert_pos, vehicle, distance_matrix, energy_config
    )
    
    print(f"\n⚡ 充电检查结果:")
    print(f"  可行性: {'✅ 可行' if feasible else '❌ 不可行'}")
    
    if charging_plan:
        print(f"  充电计划: {len(charging_plan)}个充电站 ✅")
        for i, plan in enumerate(charging_plan):
            print(f"    {i+1}. 位置{plan['position']}, 充电{plan['amount']:.1f}kWh")
    else:
        print(f"  充电计划: 无需充电 ❌ （测试失败：应该需要充电）")
        return False
    
    # 插入任务
    initial_route.insert_task(task1, insert_pos)
    
    # 插入充电站
    sorted_plan = sorted(charging_plan, key=lambda x: x['position'], reverse=True)
    for plan in sorted_plan:
        initial_route.insert_charging_visit(
            station=plan['station_node'],
            position=plan['position'],
            charge_amount=plan['amount']
        )
    
    print("\n" + "="*70)
    print("📍 最终路径（带充电）")
    print("="*70)
    print(f"节点序列：{[str(n) for n in initial_route.nodes]}")
    
    all_ok = print_route_with_battery(initial_route, vehicle, distance_matrix, energy_config)
    
    if all_ok:
        print("\n✅✅✅ 场景1测试通过：充电功能正常工作！")
        return True
    else:
        print("\n❌❌❌ 场景1测试失败：充电后仍有电量不足")
        return False


def test_scenario_2_ultra_small_battery():
    """场景2：超小电池 + 多次充电"""
    print("\n\n" + "=" * 70)
    print("测试场景2：多次充电（超小电池）")
    print("=" * 70)
    
    depot = create_depot((0, 0))
    
    # 使用相同任务
    p1, d1 = create_task_node_pair(
        task_id=1,
        pickup_id=1,
        delivery_id=2,
        pickup_coords=(60, 60),
        delivery_coords=(80, 80)
    )
    task1 = create_task(1, p1, d1)
    
    distance_matrix = create_distance_matrix_from_layout(
        depot=(0, 0),
        task_locations=[((60, 60), (80, 80))],
        charging_stations=[(40, 40)]
    )
    
    # ⭐ 超小电池（30kWh）- 应该需要多次充电
    vehicle = create_vehicle(1, battery_capacity=30.0, initial_battery=30.0)
    energy_config = EnergyConfig(consumption_rate=0.5)
    
    print(f"\n🔋 AMR参数：")
    print(f"  电池容量: {vehicle.battery_capacity}kWh （超小电池）")
    print(f"  预期: 需要多次充电")
    
    initial_route = Route(vehicle_id=1, nodes=[depot, depot])
    insert_pos = (1, 2)
    
    feasible, charging_plan = initial_route.check_energy_feasibility_for_insertion(
        task1, insert_pos, vehicle, distance_matrix, energy_config
    )
    
    print(f"\n⚡ 充电检查结果:")
    if charging_plan:
        print(f"  充电站数量: {len(charging_plan)} ✅")
        for i, plan in enumerate(charging_plan):
            print(f"    {i+1}. 位置{plan['position']}, 充电{plan['amount']:.1f}kWh")
    else:
        print(f"  ❌ 无充电计划（测试失败）")
        return False
    
    # 插入任务和充电站
    initial_route.insert_task(task1, insert_pos)
    sorted_plan = sorted(charging_plan, key=lambda x: x['position'], reverse=True)
    for plan in sorted_plan:
        initial_route.insert_charging_visit(
            station=plan['station_node'],
            position=plan['position'],
            charge_amount=plan['amount']
        )
    
    print("\n" + "="*70)
    print("📍 最终路径")
    print("="*70)
    print(f"节点数: {len(initial_route.nodes)} (包含{len(charging_plan)}个充电站)")
    
    all_ok = print_route_with_battery(initial_route, vehicle, distance_matrix, energy_config)
    
    if all_ok:
        print("\n✅✅✅ 场景2测试通过：多次充电正常工作！")
        return True
    else:
        print("\n❌❌❌ 场景2测试失败")
        return False


def print_route_with_battery(route, vehicle, distance_matrix, energy_config):
    """打印路径并模拟电量变化"""
    current_battery = vehicle.current_battery
    current_load = 0.0
    all_positive = True
    charging_count = 0
    
    print(f"\n  起点: Depot (电量: {current_battery:.1f}kWh)")
    
    for i in range(len(route.nodes) - 1):
        current_node = route.nodes[i]
        next_node = route.nodes[i + 1]
        
        distance = distance_matrix.get_distance(current_node.node_id, next_node.node_id)
        energy_needed = calculate_energy_consumption(
            distance=distance,
            load=current_load,
            config=energy_config,
            vehicle_speed=vehicle.speed,
            vehicle_capacity=vehicle.capacity
        )
        
        print(f"  └─> 移动 {distance:.1f}m，消耗 {energy_needed:.1f}kWh", end="")
        current_battery -= energy_needed
        
        if current_battery < -0.1:
            print(f" ⚠️  不足: {current_battery:.1f}kWh")
            all_positive = False
        else:
            print()
        
        if next_node.is_charging_station():
            charging_count += 1
            charge_amount = getattr(next_node, 'charge_amount', 0)
            print(f"  ⚡ 充电站{charging_count} (ID:{next_node.node_id})")
            print(f"     到达: {current_battery:.1f}kWh")
            
            actual_charge = min(charge_amount, vehicle.battery_capacity - current_battery)
            current_battery = min(current_battery + actual_charge, vehicle.battery_capacity)
            print(f"     充电: {actual_charge:.1f}kWh → 离开: {current_battery:.1f}kWh")
            
        elif next_node.is_pickup():
            print(f"  📦 Pickup (电量: {current_battery:.1f}kWh)")
            current_load += next_node.demand
            
        elif next_node.is_delivery():
            print(f"  ✅ Delivery (电量: {current_battery:.1f}kWh)")
            current_load -= next_node.demand
            
        elif next_node.is_depot():
            print(f"  🏁 返回Depot (电量: {current_battery:.1f}/{vehicle.battery_capacity}kWh)")
    
    print(f"\n📊 统计:")
    print(f"  充电次数: {charging_count}")
    print(f"  最终电量: {current_battery:.1f}kWh ({current_battery/vehicle.battery_capacity*100:.1f}%)")
    
    return all_positive


if __name__ == "__main__":
    print("\n" + "🔋" * 35)
    print("强制充电测试套件")
    print("🔋" * 35)
    
    result1 = test_scenario_1_must_charge()
    result2 = test_scenario_2_ultra_small_battery()
    
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    print(f"场景1（小电池）: {'✅ 通过' if result1 else '❌ 失败'}")
    print(f"场景2（超小电池）: {'✅ 通过' if result2 else '❌ 失败'}")
    
    if result1 and result2:
        print("\n🎉🎉🎉 所有充电测试通过！充电功能完全正常！")
    else:
        print("\n⚠️ 部分测试失败，需要调试")