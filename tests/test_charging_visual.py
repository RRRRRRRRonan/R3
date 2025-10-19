"""
可视化测试：验证充电站插入逻辑
"""
import sys
sys.path.append('src')

from core.node import create_depot, create_task_node_pair, NodeType
from core.task import create_task
from core.vehicle import create_vehicle
from core.route import Route
from physics.distance import create_distance_matrix_from_layout
from physics.energy import EnergyConfig, calculate_energy_consumption


def test_charging_insertion_sequence():
    """测试充电站插入序列验证"""
    print("=" * 70)
    print("测试：充电站插入序列验证（简化场景）")
    print("=" * 70)
    
    depot = create_depot((0, 0))
    
    # 使用合理的距离
    p1, d1 = create_task_node_pair(
        task_id=1,
        pickup_id=1,
        delivery_id=2,
        pickup_coords=(60, 60),
        delivery_coords=(80, 80)
    )
    task1 = create_task(1, p1, d1)
    
    # 创建距离矩阵
    distance_matrix = create_distance_matrix_from_layout(
        depot=(0, 0),
        task_locations=[((60, 60), (80, 80))],
        charging_stations=[(40, 40)]
    )
    
    # 合理的电池
    vehicle = create_vehicle(1, battery_capacity=80.0, initial_battery=80.0)
    energy_config = EnergyConfig(consumption_rate=0.5)
    
    # ⭐ 关键：初始路径包含往返depot
    initial_route = Route(vehicle_id=1, nodes=[depot, depot])
    
    print("\n📍 初始路径：")
    print("  Depot → Depot（空载往返）")
    print(f"\n🔋 AMR参数：")
    print(f"  电池容量: {vehicle.battery_capacity}kWh")
    print(f"  初始电量: {vehicle.current_battery}kWh")
    
    print(f"\n🎯 尝试插入任务1")
    print(f"  Pickup: {p1.coordinates}")
    print(f"  Delivery: {d1.coordinates}")
    
    # ⭐ 插入位置：在两个depot之间
    insert_pos = (1, 2)
    print(f"  插入位置: pickup at {insert_pos[0]}, delivery at {insert_pos[1]}")
    
    # 检查电量
    feasible, charging_plan = initial_route.check_energy_feasibility_for_insertion(
        task1, insert_pos, vehicle, distance_matrix, energy_config
    )
    
    print(f"\n⚡ 电量检查结果: {'✅ 可行' if feasible else '❌ 不可行'}")
    
    if not feasible:
        print("  ❌ 无法完成任务！")
        return
    
    if charging_plan:
        print(f"⚡ 需要充电：{len(charging_plan)}个充电站")
        for i, plan in enumerate(charging_plan):
            print(f"  {i+1}. 位置{plan['position']}, 充电{plan['amount']:.1f}kWh")
    else:
        print("✅ 无需充电，电量充足")
    
    # 插入任务
    initial_route.insert_task(task1, insert_pos)
    
    print("\n" + "="*70)
    print("📍 步骤1：插入任务后（未充电）")
    print("="*70)
    print(f"节点序列：{[str(n) for n in initial_route.nodes]}")
    all_ok = print_route_with_battery(initial_route, vehicle, distance_matrix, energy_config)
    
    # 插入充电站（从后往前）
    if charging_plan:
        sorted_plan = sorted(charging_plan, key=lambda x: x['position'], reverse=True)
        for plan in sorted_plan:
            initial_route.insert_charging_visit(
                station=plan['station_node'],
                position=plan['position'],
                charge_amount=plan['amount']
            )
        
        print("\n" + "="*70)
        print("📍 步骤2：插入充电站后")
        print("="*70)
        print(f"节点序列：{[str(n) for n in initial_route.nodes]}")
        all_ok = print_route_with_battery(initial_route, vehicle, distance_matrix, energy_config)
    
    if all_ok:
        print("\n✅✅✅ 测试通过：电量始终充足！")
    else:
        print("\n❌❌❌ 测试失败：仍有电量不足")


def print_route_with_battery(route, vehicle, distance_matrix, energy_config):
    """打印路径并模拟电量变化"""
    current_battery = vehicle.current_battery
    current_load = 0.0
    all_positive = True
    
    print(f"\n  起点: Depot (电量: {current_battery:.1f}kWh, 载重: {current_load:.1f}kg)")
    
    for i in range(len(route.nodes) - 1):
        current_node = route.nodes[i]
        next_node = route.nodes[i + 1]
        
        # 计算移动能耗
        distance = distance_matrix.get_distance(current_node.node_id, next_node.node_id)
        energy_needed = calculate_energy_consumption(
            distance=distance,
            load=current_load,
            config=energy_config,
            vehicle_speed=vehicle.speed,
            vehicle_capacity=vehicle.capacity
        )
        
        # 移动
        print(f"  └─> 移动 {distance:.1f}m，消耗 {energy_needed:.1f}kWh", end="")
        current_battery -= energy_needed
        
        if current_battery < -0.1:
            print(f" ⚠️  电量不足: {current_battery:.1f}kWh")
            all_positive = False
        else:
            print()
        
        # 到达节点
        if next_node.is_charging_station():
            charge_amount = getattr(next_node, 'charge_amount', 0)
            print(f"  ⚡ 充电站 {next_node.node_id}")
            print(f"     到达时: {current_battery:.1f}kWh")
            
            actual_charge = min(charge_amount, vehicle.battery_capacity - current_battery)
            current_battery = min(current_battery + actual_charge, vehicle.battery_capacity)
            print(f"     充电: {actual_charge:.1f}kWh → 离开时: {current_battery:.1f}kWh")
            
        elif next_node.is_pickup():
            print(f"  📦 Pickup {next_node.node_id} (电量: {current_battery:.1f}kWh)")
            current_load += next_node.demand
            print(f"     装载 {next_node.demand:.1f}kg → 载重: {current_load:.1f}kg")
            
        elif next_node.is_delivery():
            print(f"  ✅ Delivery {next_node.node_id} (电量: {current_battery:.1f}kWh)")
            current_load -= next_node.demand
            print(f"     卸载 {next_node.demand:.1f}kg → 载重: {current_load:.1f}kg")
            
        elif next_node.is_depot():
            print(f"  🏁 返回Depot")
            print(f"     最终电量: {current_battery:.1f}kWh / {vehicle.battery_capacity:.1f}kWh ({current_battery/vehicle.battery_capacity*100:.1f}%)")
    
    return all_positive


if __name__ == "__main__":
    test_charging_insertion_sequence()
    print("\n" + "=" * 70)