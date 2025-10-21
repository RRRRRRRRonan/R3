"""
充电逻辑测试 - 带完整调试信息
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from physics.distance import DistanceMatrix
from core.node import create_depot, create_task_node_pair
from core.task import create_task
from core.route import Route, create_empty_route
from core.vehicle import create_vehicle
from physics.energy import EnergyConfig


def test_with_full_debug():
    """测试充电逻辑（完整调试模式）"""
    print("=" * 70)
    print("充电逻辑测试 - 完整调试模式")
    print("=" * 70)
    
    # ========== 场景设置 ==========
    depot = create_depot((0, 0))
    p1, d1 = create_task_node_pair(1, 1, 3, (100, 100), (150, 150))
    p2, d2 = create_task_node_pair(2, 2, 4, (120, 50), (170, 80))
    
    task1 = create_task(1, p1, d1)
    task2 = create_task(2, p2, d2)
    
    coordinates = {
        0: (0, 0),      # depot
        1: (100, 100),  # p1
        2: (120, 50),   # p2
        3: (150, 150),  # d1
        4: (170, 80),   # d2
        5: (75, 75),    # 充电站1
        6: (100, 50),   # 充电站2
    }
    
    distance_matrix = DistanceMatrix(
        coordinates=coordinates,
        num_tasks=2,
        num_charging_stations=2
    )
    
    # ========== 创建初始路径 ==========
    route = create_empty_route(1, depot)
    route.insert_task(task1, (1, 2))
    
    print(f"\n📍 初始路径：")
    print(f"  节点序列: {[n.node_id for n in route.nodes]}")
    print(f"  说明: Depot → Pickup1 → Delivery1 → Depot")
    
    # ========== 能量参数 ==========
    total_distance = route.calculate_total_distance(distance_matrix)
    estimated_energy = total_distance / 1000.0 * 0.5
    
    # 设置70%容量，强制需要充电
    battery_capacity = estimated_energy * 0.7
    
    print(f"\n⚡ 能量参数：")
    print(f"  当前距离: {total_distance:.2f}m")
    print(f"  预估能量: {estimated_energy:.2f}kWh")
    print(f"  电池容量: {battery_capacity:.2f}kWh (70%)")
    print(f"  能量缺口: {estimated_energy - battery_capacity:.2f}kWh")
    
    if estimated_energy > battery_capacity:
        print(f"  ✅ 确认需要充电才能完成任务")
    else:
        print(f"  ⚠️  理论上可能不需要充电")
    
    # ========== 配置车辆 ==========
    vehicle = create_vehicle(
        vehicle_id=1,
        battery_capacity=battery_capacity,
        initial_battery=battery_capacity
    )
    energy_config = EnergyConfig(consumption_rate=0.5)
    
    # ========== 执行充电检查（开启调试） ==========
    print(f"\n" + "=" * 70)
    print(f"执行充电可行性检查 - 调试模式")
    print(f"=" * 70)
    print(f"\n要插入的任务: Task 2 (pickup={p2.node_id}, delivery={d2.node_id})")
    print(f"插入位置: (3, 4)")
    print(f"  - pickup将插入到索引3（在终点depot之前）")
    print(f"  - delivery将插入到索引4（在pickup之后）")
    
    is_feasible, charging_plan = route.check_energy_feasibility_for_insertion(
        task=task2,
        insert_position=(3, 4),
        vehicle=vehicle,
        distance_matrix=distance_matrix,
        energy_config=energy_config,
        debug=True  # ← 开启调试模式
    )
    
    # ========== 显示结果 ==========
    print(f"\n" + "=" * 70)
    print(f"充电检查结果")
    print(f"=" * 70)
    
    print(f"\n可行性: {'✅ 可行' if is_feasible else '❌ 不可行'}")
    
    if charging_plan:
        print(f"充电方案: 需要 {len(charging_plan)} 个充电站")
        for i, plan in enumerate(charging_plan, 1):
            print(f"  {i}. 充电站 {plan['station_node'].node_id}")
            print(f"     - 插入位置: 索引 {plan['position']}")
            print(f"     - 充电量: {plan['amount']:.2f}kWh")
        
        # ========== 实际插入并验证 ==========
        print(f"\n" + "=" * 70)
        print(f"实际插入测试")
        print(f"=" * 70)
        
        # 先插入任务
        route.insert_task(task2, (3, 4))
        print(f"\n1️⃣ 插入任务后: {[n.node_id for n in route.nodes]}")
        
        # 再插入充电站（倒序插入避免位置偏移）
        for plan in reversed(charging_plan):
            route.insert_charging_visit(
                station=plan['station_node'],
                position=plan['position'],
                charge_amount=plan['amount']
            )
        
        print(f"2️⃣ 插入充电站后: {[n.node_id for n in route.nodes]}")
        
        # 验证最终路径
        final_distance = route.calculate_total_distance(distance_matrix)
        final_energy = final_distance / 1000.0 * 0.5
        charging_count = len([n for n in route.nodes if n.is_charging_station()])
        
        print(f"\n✅ 最终路径验证：")
        print(f"  总距离: {final_distance:.2f}m")
        print(f"  预估能量: {final_energy:.2f}kWh")
        print(f"  电池容量: {battery_capacity:.2f}kWh")
        print(f"  充电站数: {charging_count}")
        
        if charging_count > 0:
            print(f"  🎉 成功插入充电站！")
            return True
        else:
            print(f"  ⚠️  没有充电站被插入")
            return False
    else:
        print(f"充电方案: 无（不需要充电或无法解决）")
        
        if not is_feasible:
            print(f"\n❌ 无法通过充电使路径可行")
            print(f"可能原因：")
            print(f"  1. 充电站位置不合适")
            print(f"  2. 电池容量过小，即使充满也无法完成")
            print(f"  3. 路径构建逻辑有错误")
        
        return False


if __name__ == "__main__":
    print("\n")
    success = test_with_full_debug()
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 测试成功！充电逻辑正常工作")
    else:
        print("❌ 测试失败，请查看上面的调试信息")
    print("=" * 70)