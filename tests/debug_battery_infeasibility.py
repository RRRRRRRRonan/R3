"""
调试脚本：详细展示PR-Fixed-30%为什么电池不可行
"""

import sys
sys.path.append('src')

from core.node import create_depot, create_task_node_pair, create_charging_node
from core.route import Route
from core.vehicle import create_vehicle
from physics.distance import DistanceMatrix
from physics.energy import EnergyConfig
from strategy.charging_strategies import (
    FullRechargeStrategy,
    PartialRechargeFixedStrategy,
    PartialRechargeMinimalStrategy
)
import random

# 设置随机种子
random.seed(42)

print("="*60)
print("调试：PR-Fixed-30%电池不可行原因分析")
print("="*60)

# 1. 创建测试场景（简化版 - 4个任务）
depot = create_depot(x=0, y=0)

task_nodes = [
    create_task_node_pair(task_id=1, x=20, y=20, demand=5, service_time=10),
    create_task_node_pair(task_id=2, x=40, y=10, demand=5, service_time=10),
    create_task_node_pair(task_id=3, x=30, y=40, demand=5, service_time=10),
    create_task_node_pair(task_id=4, x=10, y=50, demand=5, service_time=10),
]

charging_stations = [
    create_charging_node(cs_id=100, x=25, y=15, charging_rate=50.0/3600),
    create_charging_node(cs_id=101, x=20, y=35, charging_rate=50.0/3600),
]

# 2. 创建车辆和能耗配置
vehicle = create_vehicle(vehicle_id=1, capacity=100, battery_capacity=70.0)
energy_config = EnergyConfig(
    consumption_rate=0.5,  # 0.5 kWh/km
    charging_rate=50.0/3600  # 50 kW
)

# 3. 创建距离矩阵
all_nodes = [depot] + [p[0] for p in task_nodes] + [p[1] for p in task_nodes] + charging_stations
distance = DistanceMatrix(all_nodes)

# 4. 构建一个示例路径：depot -> task1 -> CS100 -> task2 -> task3 -> CS101 -> task4 -> depot
route_nodes = [
    depot,
    task_nodes[0][0],  # task1 pickup (20, 20)
    charging_stations[0],  # CS100 (25, 15)
    task_nodes[1][0],  # task2 pickup (40, 10)
    task_nodes[2][0],  # task3 pickup (30, 40)
    charging_stations[1],  # CS101 (20, 35)
    task_nodes[3][0],  # task4 pickup (10, 50)
    depot
]

route = Route()
route.nodes = route_nodes

# 5. 计算总距离和能耗
total_distance = 0.0
for i in range(len(route_nodes) - 1):
    seg_dist = distance.get_distance(route_nodes[i].node_id, route_nodes[i+1].node_id)
    total_distance += seg_dist

total_energy_demand = total_distance / 1000.0 * energy_config.consumption_rate

print(f"\n路径信息:")
print(f"  总距离: {total_distance/1000.0:.2f} km")
print(f"  总能耗需求: {total_energy_demand:.2f} kWh")
print(f"  电池容量: {vehicle.battery_capacity:.1f} kWh")
print(f"  充电站数: {len([n for n in route_nodes if n.type == 'charging'])}个")

# 6. 测试三种策略的电池可行性
strategies = [
    (FullRechargeStrategy(), "FR-完全充电"),
    (PartialRechargeFixedStrategy(charge_ratio=0.3), "PR-Fixed-30%"),
    (PartialRechargeMinimalStrategy(safety_margin=0.1), "PR-Minimal-10%"),
]

for strategy, strategy_name in strategies:
    print(f"\n{'='*60}")
    print(f"测试策略: {strategy_name}")
    print(f"{'='*60}")

    current_battery = vehicle.battery_capacity
    print(f"\n初始电量: {current_battery:.1f} kWh")

    battery_feasible = True

    for i in range(len(route_nodes) - 1):
        current_node = route_nodes[i]
        next_node = route_nodes[i + 1]

        print(f"\n--- 节点 {i}: {current_node.node_id} ({current_node.type}) ---")

        # 如果当前节点是充电站，先充电
        if current_node.type == 'charging':
            # 计算剩余路径能耗
            remaining_energy_demand = 0.0
            for j in range(i, len(route_nodes) - 1):
                seg_distance = distance.get_distance(
                    route_nodes[j].node_id,
                    route_nodes[j + 1].node_id
                )
                remaining_energy_demand += (seg_distance / 1000.0) * energy_config.consumption_rate

            charge_amount = strategy.determine_charging_amount(
                current_battery=current_battery,
                remaining_demand=remaining_energy_demand,
                battery_capacity=vehicle.battery_capacity
            )

            battery_before_charge = current_battery
            current_battery = min(vehicle.battery_capacity, current_battery + charge_amount)

            print(f"  📍 充电站 CS{current_node.node_id}")
            print(f"  充电前: {battery_before_charge:.1f} kWh")
            print(f"  剩余需求: {remaining_energy_demand:.1f} kWh")
            print(f"  充电量: {charge_amount:.1f} kWh")
            print(f"  充电后: {current_battery:.1f} kWh")

        # 计算到下一节点的距离和能耗
        seg_distance = distance.get_distance(current_node.node_id, next_node.node_id)
        energy_consumed = (seg_distance / 1000.0) * energy_config.consumption_rate

        # 移动到下一节点
        battery_before_move = current_battery
        current_battery -= energy_consumed

        print(f"  🚗 移动到节点 {next_node.node_id} ({next_node.type})")
        print(f"  距离: {seg_distance/1000.0:.2f} km")
        print(f"  能耗: {energy_consumed:.2f} kWh")
        print(f"  移动前电量: {battery_before_move:.1f} kWh")
        print(f"  移动后电量: {current_battery:.1f} kWh")

        # 检查是否电量不足
        if current_battery < 0:
            print(f"  ❌ 电池耗尽！（缺少 {-current_battery:.1f} kWh）")
            battery_feasible = False
            break
        elif current_battery < 5:
            print(f"  ⚠️  电量告急！")

    print(f"\n最终结果: {'✅ 可行' if battery_feasible else '❌ 不可行'}")
    if battery_feasible:
        print(f"最终电量: {current_battery:.1f} kWh")

print(f"\n{'='*60}")
print("总结")
print(f"{'='*60}")
print("""
关键发现：
1. PR-Fixed-30%策略充电量不足
   - 每次只充电到30%的剩余容量
   - 在长距离路段会电池耗尽

2. 电池不可行的具体含义：
   - 车辆在某个节点间移动时电量降到负数
   - 即使路径访问了充电站，但充电量太少无法完成后续行程
   - 这不是说"任务顺序不可行"，而是"充电量不足导致无法完成"

3. ALNS为什么接受不可行解：
   - 因为PR-Fixed-30%的充电策略无法生成可行解
   - ALNS只能接受带惩罚的不可行解（成本+100000）
   - 这就是为什么PR-Fixed-30%成本是339999.92而不是239999.92

4. 改进方向：
   - 增加充电比例（如50%或60%）
   - 或者在ALNS中自动插入更多充电站
   - 或者使用PR-Minimal策略（根据实际需求动态调整）
""")
