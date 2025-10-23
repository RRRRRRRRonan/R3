"""
调试PR-Minimal初始解不可行问题
"""

import sys
sys.path.append('src')

from core.node import create_depot, create_task_node_pair, create_charging_node
from core.route import Route, create_empty_route
from core.task import Task, TaskPool
from core.vehicle import create_vehicle
from physics.distance import DistanceMatrix
from physics.energy import EnergyConfig
from planner.alns import MinimalALNS, CostParameters
from strategy.charging_strategies import FullRechargeStrategy, PartialRechargeMinimalStrategy
import random

print("="*70)
print("调试：PR-Minimal初始解不可行问题")
print("="*70)

# 设置随机种子
random.seed(42)

# 创建简单场景（2个任务）
depot = create_depot(coordinates=(0, 0))

tasks = []
task_nodes = []
node_id_counter = 1

# 创建8个任务（和原始测试相同）
task_locations = [
    ((10000, 0), (15000, 0)),    # Task 1: 短距离
    ((20000, 0), (25000, 0)),    # Task 2
    ((35000, 0), (40000, 0)),    # Task 3: 中距离
    ((50000, 0), (55000, 0)),    # Task 4
    ((65000, 0), (70000, 0)),    # Task 5: 远距离
    ((80000, 0), (85000, 0)),    # Task 6
    ((45000, 10000), (50000, 10000)),  # Task 7: 不同区域
    ((30000, 10000), (35000, 10000)),  # Task 8
]

for i, (pickup_coords, delivery_coords) in enumerate(task_locations):
    pickup, delivery = create_task_node_pair(
        task_id=i+1,
        pickup_id=node_id_counter,
        delivery_id=node_id_counter + 1,
        pickup_coords=pickup_coords,
        delivery_coords=delivery_coords,
        demand=10
    )
    task_obj = Task(
        task_id=i+1,
        pickup_node=pickup,
        delivery_node=delivery,
        demand=10
    )
    tasks.append(task_obj)
    task_nodes.append((pickup, delivery))
    node_id_counter += 2

# 创建充电站（和原始测试相同）
charging_stations = [
    create_charging_node(node_id=100, coordinates=(15000, 0)),
    create_charging_node(node_id=101, coordinates=(30000, 0)),
    create_charging_node(node_id=102, coordinates=(45000, 0)),
    create_charging_node(node_id=103, coordinates=(60000, 0)),
    create_charging_node(node_id=104, coordinates=(75000, 0)),
    create_charging_node(node_id=105, coordinates=(40000, 10000)),
]

# 创建距离矩阵
coordinates = {depot.node_id: depot.coordinates}

for task in tasks:
    coordinates[task.pickup_node.node_id] = task.pickup_node.coordinates
    coordinates[task.delivery_node.node_id] = task.delivery_node.coordinates

for station in charging_stations:
    coordinates[station.node_id] = station.coordinates

distance_matrix = DistanceMatrix(
    coordinates=coordinates,
    num_tasks=len(tasks),
    num_charging_stations=len(charging_stations)
)

# 创建任务池
task_pool = TaskPool()
for task in tasks:
    task_pool.add_task(task)

# 配置参数（和原始测试相同）
vehicle = create_vehicle(vehicle_id=1, capacity=100, battery_capacity=60.0)
energy_config = EnergyConfig(
    consumption_rate=0.5,
    charging_rate=50.0/3600,
    charging_efficiency=0.9,
    critical_battery_threshold=0.2
)

cost_params = CostParameters(
    C_tr=1.0,
    C_ch=10.0,
    C_time=1.0,
    C_delay=2.0
)

print(f"\n场景信息:")
print(f"  任务数: {len(tasks)}")
print(f"  可用充电站: {len(charging_stations)}个")
print(f"  电池容量: {vehicle.battery_capacity} kWh")
print(f"  临界值: {energy_config.critical_battery_threshold*100}%")

# 测试两种策略（只测试PR-Minimal）
strategies = [
    # (FullRechargeStrategy(), "FR-完全充电"),
    (PartialRechargeMinimalStrategy(safety_margin=0.1), "PR-Minimal-10%"),
]

for strategy, strategy_name in strategies:
    print(f"\n{'='*70}")
    print(f"策略: {strategy_name}")
    print(f"{'='*70}")

    # 配置ALNS
    alns = MinimalALNS(
        distance_matrix=distance_matrix,
        task_pool=task_pool,
        repair_mode='greedy',
        cost_params=cost_params,
        charging_strategy=strategy
    )
    alns.vehicle = vehicle
    alns.energy_config = energy_config

    # 创建初始解
    initial_route = create_empty_route(1, depot)

    print(f"\n步骤1：创建空路径")
    print(f"  路径: {[n.node_id for n in initial_route.nodes]}")

    # 逐个插入任务，观察充电站插入
    for i, task in enumerate(tasks):
        print(f"\n步骤{i+2}：插入任务 {task.task_id}")

        # 插入前检查
        before_nodes = [n.node_id for n in initial_route.nodes]
        before_feasible = alns._check_battery_feasibility(initial_route, debug=True)

        print(f"  插入前路径: {before_nodes}")
        print(f"  插入前可行性: {'✓' if before_feasible else '✗'}")

        # 插入任务
        initial_route = alns.greedy_insertion(initial_route, [task.task_id])

        # 插入后检查
        after_nodes = [n.node_id for n in initial_route.nodes]
        cs_nodes = [n.node_id for n in initial_route.nodes if n.is_charging_station()]
        after_feasible = alns._check_battery_feasibility(initial_route, debug=True)

        print(f"  插入后路径: {after_nodes}")
        print(f"  充电站: {cs_nodes}")
        print(f"  插入后可行性: {'✓' if after_feasible else '✗'}")

        # 如果不可行，详细分析
        if not after_feasible:
            print(f"\n  ⚠️ 路径不可行！详细分析：")
            print(f"  路径节点详情:")
            for j, node in enumerate(initial_route.nodes):
                node_type = "Depot" if node.is_depot() else (
                    "CS" if node.is_charging_station() else (
                        "Pickup" if node.is_pickup() else "Delivery"
                    )
                )
                print(f"    [{j}] {node_type:8} id={node.node_id:3} coords={node.coordinates}")

            # 手动模拟电池消耗
            print(f"\n  电池模拟:")
            current_battery = vehicle.battery_capacity
            print(f"    起点: 电池={current_battery:.1f} kWh")

            for j in range(len(initial_route.nodes) - 1):
                curr_node = initial_route.nodes[j]
                next_node = initial_route.nodes[j + 1]

                # 充电
                if curr_node.is_charging_station():
                    # 计算剩余需求
                    remaining_demand = 0.0
                    for k in range(j, len(initial_route.nodes) - 1):
                        dist = distance_matrix.get_distance(
                            initial_route.nodes[k].node_id,
                            initial_route.nodes[k+1].node_id
                        )
                        remaining_demand += (dist / 1000.0) * energy_config.consumption_rate

                    charge_amount = strategy.determine_charging_amount(
                        current_battery=current_battery,
                        remaining_demand=remaining_demand,
                        battery_capacity=vehicle.battery_capacity
                    )
                    current_battery = min(vehicle.battery_capacity, current_battery + charge_amount)
                    print(f"    [{j}] 充电站{curr_node.node_id}: 充电{charge_amount:.1f} → {current_battery:.1f} kWh")

                # 移动
                distance = distance_matrix.get_distance(curr_node.node_id, next_node.node_id)
                energy_consumed = (distance / 1000.0) * energy_config.consumption_rate
                current_battery -= energy_consumed

                critical_threshold = energy_config.critical_battery_threshold * vehicle.battery_capacity
                is_critical = current_battery < critical_threshold

                print(f"    [{j}→{j+1}] 移动{distance/1000:.1f}km, 耗{energy_consumed:.1f}kWh → {current_battery:.1f}kWh "
                      f"{'⚠️CRITICAL' if is_critical else ''}")

                if current_battery < 0:
                    print(f"    ⚠️ 电池耗尽！")
                    break

    print(f"\n最终结果:")
    print(f"  任务数: {len(initial_route.get_served_tasks())}")
    print(f"  充电站数: {len([n for n in initial_route.nodes if n.is_charging_station()])}")
    print(f"  最终可行性: {'✓' if alns._check_battery_feasibility(initial_route) else '✗'}")
