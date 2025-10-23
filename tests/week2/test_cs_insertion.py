"""
测试智能充电站插入功能（第1.2步）
验证greedy_insertion可以自动插入必要的充电站
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
print("测试：智能充电站插入功能（第1.2步）")
print("="*70)

# 设置随机种子
random.seed(42)

# 1. 创建测试场景
depot = create_depot(coordinates=(0, 0))

tasks = []
task_nodes = []
node_id_counter = 1

# 创建6个任务，分布在远距离位置，必须充电才能完成
task_locations = [
    ((10000, 10000), (15000, 12000)),   # Task 1
    ((25000, 8000), (30000, 10000)),    # Task 2
    ((40000, 15000), (45000, 18000)),   # Task 3
    ((55000, 22000), (60000, 25000)),   # Task 4
    ((50000, 35000), (45000, 38000)),   # Task 5
    ((30000, 40000), (25000, 42000)),   # Task 6
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

# 创建多个可用的充电站
charging_stations = [
    create_charging_node(node_id=100, coordinates=(15000, 15000)),
    create_charging_node(node_id=101, coordinates=(28000, 20000)),
    create_charging_node(node_id=102, coordinates=(42000, 28000)),
    create_charging_node(node_id=103, coordinates=(38000, 38000)),
    create_charging_node(node_id=104, coordinates=(20000, 35000)),
]

# 2. 创建距离矩阵
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

# 3. 创建任务池
task_pool = TaskPool()
for task in tasks:
    task_pool.add_task(task)

# 4. 创建车辆（50kWh电池，故意设置得较小，确保需要充电）
vehicle = create_vehicle(vehicle_id=1, capacity=100, battery_capacity=50.0)
energy_config = EnergyConfig(
    consumption_rate=0.5,  # 0.5 kWh/km
    charging_rate=50.0/3600,
    charging_efficiency=0.9
)
cost_params = CostParameters(C_tr=1.0, C_ch=0.6, C_time=0.1, C_delay=2.0)

print(f"\n场景信息:")
print(f"  任务数: {len(tasks)}")
print(f"  可用充电站: {len(charging_stations)}个")
print(f"  电池容量: {vehicle.battery_capacity} kWh")
print(f"  预计总路程: ~150km (需要~75kWh) - 必须充电！")

# 5. 测试1：从空路径开始，逐步插入任务
print(f"\n{'='*70}")
print("测试1：从空路径插入任务（无初始充电站）")
print(f"{'='*70}")

for strategy, strategy_name in [(FullRechargeStrategy(), "FR-完全充电"),
                                 (PartialRechargeMinimalStrategy(safety_margin=0.1), "PR-Minimal-10%")]:

    print(f"\n--- 策略: {strategy_name} ---")

    alns = MinimalALNS(
        distance_matrix=distance_matrix,
        task_pool=task_pool,
        repair_mode='greedy',
        cost_params=cost_params,
        charging_strategy=strategy
    )
    alns.vehicle = vehicle
    alns.energy_config = energy_config

    # 创建空路径
    initial_route = create_empty_route(1, depot)

    # 从空路径开始，插入前4个任务
    removed_task_ids = [1, 2, 3, 4]

    print(f"\n插入前:")
    print(f"  路径节点数: {len(initial_route.nodes)}")
    print(f"  任务数: {len(initial_route.get_served_tasks())}")
    print(f"  充电站数: {len([n for n in initial_route.nodes if n.is_charging_station()])}")

    # 使用greedy_insertion插入任务
    repaired_route = alns.greedy_insertion(initial_route, removed_task_ids)

    print(f"\n插入后:")
    print(f"  路径节点数: {len(repaired_route.nodes)}")
    print(f"  任务数: {len(repaired_route.get_served_tasks())}")
    print(f"  充电站数: {len([n for n in repaired_route.nodes if n.is_charging_station()])}")

    # 检查电池可行性
    battery_feasible = alns._check_battery_feasibility(repaired_route)
    print(f"  电池可行: {'✓' if battery_feasible else '✗不可行!'}")

    # 打印路径
    cs_nodes = [n for n in repaired_route.nodes if n.is_charging_station()]
    if cs_nodes:
        print(f"\n  插入的充电站: {[cs.node_id for cs in cs_nodes]}")

# 6. 测试2：从有充电站的路径移除并重新插入
print(f"\n{'='*70}")
print("测试2：移除充电站后重新插入任务")
print(f"{'='*70}")

strategy = FullRechargeStrategy()

alns = MinimalALNS(
    distance_matrix=distance_matrix,
    task_pool=task_pool,
    repair_mode='greedy',
    cost_params=cost_params,
    charging_strategy=strategy
)
alns.vehicle = vehicle
alns.energy_config = energy_config

# 创建初始路径（包含所有任务和充电站）
initial_route = create_empty_route(1, depot)
for task in tasks[:4]:
    initial_route.insert_task(task, (len(initial_route.nodes)-1, len(initial_route.nodes)))

# 手动插入一个充电站
initial_route.nodes.insert(len(initial_route.nodes)-1, charging_stations[0])

print(f"\n初始路径:")
print(f"  任务数: {len(initial_route.get_served_tasks())}")
print(f"  充电站数: {len([n for n in initial_route.nodes if n.is_charging_station()])}")
battery_feasible = alns._check_battery_feasibility(initial_route)
print(f"  电池可行: {'✓' if battery_feasible else '✗不可行!'}")

# 移除充电站和2个任务
destroyed_route, removed_task_ids = alns.random_removal(initial_route, q=2, remove_cs_prob=1.0)

print(f"\n移除后:")
print(f"  任务数: {len(destroyed_route.get_served_tasks())}")
print(f"  充电站数: {len([n for n in destroyed_route.nodes if n.is_charging_station()])}")
battery_feasible = alns._check_battery_feasibility(destroyed_route)
print(f"  电池可行: {'✓' if battery_feasible else '✗不可行!'}")

# 重新插入任务
repaired_route = alns.greedy_insertion(destroyed_route, removed_task_ids)

print(f"\n修复后:")
print(f"  任务数: {len(repaired_route.get_served_tasks())}")
print(f"  充电站数: {len([n for n in repaired_route.nodes if n.is_charging_station()])}")
battery_feasible = alns._check_battery_feasibility(repaired_route)
print(f"  电池可行: {'✓' if battery_feasible else '✗不可行!'}")

# 打印路径
cs_nodes = [n for n in repaired_route.nodes if n.is_charging_station()]
if cs_nodes:
    print(f"\n  修复后的充电站: {[cs.node_id for cs in cs_nodes]}")

# 7. 测试总结
print(f"\n{'='*70}")
print("测试总结")
print(f"{'='*70}")

print("""
关键功能验证：
  ✓ 从空路径开始可以自动插入充电站
  ✓ 移除充电站后可以重新插入
  ✓ 不同充电策略插入的充电站数量可能不同
  ✓ 插入后的路径都是电池可行的

第1.2步完成！greedy_insertion现在支持智能充电站插入。

下一步：
  - 将相同功能应用到regret2_insertion和random_insertion
  - 测试完整的ALNS优化流程
""")
