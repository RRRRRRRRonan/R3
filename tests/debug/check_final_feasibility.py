"""
检查PR-Minimal最终解的电池可行性
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
from strategy.charging_strategies import PartialRechargeMinimalStrategy
import random

print("="*70)
print("检查PR-Minimal最终优化解的电池可行性")
print("="*70)

random.seed(42)

# 创建与原始测试相同的场景
depot = create_depot(coordinates=(0, 0))

tasks = []
node_id_counter = 1

task_locations = [
    ((10000, 0), (15000, 0)),
    ((20000, 0), (25000, 0)),
    ((35000, 0), (40000, 0)),
    ((50000, 0), (55000, 0)),
    ((65000, 0), (70000, 0)),
    ((80000, 0), (85000, 0)),
    ((45000, 10000), (50000, 10000)),
    ((30000, 10000), (35000, 10000)),
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
    node_id_counter += 2

charging_stations = [
    create_charging_node(node_id=100, coordinates=(15000, 0)),
    create_charging_node(node_id=101, coordinates=(30000, 0)),
    create_charging_node(node_id=102, coordinates=(45000, 0)),
    create_charging_node(node_id=103, coordinates=(60000, 0)),
    create_charging_node(node_id=104, coordinates=(75000, 0)),
    create_charging_node(node_id=105, coordinates=(40000, 10000)),
]

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

task_pool = TaskPool()
for task in tasks:
    task_pool.add_task(task)

vehicle = create_vehicle(vehicle_id=1, capacity=100, battery_capacity=60.0)
energy_config = EnergyConfig(
    consumption_rate=0.5,
    charging_rate=50.0/3600,
    charging_efficiency=0.9,
    critical_battery_threshold=0.0  # Week 2修复：禁用临界值
)

cost_params = CostParameters(
    C_tr=1.0,
    C_ch=10.0,
    C_time=1.0,
    C_delay=2.0
)

strategy = PartialRechargeMinimalStrategy(safety_margin=0.1)

alns = MinimalALNS(
    distance_matrix=distance_matrix,
    task_pool=task_pool,
    repair_mode='greedy',
    cost_params=cost_params,
    charging_strategy=strategy
)
alns.vehicle = vehicle
alns.energy_config = energy_config

# 创建初始解并优化
initial_route = create_empty_route(1, depot)

for task in tasks:
    initial_route = alns.greedy_insertion(initial_route, [task.task_id])

print(f"初始解创建:")
print(f"  充电站数: {len([n for n in initial_route.nodes if n.is_charging_station()])}")
print(f"  初始可行性: {'✓' if alns._check_battery_feasibility(initial_route) else '✗'}")

# ALNS优化
best_route = alns.optimize(initial_route, max_iterations=50)

print(f"\nALNS优化后:")
print(f"  充电站数: {len([n for n in best_route.nodes if n.is_charging_station()])}")
print(f"  充电站位置: {[n.node_id for n in best_route.nodes if n.is_charging_station()]}")

# 详细检查电池可行性
print(f"\n电池可行性详细检查:")
battery_feasible = alns._check_battery_feasibility(best_route, debug=True)
print(f"  最终可行性: {'✓' if battery_feasible else '✗'}")

# 获取成本分解
breakdown = alns.get_cost_breakdown(best_route)
print(f"\n成本分解:")
print(f"  距离成本: {breakdown['distance_cost']:.2f}")
print(f"  充电成本: {breakdown['charging_cost']:.2f} ({breakdown['total_charging']:.2f}kWh)")
print(f"  时间成本: {breakdown['time_cost']:.2f}")
print(f"  延迟成本: {breakdown['delay_cost']:.2f}")
print(f"  小计: {breakdown['distance_cost'] + breakdown['charging_cost'] + breakdown['time_cost'] + breakdown['delay_cost']:.2f}")
print(f"  总成本: {breakdown['total_cost']:.2f}")
print(f"  差额: {breakdown['total_cost'] - (breakdown['distance_cost'] + breakdown['charging_cost'] + breakdown['time_cost'] + breakdown['delay_cost']):.2f}")

# 显示充电站详细充电记录
if best_route.visits:
    print(f"\n充电站详细记录:")
    for visit in best_route.visits:
        if visit.node.is_charging_station():
            charge = visit.battery_after_service - visit.battery_after_travel
            print(f"  CS {visit.node.node_id}: 到达{visit.battery_after_travel:.1f}kWh → 充电{charge:.1f}kWh → 离开{visit.battery_after_service:.1f}kWh")
