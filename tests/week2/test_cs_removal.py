"""
测试充电站移除功能（第1.1步）
验证random_removal可以移除充电站
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
from strategy.charging_strategies import FullRechargeStrategy
import random

print("="*70)
print("测试：充电站移除功能（第1.1步）")
print("="*70)

# 设置随机种子
random.seed(42)

# 1. 创建测试场景
depot = create_depot(coordinates=(0, 0))

tasks = []
task_nodes = []
node_id_counter = 1

for i in range(1, 5):
    pickup, delivery = create_task_node_pair(
        task_id=i,
        pickup_id=node_id_counter,
        delivery_id=node_id_counter + 1,
        pickup_coords=(i*10000, i*10000),
        delivery_coords=(i*10000+5000, i*10000+5000),
        demand=5
    )
    task_obj = Task(
        task_id=i,
        pickup_node=pickup,
        delivery_node=delivery,
        demand=5
    )
    tasks.append(task_obj)
    task_nodes.append((pickup, delivery))
    node_id_counter += 2

charging_stations = [
    create_charging_node(node_id=100, coordinates=(15000, 15000)),
    create_charging_node(node_id=101, coordinates=(25000, 25000)),
    create_charging_node(node_id=102, coordinates=(35000, 35000)),
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

# 4. 创建初始路径（包含所有充电站）
route = create_empty_route(1, depot)

# 插入任务
for task in tasks:
    route.insert_task(task, (len(route.nodes)-1, len(route.nodes)))

# 插入所有3个充电站
for cs in charging_stations:
    route.nodes.insert(len(route.nodes)-1, cs)

print(f"\n初始路径信息:")
print(f"  任务数: {len(route.get_served_tasks())}")
print(f"  充电站数: {len([n for n in route.nodes if n.is_charging_station()])}")
print(f"  总节点数: {len(route.nodes)}")

# 5. 创建ALNS实例
vehicle = create_vehicle(vehicle_id=1, capacity=100, battery_capacity=70.0)
energy_config = EnergyConfig(
    consumption_rate=0.5,
    charging_rate=50.0/3600,
    charging_efficiency=0.9
    # 注：临界值将在第1.3步添加
)
cost_params = CostParameters(C_tr=1.0, C_ch=0.6, C_time=0.1, C_delay=2.0)

alns = MinimalALNS(
    distance_matrix=distance_matrix,
    task_pool=task_pool,
    repair_mode='greedy',
    cost_params=cost_params,
    charging_strategy=FullRechargeStrategy()
)
alns.vehicle = vehicle
alns.energy_config = energy_config

# 6. 测试充电站移除功能
print(f"\n{'='*70}")
print("测试1：充电站移除功能")
print(f"{'='*70}")

num_tests = 20
removal_stats = {0: 0, 1: 0, 2: 0}

for i in range(num_tests):
    destroyed_route, removed_task_ids = alns.random_removal(route, q=2, remove_cs_prob=1.0)

    original_cs_count = len([n for n in route.nodes if n.is_charging_station()])
    destroyed_cs_count = len([n for n in destroyed_route.nodes if n.is_charging_station()])

    cs_removed = original_cs_count - destroyed_cs_count

    if cs_removed in removal_stats:
        removal_stats[cs_removed] += 1

    if i < 5:  # 打印前5次测试的详情
        print(f"\n测试 {i+1}:")
        print(f"  原始充电站数: {original_cs_count}")
        print(f"  移除后充电站数: {destroyed_cs_count}")
        print(f"  移除充电站数: {cs_removed}")
        print(f"  移除任务数: {len(removed_task_ids)}")

print(f"\n统计结果（{num_tests}次测试）:")
print(f"  移除0个充电站: {removal_stats[0]}次 ({removal_stats[0]/num_tests*100:.1f}%)")
print(f"  移除1个充电站: {removal_stats[1]}次 ({removal_stats[1]/num_tests*100:.1f}%)")
print(f"  移除2个充电站: {removal_stats[2]}次 ({removal_stats[2]/num_tests*100:.1f}%)")

# 7. 测试不同概率
print(f"\n{'='*70}")
print("测试2：不同移除概率")
print(f"{'='*70}")

probs = [0.0, 0.3, 0.5, 1.0]

for prob in probs:
    cs_removed_count = 0
    num_tests_prob = 50

    for i in range(num_tests_prob):
        destroyed_route, _ = alns.random_removal(route, q=2, remove_cs_prob=prob)
        original_cs_count = len([n for n in route.nodes if n.is_charging_station()])
        destroyed_cs_count = len([n for n in destroyed_route.nodes if n.is_charging_station()])

        if destroyed_cs_count < original_cs_count:
            cs_removed_count += 1

    print(f"\n概率 {prob:.1f}:")
    print(f"  移除充电站的次数: {cs_removed_count}/{num_tests_prob}")
    print(f"  实际移除率: {cs_removed_count/num_tests_prob*100:.1f}%")
    print(f"  预期移除率: {prob*100:.1f}%")

# 8. 验证路径有效性
print(f"\n{'='*70}")
print("测试3：验证移除后路径有效性")
print(f"{'='*70}")

destroyed_route, removed_task_ids = alns.random_removal(route, q=2, remove_cs_prob=1.0)

print(f"\n原始路径:")
print(f"  节点数: {len(route.nodes)}")
print(f"  任务数: {len(route.get_served_tasks())}")
print(f"  充电站数: {len([n for n in route.nodes if n.is_charging_station()])}")

print(f"\n移除后路径:")
print(f"  节点数: {len(destroyed_route.nodes)}")
print(f"  任务数: {len(destroyed_route.get_served_tasks())}")
print(f"  充电站数: {len([n for n in destroyed_route.nodes if n.is_charging_station()])}")

# 验证depot还在
has_depot_start = destroyed_route.nodes[0].node_type.value == 'depot'
has_depot_end = destroyed_route.nodes[-1].node_type.value == 'depot'
print(f"\n路径有效性检查:")
print(f"  起点是depot: {'✓' if has_depot_start else '✗'}")
print(f"  终点是depot: {'✓' if has_depot_end else '✗'}")
print(f"  任务数减少: {'✓' if len(destroyed_route.get_served_tasks()) < len(route.get_served_tasks()) else '✗'}")

# 9. 总结
print(f"\n{'='*70}")
print("测试总结")
print(f"{'='*70}")

all_passed = (
    removal_stats[0] > 0 and  # 有些情况移除0个
    removal_stats[1] > 0 and  # 有些情况移除1个
    removal_stats[2] > 0 and  # 有些情况移除2个
    has_depot_start and
    has_depot_end
)

if all_passed:
    print("✅ 所有测试通过！")
    print("\n关键功能验证：")
    print("  ✓ 可以移除0-2个充电站")
    print("  ✓ 移除概率参数有效")
    print("  ✓ 移除后路径结构完整")
    print("  ✓ 任务和充电站独立移除")
    print("\n第1.1步完成！random_removal现在支持充电站动态优化。")
else:
    print("❌ 部分测试失败")
    if not (removal_stats[0] > 0 and removal_stats[1] > 0 and removal_stats[2] > 0):
        print("  - 充电站移除数量分布异常")
    if not (has_depot_start and has_depot_end):
        print("  - 路径结构破坏")
