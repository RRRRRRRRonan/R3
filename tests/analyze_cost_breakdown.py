"""
分析FR vs PR-Minimal的成本分解
为什么总成本相同？
"""

import sys
sys.path.append('src')

from core.node import create_depot, create_task_node_pair, create_charging_node
from core.route import Route, create_empty_route
from core.task import Task
from core.vehicle import create_vehicle
from physics.distance import DistanceMatrix
from physics.energy import EnergyConfig
from physics.time import TimeConfig
from planner.alns import MinimalALNS, CostParameters
from strategy.charging_strategies import (
    FullRechargeStrategy,
    PartialRechargeMinimalStrategy
)
from core.task_pool import TaskPool
import random

random.seed(42)

print("="*70)
print("成本分解分析：FR vs PR-Minimal")
print("="*70)

# 1. 创建简化测试场景（4任务）
depot = create_depot(coordinates=(0, 0))

tasks = []
task_nodes = []
for i in range(1, 5):
    task = Task(task_id=i, pickup_loc=(i*10, i*10), delivery_loc=(i*10+5, i*10+5), demand=5)
    tasks.append(task)
    pickup, delivery = create_task_node_pair(
        task_id=i,
        x=i*10,
        y=i*10,
        demand=5,
        service_time=10
    )
    task_nodes.append((pickup, delivery))

charging_stations = [
    create_charging_node(cs_id=100, x=20, y=20, charging_rate=50.0/3600),
    create_charging_node(cs_id=101, x=30, y=30, charging_rate=50.0/3600),
]

# 2. 配置
vehicle = create_vehicle(vehicle_id=1, capacity=100, battery_capacity=70.0)
energy_config = EnergyConfig(
    consumption_rate=0.5,
    charging_rate=50.0/3600,
    charging_efficiency=0.9
)
time_config = TimeConfig(speed=40.0)

cost_params = CostParameters(
    C_tr=1.0,      # 距离成本
    C_ch=0.6,      # 充电成本
    C_time=0.1,    # 时间成本
    C_delay=2.0
)

# 3. 创建距离矩阵
all_nodes = [depot] + [p for pair in task_nodes for p in pair] + charging_stations
distance_matrix = DistanceMatrix(all_nodes)

# 4. 创建任务池
task_pool = TaskPool()
for task in tasks:
    task_pool.add_task(task)

# 5. 测试两种策略
strategies = [
    (FullRechargeStrategy(), "FR-完全充电"),
    (PartialRechargeMinimalStrategy(safety_margin=0.1), "PR-Minimal-10%"),
]

results = []

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
    alns.time_config = time_config

    # 创建初始解
    initial_route = alns.create_initial_solution()

    # 优化（少量迭代）
    best_route = alns.optimize(initial_route, max_iterations=20)

    # 获取详细成本分解
    breakdown = alns.get_cost_breakdown(best_route)

    print(f"\n成本分解详情:")
    print(f"  距离成本: {breakdown['distance_cost']:.2f} = {breakdown['total_distance']/1000:.2f}km × {cost_params.C_tr}")
    print(f"  充电成本: {breakdown['charging_cost']:.2f} = {breakdown['charging_amount']:.2f}kWh × {cost_params.C_ch}")
    print(f"  时间成本: {breakdown['time_cost']:.2f} = {breakdown['total_time']:.2f}min × {cost_params.C_time}")
    print(f"  延迟成本: {breakdown['delay_cost']:.2f}")
    print(f"  惩罚成本: {breakdown['penalty_cost']:.2f}")
    print(f"  ---")
    print(f"  总成本: {breakdown['total_cost']:.2f}")

    results.append({
        'name': strategy_name,
        'breakdown': breakdown,
        'total_cost': breakdown['total_cost']
    })

# 6. 对比分析
print(f"\n{'='*70}")
print("对比分析")
print(f"{'='*70}")

fr_breakdown = results[0]['breakdown']
pr_breakdown = results[1]['breakdown']

print(f"\n{'成本项':<20} {'FR-完全充电':<20} {'PR-Minimal-10%':<20} {'差异':<15}")
print("-" * 75)

# 距离
fr_dist = fr_breakdown['distance_cost']
pr_dist = pr_breakdown['distance_cost']
print(f"{'距离成本':<20} {fr_dist:<20.2f} {pr_dist:<20.2f} {pr_dist - fr_dist:<15.2f}")

# 充电
fr_charge = fr_breakdown['charging_cost']
pr_charge = pr_breakdown['charging_cost']
print(f"{'充电成本':<20} {fr_charge:<20.2f} {pr_charge:<20.2f} {pr_charge - fr_charge:<15.2f}")

# 时间
fr_time = fr_breakdown['time_cost']
pr_time = pr_breakdown['time_cost']
print(f"{'时间成本':<20} {fr_time:<20.2f} {pr_time:<20.2f} {pr_time - fr_time:<15.2f}")

# 总计
fr_total = results[0]['total_cost']
pr_total = results[1]['total_cost']
print("-" * 75)
print(f"{'总成本':<20} {fr_total:<20.2f} {pr_total:<20.2f} {pr_total - fr_total:<15.2f}")

print(f"\n{'='*70}")
print("结论")
print(f"{'='*70}")

if abs(fr_total - pr_total) < 0.01:
    print("""
问题：为什么总成本几乎相同？

可能原因：
1. 充电成本权重C_ch=0.6太小，无法体现充电量差异
2. 时间成本C_time=0.1太小，无法体现充电时间差异
3. 距离成本主导了总成本（权重C_tr=1.0）

建议改进：
- 增加充电成本权重C_ch（例如从0.6增加到10.0）
- 增加时间成本权重C_time（例如从0.1增加到1.0）
- 这样可以更明显地体现PR-Minimal的节能优势
""")
else:
    savings = fr_total - pr_total
    savings_pct = (savings / fr_total) * 100
    print(f"""
✅ PR-Minimal策略节省成本: {savings:.2f} ({savings_pct:.2f}%)

成本节省主要来自：
- 充电量减少: {fr_breakdown['charging_amount'] - pr_breakdown['charging_amount']:.2f} kWh
- 充电成本减少: {fr_charge - pr_charge:.2f}
- 时间减少: {fr_breakdown['total_time'] - pr_breakdown['total_time']:.2f} min
""")

print(f"\n关键参数：")
print(f"  C_tr (距离成本权重): {cost_params.C_tr}")
print(f"  C_ch (充电成本权重): {cost_params.C_ch}")
print(f"  C_time (时间成本权重): {cost_params.C_time}")
