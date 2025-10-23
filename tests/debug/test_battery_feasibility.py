"""
测试电池可行性检查
"""

import sys
sys.path.append('src')

from core.node import create_depot, create_task_node_pair, create_charging_node
from core.route import Route, create_empty_route
from core.task import Task, TaskPool
from core.vehicle import create_vehicle
from physics.distance import DistanceMatrix
from physics.energy import EnergyConfig
from strategy.charging_strategies import FullRechargeStrategy
from planner.alns import MinimalALNS, CostParameters


def test_simple_route():
    """测试简单路径的电池可行性"""

    # 创建简单场景
    depot = create_depot((0, 0))

    # 创建2个远距离任务
    pickup1, delivery1 = create_task_node_pair(
        task_id=1, pickup_id=1, delivery_id=2,
        pickup_coords=(40000, 0), delivery_coords=(80000, 0),
        demand=20.0
    )
    task1 = Task(task_id=1, pickup_node=pickup1, delivery_node=delivery1, demand=20.0)

    # 创建充电站
    cs1 = create_charging_node(100, (40000, 0))

    # 创建距离矩阵
    coordinates = {
        depot.node_id: depot.coordinates,
        pickup1.node_id: pickup1.coordinates,
        delivery1.node_id: delivery1.coordinates,
        cs1.node_id: cs1.coordinates,
    }

    distance_matrix = DistanceMatrix(coordinates, num_tasks=1, num_charging_stations=1)

    # 创建车辆（足够的电池：80kWh可以完成任务，如果在中间充一次）
    vehicle = create_vehicle(vehicle_id=1, capacity=100.0, battery_capacity=60.0)

    energy_config = EnergyConfig(consumption_rate=0.5, charging_rate=50.0/3600)

    # 测试1：无充电站路径（不可行）
    print("=" * 60)
    print("测试1: 无充电站路径（应该不可行）")
    print("=" * 60)

    route1 = create_empty_route(1, depot)
    route1.nodes.insert(1, pickup1)
    route1.nodes.insert(2, delivery1)

    total_dist = route1.calculate_total_distance(distance_matrix) / 1000
    energy_needed = total_dist * energy_config.consumption_rate

    print(f"路径: Depot → Task1P → Task1D → Depot")
    print(f"总距离: {total_dist:.1f} km")
    print(f"能耗需求: {energy_needed:.1f} kWh")
    print(f"电池容量: {vehicle.battery_capacity} kWh")

    task_pool = TaskPool()
    task_pool.add_task(task1)

    alns = MinimalALNS(distance_matrix, task_pool, charging_strategy=FullRechargeStrategy())
    alns.vehicle = vehicle
    alns.energy_config = energy_config

    print("\n模拟过程:")
    feasible1 = alns._check_battery_feasibility(route1, debug=True)
    cost1 = alns.evaluate_cost(route1)

    print(f"\n可行性: {'✓可行' if feasible1 else '✗不可行'}")
    print(f"成本: {cost1:.2f}")

    # 测试2：有充电站路径（可行）
    print(f"\n{'=' * 60}")
    print("测试2: 有充电站路径（应该可行）")
    print("=" * 60)

    route2 = create_empty_route(1, depot)
    route2.nodes.insert(1, cs1)
    route2.nodes.insert(2, pickup1)
    route2.nodes.insert(3, delivery1)

    total_dist2 = route2.calculate_total_distance(distance_matrix) / 1000
    energy_needed2 = total_dist2 * energy_config.consumption_rate

    print(f"路径: Depot → CS1 → Task1P → Task1D → Depot")
    print(f"总距离: {total_dist2:.1f} km")
    print(f"能耗需求: {energy_needed2:.1f} kWh")
    print(f"电池容量: {vehicle.battery_capacity} kWh")

    print("\n模拟过程:")
    feasible2 = alns._check_battery_feasibility(route2, debug=True)
    cost2 = alns.evaluate_cost(route2)

    print(f"可行性: {'✓可行' if feasible2 else '✗不可行'}")
    print(f"成本: {cost2:.2f}")

    # 验证结果
    print(f"\n{'=' * 60}")
    print("结果验证")
    print("=" * 60)

    assert not feasible1, "测试1应该不可行"
    assert feasible2, "测试2应该可行"
    assert cost1 > cost2, "不可行解成本应该更高"

    print("✅ 所有测试通过!")


if __name__ == "__main__":
    test_simple_route()
