"""
测试RouteExecutor - 验证visits生成和充电记录
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
from strategy.charging_strategies import FullRechargeStrategy, PartialRechargeMinimalStrategy
from core.route_executor import RouteExecutor


def test_route_executor():
    """测试RouteExecutor生成visits和记录充电"""

    print("=" * 60)
    print("测试RouteExecutor - 充电记录")
    print("=" * 60)

    # 创建简单场景
    depot = create_depot((0, 0))

    # 创建任务（需要充电才能完成）
    pickup1, delivery1 = create_task_node_pair(
        task_id=1, pickup_id=1, delivery_id=2,
        pickup_coords=(30000, 0), delivery_coords=(60000, 0),
        demand=20.0
    )

    pickup2, delivery2 = create_task_node_pair(
        task_id=2, pickup_id=3, delivery_id=4,
        pickup_coords=(70000, 10000), delivery_coords=(80000, 20000),
        demand=20.0
    )

    # 创建充电站
    cs1 = create_charging_node(100, (30000, 0))
    cs2 = create_charging_node(101, (60000, 10000))

    # 创建距离矩阵
    coordinates = {
        depot.node_id: depot.coordinates,
        pickup1.node_id: pickup1.coordinates,
        delivery1.node_id: delivery1.coordinates,
        pickup2.node_id: pickup2.coordinates,
        delivery2.node_id: delivery2.coordinates,
        cs1.node_id: cs1.coordinates,
        cs2.node_id: cs2.coordinates,
    }

    distance_matrix = DistanceMatrix(coordinates, num_tasks=2, num_charging_stations=2)

    # 创建车辆（60kWh电池）
    vehicle = create_vehicle(vehicle_id=1, capacity=100.0, battery_capacity=60.0)

    energy_config = EnergyConfig(consumption_rate=0.5, charging_rate=50.0/3600)
    time_config = TimeConfig(vehicle_speed=10.0)

    # 创建路径：Depot → CS1 → T1P → T1D → CS2 → T2P → T2D → Depot
    route = create_empty_route(1, depot)
    route.nodes.insert(1, cs1)
    route.nodes.insert(2, pickup1)
    route.nodes.insert(3, delivery1)
    route.nodes.insert(4, cs2)
    route.nodes.insert(5, pickup2)
    route.nodes.insert(6, delivery2)

    total_dist = route.calculate_total_distance(distance_matrix) / 1000
    total_energy = total_dist * energy_config.consumption_rate

    print(f"\n路径信息:")
    print(f"  总距离: {total_dist:.1f} km")
    print(f"  总能耗需求: {total_energy:.1f} kWh")
    print(f"  电池容量: {vehicle.battery_capacity} kWh")
    print(f"  充电站数: 2个")

    # 测试1：使用FR策略
    print(f"\n{'='*60}")
    print("测试1: Full Recharge策略")
    print("=" * 60)

    executor = RouteExecutor(distance_matrix, energy_config, time_config)
    executed_route = executor.execute(
        route=route,
        vehicle=vehicle,
        charging_strategy=FullRechargeStrategy()
    )

    print(f"\n执行结果:")
    print(f"  Visits生成: {'✓' if executed_route.visits else '✗'}")
    print(f"  Visits数量: {len(executed_route.visits) if executed_route.visits else 0}")
    print(f"  可行性: {'✓可行' if executed_route.is_feasible else '✗' + executed_route.infeasibility_info}")

    # 统计充电
    if executed_route.visits:
        total_charged = 0.0
        charging_records = []

        print(f"\n充电记录:")
        for i, visit in enumerate(executed_route.visits):
            if visit.node.is_charging_station():
                charged = visit.battery_after_service - visit.battery_after_travel
                total_charged += charged
                charging_records.append({
                    'station': visit.node.node_id,
                    'battery_before': visit.battery_after_travel,
                    'charged': charged,
                    'battery_after': visit.battery_after_service
                })
                print(f"  充电站{visit.node.node_id}: "
                      f"{visit.battery_after_travel:.1f} → {visit.battery_after_service:.1f} kWh "
                      f"(充{charged:.1f} kWh)")

        print(f"\n总充电量: {total_charged:.1f} kWh")
        print(f"总时间: {executed_route.visits[-1].departure_time/60:.1f} min")

        # 验证
        assert total_charged > 0, "FR策略应该有充电量"
        assert len(charging_records) > 0, "应该有充电记录"

    # 测试2：使用PR-Minimal策略
    print(f"\n{'='*60}")
    print("测试2: PR-Minimal策略")
    print("=" * 60)

    executed_route2 = executor.execute(
        route=route,
        vehicle=vehicle,
        charging_strategy=PartialRechargeMinimalStrategy(safety_margin=0.1)
    )

    if executed_route2.visits:
        total_charged2 = 0.0
        charging_records2 = []

        print(f"\n充电记录:")
        for visit in executed_route2.visits:
            if visit.node.is_charging_station():
                charged = visit.battery_after_service - visit.battery_after_travel
                if charged > 0.01:
                    total_charged2 += charged
                    charging_records2.append({
                        'station': visit.node.node_id,
                        'charged': charged
                    })
                    print(f"  充电站{visit.node.node_id}: "
                          f"{visit.battery_after_travel:.1f} → {visit.battery_after_service:.1f} kWh "
                          f"(充{charged:.1f} kWh)")
                else:
                    print(f"  充电站{visit.node.node_id}: 跳过充电 "
                          f"(电量{visit.battery_after_travel:.1f}kWh足够)")

        print(f"\n总充电量: {total_charged2:.1f} kWh")
        print(f"对比FR: 节省{total_charged - total_charged2:.1f} kWh")

    print(f"\n{'='*60}")
    print("✅ RouteExecutor测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    test_route_executor()
