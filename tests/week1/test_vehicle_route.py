"""
Vehicle和Route模块测试
======================
测试AMR和路径的基础功能、状态管理、约束验证
"""

import sys
sys.path.append('src')

from core.node import DepotNode, create_task_node_pair
from core.task import create_task
from core.vehicle import (
    Vehicle, VehicleStatus, VehicleFleet,
    create_vehicle, create_homogeneous_fleet
)
from core.route import (
    Route, RouteNodeVisit,
    create_empty_route, create_route_from_node_sequence
)
from physics.distance import create_distance_matrix_from_layout
from physics.time import TimeWindow, TimeConfig
from physics.energy import EnergyConfig


def test_vehicle_creation():
    """测试1: Vehicle创建和基本属性"""
    print("=" * 60)
    print("测试1: Vehicle创建和基本属性")
    print("=" * 60)
    
    # 创建默认AMR
    amr1 = create_vehicle(1)
    print(f"✓ 创建AMR: {amr1}")
    print(f"  - 容量: {amr1.capacity}")
    print(f"  - 电池: {amr1.battery_capacity} kWh")
    print(f"  - 速度: {amr1.speed} m/s")
    print(f"  - 初始位置: {amr1.initial_location}")
    print(f"  - 状态: {amr1.status.value}")
    
    # 创建自定义AMR
    amr2 = create_vehicle(
        2,
        capacity=200.0,
        battery_capacity=120.0,
        initial_battery=80.0,
        initial_location=(10, 10)
    )
    print(f"\n✓ 创建自定义AMR: {amr2}")
    print(f"  - 最大载重: {amr2.capacity} kg")
    print(f"  - 电量比例: {amr2.get_battery_ratio()*100:.1f}%")
    print(f"  - 剩余容量: {amr2.get_remaining_capacity()} kg")


def test_vehicle_state_management():
    """测试2: Vehicle状态管理"""
    print("\n" + "=" * 60)
    print("测试2: Vehicle状态管理")
    print("=" * 60)
    
    amr = create_vehicle(1, capacity=150.0, battery_capacity=100.0)
    
    # 状态查询
    print(f"✓ 初始状态:")
    print(f"  - is_idle: {amr.is_idle()}")
    print(f"  - is_available: {amr.is_available()}")
    print(f"  - has_route: {amr.has_route()}")
    
    # 移动
    print(f"\n✓ 移动到 (20, 30):")
    amr.move_to((20, 30), time=50.0)
    print(f"  - 当前位置: {amr.current_location}")
    print(f"  - 当前时间: {amr.current_time}s")
    print(f"  - 状态: {amr.status.value}")
    
    # 装载
    print(f"\n✓ 装载货物 (30.0 kg):")
    amr.pickup_load(30.0)
    print(f"  - 当前载重: {amr.current_load} kg")
    print(f"  - 载重比例: {amr.get_load_ratio()*100:.1f}%")
    print(f"  - 剩余容量: {amr.get_remaining_capacity()} kg")
    
    # 消耗电量
    print(f"\n✓ 消耗电量 (15.0 kWh):")
    amr.consume_battery(15.0)
    print(f"  - 当前电量: {amr.current_battery} kWh")
    print(f"  - 电量比例: {amr.get_battery_ratio()*100:.1f}%")
    
    # 充电
    print(f"\n✓ 充电 (20.0 kWh):")
    amr.charge_battery(20.0, time=150.0)
    print(f"  - 当前电量: {amr.current_battery} kWh")
    print(f"  - 状态: {amr.status.value}")
    
    # 卸载
    print(f"\n✓ 卸载货物 (30.0 kg):")
    amr.deliver_load(30.0)
    print(f"  - 当前载重: {amr.current_load} kg")
    
    # 状态摘要
    print(f"\n✓ 状态摘要:")
    print(amr.get_state_summary())


def test_vehicle_fleet():
    """测试3: 车队管理"""
    print("\n" + "=" * 60)
    print("测试3: 车队管理")
    print("=" * 60)
    
    # 创建同质车队
    fleet = create_homogeneous_fleet(
        num_vehicles=5,
        capacity=150.0,
        battery_capacity=100.0
    )
    print(f"✓ 创建同质车队: {fleet}")
    
    # 车队统计
    stats = fleet.get_fleet_statistics()
    print(f"\n✓ 车队统计:")
    print(f"  - 总数: {stats['total']}")
    print(f"  - 可用: {stats['available']}")
    print(f"  - 空闲: {stats['idle']}")
    print(f"  - 平均电量: {stats['avg_battery_ratio']*100:.1f}%")
    
    # 分配AMR
    amr1 = fleet.get_vehicle(1)
    amr1.status = VehicleStatus.MOVING
    
    amr2 = fleet.get_vehicle(2)
    amr2.status = VehicleStatus.CHARGING
    
    print(f"\n✓ 更新部分AMR状态后:")
    stats = fleet.get_fleet_statistics()
    print(f"  - 可用: {stats['available']}")
    print(f"  - 移动中: {stats['moving']}")
    print(f"  - 充电中: {stats['charging']}")


def test_route_creation():
    """测试4: Route创建和节点操作"""
    print("\n" + "=" * 60)
    print("测试4: Route创建和节点操作")
    print("=" * 60)
    
    # 创建depot
    depot = DepotNode(coordinates=(0, 0))
    
    # 创建空路径
    route = create_empty_route(vehicle_id=1, depot_node=depot)
    print(f"✓ 创建空路径: {route}")
    print(f"  - 节点数: {route.get_num_nodes()}")
    print(f"  - 是否为空: {route.is_empty()}")
    
    # 创建任务节点
    p1, d1 = create_task_node_pair(
        1, 1, 3, (10, 20), (15, 25),
        TimeWindow(0, 100), TimeWindow(50, 150)
    )
    
    p2, d2 = create_task_node_pair(
        2, 2, 4, (30, 40), (35, 45),
        TimeWindow(0, 100), TimeWindow(50, 150)
    )
    
    # 添加节点
    print(f"\n✓ 添加任务1:")
    route.add_node(p1)
    route.add_node(d1)
    print(f"  路径: {route}")
    print(f"  节点数: {route.get_num_nodes()}")
    
    print(f"\n✓ 添加任务2:")
    route.add_node(p2)
    route.add_node(d2)
    print(f"  路径: {route}")
    print(f"  服务的任务: {route.get_served_tasks()}")
    
    # 查询
    print(f"\n✓ 路径查询:")
    print(f"  - pickup节点数: {len(route.get_pickup_nodes())}")
    print(f"  - delivery节点数: {len(route.get_delivery_nodes())}")
    print(f"  - 包含任务1: {route.contains_task(1)}")
    print(f"  - 包含任务3: {route.contains_task(3)}")


def test_route_validation():
    """测试5: Route约束验证"""
    print("\n" + "=" * 60)
    print("测试5: Route约束验证")
    print("=" * 60)
    
    depot = DepotNode(coordinates=(0, 0))
    
    # 场景1: 正确的precedence
    print(f"✓ 场景1: 正确的precedence")
    p1, d1 = create_task_node_pair(
        1, 1, 3, (10, 20), (15, 25),
        TimeWindow(0, 100), TimeWindow(50, 150)
    )
    route1 = create_route_from_node_sequence(1, [depot, p1, d1, depot])
    is_valid, error = route1.validate_precedence()
    print(f"  - 验证结果: {is_valid}")
    if error:
        print(f"  - 错误: {error}")
    
    # 场景2: 错误的precedence（delivery在pickup前）
    print(f"\n✓ 场景2: 错误的precedence (delivery在pickup前)")
    route2 = create_route_from_node_sequence(1, [depot, d1, p1, depot])
    is_valid, error = route2.validate_precedence()
    print(f"  - 验证结果: {is_valid}")
    if error:
        print(f"  - 错误: {error}")
    
    # 场景3: 缺少delivery
    print(f"\n✓ 场景3: 缺少delivery")
    route3 = create_route_from_node_sequence(1, [depot, p1, depot])
    is_valid, error = route3.validate_precedence()
    print(f"  - 验证结果: {is_valid}")
    if error:
        print(f"  - 错误: {error}")


def test_route_schedule_computation():
    """测试6: Route时间表计算"""
    print("\n" + "=" * 60)
    print("测试6: Route时间表计算")
    print("=" * 60)
    
    # 创建简单场景
    depot = DepotNode(coordinates=(0, 0))
    
    p1, d1 = create_task_node_pair(
        1, 1, 3, (10, 20), (15, 25),
        TimeWindow(0, 200), TimeWindow(100, 300)
    )
    
    p2, d2 = create_task_node_pair(
        2, 2, 4, (30, 40), (35, 45),
        TimeWindow(0, 200), TimeWindow(100, 300)
    )
    
    # 创建路径
    route = create_route_from_node_sequence(
        vehicle_id=1,
        nodes=[depot, p1, d1, p2, d2, depot]
    )
    
    # 创建距离矩阵
    dm = create_distance_matrix_from_layout(
        depot=(0, 0),
        task_locations=[
            ((10, 20), (15, 25)),
            ((30, 40), (35, 45))
        ],
        charging_stations=[]
    )
    
    # 计算时间表
    print(f"✓ 计算时间表:")
    is_feasible = route.compute_schedule(
        distance_matrix=dm,
        vehicle_capacity=150.0,
        vehicle_battery_capacity=100.0,
        initial_battery=100.0
    )
    
    print(f"  - 可行性: {is_feasible}")
    
    if is_feasible:
        # 打印详细时间表
        print(f"\n{route.get_detailed_string()}")
        
        # 计算指标
        metrics = route.get_metrics(dm)
        print(f"\n✓ 路径指标:")
        print(f"  - 总距离: {metrics['total_distance']:.2f} m")
        print(f"  - 总时间: {metrics['total_time']:.2f} s")
        print(f"  - 总能耗: {metrics['total_energy']:.2f} kWh")
        print(f"  - 任务数: {metrics['num_tasks']}")


def test_route_infeasibility():
    """测试7: Route不可行场景"""
    print("\n" + "=" * 60)
    print("测试7: Route不可行场景")
    print("=" * 60)
    
    depot = DepotNode(coordinates=(0, 0))
    
    # 场景1: 容量超限
    print(f"✓ 场景1: 容量超限")
    p1, d1 = create_task_node_pair(
        1, 1, 3, (10, 20), (15, 25),
        TimeWindow(0, 200), TimeWindow(100, 300),
        demand=200.0  # 超过容量150.0 kg
    )
    
    route1 = create_route_from_node_sequence(1, [depot, p1, d1, depot])
    
    dm = create_distance_matrix_from_layout(
        depot=(0, 0),
        task_locations=[((10, 20), (15, 25))],
        charging_stations=[]
    )
    
    is_feasible = route1.compute_schedule(
        distance_matrix=dm,
        vehicle_capacity=150.0,
        vehicle_battery_capacity=100.0,
        initial_battery=100.0
    )
    
    print(f"  - 可行性: {is_feasible}")
    print(f"  - 原因: {route1.infeasibility_info}")
    
    # 场景2: 时间窗违反（硬约束）
    print(f"\n✓ 场景2: 硬时间窗违反")
    p2, d2 = create_task_node_pair(
        1, 1, 2, (80, 90), (85, 95),  # 修改ID: task_id=1, pickup=1, delivery=2
        TimeWindow(0, 50, window_type='hard'),  # 很紧的时间窗
        TimeWindow(10, 60, window_type='hard')
    )
    
    route2 = create_route_from_node_sequence(1, [depot, p2, d2, depot])
    
    dm2 = create_distance_matrix_from_layout(
        depot=(0, 0),
        task_locations=[((80, 90), (85, 95))],
        charging_stations=[]
    )
    
    is_feasible = route2.compute_schedule(
        distance_matrix=dm2,
        vehicle_capacity=10.0,
        vehicle_battery_capacity=100.0,
        initial_battery=100.0
    )
    
    print(f"  - 可行性: {is_feasible}")
    print(f"  - 原因: {route2.infeasibility_info}")


if __name__ == "__main__":
    test_vehicle_creation()
    test_vehicle_state_management()
    test_vehicle_fleet()
    test_route_creation()
    test_route_validation()
    test_route_schedule_computation()
    test_route_infeasibility()
    
    print("\n" + "=" * 60)
    print("🎉 所有测试完成！")
    print("=" * 60)
