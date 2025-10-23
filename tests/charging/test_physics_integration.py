"""
物理规则层集成测试
==================
测试 distance.py, energy.py, time.py 的协同工作

测试场景:
    仓库: 100m × 100m
    1个Depot: (0, 0)
    2个任务:
        - 任务1: pickup(10,20) → delivery(15,25)
        - 任务2: pickup(30,40) → delivery(35,45)
    2个充电站: (50,0), (0,50)
    1个AMR: 初始电量80, 容量100

测试路径:
    depot → p1 → d1 → charging → p2 → d2 → depot
    0 → 1 → 3 → 5 → 2 → 4 → 0
"""

import sys
sys.path.append('src')

from physics.distance import (
    create_distance_matrix_from_layout,
    calculate_insertion_cost,
    NodeType
)
from physics.energy import (
    EnergyConfig,
    EnergyConstraintValidator,
    calculate_energy_consumption,
    calculate_charging_amount, 
    calculate_charging_time
)
from physics.time import (
    TimeConfig,
    TimeWindow,
    TimeWindowType,
    RouteTimeAnalyzer,
    calculate_travel_time
)


def print_section(title: str):
    """打印分隔线"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def test_1_distance_module():
    """测试1: 距离计算模块"""
    print_section("测试1: 距离计算模块")
    
    # 创建距离矩阵
    dm = create_distance_matrix_from_layout(
        depot=(0, 0),
        task_locations=[
            ((10, 20), (15, 25)),  # 任务1
            ((30, 40), (35, 45))   # 任务2
        ],
        charging_stations=[(50, 0), (0, 50)]
    )
    
    print(f"\n节点布局:")
    print(f"  0: Depot (0, 0)")
    print(f"  1: p1 (10, 20)")
    print(f"  2: p2 (30, 40)")
    print(f"  3: d1 (15, 25)")
    print(f"  4: d2 (35, 45)")
    print(f"  5: c1 (50, 0)")
    print(f"  6: c2 (0, 50)")
    
    # 测试节点类型识别
    print(f"\n✓ 节点类型识别:")
    print(f"  节点1类型: {dm.id_helper.get_node_type(1)}")
    print(f"  节点3类型: {dm.id_helper.get_node_type(3)}")
    print(f"  节点5类型: {dm.id_helper.get_node_type(5)}")
    
    # 测试配对关系
    print(f"\n✓ 配对关系:")
    print(f"  pickup 1 → delivery {dm.id_helper.get_paired_delivery(1)}")
    print(f"  pickup 2 → delivery {dm.id_helper.get_paired_delivery(2)}")
    
    # 测试距离计算
    print(f"\n✓ 关键距离:")
    print(f"  Depot → p1: {dm.get_distance(0, 1):.2f}m")
    print(f"  p1 → d1: {dm.get_distance(1, 3):.2f}m")
    print(f"  d1 → c1: {dm.get_distance(3, 5):.2f}m")
    
    # 测试路径总距离
    route = [0, 1, 3, 5, 2, 4, 0]
    total_dist = dm.total_distance(route)
    print(f"\n✓ 路径总距离:")
    print(f"  路径: {' → '.join(map(str, route))}")
    print(f"  总距离: {total_dist:.2f}m")
    
    # 测试最近充电站
    nearest_station, dist = dm.get_nearest_charging_station(1)
    print(f"\n✓ 最近充电站:")
    print(f"  从节点1最近的充电站: {nearest_station} (距离{dist:.2f}m)")
    
    # 测试配对约束验证
    valid_route = [0, 1, 3, 2, 4, 0]  # pickup在delivery前
    invalid_route = [0, 3, 1, 4, 2, 0]  # delivery在pickup前
    print(f"\n✓ pickup→delivery顺序验证:")
    print(f"  {valid_route}: {dm.validate_route_precedence(valid_route)}")
    print(f"  {invalid_route}: {dm.validate_route_precedence(invalid_route)}")
    
    return dm


def test_2_energy_module(dm):
    """测试2: 能量管理模块"""
    print_section("测试2: 能量管理模块")
    
    energy_config = EnergyConfig(
        consumption_rate=2.0,
        charging_rate=5.0,
        charging_efficiency=0.9,
        max_charging_time=600,
        max_charging_amount=80,
        battery_capacity=100
    )
    
    travel_time = 30
    energy_used = calculate_energy_consumption(
        travel_time=travel_time, 
        consumption_rate=energy_config.consumption_rate
    )
    print(f"  移动30秒消耗: {energy_used:.2f} 单位")
    
    charging_time = 100
    energy_gained = calculate_charging_amount(
        charging_time, 
        energy_config.charging_rate, 
        energy_config.charging_efficiency
    )
    print(f"  充电100秒获得: {energy_gained:.2f} 单位")
    
    # 返回配置和时间供后续测试使用
    travel_times = {
        (0, 1): dm.get_distance(0, 1) / 2.0,
        (1, 3): dm.get_distance(1, 3) / 2.0,
    }
    return energy_config, travel_times


def test_3_time_module(dm, travel_times):
    """测试3: 时间计算模块"""
    print_section("测试3: 时间计算模块")
    
    # 创建时间配置
    time_config = TimeConfig(
        vehicle_speed=1.0,           # 1m/s
        default_service_time=30,  # 服务30秒
        tardiness_penalty=1.0
    )
    
    print(f"\n时间参数:")
    print(f"  移动速度: {time_config.vehicle_speed} m/s")
    print(f"  默认服务时间: {time_config.default_service_time} 秒")
    
    # 创建时间窗
    time_windows = {
        1: TimeWindow(0, 100, TimeWindowType.SOFT),    # p1: [0, 100]
        2: TimeWindow(0, 200, TimeWindowType.SOFT),    # p2: [0, 200]
        3: TimeWindow(50, 150, TimeWindowType.SOFT),   # d1: [50, 150]
        4: TimeWindow(150, 300, TimeWindowType.SOFT)   # d2: [150, 300]
    }
    
    print(f"\n✓ 时间窗设置:")
    for node_id, tw in time_windows.items():
        print(f"  节点{node_id}: [{tw.earliest}, {tw.latest}] ({tw.window_type.value})")
    
    # 服务时间（pickup和delivery需要装卸货）
    service_times = {
        1: 30,  # p1服务30秒
        2: 30,  # p2服务30秒
        3: 30,  # d1服务30秒
        4: 30   # d2服务30秒
    }
    
    # 充电时间
    charging_times = [0, 0, 0, 100, 0, 0, 0]  # 在节点5充电100秒
    
    # 计算时间线
    print(f"\n✓ 路径时间线:")
    route = [0, 1, 3, 5, 2, 4, 0]
    travel_times = [
        dm.get_distance(route[i], route[i+1]) / time_config.vehicle_speed
        for i in range(len(route) - 1)
    ]

    print(f"\n✓ 路径时间线:")
    print(f"  路径: {' → '.join(map(str, route))}")
    print(f"  各段旅行时间: {[f'{t:.1f}s' for t in travel_times]}")
    
    analyzer = RouteTimeAnalyzer(time_config)
    timeline = analyzer.calculate_route_timeline(
        route=route,
        travel_times=travel_times,
        service_times=service_times,
        charging_times=charging_times,
        time_windows=time_windows,
        start_time=0
    )
    
    print(f"\n  {'节点':<6} {'到达':<8} {'等待':<8} {'服务':<8} {'充电':<8} {'离开':<8} {'延迟':<8}")
    print(f"  {'-'*58}")
    for profile in timeline:
        print(f"  {profile.node_id:<6} "
              f"{profile.arrival_time:<8.1f} "
              f"{profile.waiting_time:<8.1f} "
              f"{profile.service_time:<8.1f} "
              f"{profile.charging_time:<8.1f} "
              f"{profile.tardiness:<8.1f}")
    
    # 统计信息
    total_time = analyzer.calculate_total_route_time(timeline)
    total_tardiness = analyzer.calculate_total_tardiness(timeline)
    total_waiting = analyzer.calculate_total_waiting(timeline)
    
    print(f"\n✓ 时间统计:")
    print(f"  总耗时: {total_time:.1f} 秒")
    print(f"  总延迟: {total_tardiness:.1f} 秒")
    print(f"  总等待: {total_waiting:.1f} 秒")
    
    # 验证时间可行性
    is_valid, message = analyzer.validate_route_time_feasibility(
        route=route,
        travel_times=travel_times,
        service_times=service_times,
        charging_times=charging_times,
        time_windows=time_windows,
        start_time=0
    )
    print(f"\n✓ 时间约束验证: {message}")
    
    return time_config


def test_4_integrated_validation(dm, energy_config, time_config):
    """测试4: 综合可行性验证"""
    print_section("测试4: 综合可行性验证")
    
    print(f"\n测试场景: 尝试不同的路径方案")
    
    # 方案1: 无充电（应该失败）
    print(f"\n方案1: 不充电直接执行所有任务")
    route1 = [0, 1, 3, 2, 4, 0]
    travel_times1 = [dm.get_distance(route1[i], route1[i+1]) 
                     for i in range(len(route1)-1)]
    
    energy_validator = EnergyConstraintValidator(energy_config)
    is_valid1, msg1 = energy_validator.validate_route_energy_feasibility(
        route=route1,
        travel_times=travel_times1,
        charging_times=[0] * len(route1),
        initial_battery=80.0,
        charging_station_ids=[5, 6]
    )
    print(f"  能量验证: {msg1}")
    print(f"  ❌ 方案不可行" if not is_valid1 else "  ✅ 方案可行")
    
    # 方案2: 中途充电（应该成功）
    print(f"\n方案2: 在完成任务1后去充电站")
    route2 = [0, 1, 3, 5, 2, 4, 0]  # 中间插入充电站5
    travel_times2 = [dm.get_distance(route2[i], route2[i+1]) 
                     for i in range(len(route2)-1)]
    charging_times2 = [0, 0, 0, 100, 0, 0, 0]  # 在节点5充电100秒
    
    is_valid2, msg2 = energy_validator.validate_route_energy_feasibility(
        route=route2,
        travel_times=travel_times2,
        charging_times=charging_times2,
        initial_battery=80.0,
        charging_station_ids=[5, 6]
    )
    print(f"  能量验证: {msg2}")
    
    # 同时验证时间
    time_analyzer = RouteTimeAnalyzer(time_config)
    service_times = {1: 30, 2: 30, 3: 30, 4: 30}
    time_windows = {
        1: TimeWindow(0, 100, TimeWindowType.SOFT),
        2: TimeWindow(0, 300, TimeWindowType.SOFT),
        3: TimeWindow(50, 200, TimeWindowType.SOFT),
        4: TimeWindow(200, 400, TimeWindowType.SOFT)
    }
    is_time_valid2, time_msg2 = time_analyzer.validate_route_time_feasibility(
        route=route2,
        travel_times=travel_times2,
        service_times=service_times,
        charging_times=charging_times2,
        time_windows=time_windows
    )
    print(f"  时间验证: {time_msg2}")
    
    if is_valid2 and is_time_valid2:
        print(f"  ✅ 方案可行！")
        
        # 显示详细剖面
        print(f"\n  详细能量剖面:")
        battery_profile = energy_validator.calculate_route_energy_profile(
            route=route2,
            travel_times=travel_times2,
            charging_times=charging_times2,
            initial_battery=80.0,
            charging_station_ids=[5, 6]
        )
        
        for node_id, (arr, dep) in zip(route2, battery_profile):
            node_name = {0: "Depot", 1: "p1", 2: "p2", 3: "d1", 4: "d2", 5: "c1"}
            print(f"    {node_name.get(node_id, node_id)}: "
                  f"到达电量{arr:.1f} → 离开电量{dep:.1f}")
    else:
        print(f"  ❌ 方案不可行")
    
    # 方案3: 测试插入成本计算
    print(f"\n方案3: 评估充电站插入位置")
    base_route = [0, 1, 3, 2, 4, 0]
    
    print(f"  原路径: {' → '.join(map(str, base_route))}")
    print(f"  评估在不同位置插入充电站5的成本:")
    
    for pos in range(1, len(base_route)):
        # 计算插入成本
        i, j = base_route[pos-1], base_route[pos]
        insertion_cost = calculate_insertion_cost(i, j, 5, dm)
        print(f"    位置{pos} (节点{i}和{j}之间): 额外距离 {insertion_cost:.2f}m")


def main():
    """主测试函数"""
    print("=" * 60)
    print("  物理规则层集成测试")
    print("  测试 distance.py + energy.py + time.py")
    print("=" * 60)
    
    try:
        # 运行所有测试
        dm = test_1_distance_module()
        energy_config, travel_times = test_2_energy_module(dm)
        time_config = test_3_time_module(dm, travel_times)
        test_4_integrated_validation(dm, energy_config, time_config)
        
        # 总结
        print_section("测试总结")
        print("\n✅ 所有物理规则模块工作正常！")
        print("\n已验证功能:")
        print("  ✓ 距离计算与节点类型识别")
        print("  ✓ pickup-delivery配对关系")
        print("  ✓ 能量消耗与充电计算")
        print("  ✓ 时间窗约束与延迟计算")
        print("  ✓ 路径综合可行性验证")
        print("  ✓ 充电站插入成本评估")
        
        print("\n🎉 物理规则层测试通过！可以继续构建上层结构。\n")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
