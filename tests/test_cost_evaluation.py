"""
Week 1 交付物测试套件
====================
测试多目标成本函数、充电记录系统、基准充电策略

测试内容:
    1. CostParameters 配置和计算
    2. Route充电统计方法
    3. 多目标成本函数
    4. 三种基准充电策略 (FR, PR-Fixed, PR-Minimal)
"""

import sys
sys.path.append('src')

from planner.alns import CostParameters
from strategy.charging_strategies import (
    FullRechargeStrategy,
    PartialRechargeFixedStrategy,
    PartialRechargeMinimalStrategy,
    create_charging_strategy,
    compare_strategies
)
from core.route import Route, RouteNodeVisit
from core.node import create_depot, create_charging_node, NodeType


# ========== 测试1: 成本参数配置 ==========

def test_cost_parameters():
    """测试成本参数配置和计算"""
    print("=" * 60)
    print("测试1: 成本参数配置")
    print("=" * 60)

    # 默认参数
    params = CostParameters()
    print(f"\n默认参数:")
    print(f"  C_tr (距离): {params.C_tr}")
    print(f"  C_ch (充电): {params.C_ch}")
    print(f"  C_time (时间): {params.C_time}")
    print(f"  C_delay (延迟): {params.C_delay}")
    print(f"  C_wait (等待): {params.C_wait}")

    # 测试总成本计算
    total = params.get_total_cost(
        distance=1000,    # 1km
        charging=10,      # 10kWh
        time=1200,        # 20分钟 = 1200秒
        delay=300,        # 5分钟 = 300秒
        waiting=120       # 2分钟 = 120秒
    )

    print(f"\n成本计算测试 (1km距离, 10kWh充电, 20min时间, 5min延迟, 2min等待):")
    print(f"  距离成本: {1000 * params.C_tr:.2f}")
    print(f"  充电成本: {10 * params.C_ch:.2f}")
    print(f"  时间成本: {1200 * params.C_time:.2f}")
    print(f"  延迟成本: {300 * params.C_delay:.2f}")
    print(f"  等待成本: {120 * params.C_wait:.2f}")
    print(f"  总成本: {total:.2f}")

    expected = 1000*1.0 + 10*0.6 + 1200*0.1 + 300*2.0 + 120*0.05
    assert abs(total - expected) < 0.01, f"成本计算错误: {total} != {expected}"

    # 测试自定义参数
    custom_params = CostParameters(
        C_tr=2.0,
        C_ch=1.0,
        C_delay=5.0
    )
    print(f"\n自定义参数测试:")
    print(f"  C_tr: {custom_params.C_tr}")
    print(f"  C_ch: {custom_params.C_ch}")
    print(f"  C_delay: {custom_params.C_delay}")

    print("\n✅ 成本参数测试通过")


# ========== 测试2: 充电策略 ==========

def test_charging_strategies():
    """测试三种充电策略"""
    print("\n" + "=" * 60)
    print("测试2: 充电策略对比")
    print("=" * 60)

    # 场景设置
    current_battery = 20.0
    remaining_demand = 80.0
    battery_capacity = 100.0

    print(f"\n场景:")
    print(f"  当前电量: {current_battery} kWh")
    print(f"  剩余需求: {remaining_demand} kWh")
    print(f"  电池容量: {battery_capacity} kWh")

    # 策略1: FR (Full Recharge)
    fr = FullRechargeStrategy()
    fr_amount = fr.determine_charging_amount(
        current_battery, remaining_demand, battery_capacity
    )
    print(f"\n策略1 - {fr.get_strategy_name()}:")
    print(f"  充电量: {fr_amount:.2f} kWh")
    print(f"  充电后: {current_battery + fr_amount:.2f} kWh")

    assert abs(fr_amount - 80.0) < 0.01, "FR应该充满"

    # 策略2: PR-Fixed (30%)
    pr_fixed_30 = PartialRechargeFixedStrategy(charge_ratio=0.3)
    pr_30_amount = pr_fixed_30.determine_charging_amount(
        current_battery, remaining_demand, battery_capacity
    )
    print(f"\n策略2 - {pr_fixed_30.get_strategy_name()}:")
    print(f"  充电量: {pr_30_amount:.2f} kWh")
    print(f"  充电后: {current_battery + pr_30_amount:.2f} kWh")

    assert abs(pr_30_amount - 10.0) < 0.01, "PR-Fixed(30%)应该充到30kWh"

    # 策略3: PR-Fixed (80%)
    pr_fixed_80 = PartialRechargeFixedStrategy(charge_ratio=0.8)
    pr_80_amount = pr_fixed_80.determine_charging_amount(
        current_battery, remaining_demand, battery_capacity
    )
    print(f"\n策略3 - {pr_fixed_80.get_strategy_name()}:")
    print(f"  充电量: {pr_80_amount:.2f} kWh")
    print(f"  充电后: {current_battery + pr_80_amount:.2f} kWh")

    # 策略4: PR-Minimal (10% safety margin)
    pr_minimal = PartialRechargeMinimalStrategy(safety_margin=0.1)
    pr_minimal_amount = pr_minimal.determine_charging_amount(
        current_battery, remaining_demand, battery_capacity
    )
    print(f"\n策略4 - {pr_minimal.get_strategy_name()}:")
    print(f"  充电量: {pr_minimal_amount:.2f} kWh")
    print(f"  充电后: {current_battery + pr_minimal_amount:.2f} kWh")
    print(f"  说明: 需要{remaining_demand}kWh, 当前{current_battery}kWh, "
          f"缺口{remaining_demand - current_battery}kWh + 10%安全余量")

    # PR-Minimal应该充 remaining_demand - current_battery + 10% safety margin
    # = 80 - 20 + 10 = 70kWh
    expected_minimal = 70.0
    assert abs(pr_minimal_amount - expected_minimal) < 0.01, \
        f"PR-Minimal计算错误: {pr_minimal_amount} != {expected_minimal}"

    print("\n✅ 充电策略测试通过")


# ========== 测试3: 策略对比工具 ==========

def test_strategy_comparison():
    """测试策略对比工具"""
    print("\n" + "=" * 60)
    print("测试3: 策略对比工具")
    print("=" * 60)

    results = compare_strategies(
        current_battery=20.0,
        remaining_demand=80.0,
        battery_capacity=100.0
    )

    print("\n策略对比结果:")
    for strategy_name, data in results.items():
        print(f"\n  {strategy_name}:")
        print(f"    充电量: {data['charging_amount']:.2f} kWh")
        print(f"    最终电量: {data['final_battery']:.2f} kWh")

    assert len(results) == 4, "应该有4个策略"
    print("\n✅ 策略对比测试通过")


# ========== 测试4: 工厂函数 ==========

def test_charging_strategy_factory():
    """测试充电策略工厂函数"""
    print("\n" + "=" * 60)
    print("测试4: 充电策略工厂函数")
    print("=" * 60)

    # 创建FR策略
    fr = create_charging_strategy('FR')
    assert isinstance(fr, FullRechargeStrategy)
    print(f"✓ 创建FR策略: {fr.get_strategy_name()}")

    # 创建PR-Fixed策略
    pr_fixed = create_charging_strategy('PR-Fixed', charge_ratio=0.5)
    assert isinstance(pr_fixed, PartialRechargeFixedStrategy)
    print(f"✓ 创建PR-Fixed策略: {pr_fixed.get_strategy_name()}")

    # 创建PR-Minimal策略
    pr_minimal = create_charging_strategy('PR-Minimal', safety_margin=0.2)
    assert isinstance(pr_minimal, PartialRechargeMinimalStrategy)
    print(f"✓ 创建PR-Minimal策略: {pr_minimal.get_strategy_name()}")

    # 测试错误类型
    try:
        create_charging_strategy('INVALID')
        assert False, "应该抛出异常"
    except ValueError:
        print("✓ 正确处理无效策略类型")

    print("\n✅ 工厂函数测试通过")


# ========== 测试5: Route充电统计方法 ==========

def test_route_charging_statistics():
    """测试Route类的充电统计方法"""
    print("\n" + "=" * 60)
    print("测试5: Route充电统计方法")
    print("=" * 60)

    # 创建测试路径
    depot = create_depot((0, 0))
    charging_station = create_charging_node(1, (100, 100))

    route = Route(vehicle_id=1)
    route.nodes = [depot, charging_station, depot]

    # 模拟visits数据（简化版）
    visit1 = RouteNodeVisit(
        node=depot,
        battery_after_travel=80.0,
        battery_after_service=80.0
    )

    visit2 = RouteNodeVisit(
        node=charging_station,
        arrival_time=100.0,
        start_service_time=100.0,
        departure_time=200.0,  # 充电100秒
        battery_after_travel=30.0,  # 到达时30kWh
        battery_after_service=100.0  # 充满到100kWh
    )

    visit3 = RouteNodeVisit(
        node=depot,
        battery_after_travel=70.0,
        battery_after_service=70.0
    )

    route.visits = [visit1, visit2, visit3]

    # 测试充电量统计
    total_charging = route.get_total_charging_amount()
    print(f"\n总充电量: {total_charging:.2f} kWh")
    assert abs(total_charging - 70.0) < 0.01, "充电量计算错误"

    # 测试充电时间统计
    total_time = route.get_total_charging_time()
    print(f"总充电时间: {total_time:.2f} s")
    assert abs(total_time - 100.0) < 0.01, "充电时间计算错误"

    # 测试充电次数
    num_visits = route.get_num_charging_visits()
    print(f"充电次数: {num_visits}")
    assert num_visits == 1, "充电次数错误"

    # 测试充电统计
    stats = route.get_charging_statistics()
    print(f"\n充电统计:")
    print(f"  总充电量: {stats['total_amount']:.2f} kWh")
    print(f"  总充电时间: {stats['total_time']:.2f} s")
    print(f"  充电次数: {stats['num_visits']}")
    print(f"  平均充电量: {stats['avg_amount']:.2f} kWh")
    print(f"  平均充电时间: {stats['avg_time']:.2f} s")

    assert len(stats['charging_records']) == 1, "应该有1条充电记录"
    record = stats['charging_records'][0]
    print(f"\n充电记录详情:")
    print(f"  充电站ID: {record['station_id']}")
    print(f"  位置: {record['position']}")
    print(f"  充电量: {record['amount']:.2f} kWh")
    print(f"  到达电量: {record['arrival_battery']:.2f} kWh")
    print(f"  离开电量: {record['departure_battery']:.2f} kWh")

    print("\n✅ Route充电统计测试通过")


# ========== 测试6: 边界情况 ==========

def test_edge_cases():
    """测试边界情况"""
    print("\n" + "=" * 60)
    print("测试6: 边界情况")
    print("=" * 60)

    battery_capacity = 100.0

    # 情况1: 当前电量已经足够
    pr_fixed = PartialRechargeFixedStrategy(charge_ratio=0.3)
    amount = pr_fixed.determine_charging_amount(
        current_battery=50.0,  # 已经超过30%
        remaining_demand=20.0,
        battery_capacity=battery_capacity
    )
    print(f"\n情况1: 当前电量50kWh > 目标30kWh")
    print(f"  充电量: {amount:.2f} kWh (应为0)")
    assert amount == 0.0, "当前电量已够，不应充电"

    # 情况2: 剩余需求为0
    pr_minimal = PartialRechargeMinimalStrategy(safety_margin=0.1)
    amount = pr_minimal.determine_charging_amount(
        current_battery=50.0,
        remaining_demand=0.0,
        battery_capacity=battery_capacity
    )
    print(f"\n情况2: 剩余需求为0")
    print(f"  充电量: {amount:.2f} kWh")
    # 只会充安全余量
    assert amount <= 10.0, "剩余需求为0时，最多充安全余量"

    # 情况3: 电池快耗尽
    fr = FullRechargeStrategy()
    amount = fr.determine_charging_amount(
        current_battery=1.0,
        remaining_demand=50.0,
        battery_capacity=battery_capacity
    )
    print(f"\n情况3: 电池快耗尽(1kWh)")
    print(f"  FR充电量: {amount:.2f} kWh")
    assert abs(amount - 99.0) < 0.01, "应该充满"

    print("\n✅ 边界情况测试通过")


# ========== 主测试函数 ==========

def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Week 1 交付物 - 完整测试套件")
    print("=" * 60)

    tests = [
        test_cost_parameters,
        test_charging_strategies,
        test_strategy_comparison,
        test_charging_strategy_factory,
        test_route_charging_statistics,
        test_edge_cases
    ]

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"\n❌ 测试失败: {test.__name__}")
            print(f"   错误: {e}")
            raise

    print("\n" + "=" * 60)
    print("🎉 所有测试通过!")
    print("=" * 60)
    print("\nWeek 1 交付物验证完成:")
    print("  ✅ Day 1-2: 多目标成本函数实现")
    print("  ✅ Day 3-4: 充电记录系统")
    print("  ✅ Day 5-7: 基准算法实现 (FR, PR-Fixed, PR-Minimal)")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
