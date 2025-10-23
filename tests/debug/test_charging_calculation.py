"""
测试充电策略计算逻辑
"""

import sys
sys.path.append('src')

from physics.energy import calculate_minimum_charging_needed

print("="*70)
print("测试充电策略计算（考虑临界值）")
print("="*70)

# 场景：node 15的情况
current_battery = 42.5  # 当前电量
remaining_demand = 42.5  # 到终点需要的能量
battery_capacity = 60.0  # 电池容量
safety_margin = 6.0  # 10% * 60
critical_threshold_ratio = 0.2  # 20%

print(f"\n场景参数:")
print(f"  当前电量: {current_battery} kWh")
print(f"  到终点需要: {remaining_demand} kWh")
print(f"  电池容量: {battery_capacity} kWh")
print(f"  安全余量: {safety_margin} kWh ({safety_margin/battery_capacity*100:.0f}%)")
print(f"  临界值: {critical_threshold_ratio*battery_capacity} kWh ({critical_threshold_ratio*100:.0f}%)")

# 计算充电量
charge_amount = calculate_minimum_charging_needed(
    current_battery=current_battery,
    remaining_energy_demand=remaining_demand,
    battery_capacity=battery_capacity,
    safety_margin=safety_margin,
    critical_threshold_ratio=critical_threshold_ratio
)

print(f"\n充电策略计算:")
print(f"  计算出的充电量: {charge_amount} kWh")
print(f"  充电后电量: {current_battery + charge_amount} kWh")

# 模拟到达终点
final_battery = current_battery + charge_amount - remaining_demand
print(f"\n到达终点后:")
print(f"  最终电量: {final_battery} kWh")
print(f"  是否 >= 临界值: {'✓' if final_battery >= critical_threshold_ratio * battery_capacity else '✗'}")

# 测试不同场景
print(f"\n" + "="*70)
print("测试多个场景")
print(f"="*70)

test_cases = [
    # (current, remaining, desc)
    (42.5, 42.5, "Node 15场景"),
    (42.5, 30.0, "剩余需求较少"),
    (42.5, 50.0, "剩余需求较多"),
    (20.0, 40.0, "低电量"),
]

for curr, remain, desc in test_cases:
    charge = calculate_minimum_charging_needed(
        current_battery=curr,
        remaining_energy_demand=remain,
        battery_capacity=battery_capacity,
        safety_margin=safety_margin,
        critical_threshold_ratio=critical_threshold_ratio
    )
    final = curr + charge - remain
    print(f"\n{desc}:")
    print(f"  当前{curr:.1f}kWh, 需要{remain:.1f}kWh")
    print(f"  → 充电{charge:.1f}kWh → {curr+charge:.1f}kWh")
    print(f"  → 最终{final:.1f}kWh {'✓' if final >= 12 else '✗临界'}")
