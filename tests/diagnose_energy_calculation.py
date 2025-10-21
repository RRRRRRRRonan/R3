"""
诊断能量计算问题
帮助定位为什么能量消耗计算如此离谱
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from physics.distance import DistanceMatrix
from physics.energy import EnergyConfig, calculate_energy_consumption
from core.vehicle import create_vehicle


def diagnose_energy_calculation():
    """诊断能量计算函数"""
    print("=" * 70)
    print("能量计算诊断")
    print("=" * 70)
    
    # ========== 步骤1：检查基础配置 ==========
    print("\n📊 步骤1：检查基础配置")
    print("-" * 70)
    
    energy_config = EnergyConfig(consumption_rate=0.5)
    
    print(f"能量配置:")
    print(f"  consumption_rate: {energy_config.consumption_rate}")
    
    # 打印EnergyConfig的所有属性
    print(f"\n所有能量配置属性:")
    for attr in dir(energy_config):
        if not attr.startswith('_'):
            value = getattr(energy_config, attr)
            if not callable(value):
                print(f"  {attr}: {value}")
    
    # ========== 步骤2：检查车辆参数 ==========
    print(f"\n📊 步骤2：检查车辆参数")
    print("-" * 70)
    
    vehicle = create_vehicle(
        vehicle_id=1,
        battery_capacity=0.15,
        initial_battery=0.15
    )
    
    print(f"车辆参数:")
    print(f"  battery_capacity: {vehicle.battery_capacity} kWh")
    print(f"  current_battery: {vehicle.current_battery} kWh")
    print(f"  capacity (载重): {vehicle.capacity} kg")
    print(f"  speed: {vehicle.speed} m/s")
    print(f"  current_load: {vehicle.current_load} kg")
    
    # ========== 步骤3：测试简单的能量计算 ==========
    print(f"\n📊 步骤3：测试简单的能量计算")
    print("-" * 70)
    
    # 测试场景：141.4米，空载
    test_distance = 141.4  # 米
    test_load = 0.0  # 空载
    
    print(f"\n测试场景:")
    print(f"  距离: {test_distance} 米 ({test_distance/1000} 公里)")
    print(f"  载重: {test_load} kg")
    print(f"  车速: {vehicle.speed} m/s")
    print(f"  车辆容量: {vehicle.capacity} kg")
    
    print(f"\n理论计算（手动）:")
    theoretical_energy = (test_distance / 1000.0) * energy_config.consumption_rate
    print(f"  能量 = 距离(km) × 消耗率(kWh/km)")
    print(f"  能量 = {test_distance/1000:.4f} × {energy_config.consumption_rate}")
    print(f"  能量 = {theoretical_energy:.4f} kWh")
    
    print(f"\n实际调用 calculate_energy_consumption:")
    try:
        actual_energy = calculate_energy_consumption(
            distance=test_distance,
            load=test_load,
            config=energy_config,
            vehicle_speed=vehicle.speed,
            vehicle_capacity=vehicle.capacity
        )
        print(f"  返回值: {actual_energy:.4f} kWh")
        
        # 比较
        ratio = actual_energy / theoretical_energy if theoretical_energy > 0 else float('inf')
        print(f"\n📈 比较:")
        print(f"  理论值: {theoretical_energy:.4f} kWh")
        print(f"  实际值: {actual_energy:.4f} kWh")
        print(f"  比例: {ratio:.1f}x")
        
        if abs(ratio - 1.0) < 0.1:
            print(f"  ✅ 计算正常（误差小于10%）")
        elif ratio > 10:
            print(f"  ❌ 实际值远大于理论值！")
            print(f"\n可能原因:")
            print(f"  1. calculate_energy_consumption 期望距离单位是公里，但传入了米")
            print(f"  2. 载重或速度参数影响了计算，但影响过大")
            print(f"  3. 能量配置有额外的系数")
        elif ratio < 0.1:
            print(f"  ❌ 实际值远小于理论值！")
            print(f"  可能原因: 单位转换方向反了")
        else:
            print(f"  ⚠️  有一定偏差，可能是载重等因素的影响")
        
    except Exception as e:
        print(f"  ❌ 调用失败: {type(e).__name__}: {e}")
        print(f"\n错误详情:")
        import traceback
        traceback.print_exc()
        
        print(f"\n需要检查:")
        print(f"  1. calculate_energy_consumption 的函数签名是否正确")
        print(f"  2. 参数类型和单位是否匹配")
        return False
    
    # ========== 步骤4：测试不同参数的影响 ==========
    print(f"\n📊 步骤4：测试不同参数的影响")
    print("-" * 70)
    
    test_cases = [
        {"distance": 141.4, "load": 0.0, "desc": "空载，141.4米"},
        {"distance": 0.1414, "load": 0.0, "desc": "空载，0.1414公里（单位测试）"},
        {"distance": 141.4, "load": 50.0, "desc": "半载（50kg），141.4米"},
        {"distance": 141.4, "load": 100.0, "desc": "满载（100kg），141.4米"},
    ]
    
    print(f"\n测试不同参数组合:")
    for i, case in enumerate(test_cases, 1):
        try:
            energy = calculate_energy_consumption(
                distance=case["distance"],
                load=case["load"],
                config=energy_config,
                vehicle_speed=vehicle.speed,
                vehicle_capacity=vehicle.capacity
            )
            print(f"\n  测试{i}: {case['desc']}")
            print(f"    结果: {energy:.4f} kWh")
            
            # 如果是公里测试，比较结果
            if case["distance"] < 1:  # 可能是公里
                expected = case["distance"] * energy_config.consumption_rate
                if abs(energy - expected) < 0.01:
                    print(f"    💡 这个结果接近理论值！可能函数期望公里作为单位")
        except Exception as e:
            print(f"\n  测试{i}: {case['desc']}")
            print(f"    ❌ 失败: {e}")
    
    return True


if __name__ == "__main__":
    print("\n")
    success = diagnose_energy_calculation()
    
    if success:
        print("\n" + "=" * 70)
        print("诊断完成")
        print("=" * 70)
        print("\n请查看上面的输出，特别关注:")
        print("  1. 理论值 vs 实际值的比例")
        print("  2. 不同单位（米 vs 公里）的测试结果")
        print("  3. 载重对能量消耗的影响")
    else:
        print("\n" + "=" * 70)
        print("诊断过程中出现错误")
        print("=" * 70)