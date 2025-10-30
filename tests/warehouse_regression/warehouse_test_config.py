"""Shared configuration objects for the warehouse regression scenarios.

These dataclasses encode physical layouts, robot capabilities, and energy
parameters for the small, medium, and large warehouse regression scenarios so
tests can build consistent environments without duplicating constants.
"""

from dataclasses import dataclass
from typing import Tuple
import math


@dataclass
class WarehouseConfig:
    """仓库场景配置"""
    name: str
    warehouse_size: Tuple[float, float]  # (width, height) in meters
    num_tasks: int
    task_demand_range: Tuple[float, float]  # (min, max) in kg

    # 机器人参数
    robot_capacity: float  # kg
    robot_battery: float   # kWh
    robot_speed: float     # m/s

    # 能耗参数（基于实际机器人数据估算）
    # 能耗率：kWh/秒（移动 + 基础功耗）
    # 计算方式：(电池容量 / 运行时间) / 60
    # 例如：15kWh电池，满载运行4小时 = 15 / (4*3600) ≈ 0.001 kWh/s
    consumption_rate: float  # kWh/s

    # 充电参数
    num_charging_stations: int
    charging_rate: float  # kWh/s (快充功率/3600)

    def get_max_travel_distance(self) -> float:
        """计算仓库内最大可能距离（对角线）"""
        width, height = self.warehouse_size
        return math.sqrt(width**2 + height**2)

    def get_battery_range(self) -> float:
        """估算单次充电的理论行驶距离（米）"""
        # 假设恒速移动，忽略载重影响
        # 能量 / 能耗率 = 时间，时间 * 速度 = 距离
        travel_time = self.robot_battery / self.consumption_rate
        return travel_time * self.robot_speed

    def get_task_density(self) -> float:
        """计算任务密度（任务数/平方米）"""
        area = self.warehouse_size[0] * self.warehouse_size[1]
        return self.num_tasks / area


# ============================================================================
# 小规模场景配置（5-10任务）
# ============================================================================

SMALL_WAREHOUSE_5_TASKS = WarehouseConfig(
    name="小仓库_5任务_轻度",
    warehouse_size=(50.0, 50.0),
    num_tasks=5,
    task_demand_range=(10.0, 25.0),

    # MiR100类型机器人
    robot_capacity=100.0,
    robot_battery=8.0,  # 8 kWh
    robot_speed=1.5,    # 1.5 m/s

    # 能耗估算：8kWh运行4小时 = 8/(4*3600) ≈ 0.00055 kWh/s
    consumption_rate=0.0006,  # 保守估计

    num_charging_stations=0,  # 无充电站（电池足够）
    charging_rate=2.0/3600,   # 2kW充电 = 2/3600 kWh/s
)

SMALL_WAREHOUSE_10_TASKS = WarehouseConfig(
    name="小仓库_10任务_中度",
    warehouse_size=(60.0, 60.0),
    num_tasks=10,
    task_demand_range=(15.0, 30.0),

    # MiR100增强版
    robot_capacity=150.0,
    robot_battery=12.0,
    robot_speed=1.5,

    consumption_rate=0.0008,

    num_charging_stations=1,
    charging_rate=3.0/3600,  # 3kW快充
)


# ============================================================================
# 中规模场景配置（20-30任务）
# ============================================================================

MEDIUM_WAREHOUSE_20_TASKS = WarehouseConfig(
    name="中仓库_20任务_标准",
    warehouse_size=(100.0, 100.0),
    num_tasks=20,
    task_demand_range=(20.0, 40.0),

    # MiR250类型机器人
    robot_capacity=200.0,
    robot_battery=18.0,
    robot_speed=1.8,

    # 18kWh运行5小时 = 18/(5*3600) ≈ 0.001 kWh/s
    consumption_rate=0.0012,

    num_charging_stations=2,
    charging_rate=5.0/3600,  # 5kW快充
)

MEDIUM_WAREHOUSE_30_TASKS = WarehouseConfig(
    name="中仓库_30任务_密集",
    warehouse_size=(120.0, 120.0),
    num_tasks=30,
    task_demand_range=(15.0, 35.0),

    # MiR250增强版
    robot_capacity=250.0,
    robot_battery=22.0,
    robot_speed=2.0,

    consumption_rate=0.0015,

    num_charging_stations=2,
    charging_rate=6.0/3600,
)


# ============================================================================
# 大规模场景配置（50-100任务）
# ============================================================================

LARGE_WAREHOUSE_50_TASKS = WarehouseConfig(
    name="大仓库_50任务_标准",
    warehouse_size=(150.0, 150.0),
    num_tasks=50,
    task_demand_range=(20.0, 50.0),

    # Kiva类型机器人
    robot_capacity=300.0,
    robot_battery=25.0,
    robot_speed=2.0,

    # 25kWh运行6小时 = 25/(6*3600) ≈ 0.0012 kWh/s
    consumption_rate=0.0015,

    num_charging_stations=3,
    charging_rate=8.0/3600,  # 8kW快充
)

LARGE_WAREHOUSE_100_TASKS = WarehouseConfig(
    name="大仓库_100任务_高密度",
    warehouse_size=(200.0, 200.0),
    num_tasks=100,
    task_demand_range=(15.0, 45.0),

    # Kiva增强版
    robot_capacity=350.0,
    robot_battery=30.0,
    robot_speed=2.0,

    consumption_rate=0.0018,

    num_charging_stations=4,
    charging_rate=10.0/3600,  # 10kW快充
)


# ============================================================================
# 压力测试场景（极限情况）
# ============================================================================

STRESS_TEST_LOW_BATTERY = WarehouseConfig(
    name="压力测试_低电量挑战",
    warehouse_size=(100.0, 100.0),
    num_tasks=25,
    task_demand_range=(25.0, 45.0),

    # 故意设置小电池，强制充电
    robot_capacity=200.0,
    robot_battery=10.0,  # 小电池
    robot_speed=1.8,

    consumption_rate=0.0015,  # 较高能耗

    num_charging_stations=3,  # 多个充电站
    charging_rate=8.0/3600,
)

STRESS_TEST_HIGH_DENSITY = WarehouseConfig(
    name="压力测试_高密度任务",
    warehouse_size=(80.0, 80.0),  # 小空间
    num_tasks=40,                  # 多任务
    task_demand_range=(30.0, 50.0),  # 重货物

    robot_capacity=250.0,
    robot_battery=20.0,
    robot_speed=2.0,

    consumption_rate=0.0018,

    num_charging_stations=2,
    charging_rate=6.0/3600,
)


# ============================================================================
# 辅助函数
# ============================================================================

def print_config_summary(config: WarehouseConfig):
    """打印配置摘要"""
    print(f"\n{'='*70}")
    print(f"场景配置：{config.name}")
    print(f"{'='*70}")
    print(f"\n仓库尺寸：{config.warehouse_size[0]}m × {config.warehouse_size[1]}m")
    print(f"任务数量：{config.num_tasks}个")
    print(f"任务需求：{config.task_demand_range[0]}-{config.task_demand_range[1]} kg")
    print(f"任务密度：{config.get_task_density():.6f} 任务/m²")

    print(f"\n机器人参数：")
    print(f"  载重：{config.robot_capacity} kg")
    print(f"  电池：{config.robot_battery} kWh")
    print(f"  速度：{config.robot_speed} m/s")
    print(f"  能耗率：{config.consumption_rate:.6f} kWh/s ({config.consumption_rate*3600:.2f} kWh/h)")

    print(f"\n距离信息：")
    print(f"  最大距离（对角线）：{config.get_max_travel_distance():.1f} m")
    print(f"  理论续航：{config.get_battery_range():.1f} m")
    print(f"  续航倍数：{config.get_battery_range() / config.get_max_travel_distance():.1f}x")

    print(f"\n充电信息：")
    print(f"  充电站数量：{config.num_charging_stations}")
    print(f"  充电功率：{config.charging_rate*3600:.1f} kW")
    print(f"  满电时间：{config.robot_battery / config.charging_rate / 60:.1f} 分钟")
    print(f"{'='*70}")


def get_all_configs():
    """获取所有预定义配置"""
    return {
        'small_5': SMALL_WAREHOUSE_5_TASKS,
        'small_10': SMALL_WAREHOUSE_10_TASKS,
        'medium_20': MEDIUM_WAREHOUSE_20_TASKS,
        'medium_30': MEDIUM_WAREHOUSE_30_TASKS,
        'large_50': LARGE_WAREHOUSE_50_TASKS,
        'large_100': LARGE_WAREHOUSE_100_TASKS,
        'stress_battery': STRESS_TEST_LOW_BATTERY,
        'stress_density': STRESS_TEST_HIGH_DENSITY,
    }


# ============================================================================
# 测试配置展示
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("仓库机器人场景配置总览")
    print("="*70)

    configs = get_all_configs()

    print("\n📦 小规模场景（5-10任务）")
    for key in ['small_5', 'small_10']:
        print(f"\n{key}:")
        config = configs[key]
        print(f"  {config.name}")
        print(f"  仓库：{config.warehouse_size}, 任务：{config.num_tasks}, 电池：{config.robot_battery}kWh")
        print(f"  续航：{config.get_battery_range():.0f}m, 充电站：{config.num_charging_stations}")

    print("\n\n📦 中规模场景（20-30任务）")
    for key in ['medium_20', 'medium_30']:
        print(f"\n{key}:")
        config = configs[key]
        print(f"  {config.name}")
        print(f"  仓库：{config.warehouse_size}, 任务：{config.num_tasks}, 电池：{config.robot_battery}kWh")
        print(f"  续航：{config.get_battery_range():.0f}m, 充电站：{config.num_charging_stations}")

    print("\n\n📦 大规模场景（50-100任务）")
    for key in ['large_50', 'large_100']:
        print(f"\n{key}:")
        config = configs[key]
        print(f"  {config.name}")
        print(f"  仓库：{config.warehouse_size}, 任务：{config.num_tasks}, 电池：{config.robot_battery}kWh")
        print(f"  续航：{config.get_battery_range():.0f}m, 充电站：{config.num_charging_stations}")

    print("\n\n📦 压力测试场景")
    for key in ['stress_battery', 'stress_density']:
        print(f"\n{key}:")
        config = configs[key]
        print(f"  {config.name}")
        print(f"  仓库：{config.warehouse_size}, 任务：{config.num_tasks}, 电池：{config.robot_battery}kWh")
        print(f"  续航：{config.get_battery_range():.0f}m, 充电站：{config.num_charging_stations}")

    print("\n\n" + "="*70)
    print("详细配置示例：")
    print("="*70)

    # 展示一个详细配置
    print_config_summary(MEDIUM_WAREHOUSE_20_TASKS)
