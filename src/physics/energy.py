"""
 energy.py        # 能量管理
     ├─ calculate_energy_consumption()  # 行驶能耗
     ├─ calculate_charging_time()       # 充电时间
     ├─ calculate_charging_amount()     # 充电量
     └─ is_energy_feasible()            # 电量可行性检查

实现AMR电池能量的假设, 充电计算, 对应数学模型中的能量约束

数学模型映射:
    约束(9):  q_i^a = g · t_i^{ch,a}
    约束(10): B_{arr,j}^a = B_{dep,i}^a - κ·t_{move,(i,j)}^a
    约束(11): B_{dep,i}^a = B_{arr,i}^a + η·q_i^a·y_i^a
    约束(13): 0 ≤ B_{arr,i}^a, B_{dep,i}^a ≤ E_a^{max}

采用恒定功率假设：
1. AMR在移动时保持恒定速度(匀速运动)
2. 电机功率P = κ (常数，单位：能量/秒)
3. 总能耗 = 功率 * 时间 = κ * t
4. 充电站支持部分充电(Partial Recharge)
以上内容在约束(10)中有所体现。

需要在energy.py中实现约束10和11: 
1. 计算行驶能耗（基于时间）
2. 计算充电量（基于时间和效率）

参数说明:
    κ (kappa): 移动时能耗率，单位：能量单位/秒
    g: 充电速率，单位：能量单位/秒  
    η (eta): 充电效率，0 < η ≤ 1
    E^{max}: 电池最大容量
    T̄: 单次充电最大允许时间
    Q̄: 单次充电最大允许能量
"""

from typing import Tuple, Optional, List
from dataclasses import dataclass

@dataclass

#自动生成 __init__
#自动生成 __repr__（方便调试）
#类型提示清晰
#适合存储配置参数

class EnergyConfig: 
    """
    能量系统全局配置参数

    对应数学模型中的全局参数
    """

    # 移动能耗参数
    consumption_rate: float = 0.5  # κ: 能耗率 (能量单位/秒)
    
    # 充电参数
    charging_rate: float = 50.0     # g: 充电速率 (能量单位/秒)
    charging_efficiency: float = 0.9 # η: 充电效率 (0, 1]
    
    # 约束参数
    max_charging_time: float = 3600.0 # T̄: 单次充电最大时间 (秒)
    max_charging_amount: float = 100.0  # Q̄: 单次充电最大能量
    
    # 电池参数
    battery_capacity: float = 100.0# E^{max}: 电池最大容量
    
    def __post_init__(self):
        """参数合理性检查"""
        assert 0 < self.charging_efficiency <= 1, \
            f"Charging efficiency must be in the range (0,1] Current value: {self.charging_efficiency}"
        assert self.consumption_rate > 0, "Energy consumption rate must be positive."
        assert self.charging_rate > 0, "Charge rate must be positive."
        assert self.battery_capacity > 0, "Battery capacity must be positive."

def calculate_energy_consumption(distance: float = None,
                                load: float = 0.0,
                                config: EnergyConfig = None,
                                travel_time: float = None,
                                consumption_rate: float = None,
                                vehicle_speed: float = None, 
                                vehicle_capacity: float = None) -> float:
    """
    计算移动能耗（支持两种调用方式）
    
    方式1（route.py使用）: 基于距离和配置
        calculate_energy_consumption(distance=100, load=30, config=energy_config)
    
    方式2（兼容旧版）: 基于时间
        calculate_energy_consumption(travel_time=50, consumption_rate=2.0)
    
    能耗模型: E = κ * t * (1 + load_factor)
    
    参数:
        distance: 行驶距离 (m)
        load: 当前载重 (kg)
        config: 能量配置对象
        travel_time: 移动时间 (秒) - 兼容旧接口
        consumption_rate: 能耗率 - 兼容旧接口
    
    返回:
        float: 消耗的能量 (kWh)
    """
    # 方式1: 使用 config 对象（route.py的调用）
    if config is not None and distance is not None:
        # 计算旅行时间（假设恒定速度2.0 m/s）
        speed = 2.0  # 可以从 config 扩展获取
        travel_time = distance / speed
        consumption_rate = config.consumption_rate

        capacity = vehicle_capacity if vehicle_capacity is not None else 150.0
        
        # 载重影响系数 (载重越大，能耗越高)
        load_factor = 1.0 + (load / 150.0) * 0.2  # 满载增加20%能耗
        
        return consumption_rate * travel_time * load_factor
    
    # 方式2: 兼容旧接口
    elif travel_time is not None and consumption_rate is not None:
        if travel_time < 0:
            raise ValueError(f"Movement time cannot be negative: {travel_time}")
        return consumption_rate * travel_time
    
    else:
        raise ValueError("Must provide either (distance, load, config) or (travel_time, consumption_rate)")


def calculate_charging_amount(charging_time: float,
                              charging_rate: float,
                              efficiency: float) -> float:
    """
    计算充电量（考虑效率损失）
    
    对应约束(9)(11): 
        q_i^a = g · t_i^{ch,a}
        B_{dep,i}^a = B_{arr,i}^a + η·q_i^a·y_i^a
    
    物理含义:
        充电过程有能量损失（热损耗等），实际充入 = 理论值 × 效率
    
    参数:
        charging_time: 充电时间 t^{ch} (秒)
        charging_rate: 充电速率 g (能量单位/秒)
        efficiency: 充电效率 η (0~1)
    
    返回:
        float: 实际充入的能量
    
    示例:
        >>> calculate_charging_amount(100, 5.0, 0.9)
        450.0  # 100秒 × 5单位/秒 × 0.9效率 = 450单位
    """
    if charging_time < 0:
        raise ValueError(f"Charging time cannot be negative: {charging_time}")
    
    return efficiency * charging_rate * charging_time


def calculate_charging_time(desired_amount: float = None,
                            charging_rate: float = None,
                            efficiency: float = None,
                            max_time: Optional[float] = None,
                            config: EnergyConfig = None) -> float:
    """
    计算达到目标充电量所需的时间
    
    支持两种调用方式：
    1. calculate_charging_time(amount, config)  # route.py使用
    2. calculate_charging_time(amount, rate, efficiency, max_time)  # 完整参数
    
    参数:
        desired_amount: 期望充入的能量
        charging_rate: 充电速率 g (可选，从config获取)
        efficiency: 充电效率 η (可选，从config获取)
        max_time: 最大允许充电时间 (可选)
        config: 能量配置对象 (可选)
    
    返回:
        float: 所需充电时间（秒）
    """
    # 从 config 获取参数
    if config is not None:
        charging_rate = config.charging_rate
        efficiency = config.charging_efficiency
        if max_time is None:
            max_time = config.max_charging_time
    
    # 参数检查
    if desired_amount is None or charging_rate is None or efficiency is None:
        raise ValueError("Must provide desired_amount and either config or (charging_rate, efficiency)")
    
    if desired_amount < 0:
        raise ValueError(f"Charge cannot be negative: {desired_amount}")
    
    required_time = desired_amount / (charging_rate * efficiency)
    
    if max_time is not None:
        return min(required_time, max_time)
    
    return required_time

# 部分充电策略,只充"刚好够用"的电量
def calculate_minimum_charging_needed(current_battery: float,
                                     remaining_energy_demand: float,
                                     battery_capacity: float,
                                     safety_margin: float = 0.0) -> float:
    """
    计算完成剩余任务所需的最小充电量（部分充电策略核心）
    
    策略思想（来自Keskin & Çatay, 2016）:
        不充满，只充"刚好够用"的电量 → 节省时间
    
    参数:
        current_battery: 当前电量
        remaining_energy_demand: 剩余路径的能量需求
        battery_capacity: 电池容量
        safety_margin: 安全余量（预留缓冲）
    
    返回:
        float: 需要充电的量
        
    示例:
        >>> calculate_minimum_charging_needed(20, 100, 150, 10)
        90.0  # 需要100，当前20，充90就够（100-20+10安全余量）
    """
    # 计算缺口
    deficit = remaining_energy_demand - current_battery + safety_margin
    
    if deficit <= 0:
        return 0.0  # 当前电量已足够
    
    # 不能超过电池容量
    max_possible = battery_capacity - current_battery
    return min(deficit, max_possible)


def find_charging_opportunities(route: List[int],
                                current_battery: float,
                                energy_demands: List[float],
                                battery_capacity: float,
                                charging_station_ids: List[int]) -> List[Tuple[int, float]]:
    """
    识别路径中需要充电的位置（贪婪策略）
    
    算法思想:
        1. 模拟AMR沿路径移动
        2. 当电量不足时，标记需要插入充电站
        3. 返回充电位置和所需充电量
    
    参数:
        route: 节点ID序列
        current_battery: 初始电量
        energy_demands: 每段路径的能耗列表
        battery_capacity: 电池容量
        charging_station_ids: 充电站节点ID列表
    
    返回:
        List[Tuple[int, float]]: [(插入位置索引, 所需充电量), ...]
    
    示例:
        route = [0, 1, 2, 3, 0]
        返回: [(2, 50.0)]  # 在位置2后插入充电站，充50单位
    """
    charging_needed = []
    battery = current_battery
    
    for i, energy_demand in enumerate(energy_demands):
        # 检查当前电量是否足够
        if battery < energy_demand:
            # 需要充电
            deficit = energy_demand - battery
            charging_amount = min(deficit, battery_capacity - battery)
            charging_needed.append((i, charging_amount))
            battery = battery_capacity  # 假设充满后继续
        
        # 消耗电量
        battery -= energy_demand
    
    return charging_needed

# 约束验证函数
class EnergyConstraintValidator:
    """
    能量约束验证器
    
    验证路径是否满足所有能量相关约束(9)(10)(11)(13)
    """
    
    def __init__(self, config: EnergyConfig):
        self.config = config
    
    # 验证整条路径的能量可行性
    def validate_route_energy_feasibility(self,
                                         route: List[int],
                                         travel_times: List[float],
                                         charging_times: List[float],
                                         initial_battery: float,
                                         charging_station_ids: List[int]) -> Tuple[bool, str]:
        """
        验证整条路径的能量可行性
        
        检查项:
            - 约束(13): 每个节点的电量在 [0, E^{max}] 范围内
            - 不会半路没电
            - 充电站充电量不超过限制
        
        参数:
            route: 节点序列
            travel_times: 每段的移动时间
            charging_times: 每个节点的充电时间
            initial_battery: 初始电量
            charging_station_ids: 充电站ID集合
        
        返回:
            Tuple[bool, str]: (是否可行, 失败原因)
        """
        battery = initial_battery
        
        for i, node_id in enumerate(route):
            # 1. 到达节点（如果不是第一个节点）
            if i > 0:
                travel_time = travel_times[i - 1]
                energy_consumed = calculate_energy_consumption(
                    travel_time=travel_time, consumption_rate=self.config.consumption_rate
                )
                battery -= energy_consumed
                
                # 检查是否没电
                if battery < 0:
                    return False, f"Node{i} battery level is low: {battery:.2f}"
            
            # 2. 在节点充电（如果是充电站）
            if node_id in charging_station_ids:
                charging_time = charging_times[i]
                
                # 检查充电时间约束
                if charging_time > self.config.max_charging_time:
                    return False, f"Node {i} charging time overrun: {charging_time:.2f}s"
                
                # 计算充电量
                charged = calculate_charging_amount(
                    charging_time,
                    self.config.charging_rate,
                    self.config.charging_efficiency
                )
                
                # 检查充电量约束
                if charged > self.config.max_charging_amount:
                    return False, f"Node {i} charging overrun: {charged:.2f}"
                
                battery += charged
            
            # 3. 检查电量上限约束(13)
            if battery > self.config.battery_capacity:
                return False, f"Node {i} power exceeds capacity: {battery:.2f}"
        
        return True, "All energy constraints are satisfied."
    
    def calculate_route_energy_profile(self,
                                      route: List[int],
                                      travel_times: List[float],
                                      charging_times: List[float],
                                      initial_battery: float,
                                      charging_station_ids: List[int]) -> List[Tuple[float, float]]:
        """
        计算路径的完整能量曲线
        
        返回:
            List[Tuple[float, float]]: [(到达电量, 离开电量), ...] 每个节点
        """
        battery_profile = []
        battery = initial_battery
        
        for i, node_id in enumerate(route):
            # 到达电量
            if i > 0:
                travel_time = travel_times[i - 1]
                battery -= calculate_energy_consumption(
                    travel_time=travel_time, consumption_rate=self.config.consumption_rate
                )
            
            arrival_battery = battery
            
            # 充电
            if node_id in charging_station_ids:
                charging_time = charging_times[i]
                battery += calculate_charging_amount(
                    charging_time,
                    self.config.charging_rate,
                    self.config.charging_efficiency
                )
            
            departure_battery = battery
            battery_profile.append((arrival_battery, departure_battery))
        
        return battery_profile
