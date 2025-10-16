"""
AMR（车辆）数据结构模块
========================
定义AMR的状态、属性和行为

设计要点:
    - Vehicle存储静态属性（初始配置）和动态属性（执行状态）
    - 静态属性用于planning阶段（ALNS、ADP决策）
    - 动态属性用于execution阶段（CBS仿真、实际执行）
    - Vehicle拥有Route（组合关系）

对应数学模型:
    k ∈ K: AMR编号
    Q_k: AMR k的容量
    B_k: AMR k的电池容量
    v_k: AMR k的速度
"""

from typing import Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum

if TYPE_CHECKING:
    from core.route import Route


# ========== AMR状态枚举 ==========

class VehicleStatus(Enum):
    """
    AMR执行状态
    
    状态转换流程:
        IDLE → ASSIGNED → MOVING → SERVING → CHARGING → COMPLETED
               ↓
           IDLE (循环)
    """
    IDLE = "idle"               # 空闲（在depot等待任务）
    ASSIGNED = "assigned"       # 已分配任务（规划完成，未开始执行）
    MOVING = "moving"           # 移动中（前往下一个节点）
    SERVING = "serving"         # 服务中（在pickup/delivery节点执行服务）
    CHARGING = "charging"       # 充电中
    COMPLETED = "completed"     # 完成所有任务（回到depot）


# ========== AMR类 ==========

@dataclass
class Vehicle:
    """
    AMR（自主移动机器人）类
    
    功能:
        1. 存储AMR的物理属性（容量、电池、速度）
        2. 管理AMR的两种状态：
           - 静态状态：规划阶段使用（initial_*）
           - 动态状态：执行阶段使用（current_*）
        3. 关联Route对象（组合关系）
    
    用途:
        - 战略层（ADP）：查询AMR能力，决策是否接受新任务
        - 战术层（ALNS）：检查容量/电量约束，生成可行路径
        - 执行层（CBS）：实时更新位置、电量，检测冲突
    
    属性说明:
        === 身份信息 ===
        vehicle_id: AMR唯一标识
        
        === 物理属性（不可变）===
        capacity: 最大载重 (kg)
        battery_capacity: 电池容量 (kWh)
        speed: 移动速度 (m/s)
        
        === 静态状态（规划阶段使用）===
        initial_location: 初始位置坐标
        initial_battery: 初始电量
        initial_load: 初始载重（通常为0）
        
        === 动态状态（执行阶段使用）===
        current_location: 当前位置
        current_battery: 当前电量
        current_load: 当前载重
        current_time: 当前时间
        status: 当前状态（IDLE/MOVING/etc）
        
        === 关联对象 ===
        route: 分配的路径对象（Route类）
    """
    
    # === 身份信息 ===
    vehicle_id: int
    
    # === 物理属性 ===
    capacity: float = 150.0         # 最大载重 (kg)
    battery_capacity: float = 100.0  # 电池容量 (kWh)
    speed: float = 2.0              # 移动速度 (m/s)
    
    # === 静态状态（用于planning） ===
    initial_location: Tuple[float, float] = (0.0, 0.0)  # 默认在depot
    initial_battery: float = 100.0   # 默认满电
    initial_load: float = 0.0        # 默认空载
    
    # === 动态状态（用于execution） ===
    current_location: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    current_battery: float = 100.0
    current_load: float = 0.0
    current_time: float = 0.0
    status: VehicleStatus = VehicleStatus.IDLE
    
    # === 关联对象 ===
    route: Optional['Route'] = None  # Forward reference，稍后在route.py定义
    
    def __post_init__(self):
        """初始化后处理：设置动态状态等于静态状态"""
        # 将动态状态初始化为静态状态
        if self.current_location == (0.0, 0.0) and self.initial_location != (0.0, 0.0):
            self.current_location = self.initial_location
        
        if self.current_battery == 100.0 and self.initial_battery != 100.0:
            self.current_battery = self.initial_battery
        
        if self.current_load == 0.0 and self.initial_load != 0.0:
            self.current_load = self.initial_load
    
    # ========== 状态查询方法 ==========
    
    def is_idle(self) -> bool:
        """是否空闲"""
        return self.status == VehicleStatus.IDLE
    
    def is_available(self) -> bool:
        """是否可分配新任务（空闲或已完成）"""
        return self.status in [VehicleStatus.IDLE, VehicleStatus.COMPLETED]
    
    def has_route(self) -> bool:
        """是否已分配路径"""
        return self.route is not None
    
    def get_battery_ratio(self) -> float:
        """获取当前电量比例 [0, 1]"""
        return self.current_battery / self.battery_capacity
    
    def get_load_ratio(self) -> float:
        """获取当前载重比例 [0, 1]"""
        return self.current_load / self.capacity
    
    def get_remaining_capacity(self) -> float:
        """获取剩余容量"""
        return self.capacity - self.current_load
    
    def get_remaining_battery(self) -> float:
        """获取剩余电量"""
        return self.current_battery
    
    # ========== 状态更新方法（execution阶段使用）==========
    
    def reset_to_initial_state(self):
        """
        重置为初始状态
        
        用途: 
            - 新一轮规划开始前
            - 仿真实验重新开始
        """
        self.current_location = self.initial_location
        self.current_battery = self.initial_battery
        self.current_load = self.initial_load
        self.current_time = 0.0
        self.status = VehicleStatus.IDLE
        self.route = None
    
    def move_to(self, location: Tuple[float, float], time: float):
        """
        移动到新位置
        
        参数:
            location: 目标位置
            time: 到达时间
        
        注意: 不自动扣除电量，需要外部调用consume_battery()
        """
        self.current_location = location
        self.current_time = time
        self.status = VehicleStatus.MOVING
    
    def arrive_at(self, location: Tuple[float, float], time: float):
        """
        到达节点（停止移动）
        
        参数:
            location: 到达位置
            time: 到达时间
        """
        self.current_location = location
        self.current_time = time
        self.status = VehicleStatus.IDLE  # 到达后变为空闲，等待下一步操作
    
    def start_service(self, time: float):
        """
        开始服务（pickup或delivery）
        
        参数:
            time: 服务开始时间
        """
        self.current_time = time
        self.status = VehicleStatus.SERVING
    
    def finish_service(self, time: float):
        """
        完成服务
        
        参数:
            time: 服务完成时间
        """
        self.current_time = time
        self.status = VehicleStatus.IDLE
    
    def pickup_load(self, demand: float):
        """
        装载货物（pickup操作）
        
        参数:
            demand: 货物需求量
        
        异常:
            ValueError: 如果超过容量限制
        """
        new_load = self.current_load + demand
        if new_load > self.capacity + 1e-6:  # 容忍浮点误差
            raise ValueError(
                f"Load exceeds capacity: {new_load} > {self.capacity}"
            )
        self.current_load = new_load
    
    def deliver_load(self, demand: float):
        """
        卸载货物（delivery操作）
        
        参数:
            demand: 货物需求量
        
        异常:
            ValueError: 如果卸载量超过当前载重
        """
        new_load = self.current_load - demand
        if new_load < -1e-6:  # 容忍浮点误差
            raise ValueError(
                f"Cannot deliver more than current load: {demand} > {self.current_load}"
            )
        self.current_load = max(0.0, new_load)  # 避免负数
    
    def consume_battery(self, energy: float):
        """
        消耗电量
        
        参数:
            energy: 消耗的能量 (kWh)
        
        异常:
            ValueError: 如果电量不足
        """
        new_battery = self.current_battery - energy
        if new_battery < -1e-6:  # 容忍浮点误差
            raise ValueError(
                f"Insufficient battery: need {energy}, have {self.current_battery}"
            )
        self.current_battery = max(0.0, new_battery)
    
    def charge_battery(self, amount: float, time: float):
        """
        充电
        
        参数:
            amount: 充电量 (kWh)
            time: 充电完成时间
        
        注意: 不会超过电池容量
        """
        self.current_battery = min(
            self.battery_capacity,
            self.current_battery + amount
        )
        self.current_time = time
        self.status = VehicleStatus.CHARGING
    
    def assign_route(self, route: 'Route'):
        """
        分配路径
        
        参数:
            route: Route对象
        """
        self.route = route
        self.status = VehicleStatus.ASSIGNED
    
    def complete_route(self):
        """完成路径执行"""
        self.status = VehicleStatus.COMPLETED
        # 注意: 不清空route，保留用于记录和分析
    
    # ========== 验证方法 ==========
    
    def can_pickup(self, demand: float) -> bool:
        """
        检查是否可以pickup
        
        参数:
            demand: 货物需求量
        
        返回:
            bool: 是否有足够容量
        """
        return (self.current_load + demand) <= (self.capacity + 1e-6)
    
    def has_sufficient_battery(self, required_energy: float) -> bool:
        """
        检查是否有足够电量
        
        参数:
            required_energy: 需要的能量
        
        返回:
            bool: 电量是否足够
        """
        return self.current_battery >= required_energy - 1e-6
    
    # ========== 字符串表示 ==========
    
    def __str__(self) -> str:
        """简洁字符串表示"""
        return f"AMR{self.vehicle_id}"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (
            f"Vehicle(id={self.vehicle_id}, "
            f"location={self.current_location}, "
            f"battery={self.current_battery:.1f}/{self.battery_capacity}, "
            f"load={self.current_load:.1f}/{self.capacity}, "
            f"status={self.status.value})"
        )
    
    def get_state_summary(self) -> str:
        """获取状态摘要（用于调试和日志）"""
        return (
            f"AMR{self.vehicle_id} @ {self.current_location}\n"
            f"  Status: {self.status.value}\n"
            f"  Time: {self.current_time:.1f}s\n"
            f"  Battery: {self.current_battery:.1f}/{self.battery_capacity} "
            f"({self.get_battery_ratio()*100:.1f}%)\n"
            f"  Load: {self.current_load:.1f}/{self.capacity} "
            f"({self.get_load_ratio()*100:.1f}%)\n"
            f"  Route: {self.route is not None}"
        )

    def get_remaining_capacity(self) -> float:
        """获取剩余容量（ALNS计算插入可行性时用）"""
        return self.capacity - self.current_load
    
    def get_remaining_battery(self) -> float:
        """获取剩余电量（ALNS计算充电需求时用）"""
        return self.current_battery


# ========== 车队管理器 ==========

class VehicleFleet:
    """
    车队管理器
    
    功能:
        - 管理多个AMR
        - 提供车队级别的查询和统计
        - 用于战略层决策
    
    用途:
        - 战略层：查询可用AMR数量，决策是否接受新任务
        - 战术层：获取特定AMR进行路径规划
        - 监控：车队利用率、电量分布等统计
    """
    
    def __init__(self):
        self.vehicles: dict[int, Vehicle] = {}
    
    def add_vehicle(self, vehicle: Vehicle):
        """添加AMR到车队"""
        if vehicle.vehicle_id in self.vehicles:
            raise ValueError(f"Vehicle {vehicle.vehicle_id} already exists")
        self.vehicles[vehicle.vehicle_id] = vehicle
    
    def get_vehicle(self, vehicle_id: int) -> Optional[Vehicle]:
        """获取指定AMR"""
        return self.vehicles.get(vehicle_id)
    
    def get_all_vehicles(self) -> list[Vehicle]:
        """获取所有AMR"""
        return list(self.vehicles.values())
    
    def get_available_vehicles(self) -> list[Vehicle]:
        """获取可用的AMR（空闲或已完成）"""
        return [v for v in self.vehicles.values() if v.is_available()]
    
    def get_idle_vehicles(self) -> list[Vehicle]:
        """获取空闲的AMR"""
        return [v for v in self.vehicles.values() if v.is_idle()]
    
    def reset_all(self):
        """重置所有AMR到初始状态"""
        for vehicle in self.vehicles.values():
            vehicle.reset_to_initial_state()
    
    def get_fleet_statistics(self) -> dict:
        """
        获取车队统计信息
        
        返回:
            Dict: 包含各种统计指标
        """
        total = len(self.vehicles)
        if total == 0:
            return {
                "total": 0,
                "available": 0,
                "idle": 0,
                "moving": 0,
                "serving": 0,
                "charging": 0,
                "avg_battery_ratio": 0.0,
                "avg_load_ratio": 0.0
            }
        
        status_count = {status: 0 for status in VehicleStatus}
        for vehicle in self.vehicles.values():
            status_count[vehicle.status] += 1
        
        avg_battery = sum(v.get_battery_ratio() for v in self.vehicles.values()) / total
        avg_load = sum(v.get_load_ratio() for v in self.vehicles.values()) / total
        
        return {
            "total": total,
            "available": len(self.get_available_vehicles()),
            "idle": status_count[VehicleStatus.IDLE],
            "moving": status_count[VehicleStatus.MOVING],
            "serving": status_count[VehicleStatus.SERVING],
            "charging": status_count[VehicleStatus.CHARGING],
            "completed": status_count[VehicleStatus.COMPLETED],
            "avg_battery_ratio": avg_battery,
            "avg_load_ratio": avg_load
        }
    
    def __len__(self) -> int:
        """返回AMR数量"""
        return len(self.vehicles)
    
    def __contains__(self, vehicle_id: int) -> bool:
        """检查AMR是否在车队中"""
        return vehicle_id in self.vehicles
    
    def __str__(self) -> str:
        """字符串表示"""
        stats = self.get_fleet_statistics()
        return (
            f"VehicleFleet(total={stats['total']}, "
            f"available={stats['available']}, "
            f"avg_battery={stats['avg_battery_ratio']*100:.1f}%)"
        )


# ========== 便捷构造函数 ==========

def create_vehicle(vehicle_id: int,
                  capacity: float = 150.0,
                  battery_capacity: float = 100.0,
                  speed: float = 2.0,
                  initial_location: Tuple[float, float] = (0.0, 0.0),
                  initial_battery: Optional[float] = None,
                  initial_load: float = 0.0) -> Vehicle:
    """
    创建AMR的便捷函数
    
    参数:
        vehicle_id: AMR ID
        capacity: 最大载重 (kg)
        battery_capacity: 电池容量 (kWh)
        speed: 速度 (m/s)
        initial_location: 初始位置
        initial_battery: 初始电量（默认满电）
        initial_load: 初始载重 (kg)
    
    返回:
        Vehicle对象
    
    示例:
        # 创建满电AMR
        amr1 = create_vehicle(1, capacity=15.0)
        
        # 创建电量80%的AMR
        amr2 = create_vehicle(2, initial_battery=80.0)
    """
    if initial_battery is None:
        initial_battery = battery_capacity
    
    return Vehicle(
        vehicle_id=vehicle_id,
        capacity=capacity,
        battery_capacity=battery_capacity,
        speed=speed,
        initial_location=initial_location,
        initial_battery=initial_battery,
        initial_load=initial_load
    )


def create_homogeneous_fleet(num_vehicles: int,
                             capacity: float = 150.0,
                             battery_capacity: float = 100.0,
                             speed: float = 2.0,
                             depot_location: Tuple[float, float] = (0.0, 0.0)) -> VehicleFleet:
    """
    创建同质车队（所有AMR属性相同）
    
    参数:
        num_vehicles: AMR数量
        capacity: 最大载重 (kg)
        battery_capacity: 电池容量 (kWh)
        speed: 速度 (m/s)
        depot_location: Depot位置
    
    返回:
        VehicleFleet对象
    
    示例:
        fleet = create_homogeneous_fleet(5, capacity=200.0)
    """
    fleet = VehicleFleet()
    for i in range(1, num_vehicles + 1):
        vehicle = create_vehicle(
            vehicle_id=i,
            capacity=capacity,
            battery_capacity=battery_capacity,
            speed=speed,
            initial_location=depot_location
        )
        fleet.add_vehicle(vehicle)
    return fleet
