"""Time utility helpers used by the routing and charging simulators.

The functions translate physical parameters into scheduling values: they
evaluate travel times, enforce hard or soft time windows, and accumulate full
route time profiles that power the optimisation metrics.  The module is the
single source of truth for the mathematical definitions of waiting, service,
charging, and tardiness durations in seconds.
"""

from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from enum import Enum

from config import DEFAULT_TIME_SYSTEM

class TimeWindowType(Enum):
    """时间窗约束类型"""
    HARD = "hard"  # 硬约束：绝对不能违反
    SOFT = "soft"  # 软约束：可以违反，但有延迟惩罚

@dataclass
class TimeWindow:
    """
    时间窗定义
    
    对应数学模型: [e_i, l_i]
    """
    earliest: float  # e_i: 最早开始服务时间（秒）
    latest: float    # l_i: 最晚开始服务时间（秒）
    window_type: TimeWindowType = TimeWindowType.SOFT
    
    def __post_init__(self):
        """参数合理性检查"""
        if self.earliest > self.latest:
            raise ValueError(
                f"The lower bound of the time window ({self.earliest}) cannot be greater than the upper bound ({self.latest})."
            )
        if self.earliest < 0:
            raise ValueError(f"The lower bound of the time window cannot be negative: {self.earliest}")
    
    def is_hard(self) -> bool:
        """判断是否为硬时间窗"""
        return self.window_type == TimeWindowType.HARD
    
    def is_soft(self) -> bool:
        """判断是否为软时间窗"""
        return self.window_type == TimeWindowType.SOFT

    def is_within_window(self, arrival_time: float) -> bool:
        """检查到达时间是否在时间窗内"""
        return self.earliest <= arrival_time <= self.latest
    
    def calculate_waiting_time(self, arrival_time: float) -> float:
        """
        计算早到需要等待的时间
        
        如果到达时间早于时间窗下界，需要等待
        对应约束(5)中的 ω_i^a
        """
        if arrival_time < self.earliest:
            return self.earliest - arrival_time
        return 0.0
    
    def calculate_tardiness(self, arrival_time: float) -> float:
        """
        计算延迟（晚到的惩罚时间）
        
        对应约束(14): T_{tard,i}^a ≥ S_i^a - l_i
        
        返回:
            float: 延迟时间（非负）
        """
        if arrival_time > self.latest:
            return arrival_time - self.latest
        return 0.0
    
# 时间参数配置
@dataclass
class TimeConfig:
    """
    时间系统全局配置参数
    """
    # 移动速度（用于计算基准移动时间）
    vehicle_speed: float = DEFAULT_TIME_SYSTEM.vehicle_speed_m_s  # 米/秒，默认15m/s（约54km/h，适合电动车辆）

    # 服务时间（默认值，可被节点特定值覆盖）
    default_service_time: float = DEFAULT_TIME_SYSTEM.default_service_time_s  # 秒

    # 延迟惩罚系数（用于目标函数）
    tardiness_penalty: float = DEFAULT_TIME_SYSTEM.tardiness_penalty  # C^{delay}

# 核心时间计算函数
def calculate_travel_time(distance: float, speed: float) -> float:
    """
    计算基准移动时间
    
    对应约束(12): t_{move,(i,j)}^a = t̄_{(i,j)} · x_{(i,j)}^a
    其中 t̄_{(i,j)} = distance / speed
    
    参数:
        distance: 移动距离 d_{(i,j)} (米)
        speed: AMR移动速度 (米/秒)
    
    返回:
        float: 基准移动时间 t̄_{(i,j)} (秒)
    
    示例:
        >>> calculate_travel_time(100, 2.0)
        50.0  # 100米 ÷ 2m/s = 50秒
    """
    if speed <= 0:
        raise ValueError(f"Velocity must be positive: {speed}")
    if distance < 0:
        raise ValueError(f"The distance cannot be negative: {distance}")
    
    return distance / speed

def check_time_window(arrival_time: float,
                     time_window: TimeWindow) -> Tuple[bool, float, float]:
    """
    检查时间窗约束并计算等待/延迟
    
    对应约束(7): e_i ≤ S_i^a ≤ l_i + T_{tard,i}^a
    
    参数:
        arrival_time: 到达时间 S_i^a
        time_window: 时间窗 [e_i, l_i]
    
    返回:
        Tuple[bool, float, float]: (是否满足约束, 等待时间, 延迟时间)
        
    说明:
        - 早到: 需要等待到 e_i
        - 准时: 等待=0, 延迟=0
        - 晚到: 延迟 = arrival_time - l_i
    """
    waiting = time_window.calculate_waiting_time(arrival_time)
    tardiness = time_window.calculate_tardiness(arrival_time)
    
    # 硬约束：不允许延迟
    if time_window.window_type == TimeWindowType.HARD:
        is_feasible = (tardiness == 0)
    else:
        # 软约束：允许延迟
        is_feasible = True
    
    return is_feasible, waiting, tardiness

# 路径时间分析
@dataclass
class RouteTimeProfile:
    """
    单个节点的时间剖面
    
    记录AMR在某个节点的完整时间信息
    """
    node_id: int
    arrival_time: float      # S_i^a: 到达时间
    waiting_time: float      # ω_i^a: 等待时间
    service_time: float      # t_i^{serv}: 服务时间
    charging_time: float     # t_i^{ch,a}: 充电时间
    standby_time: float      # ρ_i^a: 预留时间
    departure_time: float    # T_{dep,i}^a: 离开时间
    tardiness: float         # T_{tard,i}^a: 延迟时间
    
    def total_time_at_node(self) -> float:
        """在节点停留的总时间"""
        return self.departure_time - self.arrival_time

class RouteTimeAnalyzer:
    """
    路径时间分析器
    
    计算整条路径的时间剖面，验证时间约束
    """
    
    def __init__(self, config: TimeConfig):
        self.config = config
    
    def calculate_route_timeline(self,
                                 route: List[int],
                                 travel_times: List[float],
                                 service_times: Dict[int, float],
                                 charging_times: List[float],
                                 time_windows: Dict[int, TimeWindow],
                                 start_time: float = 0.0,
                                 standby_times: Optional[List[float]] = None) -> List[RouteTimeProfile]:
        """
        计算路径的完整时间线
        
        模拟AMR沿路径执行任务，计算每个节点的时间信息
        
        参数:
            route: 节点序列 [node_0, node_1, ..., node_n]
            travel_times: 每段移动时间 [t_{0→1}, t_{1→2}, ...]
            service_times: 节点ID → 服务时间
            charging_times: 每个节点的充电时间 [t_0^ch, t_1^ch, ...]
            time_windows: 节点ID → 时间窗
            start_time: 起始时间（默认0）
            standby_times: 每个节点的预留时间（可选）
        
        返回:
            List[RouteTimeProfile]: 每个节点的时间剖面
        """
        if standby_times is None:
            standby_times = [0.0] * len(route)
        
        timeline = []
        current_time = start_time
        
        for i, node_id in enumerate(route):
            # 1. 计算到达时间
            if i == 0:
                arrival_time = start_time
            else:
                # 对应约束(15): S_j^a ≥ T_{dep,i}^a + t_{move,(i,j)}^a
                arrival_time = current_time + travel_times[i - 1]
            
            # 2. 检查时间窗，计算等待和延迟
            if node_id in time_windows:
                tw = time_windows[node_id]
                _, waiting, tardiness = check_time_window(arrival_time, tw)
            else:
                waiting, tardiness = 0.0, 0.0
            
            # 3. 获取服务时间
            service_time = service_times.get(node_id, 0.0)
            
            # 4. 获取充电时间
            charging_time = charging_times[i] if i < len(charging_times) else 0.0
            
            # 5. 获取预留时间
            standby_time = standby_times[i]

            # 6. 计算离开时间 (对应约束5)
            departure_time = (
            arrival_time + 
            waiting + 
            service_time + 
            charging_time + 
            standby_time
            )
            
            # 6. 记录节点时间剖面
            profile = RouteTimeProfile(
                node_id=node_id,
                arrival_time=arrival_time,
                waiting_time=waiting,
                service_time=service_time,
                charging_time=charging_time,
                standby_time=standby_time,
                departure_time=departure_time,
                tardiness=tardiness
            )
            timeline.append(profile)

            # 8. 更新当前时间为离开时间（用于下一个节点）
            current_time = departure_time
        
        return timeline
    
    def validate_route_time_feasibility(self,
                                       route: List[int],
                                       travel_times: List[float],
                                       service_times: Dict[int, float],
                                       charging_times: List[float],
                                       time_windows: Dict[int, TimeWindow],
                                       start_time: float = 0.0) -> Tuple[bool, str]:
        """
        验证路径是否满足所有时间约束
        
        检查项:
            - 硬时间窗不能违反
            - 时间顺序正确（约束15）
        
        返回:
            Tuple[bool, str]: (是否可行, 失败原因)
        """
        timeline = self.calculate_route_timeline(
            route, travel_times, service_times, charging_times, time_windows, start_time
        )
        
        # 检查硬时间窗违反
        for profile in timeline:
            if profile.node_id in time_windows:
                tw = time_windows[profile.node_id]
                if tw.window_type == TimeWindowType.HARD and profile.tardiness > 0:
                    return False, f"Node {profile.node_id} violates hard time window with delay {profile.tardiness:.2f} seconds."
        
        # 检查时间单调性（约束15）
        for i in range(len(timeline) - 1):
            if timeline[i].departure_time > timeline[i + 1].arrival_time:
                return False, f"Chronological error: node {i} leaves later than node {i+1} arrives."
        
        return True, "All time constraints are satisfied."
    
    def calculate_total_route_time(self, timeline: List[RouteTimeProfile]) -> float:
        """
        计算路径总耗时
        
        返回:
            float: 从起点到终点的总时间
        """
        if not timeline:
            return 0.0
        return timeline[-1].departure_time - timeline[0].arrival_time
    
    def calculate_total_tardiness(self, timeline: List[RouteTimeProfile]) -> float:
        """
        计算路径总延迟（目标函数中的延迟惩罚项）
        
        对应目标函数: C^{delay} Σ T_{tard,i}^a
        
        返回:
            float: 总延迟时间
        """
        return sum(profile.tardiness for profile in timeline)
    
    def calculate_total_waiting(self, timeline: List[RouteTimeProfile]) -> float:
        """
        计算路径总等待时间
        
        返回:
            float: 总等待时间
        """
        return sum(profile.waiting_time for profile in timeline)
    
# 高级功能：时间窗插入评估
def evaluate_node_insertion_time_impact(route: List[int],
                                       insertion_position: int,
                                       new_node_id: int,
                                       travel_times: List[float],
                                       new_travel_time_before: float,
                                       new_travel_time_after: float,
                                       service_times: Dict[int, float],
                                       time_windows: Dict[int, TimeWindow],
                                       charging_times: List[float]) -> Tuple[float, bool]:
    """
    评估插入新节点对时间的影响(用于ALNS算法)
    
    用途:
        在ALNS的"修复"算子中，评估在某位置插入节点是否可行
    
    参数:
        route: 原路径
        insertion_position: 插入位置索引
        new_node_id: 待插入的节点ID
        travel_times: 原路径的移动时间列表
        new_travel_time_before: 到新节点的移动时间
        new_travel_time_after: 从新节点出发的移动时间
        service_times: 服务时间映射
        time_windows: 时间窗映射
        charging_times: 充电时间列表
    
    返回:
        Tuple[float, bool]: (额外时间成本, 是否满足时间窗)
    """
    # 构建新路径
    new_route = route[:insertion_position] + [new_node_id] + route[insertion_position:]
    
    # 构建新的移动时间列表
    new_travel_times = (
        travel_times[:insertion_position - 1] +
        [new_travel_time_before, new_travel_time_after] +
        travel_times[insertion_position:]
    )
    
    # 构建新的充电时间列表（新节点充电时间为0）
    new_charging_times = (
        charging_times[:insertion_position] +
        [0.0] +
        charging_times[insertion_position:]
    )
    
    # 计算新时间线
    config = TimeConfig()
    analyzer = RouteTimeAnalyzer(config)
    
    new_timeline = analyzer.calculate_route_timeline(
        new_route, new_travel_times, service_times, new_charging_times, time_windows
    )
    
    # 计算额外时间成本
    original_total_time = travel_times[insertion_position - 1]
    new_total_time = new_travel_time_before + new_travel_time_after
    new_service_time = service_times.get(new_node_id, 0.0)
    
    extra_time = new_total_time - original_total_time + new_service_time
    
    # 检查时间窗可行性
    is_feasible, _ = analyzer.validate_route_time_feasibility(
        new_route, new_travel_times, service_times, new_charging_times, time_windows
    )
    
    return extra_time, is_feasible

# 便捷函数
def create_time_window(earliest: float, 
                       latest: float, 
                       window_type: TimeWindowType = TimeWindowType.SOFT) -> TimeWindow:
    """创建时间窗的便捷函数"""
    return TimeWindow(earliest, latest, window_type)


def create_flexible_time_window(center_time: float, 
                                flexibility: float,
                                window_type: TimeWindowType = TimeWindowType.SOFT) -> TimeWindow:
    """
    创建以某时间点为中心的灵活时间窗
    
    参数:
        center_time: 理想服务时间
        flexibility: 允许的偏差（±flexibility）
        window_type: 约束类型
    
    返回:
        TimeWindow: [center - flexibility, center + flexibility]
    
    示例:
        >>> create_flexible_time_window(100, 20)
        TimeWindow(earliest=80, latest=120)
    """
    return TimeWindow(
        earliest=max(0, center_time - flexibility),
        latest=center_time + flexibility,
        window_type=window_type
    )


def batch_create_time_windows(node_ids: List[int],
                              earliest_times: List[float],
                              latest_times: List[float],
                              window_type: TimeWindowType = TimeWindowType.SOFT) -> Dict[int, TimeWindow]:
    """
    批量创建时间窗
    
    参数:
        node_ids: 节点ID列表
        earliest_times: 最早时间列表
        latest_times: 最晚时间列表
        window_type: 约束类型
    
    返回:
        Dict[int, TimeWindow]: 节点ID → 时间窗映射
    """
    return {
        node_id: TimeWindow(early, late, window_type)
        for node_id, early, late in zip(node_ids, earliest_times, latest_times)
    }
