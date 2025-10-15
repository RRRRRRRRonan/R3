"""
路径（Route）数据结构模块
==========================
定义AMR的行驶路径及其验证、计算功能

设计要点:
    - Route = 节点序列 + 时间表 + 验证逻辑
    - 支持静态规划（ALNS构建路径）和动态执行（CBS仿真）
    - 提供插入/删除节点操作（ALNS的destroy/repair operators使用）
    - 集成physics层的distance、energy、time计算

功能层次:
    1. 存储层：节点序列、时间表、负载/电量轨迹
    2. 计算层：总距离、总时间、总能耗
    3. 验证层：时间窗、容量、电量、precedence约束
    4. 操作层：插入、删除、交换节点（给ALNS用）

对应数学模型:
    π_k: AMR k的路径（节点序列）
    t_i: 在节点i的到达时间
    q_i: 在节点i的载重
    b_i: 在节点i的电量
"""

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from copy import deepcopy
import sys
sys.path.append('src')

from core.node import Node, DepotNode, TaskNode, ChargingNode, NodeType
from core.task import Task
from physics.distance import DistanceMatrix
from physics.energy import (
    EnergyConfig, 
    calculate_energy_consumption,
    calculate_charging_time,
    calculate_minimum_charging_needed
)
from physics.time import (
    TimeConfig,
    calculate_travel_time
)


# ========== 路径节点访问记录 ==========

@dataclass
class RouteNodeVisit:
    """
    路径中单个节点的访问记录
    
    功能:
        记录在某个节点的详细状态
        用于验证约束、计算成本、CBS冲突检测
    
    属性:
        node: 被访问的节点对象
        arrival_time: 到达时间
        start_service_time: 开始服务时间（可能等待时间窗）
        departure_time: 离开时间
        load_after_service: 服务后的载重
        battery_after_travel: 到达时的电量（旅行后）
        battery_after_service: 服务后的电量（可能充电）
    """
    node: Node
    arrival_time: float = 0.0
    start_service_time: float = 0.0
    departure_time: float = 0.0
    load_after_service: float = 0.0
    battery_after_travel: float = 0.0
    battery_after_service: float = 0.0
    
    def get_waiting_time(self) -> float:
        """获取等待时间（到达到开始服务）"""
        return self.start_service_time - self.arrival_time
    
    def get_service_time(self) -> float:
        """获取服务时间"""
        return self.departure_time - self.start_service_time
    
    def get_delay(self) -> float:
        """获取延迟（如果晚于时间窗）"""
        if hasattr(self.node, 'time_window') and self.node.time_window:
            tw = self.node.time_window
            return max(0.0, self.arrival_time - tw.latest)
        return 0.0


# ========== 路径类 ==========

@dataclass
class Route:
    """
    路径类
    
    功能:
        1. 存储路径信息：节点序列、时间表、负载/电量轨迹
        2. 计算路径指标：总距离、总时间、总能耗
        3. 验证可行性：时间窗、容量、电量、precedence约束
        4. 支持操作：插入、删除节点（给ALNS使用）
    
    用途:
        - 战术层（ALNS）：构建和修改路径，验证可行性
        - 执行层（CBS）：按时间表执行，检测冲突
        - 评估：计算成本、延迟、资源利用率
    
    设计说明:
        - nodes存储节点序列（包括depot起点和终点）
        - visits存储每个节点的详细访问记录（懒计算，调用时生成）
        - 支持两种使用模式：
          a) 仅存储节点序列（轻量）
          b) 完整计算时间表和状态（完整验证）
    
    属性:
        vehicle_id: 所属AMR ID
        nodes: 节点序列 [depot, n1, n2, ..., nk, depot]
        visits: 节点访问记录（可选，按需计算）
        is_feasible: 路径是否可行
        infeasibility_info: 不可行原因（如果不可行）
    """
    
    vehicle_id: int
    nodes: List[Node] = field(default_factory=list)
    visits: Optional[List[RouteNodeVisit]] = None
    is_feasible: bool = True
    infeasibility_info: Optional[str] = None
    
    # ========== 基础操作 ==========
    
    def add_node(self, node: Node):
        """
        添加节点到路径末尾（depot前）
        
        注意: 如果路径以depot结尾，会在depot前插入
        """
        if len(self.nodes) > 0 and self.nodes[-1].is_depot():
            # 在结束depot前插入
            self.nodes.insert(-1, node)
        else:
            self.nodes.append(node)
        
        # 清空cached visits，需要重新计算
        self.visits = None
    
    def insert_node(self, node: Node, position: int):
        """
        在指定位置插入节点
        
        参数:
            node: 要插入的节点
            position: 插入位置索引
        """
        if position < 0 or position > len(self.nodes):
            raise ValueError(f"Invalid position: {position}")
        
        self.nodes.insert(position, node)
        self.visits = None
    
    def remove_node(self, position: int) -> Node:
        """
        删除指定位置的节点
        
        参数:
            position: 节点位置索引
        
        返回:
            被删除的节点
        """
        if position < 0 or position >= len(self.nodes):
            raise ValueError(f"Invalid position: {position}")
        
        node = self.nodes.pop(position)
        self.visits = None
        return node
    
    def get_node_position(self, node_id: int) -> Optional[int]:
        """
        查找节点在路径中的位置
        
        参数:
            node_id: 节点ID
        
        返回:
            位置索引，如果不存在返回None
        """
        for i, node in enumerate(self.nodes):
            if node.node_id == node_id:
                return i
        return None
    
    def clear(self):
        """清空路径"""
        self.nodes.clear()
        self.visits = None
        self.is_feasible = True
        self.infeasibility_info = None
    
    # ========== 查询方法 ==========
    
    def is_empty(self) -> bool:
        """路径是否为空（只有depot或完全没有节点）"""
        return len(self.nodes) <= 2  # 只有起点和终点depot
    
    def get_num_nodes(self) -> int:
        """获取节点数量（不包括depot）"""
        return len([n for n in self.nodes if not n.is_depot()])
    
    def get_task_nodes(self) -> List[TaskNode]:
        """获取所有任务节点（pickup和delivery）"""
        return [n for n in self.nodes if n.node_type in [NodeType.PICKUP, NodeType.DELIVERY]]
    
    def get_pickup_nodes(self) -> List[TaskNode]:
        """获取所有pickup节点"""
        return [n for n in self.nodes if n.is_pickup()]
    
    def get_delivery_nodes(self) -> List[TaskNode]:
        """获取所有delivery节点"""
        return [n for n in self.nodes if n.is_delivery()]
    
    def get_charging_nodes(self) -> List[ChargingNode]:
        """获取所有充电节点"""
        return [n for n in self.nodes if n.is_charging_station()]
    
    def contains_task(self, task_id: int) -> bool:
        """检查是否包含指定任务"""
        task_nodes = self.get_task_nodes()
        return any(n.task_id == task_id for n in task_nodes)
    
    def get_served_tasks(self) -> List[int]:
        """获取路径服务的所有任务ID"""
        task_ids = set()
        for node in self.get_task_nodes():
            if hasattr(node, 'task_id'):
                task_ids.add(node.task_id)
        return sorted(list(task_ids))
    
    # ========== 时间表计算 ==========
    
    def compute_schedule(self,
                        distance_matrix: DistanceMatrix,
                        vehicle_capacity: float,
                        vehicle_battery_capacity: float,
                        initial_battery: float,
                        time_config: TimeConfig = None,
                        energy_config: EnergyConfig = None) -> bool:
        """
        计算路径的完整时间表和状态轨迹
        
        功能:
            1. 计算每个节点的到达/离开时间
            2. 计算载重和电量轨迹
            3. 验证所有约束
        
        参数:
            distance_matrix: 距离矩阵
            vehicle_capacity: AMR容量
            vehicle_battery_capacity: 电池容量
            initial_battery: 初始电量
            time_config: 时间配置
            energy_config: 能量配置
        
        返回:
            bool: 是否可行
        
        副作用:
            - 更新self.visits
            - 更新self.is_feasible和self.infeasibility_info
        """
        if time_config is None:
            time_config = TimeConfig()
        if energy_config is None:
            energy_config = EnergyConfig()
        
        if len(self.nodes) == 0:
            self.is_feasible = True
            self.visits = []
            return True
        
        # 初始化
        self.visits = []
        current_time = 0.0
        current_load = 0.0
        current_battery = initial_battery
        
        for i, node in enumerate(self.nodes):
            visit = RouteNodeVisit(node=node)
            
            # 1. 计算旅行能耗和时间
            if i > 0:
                prev_node = self.nodes[i-1]
                distance = distance_matrix.get_distance(
                    prev_node.node_id,
                    node.node_id
                )
                
                # 旅行时间
                travel_time = calculate_travel_time(
                    distance,
                    time_config.vehicle_speed
                )
                
                # 能量消耗
                energy_consumed = calculate_energy_consumption(
                    distance=distance,
                    load=current_load,
                    config=energy_config, 
                    vehicle_speed=time_config.vehicle_speed,
                    vehicle_capacity=vehicle_capacity
                )
                
                # 更新电量
                current_battery -= energy_consumed
                
                # 检查电量
                if current_battery < -1e-6:
                    self.is_feasible = False
                    self.infeasibility_info = (
                        f"Insufficient battery at node {node.node_id}: "
                        f"need {energy_consumed:.2f}, have {current_battery + energy_consumed:.2f}"
                    )
                    return False
                
                current_battery = max(0.0, current_battery)
                visit.battery_after_travel = current_battery
                
                # 更新时间
                current_time += travel_time
            else:
                visit.battery_after_travel = current_battery
            
            visit.arrival_time = current_time
            
            # 2. 处理时间窗（等待或延迟）
            if hasattr(node, 'time_window') and node.time_window:
                tw = node.time_window
                
                if current_time < tw.earliest:
                    # 早到，等待
                    visit.start_service_time = tw.earliest
                    current_time = tw.earliest
                elif current_time <= tw.latest:
                    # 准时
                    visit.start_service_time = current_time
                else:
                    # 晚到
                    if tw.is_hard():
                        self.is_feasible = False
                        self.infeasibility_info = (
                            f"Time window violation at node {node.node_id}: "
                            f"arrive {current_time:.1f} > latest {tw.latest}"
                        )
                        return False
                    else:
                        # Soft time window，允许延迟
                        visit.start_service_time = current_time
            else:
                visit.start_service_time = current_time
            
            # 3. 执行服务
            service_time = node.service_time if hasattr(node, 'service_time') else 0.0
            current_time = visit.start_service_time + service_time
            visit.departure_time = current_time
            
            # 4. 更新载重
            if node.is_pickup():
                current_load += node.demand
                if current_load > vehicle_capacity + 1e-6:
                    self.is_feasible = False
                    self.infeasibility_info = (
                        f"Capacity violation at node {node.node_id}: "
                        f"load {current_load:.2f} > capacity {vehicle_capacity}"
                    )
                    return False
            elif node.is_delivery():
                current_load -= node.demand
                current_load = max(0.0, current_load)
            
            visit.load_after_service = current_load
            
            # 5. 充电
            if node.is_charging_station():
                # 充满电
                charge_amount = vehicle_battery_capacity - current_battery
                if charge_amount > 0:
                    charging_time = calculate_charging_time(
                        charge_amount,
                        energy_config
                    )
                    current_time += charging_time
                    current_battery = vehicle_battery_capacity
                    visit.departure_time = current_time
            
            visit.battery_after_service = current_battery
            self.visits.append(visit)
        
        self.is_feasible = True
        self.infeasibility_info = None
        return True
    
    # ========== 约束验证 ==========
    
    def validate_precedence(self) -> Tuple[bool, Optional[str]]:
        """
        验证pickup-delivery precedence约束
        
        返回:
            (is_valid, error_message)
        """
        pickup_positions = {}
        delivery_positions = {}
        
        for i, node in enumerate(self.nodes):
            if node.is_pickup():
                if not hasattr(node, 'task_id'):
                    continue
                pickup_positions[node.task_id] = i
            elif node.is_delivery():
                if not hasattr(node, 'task_id'):
                    continue
                delivery_positions[node.task_id] = i
        
        # 检查每个任务的pickup必须在delivery之前
        for task_id in pickup_positions:
            if task_id not in delivery_positions:
                return False, f"Task {task_id}: pickup exists but delivery missing"
            
            if pickup_positions[task_id] >= delivery_positions[task_id]:
                return False, (
                    f"Task {task_id}: pickup at position {pickup_positions[task_id]} "
                    f">= delivery at position {delivery_positions[task_id]}"
                )
        
        # 检查是否有孤立的delivery
        for task_id in delivery_positions:
            if task_id not in pickup_positions:
                return False, f"Task {task_id}: delivery exists but pickup missing"
        
        return True, None
    
    def validate_structure(self) -> Tuple[bool, Optional[str]]:
        """
        验证路径结构
        
        检查:
            - 必须以depot开始和结束
            - precedence约束
        
        返回:
            (is_valid, error_message)
        """
        if len(self.nodes) == 0:
            return True, None
        
        # 检查起点和终点
        if not self.nodes[0].is_depot():
            return False, "Route must start with depot"
        
        if not self.nodes[-1].is_depot():
            return False, "Route must end with depot"
        
        # 检查precedence
        return self.validate_precedence()
    
    # ========== 成本计算 ==========
    
    def calculate_total_distance(self, distance_matrix: DistanceMatrix) -> float:
        """
        计算总距离
        
        参数:
            distance_matrix: 距离矩阵
        
        返回:
            总距离 (m)
        """
        if len(self.nodes) <= 1:
            return 0.0
        
        total = 0.0
        for i in range(len(self.nodes) - 1):
            total += distance_matrix.get_distance(
                self.nodes[i].node_id,
                self.nodes[i+1].node_id
            )
        return total
    
    def calculate_total_time(self) -> float:
        """
        计算总时间（需要先调用compute_schedule）
        
        返回:
            总时间 (s)
        """
        if not self.visits or len(self.visits) == 0:
            return 0.0
        
        return self.visits[-1].departure_time - self.visits[0].arrival_time
    
    def calculate_total_energy(self, energy_config: EnergyConfig = None) -> float:
        """
        计算总能耗（需要先调用compute_schedule）
        
        返回:
            总能耗 (kWh)
        """
        if not self.visits or len(self.visits) <= 1:
            return 0.0
        
        if energy_config is None:
            energy_config = EnergyConfig()
        
        initial_battery = self.visits[0].battery_after_travel
        
        # 找到最后一次充电后的电量
        final_battery = self.visits[-1].battery_after_service
        
        # 总能耗 = 初始电量 - 最终电量 + 所有充电量
        total_charged = 0.0
        for visit in self.visits:
            if visit.node.is_charging_station():
                charged = visit.battery_after_service - visit.battery_after_travel
                total_charged += charged
        
        return initial_battery - final_battery + total_charged
    
    def calculate_total_delay(self) -> float:
        """
        计算总延迟（所有节点的延迟之和）
        
        返回:
            总延迟 (s)
        """
        if not self.visits:
            return 0.0
        
        return sum(visit.get_delay() for visit in self.visits)
    
    def get_metrics(self, distance_matrix: DistanceMatrix) -> Dict:
        """
        获取路径的所有指标
        
        注意: 需要先调用compute_schedule()
        
        返回:
            Dict: 包含各种指标
        """
        return {
            "num_nodes": self.get_num_nodes(),
            "num_tasks": len(self.get_served_tasks()),
            "num_charging_stops": len(self.get_charging_nodes()),
            "total_distance": self.calculate_total_distance(distance_matrix),
            "total_time": self.calculate_total_time(),
            "total_energy": self.calculate_total_energy(),
            "total_delay": self.calculate_total_delay(),
            "is_feasible": self.is_feasible
        }
    
    # ========== 字符串表示 ==========
    
    def __str__(self) -> str:
        """简洁字符串表示"""
        node_ids = [n.node_id for n in self.nodes]
        return f"Route(AMR{self.vehicle_id}): {node_ids}"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (
            f"Route(vehicle_id={self.vehicle_id}, "
            f"num_nodes={len(self.nodes)}, "
            f"feasible={self.is_feasible})"
        )
    
    def get_detailed_string(self) -> str:
        """获取详细的路径描述（包括时间表）"""
        if not self.visits:
            return str(self)
        
        lines = [f"Route for AMR{self.vehicle_id}:"]
        lines.append("=" * 60)
        
        for i, visit in enumerate(self.visits):
            node = visit.node
            lines.append(
                f"  {i}. Node {node.node_id} ({node.node_type.value})"
            )
            lines.append(f"     Arrive: {visit.arrival_time:.1f}s")
            lines.append(f"     Start Service: {visit.start_service_time:.1f}s")
            lines.append(f"     Depart: {visit.departure_time:.1f}s")
            lines.append(f"     Load: {visit.load_after_service:.1f}")
            lines.append(f"     Battery: {visit.battery_after_service:.1f}")
            
            if visit.get_delay() > 0:
                lines.append(f"     ⚠ Delay: {visit.get_delay():.1f}s")
        
        return "\n".join(lines)


# ========== 便捷构造函数 ==========

def create_empty_route(vehicle_id: int, 
                      depot_node: DepotNode) -> Route:
    """
    创建空路径（只有起点和终点depot）
    
    参数:
        vehicle_id: AMR ID
        depot_node: Depot节点
    
    返回:
        Route对象
    """
    route = Route(vehicle_id=vehicle_id)
    route.nodes = [depot_node, deepcopy(depot_node)]  # 起点和终点
    return route


def create_route_from_node_sequence(vehicle_id: int,
                                   nodes: List[Node]) -> Route:
    """
    从节点序列创建路径
    
    参数:
        vehicle_id: AMR ID
        nodes: 节点序列（应该包括起点和终点depot）
    
    返回:
        Route对象
    """
    route = Route(vehicle_id=vehicle_id, nodes=nodes.copy())
    
    # 验证结构
    is_valid, error = route.validate_structure()
    if not is_valid:
        route.is_feasible = False
        route.infeasibility_info = error
    
    return route