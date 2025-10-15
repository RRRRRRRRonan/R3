#NodeType(DEPOT, PICKUP, DELIVERY, CHARGING), Node（基类，所有结点的基础）, DepotNode(仓库节点), 
#TaskNode(任务节点，pickup/delivery), ChargingNode

"""
节点数据结构模块
================
定义仓库环境中的所有节点类型

节点ID编号约定（与数学模型一致）:
    0: Depot节点
    1 ~ n: Pickup节点
    n+1 ~ 2n: Delivery节点（pickup i 对应 delivery i+n）
    2n+1 ~ 2n+m: Charging站节点

设计模式:
    使用继承体系：基类定义共有属性，子类添加特定属性
"""

from typing import Tuple, Optional
from enum import Enum
from dataclasses import dataclass
import sys
sys.path.append('.')

from physics.time import TimeWindow, TimeWindowType


# ========== 节点类型枚举 ==========

class NodeType(Enum):
    """节点类型枚举"""
    DEPOT = "depot"
    PICKUP = "pickup"
    DELIVERY = "delivery"
    CHARGING = "charging"


# ========== 节点基类 ==========

@dataclass(frozen=True)  # frozen=True 使对象不可变
class Node:
    """
    节点基类
    
    所有节点共有的属性：
        - 唯一标识符
        - 空间位置
        - 节点类型
    
    设计为不可变(immutable)：
        节点的基本属性在创建后不应改变
    """
    node_id: int                      # 节点ID
    coordinates: Tuple[float, float]  # (x, y) 坐标
    node_type: NodeType               # 节点类型
    
    def __post_init__(self):
        """节点合法性检查"""
        if self.node_id < 0:
            raise ValueError(f"节点ID不能为负: {self.node_id}")
    
    def is_depot(self) -> bool:
        """判断是否为depot节点"""
        return self.node_type == NodeType.DEPOT
    
    def is_pickup(self) -> bool:
        """判断是否为pickup节点"""
        return self.node_type == NodeType.PICKUP
    
    def is_delivery(self) -> bool:
        """判断是否为delivery节点"""
        return self.node_type == NodeType.DELIVERY
    
    def is_charging_station(self) -> bool:
        """判断是否为充电站节点"""
        return self.node_type == NodeType.CHARGING
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.node_type.value.capitalize()}{self.node_id}"
    
    def __repr__(self) -> str:
        """调试表示"""
        return f"{self.__class__.__name__}(id={self.node_id}, pos={self.coordinates})"


# ========== 子类：Depot节点 ==========

@dataclass(frozen=True)
class DepotNode(Node):
    """
    仓库depot节点
    
    特点:
        - 固定为节点ID 0
        - AMR的起点和终点
        - 无服务时间、无时间窗
    
    对应数学模型: o_a, d_a
    """
    
    def __init__(self, coordinates: Tuple[float, float]):
        """
        创建Depot节点
        
        参数:
            coordinates: 仓库坐标
        """
        # 调用父类初始化，depot固定为ID 0
        object.__setattr__(self, 'node_id', 0)
        object.__setattr__(self, 'coordinates', coordinates)
        object.__setattr__(self, 'node_type', NodeType.DEPOT)


# ========== 子类：任务节点（Pickup/Delivery）==========

@dataclass(frozen=True)
class TaskNode(Node):
    """
    任务节点（Pickup或Delivery）
    
    额外属性:
        - 所属任务ID（用于关联配对节点）
        - 时间窗约束
        - 服务时间
        - 货物需求量
    
    对应数学模型:
        - P: pickup节点集合
        - D: delivery节点集合
        - e_i, l_i: 时间窗
        - t_i^{serv}: 服务时间
        - D_i^w: 需求量
    """
    task_id: int                           # 所属任务ID
    time_window: TimeWindow                # 时间窗 [e_i, l_i]
    service_time: float                    # 服务时间 t_i^{serv}
    demand: float                          # 需求量（货物重量）
    
    def __post_init__(self):
        """任务节点合法性检查"""
        super().__post_init__()
        
        if self.task_id <= 0:
            raise ValueError(f"任务ID必须为正: {self.task_id}")
        
        if self.service_time < 0:
            raise ValueError(f"服务时间不能为负: {self.service_time}")
        
        if self.demand < 0:
            raise ValueError(f"需求量不能为负: {self.demand}")
        
        # 验证节点类型
        if self.node_type not in [NodeType.PICKUP, NodeType.DELIVERY]:
            raise ValueError(f"TaskNode的类型必须是PICKUP或DELIVERY")
    
    def is_pickup(self) -> bool:
        """判断是否为pickup节点"""
        return self.node_type == NodeType.PICKUP
    
    def is_delivery(self) -> bool:
        """判断是否为delivery节点"""
        return self.node_type == NodeType.DELIVERY
    
    def get_paired_node_id(self, num_tasks: int) -> int:
        """
        获取配对节点ID
        
        数学模型规则:
            - pickup i → delivery i+n
            - delivery i+n → pickup i
        
        参数:
            num_tasks: 总任务数 n
        
        返回:
            int: 配对节点ID
        """
        if self.is_pickup():
            return self.node_id + num_tasks
        else:
            return self.node_id - num_tasks


# ========== 子类：充电站节点 ==========

@dataclass(frozen=True)
class ChargingNode(Node):
    """
    充电站节点
    
    特点:
        - 可以为AMR充电
        - 无服务时间（充电时间单独计算）
        - 通常无时间窗约束
    
    对应数学模型: C (充电站集合)
    """
    
    # 充电站可以有名称（可选）
    station_name: Optional[str] = None
    
    def __post_init__(self):
        """充电站节点合法性检查"""
        super().__post_init__()
        
        if self.node_type != NodeType.CHARGING:
            raise ValueError("ChargingNode的类型必须是CHARGING")
    
    def __str__(self) -> str:
        if self.station_name:
            return f"Charging({self.station_name})"
        return f"Charging{self.node_id}"


# ========== 便捷构造函数 ==========

def create_depot(coordinates: Tuple[float, float] = (0, 0)) -> DepotNode:
    """
    创建Depot节点的便捷函数
    
    示例:
        depot = create_depot((0, 0))
    """
    return DepotNode(coordinates)


def create_pickup_node(node_id: int,
                       coordinates: Tuple[float, float],
                       task_id: int,
                       time_window: TimeWindow,
                       service_time: float = 30.0,
                       demand: float = 1.0) -> TaskNode:
    """
    创建Pickup节点的便捷函数
    
    参数:
        node_id: 节点ID（1 ~ n）
        coordinates: 坐标
        task_id: 所属任务ID
        time_window: 时间窗
        service_time: 服务时间（默认30秒）
        demand: 需求量（默认1单位）
    
    示例:
        pickup = create_pickup_node(
            node_id=1,
            coordinates=(10, 20),
            task_id=1,
            time_window=TimeWindow(0, 100)
        )
    """
    return TaskNode(
        node_id=node_id,
        coordinates=coordinates,
        node_type=NodeType.PICKUP,
        task_id=task_id,
        time_window=time_window,
        service_time=service_time,
        demand=demand
    )


def create_delivery_node(node_id: int,
                        coordinates: Tuple[float, float],
                        task_id: int,
                        time_window: TimeWindow,
                        service_time: float = 30.0,
                        demand: float = 1.0) -> TaskNode:
    """
    创建Delivery节点的便捷函数
    
    参数:
        node_id: 节点ID（n+1 ~ 2n）
        coordinates: 坐标
        task_id: 所属任务ID
        time_window: 时间窗
        service_time: 服务时间（默认30秒）
        demand: 需求量（默认1单位，应与对应pickup相同）
    
    示例:
        delivery = create_delivery_node(
            node_id=4,  # 如果pickup是1，且n=3，则delivery是4
            coordinates=(15, 25),
            task_id=1,
            time_window=TimeWindow(50, 150)
        )
    """
    return TaskNode(
        node_id=node_id,
        coordinates=coordinates,
        node_type=NodeType.DELIVERY,
        task_id=task_id,
        time_window=time_window,
        service_time=service_time,
        demand=demand
    )


def create_charging_node(node_id: int,
                        coordinates: Tuple[float, float],
                        station_name: Optional[str] = None) -> ChargingNode:
    """
    创建Charging站节点的便捷函数
    
    参数:
        node_id: 节点ID（2n+1 ~ 2n+m）
        coordinates: 坐标
        station_name: 充电站名称（可选）
    
    示例:
        charging = create_charging_node(
            node_id=7,
            coordinates=(50, 0),
            station_name="Station-A"
        )
    """
    return ChargingNode(
        node_id=node_id,
        coordinates=coordinates,
        node_type=NodeType.CHARGING,
        station_name=station_name
    )


def create_task_node_pair(task_id: int,
                          pickup_id: int,
                          delivery_id: int,
                          pickup_coords: Tuple[float, float],
                          delivery_coords: Tuple[float, float],
                          pickup_time_window: TimeWindow,
                          delivery_time_window: TimeWindow,
                          service_time: float = 30.0,
                          demand: float = 1.0) -> Tuple[TaskNode, TaskNode]:
    """
    同时创建配对的pickup和delivery节点
    
    返回:
        Tuple[TaskNode, TaskNode]: (pickup节点, delivery节点)
    
    示例:
        pickup, delivery = create_task_node_pair(
            task_id=1,
            pickup_id=1,
            delivery_id=4,
            pickup_coords=(10, 20),
            delivery_coords=(15, 25),
            pickup_time_window=TimeWindow(0, 100),
            delivery_time_window=TimeWindow(50, 150)
        )
    """
    pickup = create_pickup_node(
        pickup_id, pickup_coords, task_id,
        pickup_time_window, service_time, demand
    )
    
    delivery = create_delivery_node(
        delivery_id, delivery_coords, task_id,
        delivery_time_window, service_time, demand
    )
    
    return pickup, delivery