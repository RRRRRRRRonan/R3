#TaskStatus (PENDING, ASSIGNED, IN_PROGRESS, COMPLETED), Task (paired with pickup and delivery),
#TaskPool (Manages all tasks)
"""
任务数据结构模块
================
定义任务（Task）及任务池（TaskPool）管理

任务定义:
    每个任务包含一对配对的pickup和delivery节点
    对应数学模型中的 r ∈ R (request/task)

设计要点:
    - Task本身是不可变的（任务属性创建后不变）
    - TaskStatus跟踪任务的执行状态
    - TaskPool管理所有任务的集合
"""

from typing import List, Dict, Optional, Set
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

from core.node import TaskNode, NodeType


# ========== 任务状态枚举 ==========

class TaskStatus(Enum):
    """
    任务执行状态
    
    状态转换流程:
        PENDING → ASSIGNED → IN_PROGRESS → COMPLETED
                     ↓
                  REJECTED
    """
    PENDING = "pending"           # 待分配（新到达的任务）
    ASSIGNED = "assigned"         # 已分配给某个AMR
    IN_PROGRESS = "in_progress"   # 执行中（已pickup，未delivery）
    COMPLETED = "completed"       # 已完成
    REJECTED = "rejected"         # 已拒绝（战略层决定不接受）
    CANCELLED = "cancelled"       # 已取消（客户取消）


# ========== 任务类 ==========

@dataclass(frozen=True)
class Task:
    """
    任务类
    
    对应数学模型:
        r ∈ R: 一个任务请求
        p_r: 任务r的pickup节点
        d_r: 任务r的delivery节点
    
    属性说明:
        task_id: 任务唯一标识
        pickup_node: pickup节点对象（存储对象，方便访问属性）
        delivery_node: delivery节点对象
        demand: 货物重量/体积（与TaskNode中的demand一致）
        priority: 任务优先级（高优先级任务可优先处理）
        arrival_time: 任务到达系统的时间（动态场景）
    """
    task_id: int
    pickup_node: TaskNode
    delivery_node: TaskNode
    demand: float = 1.0
    priority: int = 0              # 默认优先级为0
    arrival_time: float = 0.0      # 默认时间为0（静态场景）
    
    def __post_init__(self):
        """任务合法性检查"""
        # 检查task_id一致性
        if self.pickup_node.task_id != self.delivery_node.task_id:
            raise ValueError(
                f"Inconsistency between Pickup and Delivery task_id:"
                f"{self.pickup_node.task_id} vs {self.delivery_node.task_id}"
            )
        
        if self.task_id != self.pickup_node.task_id:
            raise ValueError(
                f"Task's ID does not match the node's task_id:"
                f"{self.task_id} vs {self.pickup_node.task_id}"
            )
        
        # 检查节点类型
        if not self.pickup_node.is_pickup():
            raise ValueError(f"pickup_node must be of type PICKUP")
        
        if not self.delivery_node.is_delivery():
            raise ValueError(f"delivery_node must be of type DELIVERY")
        
        # 检查需求量一致性
        if abs(self.pickup_node.demand - self.delivery_node.demand) > 1e-6:
            raise ValueError(
                f"Pickup and Delivery requirements must be the same:"
                f"{self.pickup_node.demand} vs {self.delivery_node.demand}"
            )
        
        if abs(self.demand - self.pickup_node.demand) > 1e-6:
            raise ValueError(
                f"Task's demand does not match node's demand:"
                f"{self.demand} vs {self.pickup_node.demand}"
            )
    
    @property
    def pickup_id(self) -> int:
        """快捷访问pickup节点ID"""
        return self.pickup_node.node_id
    
    @property
    def delivery_id(self) -> int:
        """快捷访问delivery节点ID"""
        return self.delivery_node.node_id
    
    @property
    def pickup_coordinates(self) -> tuple:
        """快捷访问pickup坐标"""
        return self.pickup_node.coordinates
    
    @property
    def delivery_coordinates(self) -> tuple:
        """快捷访问delivery坐标"""
        return self.delivery_node.coordinates
    
    @property
    def pickup_time_window(self):
        """快捷访问pickup时间窗"""
        return self.pickup_node.time_window
    
    @property
    def delivery_time_window(self):
        """快捷访问delivery时间窗"""
        return self.delivery_node.time_window
    
    def get_total_service_time(self) -> float:
        """获取任务的总服务时间(pickup + delivery)"""
        return self.pickup_node.service_time + self.delivery_node.service_time
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"Task{self.task_id}(p{self.pickup_id}→d{self.delivery_id})"
    
    def __repr__(self) -> str:
        """调试表示"""
        return (f"Task(id={self.task_id}, "
                f"pickup={self.pickup_id}, "
                f"delivery={self.delivery_id}, "
                f"demand={self.demand})")


# ========== 任务状态跟踪器 ==========

@dataclass
class TaskStateTracker:
    """
    任务状态跟踪器
    
    跟踪单个任务在系统中的状态变化
    与Task分离, 因为状态是可变的, 而Task是不可变的
    """
    task: Task
    status: TaskStatus = TaskStatus.PENDING
    assigned_vehicle_id: Optional[int] = None  # 分配给哪个AMR
    pickup_time: Optional[float] = None        # 实际pickup时间
    delivery_time: Optional[float] = None      # 实际delivery时间
    
    def assign_to_vehicle(self, vehicle_id: int):
        """分配任务给AMR"""
        if self.status != TaskStatus.PENDING:
            raise ValueError(f"Only tasks with PENDING status can be assigned, current status: {self.status}")
        self.status = TaskStatus.ASSIGNED
        self.assigned_vehicle_id = vehicle_id
    
    def start_execution(self, pickup_time: float):
        """开始执行任务（完成pickup）"""
        if self.status != TaskStatus.ASSIGNED:
            raise ValueError(f"Only tasks with ASSIGNED status can be started, current status: {self.status}")
        self.status = TaskStatus.IN_PROGRESS
        self.pickup_time = pickup_time
    
    def complete(self, delivery_time: float):
        """完成任务（完成delivery）"""
        if self.status != TaskStatus.IN_PROGRESS:
            raise ValueError(f"Only tasks with IN_PROGRESS status can be accomplished, current status: {self.status}")
        self.status = TaskStatus.COMPLETED
        self.delivery_time = delivery_time
    
    def reject(self):
        """拒绝任务（战略层决策）"""
        if self.status != TaskStatus.PENDING:
            raise ValueError(f"Only tasks with PENDING status can be rejected, current status: {self.status}")
        self.status = TaskStatus.REJECTED
    
    def is_completed(self) -> bool:
        """判断任务是否完成"""
        return self.status == TaskStatus.COMPLETED
    
    def is_in_progress(self) -> bool:
        """判断任务是否执行中"""
        return self.status == TaskStatus.IN_PROGRESS
    
    def get_execution_time(self) -> Optional[float]:
        """获取任务执行时间（delivery_time - pickup_time）"""
        if self.pickup_time is not None and self.delivery_time is not None:
            return self.delivery_time - self.pickup_time
        return None


# ========== 任务池管理器 ==========

class TaskPool:
    """
    任务池管理器
    
    管理系统中所有任务的集合，提供查询和过滤功能
    
    用途:
        - 战略层：查询待分配的任务
        - 战术层：获取分配给特定AMR的任务
        - 统计：计算完成率、拒绝率等指标
    """
    
    def __init__(self):
        self.tasks: Dict[int, Task] = {}              # task_id → Task
        self.trackers: Dict[int, TaskStateTracker] = {}  # task_id → Tracker
    
    def add_task(self, task: Task):
        """
        添加新任务到池中
        
        参数:
            task: 任务对象
        """
        if task.task_id in self.tasks:
            raise ValueError(f"Task ID {task.task_id} exists in the pool.")
        
        self.tasks[task.task_id] = task
        self.trackers[task.task_id] = TaskStateTracker(task)
    
    def add_tasks(self, tasks: List[Task]):
        """批量添加任务"""
        for task in tasks:
            self.add_task(task)
    
    def get_task(self, task_id: int) -> Optional[Task]:
        """根据ID获取任务"""
        return self.tasks.get(task_id)
    
    def get_tracker(self, task_id: int) -> Optional[TaskStateTracker]:
        """根据ID获取任务状态跟踪器"""
        return self.trackers.get(task_id)
    
    def get_all_tasks(self) -> List[Task]:
        """获取所有任务"""
        return list(self.tasks.values())
    
    def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """
        获取指定状态的所有任务
        
        示例:
            pending_tasks = pool.get_tasks_by_status(TaskStatus.PENDING)
        """
        return [
            task for task_id, task in self.tasks.items()
            if self.trackers[task_id].status == status
        ]
    
    def get_pending_tasks(self) -> List[Task]:
        """获取待分配的任务（战略层常用）"""
        return self.get_tasks_by_status(TaskStatus.PENDING)
    
    def get_assigned_tasks(self) -> List[Task]:
        """获取已分配的任务"""
        return self.get_tasks_by_status(TaskStatus.ASSIGNED)
    
    def get_completed_tasks(self) -> List[Task]:
        """获取已完成的任务"""
        return self.get_tasks_by_status(TaskStatus.COMPLETED)
    
    def get_tasks_for_vehicle(self, vehicle_id: int) -> List[Task]:
        """
        获取分配给指定AMR的所有任务
        
        用途: 战术层规划单个AMR的路径
        """
        return [
            task for task_id, task in self.tasks.items()
            if self.trackers[task_id].assigned_vehicle_id == vehicle_id
            and self.trackers[task_id].status in [TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS]
        ]
    
    def assign_task(self, task_id: int, vehicle_id: int):
        """将任务分配给AMR"""
        tracker = self.trackers.get(task_id)
        if tracker is None:
            raise ValueError(f"任务 {task_id} 不存在")
        tracker.assign_to_vehicle(vehicle_id)
    
    def reject_task(self, task_id: int):
        """拒绝任务（战略层决策）"""
        tracker = self.trackers.get(task_id)
        if tracker is None:
            raise ValueError(f"任务 {task_id} 不存在")
        tracker.reject()
    
    def get_statistics(self) -> Dict[str, any]:
        """
        获取任务池统计信息
        
        返回:
            Dict: 包含各种统计指标
        """
        total = len(self.tasks)
        if total == 0:
            return {
                "total": 0,
                "pending": 0,
                "assigned": 0,
                "in_progress": 0,
                "completed": 0,
                "rejected": 0,
                "completion_rate": 0.0,
                "rejection_rate": 0.0
            }
        
        status_counts = {status: 0 for status in TaskStatus}
        for tracker in self.trackers.values():
            status_counts[tracker.status] += 1
        
        completed = status_counts[TaskStatus.COMPLETED]
        rejected = status_counts[TaskStatus.REJECTED]
        
        return {
            "total": total,
            "pending": status_counts[TaskStatus.PENDING],
            "assigned": status_counts[TaskStatus.ASSIGNED],
            "in_progress": status_counts[TaskStatus.IN_PROGRESS],
            "completed": completed,
            "rejected": rejected,
            "completion_rate": completed / total if total > 0 else 0.0,
            "rejection_rate": rejected / total if total > 0 else 0.0
        }
    
    def __len__(self) -> int:
        """返回任务总数"""
        return len(self.tasks)
    
    def __contains__(self, task_id: int) -> bool:
        """检查任务是否在池中"""
        return task_id in self.tasks
    
    def __str__(self) -> str:
        """字符串表示"""
        stats = self.get_statistics()
        return (f"TaskPool(total={stats['total']}, "
                f"pending={stats['pending']}, "
                f"completed={stats['completed']})")


# ========== 便捷构造函数 ==========

def create_task(task_id: int,
               pickup_node: TaskNode,
               delivery_node: TaskNode,
               demand: Optional[float] = None,
               priority: int = 0,
               arrival_time: float = 0.0) -> Task:
    """
    创建任务的便捷函数
    
    参数:
        task_id: 任务ID
        pickup_node: pickup节点
        delivery_node: delivery节点
        demand: 需求量（如果不提供，使用pickup_node的demand）
        priority: 优先级
        arrival_time: 到达时间
    
    示例:
        task = create_task(
            task_id=1,
            pickup_node=pickup1,
            delivery_node=delivery1,
            priority=5
        )
    """
    # 如果没有提供demand，使用pickup_node的demand
    if demand is None:
        demand = pickup_node.demand
    
    return Task(
        task_id=task_id,
        pickup_node=pickup_node,
        delivery_node=delivery_node,
        demand=demand,
        priority=priority,
        arrival_time=arrival_time
    )


def create_task_from_node_pair(task_id: int,
                               pickup_node: TaskNode,
                               delivery_node: TaskNode,
                               **kwargs) -> Task:
    """
    从节点对创建任务（别名函数）
    
    与create_task功能相同，提供更明确的命名
    """
    return create_task(task_id, pickup_node, delivery_node, **kwargs)
