"""Distance helpers shared by the routing, energy, and optimisation layers.

It provides both Manhattan and Euclidean metrics plus a precomputed
``DistanceMatrix`` that caches all pairwise distances, exposes nearest charging
station queries, and measures incremental route deltas for destroy/repair
operators.
"""

from typing import Tuple, List, Dict, Optional, Set
from enum import Enum
import math 

class NodeType(Enum):
    """节点类型枚举"""
    DEPOT = "depot"
    PICKUP = "pickup"
    DELIVERY = "delivery"
    CHARGING = "charging"

class NodeIDHelper: 
    """
    节点ID管理辅助类
    
    根据数学模型的编号约定, 提供ID查询和转换功能
    """
    
    def __init__(self, num_tasks: int, num_charging_stations: int):
        """
        参数:
            num_tasks: 任务数量 n
            num_charging_stations: 充电站数量 m
        """
        self.num_tasks = num_tasks
        self.num_charging_stations = num_charging_stations
        # 计算各类节点的ID范围
        self.depot_id = 0
        self.pickup_range = (1, num_tasks)  # [1, n]
        self.delivery_range = (num_tasks + 1, 2 * num_tasks)  # [n+1, 2n]
        self.charging_range = (2 * num_tasks + 1, 
                              2 * num_tasks + num_charging_stations)  # [2n+1, 2n+m]
    
    def get_node_type(self, node_id: int) -> NodeType:
        """
        根据节点ID判断节点类型
        
        参数:
            node_id: 节点ID
        
        返回:
            NodeType: 节点类型
        """
        if node_id == 0:
            return NodeType.DEPOT
        elif self.pickup_range[0] <= node_id <= self.pickup_range[1]:
            return NodeType.PICKUP
        elif self.delivery_range[0] <= node_id <= self.delivery_range[1]:
            return NodeType.DELIVERY
        elif self.charging_range[0] <= node_id <= self.charging_range[1]:
            return NodeType.CHARGING
        else:
            raise ValueError(f"Invalid node_id: {node_id}")
        
    def get_paired_delivery(self, pickup_id: int) -> int:
        """
        获取pickup节点对应的delivery节点ID
        
        数学模型: delivery_id = pickup_id + n
        
        参数:
            pickup_id: pickup节点ID (1 ~ n)
        
        返回:
            int: 对应的delivery节点ID (n+1 ~ 2n)
        """
        if not (self.pickup_range[0] <= pickup_id <= self.pickup_range[1]):
            raise ValueError(f"Node {pickup_id} is not a pickup node")
        return pickup_id + self.num_tasks
    
    def get_paired_pickup(self, delivery_id: int) -> int:
        """
        获取delivery节点对应的pickup节点ID
        
        数学模型: pickup_id = delivery_id - n
        
        参数:
            delivery_id: delivery节点ID (n+1 ~ 2n)
        
        返回:
            int: 对应的pickup节点ID (1 ~ n)
        """
        if not (self.delivery_range[0] <= delivery_id <= self.delivery_range[1]):
            raise ValueError(f"Node {delivery_id} is not a delivery node")
        return delivery_id - self.num_tasks
    
    def get_task_id(self, node_id: int) -> int:
        """
        获取任务节点所属的任务编号
        
        参数:
            node_id: pickup或delivery节点ID
        
        返回:
            int: 任务编号 r (1 ~ n)
        """
        node_type = self.get_node_type(node_id)
        if node_type == NodeType.PICKUP:
            return node_id
        elif node_type == NodeType.DELIVERY:
            return self.get_paired_pickup(node_id)
        else:
            raise ValueError(f"Node {node_id} is not a task node")
    
    def get_all_pickup_ids(self) -> List[int]:
        """获取所有pickup节点ID列表"""
        return list(range(self.pickup_range[0], self.pickup_range[1] + 1))
    
    def get_all_delivery_ids(self) -> List[int]:
        """获取所有delivery节点ID列表"""
        return list(range(self.delivery_range[0], self.delivery_range[1] + 1))
    
    def get_all_charging_ids(self) -> List[int]:
        """获取所有充电站节点ID列表"""
        return list(range(self.charging_range[0], self.charging_range[1] + 1))
    
    def is_task_pair(self, node_i: int, node_j: int) -> bool:
        """
        判断两个节点是否为配对的pickup-delivery
        
        用途: 检查约束(3)和约束(6)的前提条件
        """
        try:
            if self.get_node_type(node_i) == NodeType.PICKUP:
                return self.get_paired_delivery(node_i) == node_j
            elif self.get_node_type(node_i) == NodeType.DELIVERY:
                return self.get_paired_pickup(node_i) == node_j
            return False
        except ValueError:
            return False
    


# Basic distance calculation functions
def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    计算两点间的欧几里得距离(直线距离)
    
    数学公式: d = √[(x2-x1)² + (y2-y1)²]
    
    参数:
        x1, y1: 起点坐标
        x2, y2: 终点坐标
    
    返回:
        float: 两点间的欧几里得距离
    """
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def manhattan_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    计算两点间的曼哈顿距离(网格距离)
    
    数学公式: d = |x2-x1| + |y2-y1|
    
    参数:
        x1, y1: 起点坐标
        x2, y2: 终点坐标
    
    返回:
        float: 两点间的曼哈顿距离
    """
    return abs(x2 - x1) + abs(y2 - y1)

# Distance Matrix class for precomputed distances
class DistanceMatrix:
    """
    距离矩阵 - 预计算所有节点间距离
    
    性能优化: O(√n) 计算 → O(1) 查表
    """
    
    def __init__(self, 
                 coordinates: Dict[int, Tuple[float, float]],
                 num_tasks: int,
                 num_charging_stations: int,
                 distance_func=euclidean_distance):
        """
        初始化距离矩阵
        
        参数:
            coordinates: 节点ID → (x, y)坐标的映射
            num_tasks: 任务数量 n
            num_charging_stations: 充电站数量 m
            distance_func: 距离计算函数
        
        示例:
            coordinates = {
                0: (0, 0),      # Depot
                1: (10, 20),    # p_1
                2: (30, 40),    # p_2
                3: (15, 25),    # d_1 (配对1)
                4: (35, 45),    # d_2 (配对2)
                5: (50, 0),     # c_1 (charging)
                6: (0, 50)      # c_2 (charging)
            }
            dm = DistanceMatrix(coordinates, num_tasks=2, num_charging_stations=2)
        """
        self.coordinates = coordinates
        self.num_tasks = num_tasks
        self.num_charging_stations = num_charging_stations
        self.distance_func = distance_func
        
        # 初始化节点ID管理器
        self.id_helper = NodeIDHelper(num_tasks, num_charging_stations)
        
        # 构建距离矩阵
        self._matrix: Dict[Tuple[int, int], float] = {}
        self._build_matrix()
    
    def _build_matrix(self):
        """构建完整距离矩阵"""
        node_ids = list(self.coordinates.keys())
        
        for i in node_ids:
            for j in node_ids:
                if i == j:
                    self._matrix[(i, j)] = 0.0
                else:
                    x1, y1 = self.coordinates[i]
                    x2, y2 = self.coordinates[j]
                    distance = self.distance_func(x1, y1, x2, y2)
                    self._matrix[(i, j)] = distance
    
    def get_distance(self, node_i: int, node_j: int) -> float:
        """获取两节点间距离 (O(1)查找)"""
        return self._matrix[(node_i, node_j)]
    
    def get_nearest_nodes(self, node_id: int, k: int = 5, 
                          node_types: Optional[Set[NodeType]] = None) -> List[Tuple[int, float]]:
        """
        获取距离某节点最近的k个节点
        
        参数:
            node_id: 参考节点ID
            k: 返回最近的k个节点
            node_types: 节点类型过滤器，只返回指定类型的节点
        
        返回:
            List[Tuple[int, float]]: [(节点ID, 距离), ...] 按距离升序
        
        示例:
            # 找最近的充电站
            dm.get_nearest_nodes(1, k=2, node_types={NodeType.CHARGING})
        """
        # 确定候选节点
        if node_types is not None:
            candidate_nodes = [
                n for n in self.coordinates 
                if n != node_id and self.id_helper.get_node_type(n) in node_types
            ]
        else:
            candidate_nodes = [n for n in self.coordinates if n != node_id]
        
        # 计算距离并排序
        distances = [(n, self.get_distance(node_id, n)) for n in candidate_nodes]
        distances.sort(key=lambda x: x[1])
        return distances[:k]
    
    def get_nearest_charging_station(self, node_id: int) -> Tuple[int, float]:
        """
        获取最近的充电站（便捷方法）
        
        用途: 贪婪充电站插入算法
        
        返回:
            Tuple[int, float]: (充电站ID, 距离)
        """
        nearest = self.get_nearest_nodes(node_id, k=1, 
                                        node_types={NodeType.CHARGING})
        if not nearest:
            raise ValueError("No charging stations available")
        return nearest[0]
    
    def total_distance(self, route: List[int]) -> float:
        """
        计算路径总距离
        
        参数:
            route: 节点ID序列
        """
        if len(route) < 2:
            return 0.0
        
        total = 0.0
        for i in range(len(route) - 1):
            total += self.get_distance(route[i], route[i + 1])
        return total
    
    def distance_from_depot(self, node_id: int) -> float:
        """计算节点到Depot的距离"""
        return self.get_distance(0, node_id)
    
    def validate_route_precedence(self, route: List[int]) -> bool:
        """
        验证路径是否满足pickup在delivery之前的约束
        
        对应数学模型约束(6):
            S_{i+n}^a ≥ T_{dep,i}^a + t_{move,(i,i+n)}^a
        
        参数:
            route: 节点ID序列
        
        返回:
            bool: True表示满足约束
        """
        visited_pickups = set()
        
        for node_id in route:
            node_type = self.id_helper.get_node_type(node_id)
            
            if node_type == NodeType.PICKUP:
                visited_pickups.add(node_id)
            elif node_type == NodeType.DELIVERY:
                # 检查对应的pickup是否已访问
                paired_pickup = self.id_helper.get_paired_pickup(node_id)
                if paired_pickup not in visited_pickups:
                    return False  # 违反约束
        
        return True
    
# ========== 辅助函数 ==========

def calculate_insertion_cost(i: int, j: int, k: int, 
                             dist_matrix: DistanceMatrix) -> float:
    """
    计算在i和j之间插入节点k的额外距离成本
    
    公式: cost = d(i,k) + d(k,j) - d(i,j)
    """
    d_ik = dist_matrix.get_distance(i, k)
    d_kj = dist_matrix.get_distance(k, j)
    d_ij = dist_matrix.get_distance(i, j)
    return (d_ik + d_kj) - d_ij


def calculate_removal_savings(i: int, k: int, j: int,
                              dist_matrix: DistanceMatrix) -> float:
    """
    计算移除节点k的距离节约
    
    当前路径: i → k → j
    移除后: i → j
    节约: d(i,k) + d(k,j) - d(i,j)
    """
    return -calculate_insertion_cost(i, j, k, dist_matrix)


# ========== 便捷创建函数 ==========

def create_distance_matrix_from_layout(
        depot: Tuple[float, float],
        task_locations: List[Tuple[Tuple[float, float], Tuple[float, float]]],
        charging_stations: List[Tuple[float, float]],
        use_manhattan: bool = False) -> DistanceMatrix:
    """
    从仓库布局创建距离矩阵
    
    参数:
        depot: Depot坐标
        task_locations: 任务位置列表，每个元素为 (pickup坐标, delivery坐标)
        charging_stations: 充电站坐标列表
        use_manhattan: 是否使用曼哈顿距离
    
    返回:
        DistanceMatrix: 构建好的距离矩阵
    
    示例:
        dm = create_distance_matrix_from_layout(
            depot=(0, 0),
            task_locations=[
                ((10, 20), (15, 25)),  # 任务1: p_1, d_1
                ((30, 40), (35, 45))   # 任务2: p_2, d_2
            ],
            charging_stations=[(50, 0), (0, 50)]
        )
        
        生成的节点ID:
        0: Depot (0, 0)
        1: p_1 (10, 20)
        2: p_2 (30, 40)
        3: d_1 (15, 25)
        4: d_2 (35, 45)
        5: c_1 (50, 0)
        6: c_2 (0, 50)
    """
    coordinates = {}
    num_tasks = len(task_locations)
    num_charging_stations = len(charging_stations)
    
    # ID 0: Depot
    coordinates[0] = depot
    
    # ID 1~n: Pickup节点
    for task_id, (pickup_loc, _) in enumerate(task_locations, start=1):
        coordinates[task_id] = pickup_loc
    
    # ID n+1~2n: Delivery节点
    for task_id, (_, delivery_loc) in enumerate(task_locations, start=1):
        delivery_id = task_id + num_tasks
        coordinates[delivery_id] = delivery_loc
    
    # ID 2n+1~2n+m: 充电站节点
    charging_start_id = 2 * num_tasks + 1
    for idx, station_loc in enumerate(charging_stations, start=charging_start_id):
        coordinates[idx] = station_loc
    
    distance_func = manhattan_distance if use_manhattan else euclidean_distance
    
    return DistanceMatrix(
        coordinates=coordinates,
        num_tasks=num_tasks,
        num_charging_stations=num_charging_stations,
        distance_func=distance_func
    )
