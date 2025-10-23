"""
ALNS (Adaptive Large Neighborhood Search) 优化器
==============================================
用于单AMR路径规划 + 局部充电优化
"""

import random
import math
import time
from typing import List, Tuple, Dict
from dataclasses import dataclass
import sys
sys.path.append('src')

from core.route import Route
from core.task import Task
from core.vehicle import Vehicle, create_vehicle
from physics.energy import EnergyConfig
from physics.distance import DistanceMatrix


# ========== 成本参数配置 ==========
@dataclass
class CostParameters:
    """
    多目标成本函数参数配置
    支持距离、充电、时间、延迟多目标优化
    """
    # 基础成本权重
    C_tr: float = 1.0      # 距离成本 ($/m)
    C_ch: float = 0.6      # 充电成本 ($/kWh)
    C_time: float = 0.1    # 时间成本 ($/s)
    C_delay: float = 2.0   # 延迟惩罚 ($/s)
    C_wait: float = 0.05   # 等待成本 ($/s)

    # 惩罚项
    C_missing_task: float = 10000.0  # 任务丢失惩罚
    C_infeasible: float = 10000.0    # 不可行解惩罚

    def get_total_cost(self, distance: float, charging: float,
                      time: float, delay: float, waiting: float) -> float:
        """
        计算加权总成本

        参数:
            distance: 总距离 (m)
            charging: 总充电量 (kWh)
            time: 总时间 (s)
            delay: 总延迟 (s)
            waiting: 总等待 (s)

        返回:
            float: 加权总成本
        """
        return (self.C_tr * distance +
                self.C_ch * charging +
                self.C_time * time +
                self.C_delay * delay +
                self.C_wait * waiting)

class MinimalALNS:
    """
    最简ALNS实现

    第一版功能：
    - Random Removal (destroy)
    - Greedy Insertion (repair)
    - 模拟退火接受准则
    - 充电约束集成

    Week 1 改进:
    - 多目标成本函数 (距离 + 充电 + 时间 + 延迟)
    """

    def __init__(self, distance_matrix: DistanceMatrix, task_pool,
                 repair_mode='mixed', cost_params: CostParameters = None):
        """
        参数：
            distance_matrix: 距离矩阵（用于计算成本）
            task_pool: 任务池
            repair_mode: 修复算子模式 ('greedy', 'regret2', 'mixed')
            cost_params: 成本参数配置
        """
        self.distance = distance_matrix
        self.task_pool = task_pool
        self.repair_mode = repair_mode

        # Week 1: 成本参数
        self.cost_params = cost_params or CostParameters()

        # 模拟退火参数
        self.initial_temp = 100.0
        self.cooling_rate = 0.995
    
    def optimize(self, 
                 initial_route: Route,
                 max_iterations: int = 100) -> Route:
        """
        ALNS主循环
        
        参数：
            initial_route: 初始路径
            max_iterations: 迭代次数
        
        返回：
            优化后的最佳路径
        """
        current_route = initial_route.copy()
        best_route = initial_route.copy()
        best_cost = self.evaluate_cost(best_route)

        temperature = self.initial_temp

        greedy_count = 0
        regret_count = 0

        print(f"初始成本: {best_cost:.2f}m")
        print(f"总迭代次数: {max_iterations}")

        for iteration in range(max_iterations):
            destroyed_route, removed_task_ids = self.random_removal(current_route, q=2)
            
            if self.repair_mode == 'greedy':
                candidate_route = self.greedy_insertion(destroyed_route, removed_task_ids)
                greedy_count += 1
            elif self.repair_mode == 'regret2':
                candidate_route = self.regret2_insertion(destroyed_route, removed_task_ids)
                regret_count += 1
            else:
                repair_choice = random.random()
                if repair_choice < 0.5:
                    candidate_route = self.greedy_insertion(destroyed_route, removed_task_ids)
                    greedy_count += 1
                else:
                    candidate_route = self.regret2_insertion(destroyed_route, removed_task_ids)
                    regret_count += 1
            
            candidate_cost = self.evaluate_cost(candidate_route)
            current_cost = self.evaluate_cost(current_route)
            
            if self.accept_solution(candidate_cost, current_cost, temperature):
                current_route = candidate_route
                if candidate_cost < best_cost:
                    best_route = candidate_route
                    best_cost = candidate_cost
                    print(f"迭代 {iteration+1}: 新最优成本 {best_cost:.2f}m")
            
            temperature *= self.cooling_rate
            
            if (iteration + 1) % 50 == 0:
                print(f"  [进度] 已完成 {iteration+1}/{max_iterations} 次迭代, 当前最优: {best_cost:.2f}m")
        
        print(f"算子使用统计: Greedy={greedy_count}, Regret-2={regret_count}")
        print(f"最终最优成本: {best_cost:.2f}m (改进 {self.evaluate_cost(initial_route)-best_cost:.2f}m)")
        return best_route
    
    def random_removal(self, route: Route, q: int = 2) -> Tuple[Route, List[int]]:
        """
        Destroy算子：随机移除q个任务
        """
        task_ids = route.get_served_tasks()
        
        if len(task_ids) < q:
            q = max(1, len(task_ids))
        
        if len(task_ids) == 0:
            return route.copy(), []
        
        removed_task_ids = random.sample(task_ids, q)
        
        destroyed_route = route.copy()
        for task_id in removed_task_ids:
            task = self.task_pool.get_task(task_id)
            destroyed_route.remove_task(task)
        
        return destroyed_route, removed_task_ids

    def partial_removal(self, route: Route, q: int = 2) -> Tuple[Route, List[int]]:
        """
        Destroy算子：只移除delivery节点（Week 3步骤2.2）

        功能:
            - 随机选择q个任务
            - 只移除这些任务的delivery节点
            - 保留pickup节点在路径中
            - 允许repair阶段重新选择delivery位置

        返回:
            (destroyed_route, removed_task_ids)
            其中removed_task_ids表示需要重新插入delivery的任务
        """
        task_ids = route.get_served_tasks()

        if len(task_ids) < q:
            q = max(1, len(task_ids))

        if len(task_ids) == 0:
            return route.copy(), []

        # 随机选择要移除delivery的任务
        selected_task_ids = random.sample(task_ids, q)

        destroyed_route = route.copy()
        for task_id in selected_task_ids:
            task = self.task_pool.get_task(task_id)

            # 只移除delivery节点，保留pickup节点
            # 找到delivery节点并移除
            delivery_node_id = task.delivery_node.node_id
            destroyed_route.nodes = [
                node for node in destroyed_route.nodes
                if node.node_id != delivery_node_id
            ]

        return destroyed_route, selected_task_ids

    def pair_exchange(self, route: Route) -> Route:
        """
        Local search算子：交换两个任务的位置（Week 3步骤2.3）

        功能:
            - 随机选择两个任务
            - 交换它们在路径中的位置
            - 保持precedence约束（pickup在delivery之前）
            - 这是一个2-opt类型的local search

        返回:
            交换后的路径
        """
        task_ids = route.get_served_tasks()

        if len(task_ids) < 2:
            return route.copy()  # 少于2个任务，无法交换

        # 随机选择两个不同的任务
        task1_id, task2_id = random.sample(task_ids, 2)
        task1 = self.task_pool.get_task(task1_id)
        task2 = self.task_pool.get_task(task2_id)

        # 找到两个任务在路径中的位置
        task1_pickup_pos = None
        task1_delivery_pos = None
        task2_pickup_pos = None
        task2_delivery_pos = None

        for i, node in enumerate(route.nodes):
            if hasattr(node, 'task_id'):
                if node.task_id == task1_id:
                    if node.is_pickup():
                        task1_pickup_pos = i
                    elif node.is_delivery():
                        task1_delivery_pos = i
                elif node.task_id == task2_id:
                    if node.is_pickup():
                        task2_pickup_pos = i
                    elif node.is_delivery():
                        task2_delivery_pos = i

        # 检查是否找到所有节点
        if None in [task1_pickup_pos, task1_delivery_pos, task2_pickup_pos, task2_delivery_pos]:
            return route.copy()  # 无法找到完整任务，返回原路径

        # 创建新路径进行交换
        exchanged_route = route.copy()

        # 提取节点
        task1_pickup = task1.pickup_node
        task1_delivery = task1.delivery_node
        task2_pickup = task2.pickup_node
        task2_delivery = task2.delivery_node

        # 移除所有四个节点（从后往前移除，避免索引变化）
        positions = sorted([task1_pickup_pos, task1_delivery_pos, task2_pickup_pos, task2_delivery_pos], reverse=True)
        for pos in positions:
            exchanged_route.nodes.pop(pos)

        # 重新插入，交换两个任务的相对顺序
        # 策略：保持pickup-delivery的相对距离，但交换任务顺序

        # 简化策略：将task2插入到task1原来的位置，task1插入到task2原来的位置
        # 需要确保precedence约束

        # 找到最小和最大位置
        min_pos = min(task1_pickup_pos, task2_pickup_pos)

        # 按照新顺序插入
        if task1_pickup_pos < task2_pickup_pos:
            # 原顺序：task1在前，task2在后
            # 交换后：task2在前，task1在后
            # 在min_pos位置插入task2
            exchanged_route.nodes.insert(min_pos, task2_pickup)
            exchanged_route.nodes.insert(min_pos + 1, task2_delivery)
            exchanged_route.nodes.insert(min_pos + 2, task1_pickup)
            exchanged_route.nodes.insert(min_pos + 3, task1_delivery)
        else:
            # 原顺序：task2在前，task1在后
            # 交换后：task1在前，task2在后
            exchanged_route.nodes.insert(min_pos, task1_pickup)
            exchanged_route.nodes.insert(min_pos + 1, task1_delivery)
            exchanged_route.nodes.insert(min_pos + 2, task2_pickup)
            exchanged_route.nodes.insert(min_pos + 3, task2_delivery)

        return exchanged_route

    def greedy_insertion(self, route: Route, removed_task_ids: List[int]) -> Route:
        """
        贪心插入算子 + 充电支持

        Week 3改进（步骤2.1）：
        - 支持pickup/delivery分离插入（可以先集中取货，再集中送货）
        - 增加容量约束检查，防止超载

        策略：
        1. 对每个任务，找到成本最小的插入位置
        2. Week 3新增：检查容量可行性（避免超载）
        3. 如果需要充电，在总成本中加入充电惩罚
        4. 插入成本 = 距离增量 + 充电惩罚
        """
        from core.vehicle import create_vehicle
        from physics.energy import EnergyConfig

        repaired_route = route.copy()

        if not hasattr(self, 'vehicle') or self.vehicle is None:
            raise ValueError("必须设置vehicle属性才能进行充电约束规划")
        if not hasattr(self, 'energy_config') or self.energy_config is None:
            raise ValueError("必须设置energy_config属性才能进行充电约束规划")

        vehicle = self.vehicle
        energy_config = self.energy_config

        for task_id in removed_task_ids:
            task = self.task_pool.get_task(task_id)

            # Week 3步骤2.2：检查pickup是否已在路径中
            pickup_in_route = False
            pickup_position = None
            for i, node in enumerate(repaired_route.nodes):
                if hasattr(node, 'task_id') and node.task_id == task_id and node.is_pickup():
                    pickup_in_route = True
                    pickup_position = i
                    break

            best_cost = float('inf')
            best_position = None
            best_charging_plan = None

            if pickup_in_route:
                # 只需要插入delivery节点
                for delivery_pos in range(pickup_position + 1, len(repaired_route.nodes) + 1):
                    # 创建临时路径测试插入
                    temp_route = repaired_route.copy()
                    temp_route.nodes.insert(delivery_pos, task.delivery_node)

                    # Week 3新增：检查容量可行性
                    capacity_feasible, capacity_error = temp_route.check_capacity_feasibility(
                        vehicle.capacity,
                        debug=False
                    )

                    if not capacity_feasible:
                        continue

                    # 计算插入delivery的成本增量
                    if delivery_pos == 0:
                        cost_delta = 0.0
                    else:
                        prev_node = repaired_route.nodes[delivery_pos - 1]
                        if delivery_pos < len(repaired_route.nodes):
                            next_node = repaired_route.nodes[delivery_pos]
                            # 移除原边，添加新边
                            old_dist = self.distance.get_distance(prev_node.node_id, next_node.node_id)
                            new_dist = (self.distance.get_distance(prev_node.node_id, task.delivery_node.node_id) +
                                       self.distance.get_distance(task.delivery_node.node_id, next_node.node_id))
                            cost_delta = new_dist - old_dist
                        else:
                            # 插入到末尾
                            cost_delta = self.distance.get_distance(prev_node.node_id, task.delivery_node.node_id)

                    if cost_delta < best_cost:
                        best_cost = cost_delta
                        best_position = ('delivery_only', delivery_pos)
            else:
                # 需要插入完整任务（pickup和delivery）
                # Week 3改进：遍历所有pickup和delivery位置组合
                # 允许pickup和delivery之间有间隔（支持分离插入）
                for pickup_pos in range(1, len(repaired_route.nodes)):
                    for delivery_pos in range(pickup_pos + 1, len(repaired_route.nodes) + 1):

                        # 创建临时路径测试插入
                        temp_route = repaired_route.copy()
                        temp_route.insert_task(task, (pickup_pos, delivery_pos))

                    # Week 3新增：检查容量可行性
                    capacity_feasible, capacity_error = temp_route.check_capacity_feasibility(
                        vehicle.capacity,
                        debug=False
                    )

                    if not capacity_feasible:
                        # 容量不可行，跳过此位置
                        continue

                    cost_delta = repaired_route.calculate_insertion_cost_delta(
                        task,
                        (pickup_pos, delivery_pos),
                        self.distance
                    )

                    feasible, charging_plan = repaired_route.check_energy_feasibility_for_insertion(
                        task,
                        (pickup_pos, delivery_pos),
                        vehicle,
                        self.distance,
                        energy_config
                    )

                    if not feasible:
                        continue

                    if charging_plan:
                        charging_penalty_per_station = 100.0
                        total_charging_penalty = len(charging_plan) * charging_penalty_per_station
                        cost_delta += total_charging_penalty

                    if cost_delta < best_cost:
                        best_cost = cost_delta
                        best_position = (pickup_pos, delivery_pos)
                        best_charging_plan = charging_plan

            # Week 3步骤2.2：根据best_position类型执行不同的插入
            if best_position is not None:
                if isinstance(best_position, tuple) and len(best_position) == 2:
                    if best_position[0] == 'delivery_only':
                        # 只插入delivery节点
                        delivery_pos = best_position[1]
                        repaired_route.nodes.insert(delivery_pos, task.delivery_node)
                    else:
                        # 插入完整任务（pickup和delivery）
                        repaired_route.insert_task(task, best_position)

                        if best_charging_plan:
                            sorted_plans = sorted(best_charging_plan, key=lambda x: x['position'], reverse=True)
                            for plan in sorted_plans:
                                repaired_route.insert_charging_visit(
                                    station=plan['station_node'],
                                    position=plan['position'],
                                    charge_amount=plan['amount']
                                )

        return repaired_route
    
    def evaluate_cost(self, route: Route) -> float:
        """
        多目标成本评估

        成本组成:
        1. 距离成本 = 总距离 × C_tr
        2. 充电成本 = 总充电量 × C_ch (从visits推导)
        3. 时间成本 = 总时间 × C_time (可选)
        4. 延迟成本 = 总延迟 × C_delay (可选)
        5. 任务丢失惩罚
        6. 不可行解惩罚

        返回:
            float: 加权总成本
        """
        # 1. 距离成本
        total_distance = route.calculate_total_distance(self.distance)
        distance_cost = total_distance * self.cost_params.C_tr

        # 2. 充电成本 (从visits推导)
        charging_amount = 0.0
        if route.visits:
            for visit in route.visits:
                if visit.node.is_charging_station():
                    # 充电量 = 离开时电量 - 到达时电量
                    charged = visit.battery_after_service - visit.battery_after_travel
                    charging_amount += max(0.0, charged)
        charging_cost = charging_amount * self.cost_params.C_ch

        # 3. 时间成本 (可选)
        time_cost = 0.0
        if route.visits and len(route.visits) > 0:
            total_time = route.visits[-1].departure_time - route.visits[0].arrival_time
            time_cost = total_time * self.cost_params.C_time

        # 4. 延迟成本 (可选)
        delay_cost = 0.0
        if route.visits:
            for visit in route.visits:
                delay_cost += visit.get_delay() * self.cost_params.C_delay

        # 5. 任务丢失惩罚
        served_tasks = set(route.get_served_tasks())
        all_tasks = self.task_pool.get_all_tasks()
        expected_tasks = set(task.task_id for task in all_tasks)
        missing_tasks = expected_tasks - served_tasks
        missing_penalty = len(missing_tasks) * self.cost_params.C_missing_task

        # 6. 不可行解惩罚
        infeasible_penalty = 0.0
        if route.is_feasible is False:
            infeasible_penalty = self.cost_params.C_infeasible

        total_cost = (distance_cost + charging_cost + time_cost +
                     delay_cost + missing_penalty + infeasible_penalty)

        return total_cost

    def get_cost_breakdown(self, route: Route) -> Dict:
        """
        获取成本分解（用于分析和调试）

        返回:
            Dict: 各项成本明细
        """
        distance = route.calculate_total_distance(self.distance)

        # 充电量统计
        charging_amount = 0.0
        num_charging_stops = 0
        if route.visits:
            for visit in route.visits:
                if visit.node.is_charging_station():
                    charged = visit.battery_after_service - visit.battery_after_travel
                    charging_amount += max(0.0, charged)
                    num_charging_stops += 1

        # 时间统计
        total_time = 0.0
        if route.visits and len(route.visits) > 0:
            total_time = route.visits[-1].departure_time - route.visits[0].arrival_time

        # 延迟统计
        total_delay = 0.0
        if route.visits:
            for visit in route.visits:
                total_delay += visit.get_delay()

        return {
            'total_distance': distance,
            'total_charging': charging_amount,
            'total_time': total_time,
            'total_delay': total_delay,
            'num_charging_stops': num_charging_stops,
            'distance_cost': distance * self.cost_params.C_tr,
            'charging_cost': charging_amount * self.cost_params.C_ch,
            'time_cost': total_time * self.cost_params.C_time,
            'delay_cost': total_delay * self.cost_params.C_delay,
            'total_cost': self.evaluate_cost(route),
            'cost_per_km': self.evaluate_cost(route) / (distance / 1000) if distance > 0 else 0
        }
    
    def accept_solution(self, 
                       new_cost: float, 
                       current_cost: float, 
                       temperature: float) -> bool:
        """
        模拟退火接受准则
        """
        if new_cost < current_cost:
            return True
        else:
            probability = math.exp(-(new_cost - current_cost) / temperature)
            return random.random() < probability
        
    def regret2_insertion(self,
                      route: Route,
                      removed_task_ids: List[int]) -> Route:
        """
        Regret-2插入算子+充电支持（Week 3步骤2.4改进）

        Week 3改进：
        - 支持容量约束检查
        - 支持partial delivery插入
        - 更智能的位置评估
        """
        repaired_route = route.copy()
        remaining_tasks = removed_task_ids.copy()

        if not hasattr(self, 'vehicle') or self.vehicle is None:
            raise ValueError("必须设置vehicle属性才能进行充电约束规划")
        if not hasattr(self, 'energy_config') or self.energy_config is None:
            raise ValueError("必须设置energy_config属性才能进行充电约束规划")

        vehicle = self.vehicle
        energy_config = self.energy_config

        while remaining_tasks:
            best_regret = -float('inf')
            best_task_id = None
            best_position = None
            best_charging_plan = None

            for task_id in remaining_tasks:
                task = self.task_pool.get_task(task_id)

                # Week 3步骤2.2：检查pickup是否已在路径中
                pickup_in_route = False
                pickup_position = None
                for i, node in enumerate(repaired_route.nodes):
                    if hasattr(node, 'task_id') and node.task_id == task_id and node.is_pickup():
                        pickup_in_route = True
                        pickup_position = i
                        break

                feasible_insertions = []

                if pickup_in_route:
                    # 只需要插入delivery节点
                    for delivery_pos in range(pickup_position + 1, len(repaired_route.nodes) + 1):
                        temp_route = repaired_route.copy()
                        temp_route.nodes.insert(delivery_pos, task.delivery_node)

                        # Week 3新增：检查容量可行性
                        capacity_feasible, _ = temp_route.check_capacity_feasibility(vehicle.capacity, debug=False)
                        if not capacity_feasible:
                            continue

                        # 计算成本增量
                        prev_node = repaired_route.nodes[delivery_pos - 1]
                        if delivery_pos < len(repaired_route.nodes):
                            next_node = repaired_route.nodes[delivery_pos]
                            old_dist = self.distance.get_distance(prev_node.node_id, next_node.node_id)
                            new_dist = (self.distance.get_distance(prev_node.node_id, task.delivery_node.node_id) +
                                       self.distance.get_distance(task.delivery_node.node_id, next_node.node_id))
                            cost_delta = new_dist - old_dist
                        else:
                            cost_delta = self.distance.get_distance(prev_node.node_id, task.delivery_node.node_id)

                        feasible_insertions.append({
                            'cost': cost_delta,
                            'position': ('delivery_only', delivery_pos),
                            'charging_plan': None
                        })
                else:
                    # 需要插入完整任务
                    for pickup_pos in range(1, len(repaired_route.nodes)):
                        for delivery_pos in range(pickup_pos + 1, len(repaired_route.nodes) + 1):
                            temp_route = repaired_route.copy()
                            temp_route.insert_task(task, (pickup_pos, delivery_pos))

                            # Week 3新增：检查容量可行性
                            capacity_feasible, _ = temp_route.check_capacity_feasibility(vehicle.capacity, debug=False)
                            if not capacity_feasible:
                                continue

                            cost_delta = repaired_route.calculate_insertion_cost_delta(
                                task,
                                (pickup_pos, delivery_pos),
                                self.distance
                            )

                            feasible, charging_plan = repaired_route.check_energy_feasibility_for_insertion(
                                task,
                                (pickup_pos, delivery_pos),
                                vehicle,
                                self.distance,
                                energy_config
                            )

                            if not feasible:
                                continue

                            if charging_plan:
                                cost_delta += len(charging_plan) * 50.0

                            feasible_insertions.append({
                                'cost': cost_delta,
                                'position': (pickup_pos, delivery_pos),
                                'charging_plan': charging_plan
                            })
                
                if len(feasible_insertions) >= 2:
                    feasible_insertions.sort(key=lambda x: x['cost'])
                    
                    best_cost = feasible_insertions[0]['cost']
                    second_best_cost = feasible_insertions[1]['cost']
                    
                    regret = second_best_cost - best_cost
                    
                    if regret > best_regret:
                        best_regret = regret
                        best_task_id = task_id
                        best_position = feasible_insertions[0]['position']
                        best_charging_plan = feasible_insertions[0]['charging_plan']
                
                elif len(feasible_insertions) == 1:
                    if best_regret < float('inf'):
                        best_regret = float('inf')
                        best_task_id = task_id
                        best_position = feasible_insertions[0]['position']
                        best_charging_plan = feasible_insertions[0]['charging_plan']
            
            if best_task_id:
                task = self.task_pool.get_task(best_task_id)

                # Week 3步骤2.2：根据best_position类型执行不同的插入
                if isinstance(best_position, tuple) and len(best_position) == 2:
                    if best_position[0] == 'delivery_only':
                        # 只插入delivery节点
                        delivery_pos = best_position[1]
                        repaired_route.nodes.insert(delivery_pos, task.delivery_node)
                    else:
                        # 插入完整任务
                        repaired_route.insert_task(task, best_position)

                        if best_charging_plan:
                            sorted_plans = sorted(best_charging_plan,
                                                key=lambda x: x['position'],
                                                reverse=True)
                            for plan in sorted_plans:
                                repaired_route.insert_charging_visit(
                                    station=plan['station_node'],
                                    position=plan['position'],
                                    charge_amount=plan['amount']
                                )

                remaining_tasks.remove(best_task_id)
            else:
                break
        
        return repaired_route