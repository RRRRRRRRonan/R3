"""
ALNS (Adaptive Large Neighborhood Search) 优化器
==============================================
用于单AMR路径规划 + 局部充电优化
"""

import random
import math
import time
from typing import List, Tuple
import sys
sys.path.append('src')

from core.route import Route
from core.task import Task
from core.vehicle import Vehicle, create_vehicle
from physics.energy import EnergyConfig
from physics.distance import DistanceMatrix

class MinimalALNS:
    """
    最简ALNS实现
    
    第一版功能：
    - Random Removal (destroy)
    - Greedy Insertion (repair)
    - 模拟退火接受准则
    - 充电约束集成
    """
    
    def __init__(self, distance_matrix: DistanceMatrix, task_pool, repair_mode='mixed'):
        """
        参数：
            distance_matrix: 距离矩阵（用于计算成本）
        """
        self.distance = distance_matrix
        self.task_pool = task_pool
        self.repair_mode = repair_mode
        
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
    
    def greedy_insertion(self, route: Route, removed_task_ids: List[int]) -> Route:
        """
        贪心插入算子 + 充电支持

        策略：
        1. 对每个任务，找到成本最小的插入位置
        2. 如果需要充电，在总成本中加入充电惩罚
        3. 插入成本 = 距离增量 + 充电惩罚
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
            
            best_cost = float('inf')
            best_position = None
            best_charging_plan = None
            
            for pickup_pos in range(1, len(repaired_route.nodes)):
                for delivery_pos in range(pickup_pos + 1, len(repaired_route.nodes) + 1):
                    
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
            
            if best_position is not None:
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
        评估路径成本
        
        成本 = 距离 + 任务丢失惩罚 + 能量不可行惩罚
        """
        distance_cost = route.calculate_total_distance(self.distance)
        
        served_tasks = set(route.get_served_tasks())
        all_tasks = self.task_pool.get_all_tasks()
        expected_tasks = set(task.task_id for task in all_tasks)
        missing_tasks = expected_tasks - served_tasks
        
        missing_penalty = len(missing_tasks) * 10000.0
        
        # 能量可行性检查
        total_distance = route.calculate_total_distance(self.distance)
        estimated_energy = (total_distance / 1000.0) * self.energy_config.consumption_rate
        
        from core.node import NodeType
        num_charging_stations = len([n for n in route.nodes if n.node_type == NodeType.CHARGING])
        
        if estimated_energy > self.vehicle.battery_capacity and num_charging_stations == 0:
            energy_penalty = 10000.0
        else:
            energy_penalty = 0.0
        
        return distance_cost + missing_penalty + energy_penalty
    
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
        Regret-2插入算子+充电支持
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
                
                feasible_insertions = []

                for pickup_pos in range(1, len(repaired_route.nodes)):
                    for delivery_pos in range(pickup_pos + 1, len(repaired_route.nodes) + 1):
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