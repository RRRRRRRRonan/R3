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

    Week 1 改进: 支持距离、充电、时间、延迟多目标优化
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
                 repair_mode='mixed', cost_params: CostParameters = None,
                 charging_strategy=None):
        """
        参数：
            distance_matrix: 距离矩阵（用于计算成本）
            task_pool: 任务池
            repair_mode: 修复算子模式 ('greedy', 'regret2', 'random', 'mixed')
            cost_params: 成本参数配置
            charging_strategy: 充电策略对象 (Week 2新增)
        """
        self.distance = distance_matrix
        self.task_pool = task_pool
        self.repair_mode = repair_mode

        # Week 1: 成本参数
        self.cost_params = cost_params or CostParameters()

        # Week 2: 充电策略
        self.charging_strategy = charging_strategy

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

        # Week 2: 添加random模式统计
        random_count = 0

        for iteration in range(max_iterations):
            destroyed_route, removed_task_ids = self.random_removal(current_route, q=2)

            if self.repair_mode == 'greedy':
                candidate_route = self.greedy_insertion(destroyed_route, removed_task_ids)
                greedy_count += 1
            elif self.repair_mode == 'regret2':
                candidate_route = self.regret2_insertion(destroyed_route, removed_task_ids)
                regret_count += 1
            elif self.repair_mode == 'random':
                candidate_route = self.random_insertion(destroyed_route, removed_task_ids)
                random_count += 1
            else:  # mixed mode
                repair_choice = random.random()
                if repair_choice < 0.33:
                    candidate_route = self.greedy_insertion(destroyed_route, removed_task_ids)
                    greedy_count += 1
                elif repair_choice < 0.67:
                    candidate_route = self.regret2_insertion(destroyed_route, removed_task_ids)
                    regret_count += 1
                else:
                    candidate_route = self.random_insertion(destroyed_route, removed_task_ids)
                    random_count += 1
            
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
        
        print(f"算子使用统计: Greedy={greedy_count}, Regret-2={regret_count}, Random={random_count}")
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
        多目标成本评估（Week 2 改进）

        成本组成:
        1. 距离成本 = 总距离 × C_tr
        2. 充电成本 = 总充电量 × C_ch (从visits推导)
        3. 时间成本 = 总时间 × C_time (可选)
        4. 延迟成本 = 总延迟 × C_delay (可选)
        5. 任务丢失惩罚
        6. 不可行解惩罚
        7. 电池可行性惩罚 (Week 2新增)

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

        # 7. Week 2新增：电池可行性检查
        battery_penalty = 0.0
        if hasattr(self, 'vehicle') and hasattr(self, 'energy_config'):
            battery_feasible = self._check_battery_feasibility(route)
            if not battery_feasible:
                # 不可行解给予极大惩罚
                battery_penalty = self.cost_params.C_infeasible * 10.0

        total_cost = (distance_cost + charging_cost + time_cost +
                     delay_cost + missing_penalty + infeasible_penalty +
                     battery_penalty)

        return total_cost

    def _check_battery_feasibility(self, route: Route, debug=False) -> bool:
        """
        检查路径的电池可行性 (Week 2新增)

        模拟整个路径的电池消耗，检查是否会出现电量不足

        返回:
            bool: True表示可行，False表示不可行
        """
        if not hasattr(self, 'vehicle') or not hasattr(self, 'energy_config'):
            return True  # 没有约束则认为可行

        vehicle = self.vehicle
        energy_config = self.energy_config

        # 模拟电池消耗
        current_battery = vehicle.battery_capacity  # 满电出发

        for i in range(len(route.nodes) - 1):
            current_node = route.nodes[i]
            next_node = route.nodes[i + 1]

            # 如果当前节点是充电站，先充电
            if current_node.is_charging_station():
                # 使用充电策略决定充电量
                if self.charging_strategy:
                    # 计算剩余路径能耗（包括当前到下一节点的距离）
                    remaining_energy_demand = 0.0
                    for j in range(i, len(route.nodes) - 1):
                        seg_distance = self.distance.get_distance(
                            route.nodes[j].node_id,
                            route.nodes[j + 1].node_id
                        )
                        remaining_energy_demand += (seg_distance / 1000.0) * energy_config.consumption_rate

                    charge_amount = self.charging_strategy.determine_charging_amount(
                        current_battery=current_battery,
                        remaining_demand=remaining_energy_demand,
                        battery_capacity=vehicle.battery_capacity
                    )
                    if debug:
                        print(f"  CS at node {i}: battery={current_battery:.1f}, demand={remaining_energy_demand:.1f}, charge={charge_amount:.1f}")
                    current_battery = min(vehicle.battery_capacity, current_battery + charge_amount)
                else:
                    # 没有充电策略，默认充满
                    if debug:
                        print(f"  CS at node {i}: charging to full")
                    current_battery = vehicle.battery_capacity

            # 计算到下一节点的距离和能耗
            distance = self.distance.get_distance(
                current_node.node_id,
                next_node.node_id
            )
            energy_consumed = (distance / 1000.0) * energy_config.consumption_rate

            # 消耗能量前往下一节点
            current_battery -= energy_consumed

            if debug:
                print(f"  After move to node {i+1}: battery={current_battery:.1f}, consumed={energy_consumed:.1f}")

            # 检查是否电量不足
            if current_battery < 0:
                if debug:
                    print(f"  ✗ Battery depleted at node {i+1}!")
                return False  # 电量不足，不可行

        if debug:
            print(f"  ✓ Route feasible, final battery={current_battery:.1f}")
        return True  # 整个路径可行

    def get_cost_breakdown(self, route: Route) -> Dict:
        """
        获取成本分解（用于分析和调试）

        Week 2改进：
        - 使用RouteExecutor执行路径生成visits
        - 准确记录充电量和充电次数

        返回:
            Dict: 各项成本明细
        """
        from core.route_executor import RouteExecutor

        distance = route.calculate_total_distance(self.distance)

        # Week 2新增：执行路径生成visits（如果有vehicle和energy_config）
        if hasattr(self, 'vehicle') and hasattr(self, 'energy_config'):
            executor = RouteExecutor(
                distance_matrix=self.distance,
                energy_config=self.energy_config,
                time_config=getattr(self, 'time_config', None)
            )
            # 执行路径并生成visits
            executed_route = executor.execute(
                route=route,
                vehicle=self.vehicle,
                charging_strategy=self.charging_strategy
            )
            # 使用执行后的路径
            route_to_analyze = executed_route
        else:
            route_to_analyze = route

        # 充电量统计
        charging_amount = 0.0
        num_charging_stops = 0
        if route_to_analyze.visits:
            for visit in route_to_analyze.visits:
                if visit.node.is_charging_station():
                    charged = visit.battery_after_service - visit.battery_after_travel
                    if charged > 0.01:  # 只计算实际充电的
                        charging_amount += charged
                        num_charging_stops += 1

        # 时间统计
        total_time = 0.0
        if route_to_analyze.visits and len(route_to_analyze.visits) > 0:
            total_time = route_to_analyze.visits[-1].departure_time - route_to_analyze.visits[0].arrival_time

        # 延迟统计
        total_delay = 0.0
        if route_to_analyze.visits:
            for visit in route_to_analyze.visits:
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

    def random_insertion(self, route: Route, removed_task_ids: List[int]) -> Route:
        """
        随机插入算子 + 充电支持 (Week 2新增)

        策略：
        1. 对每个任务，随机选择一个可行的插入位置
        2. 可行性检查：必须满足充电约束
        3. 如果没有可行位置，跳过该任务
        """
        repaired_route = route.copy()

        if not hasattr(self, 'vehicle') or self.vehicle is None:
            raise ValueError("必须设置vehicle属性才能进行充电约束规划")
        if not hasattr(self, 'energy_config') or self.energy_config is None:
            raise ValueError("必须设置energy_config属性才能进行充电约束规划")

        vehicle = self.vehicle
        energy_config = self.energy_config

        for task_id in removed_task_ids:
            task = self.task_pool.get_task(task_id)

            # 收集所有可行的插入位置
            feasible_insertions = []

            for pickup_pos in range(1, len(repaired_route.nodes)):
                for delivery_pos in range(pickup_pos + 1, len(repaired_route.nodes) + 1):

                    feasible, charging_plan = repaired_route.check_energy_feasibility_for_insertion(
                        task,
                        (pickup_pos, delivery_pos),
                        vehicle,
                        self.distance,
                        energy_config
                    )

                    if feasible:
                        feasible_insertions.append({
                            'position': (pickup_pos, delivery_pos),
                            'charging_plan': charging_plan
                        })

            # 如果有可行位置，随机选择一个
            if feasible_insertions:
                chosen = random.choice(feasible_insertions)
                repaired_route.insert_task(task, chosen['position'])

                if chosen['charging_plan']:
                    sorted_plans = sorted(chosen['charging_plan'],
                                        key=lambda x: x['position'],
                                        reverse=True)
                    for plan in sorted_plans:
                        repaired_route.insert_charging_visit(
                            station=plan['station_node'],
                            position=plan['position'],
                            charge_amount=plan['amount']
                        )

        return repaired_route