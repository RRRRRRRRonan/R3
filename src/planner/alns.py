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
from core.vehicle import Vehicle
from physics.distance import DistanceMatrix

class MinimalALNS:
    """
    最简ALNS实现
    
    第一版功能：
    - Random Removal (destroy)
    - Greedy Insertion (repair)
    - 模拟退火接受准则
    - 不考虑充电（Week 3再加）
    """
    
    def __init__(self, distance_matrix: DistanceMatrix, task_pool, repair_mode='mixed'):
        """
        参数：
            distance_matrix: 距离矩阵（用于计算成本）
        """
        self.distance = distance_matrix
        self.task_pool = task_pool  # 任务池（用于获取Task对象）
        self.repair_mode = repair_mode  # 'greedy', 'regret2', 'mixed'
        
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
        # 初始化三个解
        current_route = initial_route.copy()
        best_route = initial_route.copy()
        best_cost = self.evaluate_cost(best_route)

        temperature = self.initial_temp

        greedy_count = 0
        regret_count = 0

        print(f"初始成本: {best_cost:.2f}m")
        print(f"总迭代次数: {max_iterations}")

        for iteration in range(max_iterations):
            # 1. Destroy:移除2个任务
            destroyed_route, removed_task_ids = self.random_removal(current_route, q=2)
            # 2. Repair: 随机选择repair算子
            if self.repair_mode == 'greedy':
                candidate_route = self.greedy_insertion(destroyed_route, removed_task_ids)
                greedy_count += 1
            elif self.repair_mode == 'regret2':
                candidate_route = self.regret2_insertion(destroyed_route, removed_task_ids)
                regret_count += 1
            else:  # mixed
                repair_choice = random.random()
                if repair_choice < 0.5:
                    # 50%概率使用贪心插入
                    candidate_route = self.greedy_insertion(destroyed_route, removed_task_ids)
                    greedy_count += 1
                else:
                    # 50%概率使用Regret-2插入
                    candidate_route = self.regret2_insertion(destroyed_route, removed_task_ids)
                    regret_count += 1
            # 3. 评估新解成本
            candidate_cost = self.evaluate_cost(candidate_route)
            current_cost = self.evaluate_cost(current_route)
            # 4. 接受准则
            if self.accept_solution(candidate_cost, current_cost, temperature):
                current_route = candidate_route
                # 更新最优解
                if candidate_cost < best_cost:
                    best_route = candidate_route
                    best_cost = candidate_cost
                    print(f"迭代 {iteration+1}: 新最优成本 {best_cost:.2f}m")
            # 5. 降温
            temperature *= self.cooling_rate
            if (iteration + 1) % 50 == 0:  # ← 添加这个进度监控
                print(f"  [进度] 已完成 {iteration+1}/{max_iterations} 次迭代, 当前最优: {best_cost:.2f}m")
        print(f"算子使用统计: Greedy={greedy_count}, Regret-2={regret_count}")
        print(f"最终最优成本: {best_cost:.2f}m (改进 {self.evaluate_cost(initial_route)-best_cost:.2f}m)")
        return best_route
    
    def random_removal(self, route: Route, q: int = 2) -> Tuple[Route, List[int]]:
        """
        Destroy算子：随机移除q个任务
        
        参数：
            route: 当前路径
            q: 移除任务数量
        
        返回：
            (破坏后的路径, 被移除的任务ID列表)
        """
        # 获取当前路径中的任务ID
        task_ids = route.get_served_tasks()
        # 如果任务数太少，减少移除数量
        if len(task_ids) < q:
            q = max(1, len(task_ids) - 1)
        # 随机选择q个任务
        removed_task_ids = random.sample(task_ids, q)
        # 复制路径并移除任务
        destroyed_route = route.copy()
        for task_id in removed_task_ids:
            task = self.task_pool.get_task(task_id)  # 从任务池获取Task对象
            destroyed_route.remove_task(task)
        return destroyed_route, removed_task_ids
    
    def greedy_insertion(self, 
                        route: Route, 
                        removed_task_ids: List[int]) -> Route:
        """
        Repair算子：贪心插入移除的任务
        
        参数：
            route: 被破坏的路径
            removed_task_ids: 需要重新插入的任务ID
            task_pool: 任务池（用于获取Task对象）
        
        返回：
            修复后的路径
        """
        repaired_route = route.copy()

        # 对每个被移除的任务
        for task_id in removed_task_ids: 
            task = self.task_pool.get_task(task_id)  # 从任务池获取Task对象
            best_cost = float('inf')
            best_position = None
            # 尝试所有可能的插入位置
            num_nodes = len(repaired_route.nodes)
            #pickup可以插在任意位置 （除了第一个depot）
            for pickup_pos in range(1, num_nodes):
                #delivery必须在pickup之后
                for delivery_pos in range(pickup_pos + 1, num_nodes + 1):
                    # 计算插入这个位置的成本增量
                    cost_delta = self._calculate_insertion_cost(
                        repaired_route, task, pickup_pos, delivery_pos
                    )
                    if cost_delta < best_cost:
                        best_cost = cost_delta
                        best_position = (pickup_pos, delivery_pos)
            # 执行最优插入
            if best_position:
                repaired_route.insert_task(task, best_position)
        return repaired_route
    
    def _calculate_insertion_cost(self, route: Route, task: Task, pickup_pos: int, delivery_pos: int) -> float:
        """
        计算插入成本（假设）
        方法：创建临时路径，插入，计算成本差
        """
        # 创建临时路径
        temp_route = route.copy()
        temp_route.insert_task(task, (pickup_pos, delivery_pos))
        # 计算成本差
        original_cost = self.evaluate_cost(route)
        new_cost = self.evaluate_cost(temp_route)

        return new_cost - original_cost
    
    def evaluate_cost(self, route: Route) -> float:
        """
        评估路径成本（目前只考虑距离）
        
        参数：
            route: 路径
        
        返回：
            成本值（越小越好）
        """
        return route.calculate_total_distance(self.distance)
    
    def accept_solution(self, 
                       new_cost: float, 
                       current_cost: float, 
                       temperature: float) -> bool:
        """
        模拟退火接受准则
        
        参数：
            new_cost: 新解的成本
            current_cost: 当前解的成本
            temperature: 当前温度
        
        返回：
            是否接受新解
        """
        if new_cost < current_cost:
            return True  # 更好的解，一定接受
        else:
            # 更差的解，以概率接受（避免局部最优）
            probability = math.exp(-(new_cost - current_cost) / temperature)
            return random.random() < probability
        
    def regret2_insertion(self, 
                      route: Route, 
                      removed_task_ids: List[int]) -> Route:
        """
        Regret-2插入算子：防止贪心的短视
        
        核心思想：
        优先插入"后悔值"大的任务，即如果现在不插入，
        下次插入会变得很贵的任务。
        
        参数：
            route: 被破坏的路径
            removed_task_ids: 需要重新插入的任务ID列表
        
        返回：
            修复后的路径
        """
        repaired_route = route.copy()
        remaining_tasks = removed_task_ids.copy()
        
        # 迭代插入，每次选择regret值最大的任务
        while remaining_tasks:
            best_regret = -float('inf')  # 找最大regret
            best_task_id = None
            best_position = None
            
            # 对每个剩余任务，计算其regret值
            for task_id in remaining_tasks:
                task = self.task_pool.get_task(task_id)
                
                # 找到该任务的最优和次优插入位置
                insertion_costs = []  # 存储所有可能的插入成本
                
                num_nodes = len(repaired_route.nodes)
                
                # 遍历所有可能的插入位置
                for pickup_pos in range(1, num_nodes):
                    for delivery_pos in range(pickup_pos + 1, num_nodes + 1):
                        cost_delta = repaired_route.calculate_insertion_cost_delta(
                            task, 
                            (pickup_pos, delivery_pos),
                            self.distance
                        )
                        insertion_costs.append({
                            'cost': cost_delta,
                            'position': (pickup_pos, delivery_pos)
                        })
                
                # 按成本排序
                insertion_costs.sort(key=lambda x: x['cost'])
                
                # 计算regret值
                if len(insertion_costs) >= 2:
                    c1 = insertion_costs[0]['cost']  # 最优
                    c2 = insertion_costs[1]['cost']  # 次优
                    regret = c2 - c1  # regret-2值
                    
                    # 更新最大regret的任务
                    if regret > best_regret:
                        best_regret = regret
                        best_task_id = task_id
                        best_position = insertion_costs[0]['position']
                
                elif len(insertion_costs) == 1:
                    # 只有一个可行位置，regret为无穷大（必须插这里）
                    regret = float('inf')
                    if regret > best_regret:
                        best_regret = regret
                        best_task_id = task_id
                        best_position = insertion_costs[0]['position']
            
            # 插入选中的任务
            if best_task_id is not None:
                task = self.task_pool.get_task(best_task_id)
                repaired_route.insert_task(task, best_position)
                remaining_tasks.remove(best_task_id)
            else:
                # 没有可行插入（理论上不应该发生）
                break
        
        return repaired_route
            
