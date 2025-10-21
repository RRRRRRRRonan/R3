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
        """
        task_ids = route.get_served_tasks()
        
        # ⭐ 关键修复：确保移除数量不超过现有任务数
        if len(task_ids) < q:
            q = max(1, len(task_ids))  # 至少移除1个，最多移除全部
        
        # ⭐ 如果任务数为0，直接返回空列表
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
        
        # 需要vehicle和energy_config参数
        if not hasattr(self, 'vehicle') or self.vehicle is None:
            raise ValueError("必须设置vehicle属性才能进行充电约束规划")
        if not hasattr(self, 'energy_config') or self.energy_config is None:
            raise ValueError("必须设置energy_config属性才能进行充电约束规划")

        vehicle = self.vehicle
        energy_config = self.energy_config

        print(f"\n🔧 [DEBUG] Greedy Insertion 开始")
        print(f"  当前路径节点数: {len(repaired_route.nodes)}")
        print(f"  需要插入的任务: {removed_task_ids}")
        print(f"  使用的电池容量: {vehicle.battery_capacity}kWh")
        
        successfully_inserted = []
        failed_to_insert = []
        
        for task_id in removed_task_ids:
            task = self.task_pool.get_task(task_id)
            
            best_cost = float('inf')
            best_position = None
            best_charging_plan = None

            feasible_positions_count = 0
            total_positions_checked = 0
            
            # 遍历所有可能的插入位置
            for pickup_pos in range(1, len(repaired_route.nodes)):
                for delivery_pos in range(pickup_pos + 1, len(repaired_route.nodes) + 1):
                    total_positions_checked += 1

                    # 1️⃣ 计算基础插入成本（距离）
                    cost_delta = repaired_route.calculate_insertion_cost_delta(
                        task, 
                        (pickup_pos, delivery_pos),
                        self.distance
                    )
                    
                    # 2️⃣ 检查能量可行性和充电需求
                    feasible, charging_plan = repaired_route.check_energy_feasibility_for_insertion(
                        task,
                        (pickup_pos, delivery_pos),
                        vehicle,
                        self.distance,
                        energy_config
                    )

                    if feasible:
                        feasible_positions_count += 1
                    
                    if not feasible:
                        continue  # 这个位置不可行，跳过
                    
                    # 3️⃣ 如果需要充电，增加充电成本
                    if charging_plan:
                        # 每个充电站的惩罚成本
                        # 这个值可以调整，代表充电的综合成本（时间、绕路、运营中断等）
                        charging_penalty_per_station = 100.0  # 相当于100米的距离成本
                    
                        total_charging_penalty = len(charging_plan) * charging_penalty_per_station
                        cost_delta += total_charging_penalty
                    
                    # 4️⃣ 更新最佳方案
                    if cost_delta < best_cost:
                        best_cost = cost_delta
                        best_position = (pickup_pos, delivery_pos)
                        best_charging_plan = charging_plan
            
            print(f"\n  任务 {task_id} 分析:")
            print(f"    检查的位置数: {total_positions_checked}")
            print(f"    可行的位置数: {feasible_positions_count}")

            if best_position is not None:
                print(f"    ✅ 找到最佳位置: {best_position}")
                print(f"    插入成本: {best_cost:.2f}")
                if best_charging_plan:
                    print(f"    🔋 需要充电: {len(best_charging_plan)}个充电站")
                    for i, plan in enumerate(best_charging_plan):
                        print(f"      充电站{i+1}: 位置{plan['position']}, 充电{plan['amount']:.2f}kWh")
                else:
                    print(f"    🔋 不需要充电")
                
                repaired_route.insert_task(task, best_position)

                # 从后往前插入充电站（避免位置偏移）
                if best_charging_plan:
                    sorted_plans = sorted(best_charging_plan, key=lambda x: x['position'], reverse=True)
                    for plan in sorted_plans:
                        repaired_route.insert_charging_visit(
                            station=plan['station_node'],
                            position=plan['position'],
                            charge_amount=plan['amount']
                        )
                
                successfully_inserted.append(task_id)
            else:
                print(f"    ❌ 无法找到可行的插入位置！")
                failed_to_insert.append(task_id)
        
        print(f"\n  插入总结:")
        print(f"    成功插入: {successfully_inserted}")
        print(f"    插入失败: {failed_to_insert}")
        print(f"    最终路径节点数: {len(repaired_route.nodes)}")
        
        if failed_to_insert:
            print(f"\n⚠️  警告：有 {len(failed_to_insert)} 个任务无法插入！")
            print(f"   失败的任务: {failed_to_insert}")
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
        评估路径成本
        
        成本 = 距离 + 任务丢失惩罚
        """
        # 基础距离成本
        distance_cost = route.calculate_total_distance(self.distance)
        
        # 任务完整性检查
        served_tasks = set(route.get_served_tasks())
        all_tasks = self.task_pool.get_all_tasks()
        expected_tasks = set(task.task_id for task in all_tasks)
        missing_tasks = expected_tasks - served_tasks
        
        # 每个丢失的任务，施加巨大惩罚（比如10000米）
        missing_penalty = len(missing_tasks) * 10000.0
        
        return distance_cost + missing_penalty
    
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
        Regret-2插入算子+充电支持：防止贪心的短视
        
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

        if not hasattr(self, 'vehicle') or self.vehicle is None:
            raise ValueError("必须设置vehicle属性才能进行充电约束规划")
        if not hasattr(self, 'energy_config') or self.energy_config is None:
            raise ValueError("必须设置energy_config属性才能进行充电约束规划")

        vehicle = self.vehicle
        energy_config = self.energy_config
        
        # 迭代插入，每次选择regret值最大的任务
        while remaining_tasks:
            best_regret = -float('inf')  # 找最大regret
            best_task_id = None
            best_position = None
            best_charging_plan = None
            
            # 对每个剩余任务，计算其regret值
            for task_id in remaining_tasks:
                task = self.task_pool.get_task(task_id)
                
                # 存储所有插入位置的成本
                feasible_insertions = []

                # 遍历所有可能的插入位置
                for pickup_pos in range(1, len(repaired_route.nodes)):
                    for delivery_pos in range(pickup_pos + 1, len(repaired_route.nodes) + 1):
                        cost_delta = repaired_route.calculate_insertion_cost_delta(
                            task, 
                            (pickup_pos, delivery_pos),
                            self.distance
                        )
                        # 检查能量可行性
                        feaasible, charging_plan = repaired_route.check_energy_feasibility_for_insertion(
                            task,
                            (pickup_pos, delivery_pos),
                            vehicle,
                            self.distance,
                            energy_config
                        )
                        if not feaasible:
                            continue  # 不可行，跳过

                        # 加入充电成本
                        if charging_plan:
                            cost_delta += len(charging_plan) * 50.0  # 充电惩罚
                        feasible_insertions.append({
                            'cost': cost_delta,
                            'position': (pickup_pos, delivery_pos),
                            'charging_plan': charging_plan
                        })
                
                # 计算regret值
                if len(feasible_insertions) >= 2:
                    # 按成本排序
                    feasible_insertions.sort(key=lambda x: x['cost'])
                    
                    best_cost = feasible_insertions[0]['cost']
                    second_best_cost = feasible_insertions[1]['cost']
                    
                    regret = second_best_cost - best_cost  # regret值
                    
                    if regret > best_regret:
                        best_regret = regret
                        best_task_id = task_id
                        best_position = feasible_insertions[0]['position']
                        best_charging_plan = feasible_insertions[0]['charging_plan']
                
                elif len(feasible_insertions) == 1:
                    # 只有一个可行位置，regret = 无穷大（优先插入）
                    if best_regret < float('inf'):
                        best_regret = float('inf')
                        best_task_id = task_id
                        best_position = feasible_insertions[0]['position']
                        best_charging_plan = feasible_insertions[0]['charging_plan']
            
            # 插入regret值最大的任务
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
                break  # 无可行插入
        
        return repaired_route
