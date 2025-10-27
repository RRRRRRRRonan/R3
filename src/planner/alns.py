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


# ========== 自适应算子选择器 ==========
class AdaptiveOperatorSelector:
    """
    自适应算子选择器

    功能：
    - 跟踪每个算子的历史表现（成功率、平均改进）
    - 根据表现动态调整选择概率
    - 使用轮盘赌选择机制

    实现参考：
    Ropke & Pisinger (2006) - An adaptive large neighborhood search heuristic
    """

    def __init__(self, operators: List[str], initial_weight: float = 1.0,
                 decay_factor: float = 0.8):
        """
        初始化自适应算子选择器

        参数:
            operators: 算子名称列表，如 ['greedy', 'regret2', 'random']
            initial_weight: 初始权重
            decay_factor: 权重衰减因子（历史表现的影响衰减速度）
        """
        self.operators = operators
        self.decay_factor = decay_factor

        # 权重（影响选择概率）
        self.weights = {op: initial_weight for op in operators}

        # 统计信息
        self.usage_count = {op: 0 for op in operators}  # 使用次数
        self.success_count = {op: 0 for op in operators}  # 成功次数（找到更好解）
        self.total_improvement = {op: 0.0 for op in operators}  # 累计改进

        # 奖励分数（用于调整权重）
        self.sigma1 = 33  # 找到新的全局最优解
        self.sigma2 = 9   # 接受的解但不是全局最优
        self.sigma3 = 13  # 找到更好的解（即使没接受）

    def select_operator(self) -> str:
        """
        使用轮盘赌方法选择算子

        返回:
            选中的算子名称
        """
        # 计算总权重
        total_weight = sum(self.weights.values())

        if total_weight == 0:
            # 如果所有权重都是0，均匀选择
            return random.choice(self.operators)

        # 轮盘赌选择
        rand_val = random.random() * total_weight
        cumulative = 0.0

        for op in self.operators:
            cumulative += self.weights[op]
            if rand_val <= cumulative:
                self.usage_count[op] += 1
                return op

        # 理论上不应该到达这里，但作为后备
        return self.operators[-1]

    def update_weights(self, operator: str, improvement: float,
                       is_new_best: bool, is_accepted: bool):
        """
        根据算子表现更新权重

        参数:
            operator: 算子名称
            improvement: 成本改进量（正值表示改进）
            is_new_best: 是否找到新的全局最优解
            is_accepted: 解是否被接受
        """
        # 确定奖励分数
        if is_new_best:
            score = self.sigma1  # 最高奖励
        elif is_accepted:
            score = self.sigma2  # 中等奖励
        elif improvement > 0:
            score = self.sigma3  # 找到改进但未接受
        else:
            score = 0  # 没有改进

        # 更新统计
        if improvement > 0:
            self.success_count[operator] += 1
            self.total_improvement[operator] += improvement

        # 更新权重（带衰减）
        # 新权重 = 旧权重 * decay + 当前分数
        self.weights[operator] = (self.weights[operator] * self.decay_factor +
                                  score * (1 - self.decay_factor))

    def get_statistics(self) -> Dict[str, Dict]:
        """
        获取算子统计信息

        返回:
            包含每个算子统计数据的字典
        """
        stats = {}
        for op in self.operators:
            usage = self.usage_count[op]
            success = self.success_count[op]
            stats[op] = {
                'usage_count': usage,
                'success_count': success,
                'success_rate': success / usage if usage > 0 else 0.0,
                'avg_improvement': (self.total_improvement[op] / success
                                   if success > 0 else 0.0),
                'weight': self.weights[op]
            }
        return stats

    def print_statistics(self):
        """打印算子统计信息"""
        print("\n" + "=" * 70)
        print("自适应算子选择统计")
        print("=" * 70)
        print(f"{'算子':<15} {'使用次数':<10} {'成功次数':<10} {'成功率':<10} {'平均改进':<12} {'当前权重':<10}")
        print("-" * 70)

        stats = self.get_statistics()
        for op, data in stats.items():
            print(f"{op:<15} {data['usage_count']:<10} {data['success_count']:<10} "
                  f"{data['success_rate']:<10.2%} {data['avg_improvement']:<12.2f} "
                  f"{data['weight']:<10.2f}")
        print("=" * 70)


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
                 repair_mode='mixed', cost_params: CostParameters = None,
                 charging_strategy=None, use_adaptive: bool = True):
        """
        参数：
            distance_matrix: 距离矩阵（用于计算成本）
            task_pool: 任务池
            repair_mode: 修复算子模式 ('greedy', 'regret2', 'random', 'mixed', 'adaptive')
            cost_params: 成本参数配置
            charging_strategy: 充电策略对象 (Week 2新增)
            use_adaptive: 是否使用自适应算子选择（Week 4新增）
        """
        self.distance = distance_matrix
        self.task_pool = task_pool
        self.repair_mode = repair_mode

        # Week 1: 成本参数
        self.cost_params = cost_params or CostParameters()

        # Week 2: 充电策略
        self.charging_strategy = charging_strategy

        # Week 4: 自适应算子选择（Repair算子）
        self.use_adaptive = use_adaptive or repair_mode == 'adaptive'
        if self.use_adaptive:
            # Repair算子自适应选择器
            self.adaptive_repair_selector = AdaptiveOperatorSelector(
                operators=['greedy', 'regret2', 'random'],
                initial_weight=1.0,
                decay_factor=0.8
            )
            # Destroy算子自适应选择器（新增）
            self.adaptive_destroy_selector = AdaptiveOperatorSelector(
                operators=['random_removal', 'partial_removal'],
                initial_weight=1.0,
                decay_factor=0.8
            )
        else:
            self.adaptive_repair_selector = None
            self.adaptive_destroy_selector = None

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

        # 算子使用统计（Repair）
        greedy_count = 0
        regret_count = 0
        random_count = 0

        # 算子使用统计（Destroy）
        random_removal_count = 0
        partial_removal_count = 0

        print(f"初始成本: {best_cost:.2f}m")
        print(f"总迭代次数: {max_iterations}")
        if self.use_adaptive:
            print("使用自适应算子选择 ✓ (Destroy + Repair)")

        for iteration in range(max_iterations):
            # Destroy阶段 - 使用自适应选择或固定模式
            if self.use_adaptive and self.adaptive_destroy_selector is not None:
                # 自适应选择destroy算子
                selected_destroy = self.adaptive_destroy_selector.select_operator()

                if selected_destroy == 'random_removal':
                    destroyed_route, removed_task_ids = self.random_removal(current_route, q=2)
                    random_removal_count += 1
                else:  # partial_removal
                    destroyed_route, removed_task_ids = self.partial_removal(current_route, q=2)
                    partial_removal_count += 1
            else:
                # 默认使用random_removal
                destroyed_route, removed_task_ids = self.random_removal(current_route, q=2)
                selected_destroy = 'random_removal'
                random_removal_count += 1

            # Repair阶段 - 使用自适应选择或固定模式
            if self.use_adaptive and self.adaptive_repair_selector is not None:
                # 自适应选择repair算子
                selected_repair = self.adaptive_repair_selector.select_operator()

                if selected_repair == 'greedy':
                    candidate_route = self.greedy_insertion(destroyed_route, removed_task_ids)
                    greedy_count += 1
                elif selected_repair == 'regret2':
                    candidate_route = self.regret2_insertion(destroyed_route, removed_task_ids)
                    regret_count += 1
                else:  # random
                    candidate_route = self.random_insertion(destroyed_route, removed_task_ids)
                    random_count += 1
            else:
                # 固定模式选择
                if self.repair_mode == 'greedy':
                    candidate_route = self.greedy_insertion(destroyed_route, removed_task_ids)
                    greedy_count += 1
                    selected_repair = 'greedy'
                elif self.repair_mode == 'regret2':
                    candidate_route = self.regret2_insertion(destroyed_route, removed_task_ids)
                    regret_count += 1
                    selected_repair = 'regret2'
                elif self.repair_mode == 'random':
                    candidate_route = self.random_insertion(destroyed_route, removed_task_ids)
                    random_count += 1
                    selected_repair = 'random'
                else:  # mixed mode
                    repair_choice = random.random()
                    if repair_choice < 0.33:
                        candidate_route = self.greedy_insertion(destroyed_route, removed_task_ids)
                        greedy_count += 1
                        selected_repair = 'greedy'
                    elif repair_choice < 0.67:
                        candidate_route = self.regret2_insertion(destroyed_route, removed_task_ids)
                        regret_count += 1
                        selected_repair = 'regret2'
                    else:
                        candidate_route = self.random_insertion(destroyed_route, removed_task_ids)
                        random_count += 1
                        selected_repair = 'random'

            # 评估成本
            candidate_cost = self.evaluate_cost(candidate_route)
            current_cost = self.evaluate_cost(current_route)

            # 计算改进量
            improvement = current_cost - candidate_cost

            # 接受准则
            is_accepted = self.accept_solution(candidate_cost, current_cost, temperature)
            is_new_best = False

            if is_accepted:
                current_route = candidate_route
                if candidate_cost < best_cost:
                    best_route = candidate_route
                    best_cost = candidate_cost
                    is_new_best = True
                    print(f"迭代 {iteration+1}: 新最优成本 {best_cost:.2f}m")

            # 更新自适应权重
            if self.use_adaptive:
                # 更新repair算子权重
                self.adaptive_repair_selector.update_weights(
                    operator=selected_repair,
                    improvement=improvement,
                    is_new_best=is_new_best,
                    is_accepted=is_accepted
                )
                # 更新destroy算子权重
                if self.adaptive_destroy_selector is not None:
                    self.adaptive_destroy_selector.update_weights(
                        operator=selected_destroy,
                        improvement=improvement,
                        is_new_best=is_new_best,
                        is_accepted=is_accepted
                    )

            # 降温
            temperature *= self.cooling_rate

            # 进度报告
            if (iteration + 1) % 50 == 0:
                print(f"  [进度] 已完成 {iteration+1}/{max_iterations} 次迭代, 当前最优: {best_cost:.2f}m")

        # 最终统计
        print(f"\n算子使用统计:")
        print(f"  Repair: Greedy={greedy_count}, Regret-2={regret_count}, Random={random_count}")
        print(f"  Destroy: Random-Removal={random_removal_count}, Partial-Removal={partial_removal_count}")
        print(f"最终最优成本: {best_cost:.2f}m (改进 {self.evaluate_cost(initial_route)-best_cost:.2f}m)")

        # 打印自适应统计
        if self.use_adaptive:
            print("\n" + "="*70)
            print("Repair算子自适应统计")
            print("="*70)
            self.adaptive_repair_selector.print_statistics()

            if self.adaptive_destroy_selector is not None:
                print("\n" + "="*70)
                print("Destroy算子自适应统计")
                print("="*70)
                self.adaptive_destroy_selector.print_statistics()

        return best_route
    
    def random_removal(self, route: Route, q: int = 2, remove_cs_prob: float = 0.3) -> Tuple[Route, List[int]]:
        """
        Destroy算子：随机移除q个任务 + 可选地移除充电站

        Week 2改进：支持充电站动态优化

        参数:
            route: 当前路径
            q: 移除的任务数量
            remove_cs_prob: 移除充电站的概率 (0.0-1.0)

        返回:
            (destroyed_route, removed_task_ids)
        """
        task_ids = route.get_served_tasks()

        if len(task_ids) < q:
            q = max(1, len(task_ids))

        if len(task_ids) == 0:
            return route.copy(), []

        removed_task_ids = random.sample(task_ids, q)

        destroyed_route = route.copy()

        # 移除任务
        for task_id in removed_task_ids:
            task = self.task_pool.get_task(task_id)
            destroyed_route.remove_task(task)

        # Week 2新增：可选地移除充电站
        if random.random() < remove_cs_prob:
            cs_nodes = [n for n in destroyed_route.nodes if n.is_charging_station()]

            if len(cs_nodes) > 0:
                # 随机决定移除多少个充电站（0-2个）
                num_to_remove = random.randint(0, min(2, len(cs_nodes)))

                if num_to_remove > 0:
                    removed_cs = random.sample(cs_nodes, num_to_remove)

                    # 移除充电站
                    for cs in removed_cs:
                        destroyed_route.nodes.remove(cs)

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

                    # Week 3新增：检查时间窗可行性
                    time_feasible, delay_cost = self._check_time_window_feasibility_fast(temp_route)
                    if not time_feasible:
                        # 违反硬时间窗，跳过
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

                        # Week 3新增：检查时间窗可行性
                        time_feasible, delay_cost = self._check_time_window_feasibility_fast(temp_route)
                        if not time_feasible:
                            # 违反硬时间窗，跳过
                            continue

                        cost_delta = repaired_route.calculate_insertion_cost_delta(
                            task,
                            (pickup_pos, delivery_pos),
                            self.distance
                        )

                        # 加入时间窗延迟成本
                        cost_delta += delay_cost

                        feasible, charging_plan = repaired_route.check_energy_feasibility_for_insertion(
                            task,
                            (pickup_pos, delivery_pos),
                            vehicle,
                            self.distance,
                            energy_config, 
                            charging_strategy=self.charging_strategy
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
        多目标成本评估（Week 2 改进）

        成本组成:
        1. 距离成本 = 总距离 × C_tr
        2. 充电成本 = 总充电量 × C_ch (从visits推导或电池模拟估算)
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

        # 2. 充电成本和时间成本（优先从visits获取，否则通过电池模拟估算）
        charging_amount = 0.0
        total_time = 0.0

        if route.visits:
            # 有visits，从visits精确计算
            for visit in route.visits:
                if visit.node.is_charging_station():
                    charged = visit.battery_after_service - visit.battery_after_travel
                    charging_amount += max(0.0, charged)
            if len(route.visits) > 0:
                total_time = route.visits[-1].departure_time - route.visits[0].arrival_time
        elif hasattr(self, 'vehicle') and hasattr(self, 'energy_config') and self.charging_strategy:
            # 没有visits，通过电池模拟估算充电量和时间
            charging_amount, total_time = self._estimate_charging_and_time(route)

        charging_cost = charging_amount * self.cost_params.C_ch
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
            vehicle_speed = 1.5  # m/s
            distance = self.distance.get_distance(
                current_node.node_id,
                next_node.node_id
            )
            travel_time = distance / vehicle_speed
            energy_consumed = energy_config.consumption_rate * travel_time
            safety_threshold_value = energy_config.safety_threshold * vehicle.battery_capacity
            # 如果当前节点是充电站，先充电
            if current_node.is_charging_station():
                # 使用充电策略决定充电量
                if self.charging_strategy:
                    # 计算剩余路径能耗（包括当前到下一节点的距离）
                    # 正确公式：energy = consumption_rate(kWh/s) * time(s)
                    remaining_energy_demand = 0.0
                    energy_to_next_stop = 0.0
                    next_stop_is_cs = False
                    for j in range(i, len(route.nodes) - 1):
                        seg_distance = self.distance.get_distance(
                            route.nodes[j].node_id,
                            route.nodes[j + 1].node_id
                        )
                        travel_time = seg_distance / vehicle_speed
                        seg_energy = energy_config.consumption_rate * travel_time
                        remaining_energy_demand += seg_energy
                        energy_to_next_stop += seg_energy

                        if route.nodes[j + 1].is_charging_station() and j >= i:
                            next_stop_is_cs = True
                            break

                    if not next_stop_is_cs:
                        # 继续累加剩余路径以便返回仓库
                        energy_to_next_stop = remaining_energy_demand

                    target_energy_demand = (energy_to_next_stop
                                            if next_stop_is_cs
                                            else remaining_energy_demand)

                    charge_amount = self.charging_strategy.determine_charging_amount(
                        current_battery=current_battery,
                        remaining_demand=target_energy_demand,
                        battery_capacity=vehicle.battery_capacity
                    )
                    current_battery = min(
                        vehicle.battery_capacity,
                        current_battery + max(0.0, charge_amount)
                    )
                else:
                    # 没有充电策略，默认充满
                    current_battery = vehicle.battery_capacity
                # 根据下一段行程的能量需求，确保最低出发电量
                min_departure_energy = energy_consumed

                # 确保具备到达下一充电站或终点的能量
                if self.charging_strategy and current_node.is_charging_station():
                    required_for_next_stop = energy_to_next_stop
                    if not next_stop_is_cs:
                        required_for_next_stop += safety_threshold_value
                    min_departure_energy = max(min_departure_energy, required_for_next_stop)
                elif not next_node.is_charging_station():
                    min_departure_energy += safety_threshold_value

                if min_departure_energy > vehicle.battery_capacity:
                    return False

                if current_battery < min_departure_energy:
                    current_battery = min(vehicle.battery_capacity, min_departure_energy)

            # 计算到下一节点的距离和能耗
            # 消耗能量前往下一节点
            current_battery -= energy_consumed


            # 检查是否电量不足
            if current_battery < 0:
                return False  # 电量不足，不可行
            
            # Week 4修复：如果刚到达充电站，应允许先充电再检查阈值
            # 之前的逻辑会在到站前执行阈值检查，导致即使成功抵达充电站
            #（电量>=0）也会因为低于警告/安全阈值被判为不可行
            # 这会让大量候选解被提前否决，使优化率显著下降。
            if next_node.is_charging_station():
                # 到达充电站后，下一轮循环会触发充电逻辑
                # 因此这里无需再进行阈值检查
                continue

            # Week 4改进：策略感知的分层临界值检查（软约束）
            # 安全层：绝对最低电量（5%），硬约束
            safety_threshold = energy_config.safety_threshold * vehicle.battery_capacity
            if current_battery < safety_threshold:
                if debug:
                    print(f"  ✗ Safety threshold violated at node {i+1}! ({current_battery:.1f} < {safety_threshold:.1f})")
                return False  # 低于安全层，绝对不可行

            # 警告层：策略感知的建议充电阈值（软约束+前瞻性检查）
            if self.charging_strategy:
                warning_threshold_ratio = self.charging_strategy.get_warning_threshold()
                warning_threshold = warning_threshold_ratio * vehicle.battery_capacity

                if current_battery < warning_threshold:
                    # 低于警告阈值，进行前瞻性检查
                    # 检查接下来是否有充电站，以及能否安全到达
                    next_cs_index = -1
                    energy_to_next_cs = 0.0

                    # 查找前方的第一个充电站（不能只看固定窗口，
                    # 否则真实存在但距离较远的充电站会被忽略，
                    # 导致错误地判定路径不可行）
                    for j in range(i + 2, len(route.nodes)):
                        if route.nodes[j].is_charging_station():
                            next_cs_index = j
                            break

                    if next_cs_index != -1:
                        # 计算到达下一个充电站需要的能量
                        for j in range(i + 1, next_cs_index):
                            seg_distance = self.distance.get_distance(
                                route.nodes[j].node_id,
                                route.nodes[j + 1].node_id
                            )
                            travel_time = seg_distance / vehicle_speed
                            energy_to_next_cs += energy_config.consumption_rate * travel_time

                        # 预估到达充电站时的电量
                        predicted_battery_at_cs = current_battery - energy_to_next_cs

                        # 如果预估电量为负，说明无法抵达下一个充电站
                        if predicted_battery_at_cs < 0:
                            return False  # 无法安全到达下一个充电站
                    else:
                        # 前方没有充电站，检查能否到达终点
                        remaining_energy_to_depot = 0.0
                        for j in range(i + 1, len(route.nodes) - 1):
                            seg_distance = self.distance.get_distance(
                                route.nodes[j].node_id,
                                route.nodes[j + 1].node_id
                            )
                            travel_time = seg_distance / vehicle_speed
                            remaining_energy_to_depot += energy_config.consumption_rate * travel_time

                        predicted_battery_at_depot = current_battery - remaining_energy_to_depot

                        # 如果无法安全到达终点，不可行
                        if predicted_battery_at_depot < 0:
                            return False  # 需要充电站但前方没有

        return True  # 整个路径可行

    def _check_time_window_feasibility_fast(self, temp_route: Route, vehicle_speed: float = 1.5) -> Tuple[bool, float]:
        """
        快速检查路径的时间窗可行性（Week 3新增）

        简化版：只检查硬时间窗违反，计算软时间窗延迟成本

        参数:
            temp_route: 待检查的路径
            vehicle_speed: 车辆速度 (m/s)，默认1.5m/s

        返回:
            Tuple[bool, float]: (是否满足硬时间窗, 延迟惩罚成本)
        """
        current_time = 0.0
        total_tardiness = 0.0

        for i in range(len(temp_route.nodes)):
            node = temp_route.nodes[i]

            # 1. 到达当前节点
            if i > 0:
                prev_node = temp_route.nodes[i - 1]
                distance = self.distance.get_distance(prev_node.node_id, node.node_id)
                travel_time = distance / vehicle_speed  # 秒
                current_time += travel_time

            # 2. 检查时间窗
            if hasattr(node, 'time_window') and node.time_window:
                tw = node.time_window

                if current_time < tw.earliest:
                    # 早到，等待
                    current_time = tw.earliest
                elif current_time > tw.latest:
                    # 晚到
                    tardiness = current_time - tw.latest

                    if tw.is_hard():
                        # 硬时间窗违反，不可行
                        return False, float('inf')
                    else:
                        # 软时间窗，累计延迟
                        total_tardiness += tardiness

            # 3. 服务时间
            service_time = node.service_time if hasattr(node, 'service_time') else 0.0
            current_time += service_time

            # 4. 充电时间（如果是充电站）
            if node.is_charging_station():
                # 简化：假设充电10秒（实际应根据充电量计算）
                if hasattr(node, 'charge_amount'):
                    charging_rate = self.energy_config.charging_rate if hasattr(self, 'energy_config') else 0.001
                    charge_time = node.charge_amount / charging_rate if charging_rate > 0 else 10.0
                    current_time += charge_time
                else:
                    current_time += 10.0  # 默认充电时间

        # 计算延迟惩罚成本
        delay_cost = total_tardiness * self.cost_params.C_delay

        return True, delay_cost

    def _find_battery_depletion_position(self, route: Route) -> int:
        """
        找到电池耗尽的位置（Week 2新增 - 第1.2步）

        返回第一个电量不足的节点索引
        如果路径可行，返回-1
        """
        if not hasattr(self, 'vehicle') or not hasattr(self, 'energy_config'):
            return -1

        vehicle = self.vehicle
        energy_config = self.energy_config
        current_battery = vehicle.battery_capacity

        for i in range(len(route.nodes) - 1):
            current_node = route.nodes[i]
            next_node = route.nodes[i + 1]

            # 如果当前节点是充电站，先充电
            if current_node.is_charging_station():
                if self.charging_strategy:
                    # 正确公式：energy = consumption_rate(kWh/s) * time(s)
                    vehicle_speed = 1.5  # m/s
                    remaining_energy_demand = 0.0
                    energy_to_next_stop = 0.0
                    next_stop_is_cs = False
                    for j in range(i, len(route.nodes) - 1):
                        seg_distance = self.distance.get_distance(
                            route.nodes[j].node_id,
                            route.nodes[j + 1].node_id
                        )
                        travel_time = seg_distance / vehicle_speed
                        seg_energy = energy_config.consumption_rate * travel_time
                        remaining_energy_demand += seg_energy
                        energy_to_next_stop += seg_energy

                        if route.nodes[j + 1].is_charging_station() and j >= i:
                            next_stop_is_cs = True
                            break

                    if not next_stop_is_cs:
                        energy_to_next_stop = remaining_energy_demand

                    target_energy_demand = (energy_to_next_stop
                                            if next_stop_is_cs
                                            else remaining_energy_demand)

                    charge_amount = self.charging_strategy.determine_charging_amount(
                        current_battery=current_battery,
                        remaining_demand=target_energy_demand,
                        battery_capacity=vehicle.battery_capacity
                    )
                    current_battery = min(vehicle.battery_capacity, current_battery + charge_amount)
                else:
                    current_battery = vehicle.battery_capacity

            # 计算到下一节点的能耗
            # 正确公式：energy = consumption_rate(kWh/s) * time(s)
            distance = self.distance.get_distance(current_node.node_id, next_node.node_id)
            vehicle_speed = 1.5  # m/s
            travel_time = distance / vehicle_speed
            energy_consumed = energy_config.consumption_rate * travel_time
            current_battery -= energy_consumed

            # 检查是否电量不足（只检查耗尽，不检查临界值）
            # Week 2注释：初始解生成时允许略微违反临界值，在ALNS优化中自然修复
            if current_battery < 0:
                return i + 1  # 返回无法到达的节点位置

        return -1  # 路径可行

    def _get_available_charging_stations(self, route: Route):
        """
        获取可用的充电站列表（Week 2新增 - 第1.2步）

        从距离矩阵中获取所有充电站节点
        排除已经在路径中的充电站
        """
        # 获取路径中已有的充电站ID
        existing_cs_ids = set(n.node_id for n in route.nodes if n.is_charging_station())

        # 从距离矩阵的coordinates中找出所有充电站
        # 充电站节点ID通常 >= 100
        available_stations = []

        if hasattr(self.distance, 'coordinates'):
            for node_id, coords in self.distance.coordinates.items():
                # 假设充电站ID >= 100，任务节点ID < 100
                if node_id >= 100 and node_id not in existing_cs_ids:
                    # 创建充电站节点
                    from core.node import create_charging_node
                    cs_node = create_charging_node(node_id=node_id, coordinates=coords)
                    available_stations.append(cs_node)

        return available_stations

    def _find_best_charging_station(self, route: Route, position: int):
        """
        找到最优的充电站插入位置（Week 2新增 - 第1.2步）

        策略：选择绕路成本最小的充电站

        参数:
            route: 当前路径
            position: 需要插入充电站的位置

        返回:
            (best_station, best_insert_pos): 最优充电站和插入位置
        """
        available_stations = self._get_available_charging_stations(route)

        if not available_stations:
            return None, None

        best_detour_cost = float('inf')
        best_station = None
        best_insert_pos = None

        # 在position前后尝试插入充电站
        for insert_pos in range(max(1, position - 2), min(len(route.nodes), position + 2)):
            if insert_pos <= 0 or insert_pos >= len(route.nodes):
                continue

            prev_node = route.nodes[insert_pos - 1]
            next_node = route.nodes[insert_pos]

            # 原始距离
            original_distance = self.distance.get_distance(prev_node.node_id, next_node.node_id)

            # 尝试每个充电站
            for station in available_stations:
                # 绕路距离 = (prev -> station) + (station -> next) - (prev -> next)
                detour_distance = (
                    self.distance.get_distance(prev_node.node_id, station.node_id) +
                    self.distance.get_distance(station.node_id, next_node.node_id) -
                    original_distance
                )

                if detour_distance < best_detour_cost:
                    best_detour_cost = detour_distance
                    best_station = station
                    best_insert_pos = insert_pos

        return best_station, best_insert_pos

    def _insert_necessary_charging_stations(self, route: Route, max_attempts: int = 10) -> Route:
        """
        自动插入必要的充电站（Week 2新增 - 第1.2步）

        策略：
        1. 检查电池可行性（包括临界值）
        2. 如果不可行，找到电量耗尽或临界位置
        3. 在该位置前插入最优充电站
        4. 重复直到路径可行或达到最大尝试次数

        Week 2修复：增加max_attempts到10，更积极地修复

        参数:
            route: 当前路径
            max_attempts: 最大尝试次数（防止无限循环）

        返回:
            修复后的路径
        """
        attempts = 0

        while attempts < max_attempts:
            # Week 2修复：优先检查完整电池可行性（包括临界值）
            if self._check_battery_feasibility(route):
                return route  # 已经可行

            # 找到电量耗尽或临界位置
            depletion_pos = self._find_battery_depletion_position(route)

            if depletion_pos == -1:
                # 找不到耗尽位置，但可行性检查失败，可能是临界值问题
                # 尝试在路径末尾附近插入充电站
                depletion_pos = len(route.nodes) - 1

            # 找到最优充电站
            best_station, best_insert_pos = self._find_best_charging_station(route, depletion_pos)

            if best_station is None or best_insert_pos is None:
                # 找不到可用的充电站，无法修复
                return route

            # 插入充电站
            route.nodes.insert(best_insert_pos, best_station)

            attempts += 1

        return route

    def _estimate_charging_and_time(self, route: Route) -> tuple[float, float]:
        """
        估算路径的充电量和总时间（当visits不可用时）

        通过模拟电池消耗和充电过程，估算：
        1. 总充电量 (kWh)
        2. 总时间 (秒或分钟，取决于time_config)

        返回:
            tuple: (total_charging_amount, total_time)
        """
        if not hasattr(self, 'vehicle') or not hasattr(self, 'energy_config'):
            return 0.0, 0.0

        vehicle = self.vehicle
        energy_config = self.energy_config
        time_config = getattr(self, 'time_config', None)

        current_battery = vehicle.battery_capacity
        total_charging = 0.0
        total_time = 0.0

        for i in range(len(route.nodes) - 1):
            current_node = route.nodes[i]
            next_node = route.nodes[i + 1]

            # 如果当前节点是充电站，计算充电量和充电时间
            if current_node.is_charging_station():
                if self.charging_strategy:
                    # 计算剩余能耗
                    # 正确公式：energy = consumption_rate(kWh/s) * time(s)
                    vehicle_speed = 1.5  # m/s
                    remaining_energy_demand = 0.0
                    energy_to_next_stop = 0.0
                    next_stop_is_cs = False
                    for j in range(i, len(route.nodes) - 1):
                        seg_distance = self.distance.get_distance(
                            route.nodes[j].node_id,
                            route.nodes[j + 1].node_id
                        )
                        travel_time = seg_distance / vehicle_speed
                        seg_energy = energy_config.consumption_rate * travel_time
                        remaining_energy_demand += seg_energy
                        energy_to_next_stop += seg_energy

                        if route.nodes[j + 1].is_charging_station() and j >= i:
                            next_stop_is_cs = True
                            break

                    if not next_stop_is_cs:
                        energy_to_next_stop = remaining_energy_demand

                    target_energy_demand = (energy_to_next_stop
                                            if next_stop_is_cs
                                            else remaining_energy_demand)

                    # 决定充电量
                    charge_amount = self.charging_strategy.determine_charging_amount(
                        current_battery=current_battery,
                        remaining_demand=target_energy_demand,
                        battery_capacity=vehicle.battery_capacity
                    )

                    # 累计充电量
                    total_charging += charge_amount

                    # 计算充电时间
                    if charge_amount > 0:
                        charging_time = charge_amount / energy_config.charging_rate
                        total_time += charging_time

                    # 更新电池
                    current_battery = min(vehicle.battery_capacity, current_battery + charge_amount)

            # 计算行驶距离和时间
            distance = self.distance.get_distance(current_node.node_id, next_node.node_id)
            # 正确公式：energy = consumption_rate(kWh/s) * time(s)
            vehicle_speed = 1.5  # m/s
            travel_time = distance / vehicle_speed
            energy_consumed = energy_config.consumption_rate * travel_time

            # 行驶时间
            if time_config:
                travel_time = distance / time_config.speed
                total_time += travel_time

            # 服务时间（如果是任务节点）
            if hasattr(current_node, 'service_time') and current_node.service_time > 0:
                total_time += current_node.service_time

            # 更新电池
            current_battery -= energy_consumed

        return total_charging, total_time

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

        # 计算总成本（使用executed route以确保包含visits）
        total_cost_value = self.evaluate_cost(route_to_analyze)

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
            'total_cost': total_cost_value,
            'cost_per_km': total_cost_value / (distance / 1000) if distance > 0 else 0
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

                        # Week 3新增：检查时间窗可行性
                        time_feasible, delay_cost = self._check_time_window_feasibility_fast(temp_route)
                        if not time_feasible:
                            # 违反硬时间窗，跳过
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

                        # 加入时间窗延迟成本
                        cost_delta += delay_cost

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

                            # Week 3新增：检查时间窗可行性
                            time_feasible, delay_cost = self._check_time_window_feasibility_fast(temp_route)
                            if not time_feasible:
                                # 违反硬时间窗，跳过
                                continue

                            cost_delta = repaired_route.calculate_insertion_cost_delta(
                                task,
                                (pickup_pos, delivery_pos),
                                self.distance
                            )

                            # 加入时间窗延迟成本
                            cost_delta += delay_cost

                            feasible, charging_plan = repaired_route.check_energy_feasibility_for_insertion(
                                task,
                                (pickup_pos, delivery_pos),
                                vehicle,
                                self.distance,
                                energy_config, 
                                charging_strategy=self.charging_strategy
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

                elif len(feasible_insertions) == 1:
                    # 只有一个可行位置，regret = inf
                    if best_regret < float('inf'):
                        best_regret = float('inf')
                        best_task_id = task_id
                        best_position = feasible_insertions[0]['position']

            # 插入regret最大的任务
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

    def random_insertion(self, route: Route, removed_task_ids: List[int]) -> Route:
        """
        随机插入算子 + 智能充电站插入 (Week 2改进 - 第1.2步)

        策略：
        1. 对每个任务，随机选择一个插入位置
        2. 插入任务后，自动插入必要的充电站
        """
        repaired_route = route.copy()

        if not hasattr(self, 'vehicle') or self.vehicle is None:
            raise ValueError("必须设置vehicle属性才能进行充电约束规划")
        if not hasattr(self, 'energy_config') or self.energy_config is None:
            raise ValueError("必须设置energy_config属性才能进行充电约束规划")

        for task_id in removed_task_ids:
            task = self.task_pool.get_task(task_id)

            # 收集所有可能的插入位置
            possible_positions = []

            for pickup_pos in range(1, len(repaired_route.nodes)):
                for delivery_pos in range(pickup_pos + 1, len(repaired_route.nodes) + 1):
                    possible_positions.append((pickup_pos, delivery_pos))

            # 随机选择一个位置
            if possible_positions:
                chosen_position = random.choice(possible_positions)
                repaired_route.insert_task(task, chosen_position)

                # Week 2新增：插入任务后，自动插入必要的充电站
                repaired_route = self._insert_necessary_charging_stations(repaired_route)

        return repaired_route