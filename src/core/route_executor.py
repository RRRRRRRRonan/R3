"""
路径执行器 (Route Executor)
============================
执行路径并生成详细的visits记录，包括充电决策

功能:
- 模拟车辆沿路径行驶
- 在充电站使用充电策略决定充电量
- 生成RouteNodeVisit记录
- 计算时间、电量、负载轨迹
"""

import sys
sys.path.append('src')

from typing import List, Optional
from core.route import Route, RouteNodeVisit
from core.node import Node
from core.vehicle import Vehicle
from physics.distance import DistanceMatrix
from physics.energy import EnergyConfig
from physics.time import TimeConfig


class RouteExecutor:
    """
    路径执行器

    负责执行路径并生成详细的访问记录（visits）
    """

    def __init__(self,
                 distance_matrix: DistanceMatrix,
                 energy_config: EnergyConfig,
                 time_config: Optional[TimeConfig] = None):
        """
        初始化路径执行器

        参数:
            distance_matrix: 距离矩阵
            energy_config: 能量配置
            time_config: 时间配置（可选）
        """
        self.distance = distance_matrix
        self.energy_config = energy_config
        self.time_config = time_config or TimeConfig()

    def execute(self,
                route: Route,
                vehicle: Vehicle,
                charging_strategy=None,
                start_time: float = 0.0) -> Route:
        """
        执行路径并生成visits记录

        参数:
            route: 要执行的路径
            vehicle: 车辆对象
            charging_strategy: 充电策略（可选）
            start_time: 起始时间

        返回:
            Route: 带有visits记录的路径副本
        """
        executed_route = route.copy()
        visits = []

        # 初始状态
        current_time = start_time
        current_battery = vehicle.battery_capacity  # 满电出发
        current_load = 0.0

        for i, node in enumerate(executed_route.nodes):
            # 到达时间和电量
            if i == 0:
                # 起点（depot）
                arrival_time = current_time
                battery_after_travel = current_battery
            else:
                # 计算从上一节点到当前节点的距离和能耗
                prev_node = executed_route.nodes[i-1]
                distance = self.distance.get_distance(prev_node.node_id, node.node_id)
                travel_time = distance / self.time_config.vehicle_speed
                energy_consumed = (distance / 1000.0) * self.energy_config.consumption_rate

                arrival_time = current_time + travel_time
                battery_after_travel = current_battery - energy_consumed

            # 服务节点（更新负载和电量）
            battery_after_service = battery_after_travel
            load_after_service = current_load

            # 如果是充电站，充电
            if node.is_charging_station():
                if charging_strategy:
                    # 计算剩余路径能量需求
                    remaining_demand = self._calculate_remaining_demand(
                        executed_route, i, self.energy_config
                    )

                    # 使用充电策略决定充电量
                    charge_amount = charging_strategy.determine_charging_amount(
                        current_battery=battery_after_travel,
                        remaining_demand=remaining_demand,
                        battery_capacity=vehicle.battery_capacity
                    )

                    battery_after_service = min(
                        vehicle.battery_capacity,
                        battery_after_travel + charge_amount
                    )
                else:
                    # 没有充电策略，默认充满
                    battery_after_service = vehicle.battery_capacity

            # 如果是任务节点，更新负载
            if node.is_pickup():
                load_after_service = current_load + node.demand
            elif node.is_delivery():
                load_after_service = max(0, current_load - node.demand)

            # 服务时间
            service_duration = node.service_duration if hasattr(node, 'service_duration') else 0.0
            departure_time = arrival_time + service_duration

            # 创建visit记录
            visit = RouteNodeVisit(
                node=node,
                arrival_time=arrival_time,
                start_service_time=arrival_time,
                departure_time=departure_time,
                load_after_service=load_after_service,
                battery_after_travel=battery_after_travel,
                battery_after_service=battery_after_service
            )
            visits.append(visit)

            # 更新当前状态
            current_time = departure_time
            current_battery = battery_after_service
            current_load = load_after_service

        # 更新路径的visits
        executed_route.visits = visits

        # 检查可行性
        if any(v.battery_after_travel < 0 for v in visits):
            executed_route.is_feasible = False
            executed_route.infeasibility_info = "Battery depleted"
        elif any(v.load_after_service > vehicle.capacity for v in visits):
            executed_route.is_feasible = False
            executed_route.infeasibility_info = "Capacity exceeded"
        else:
            executed_route.is_feasible = True

        return executed_route

    def _calculate_remaining_demand(self,
                                    route: Route,
                                    current_index: int,
                                    energy_config: EnergyConfig) -> float:
        """
        计算从当前位置到终点的剩余能量需求

        参数:
            route: 路径
            current_index: 当前节点索引
            energy_config: 能量配置

        返回:
            float: 剩余能量需求（kWh）
        """
        remaining_distance = 0.0

        for j in range(current_index, len(route.nodes) - 1):
            distance = self.distance.get_distance(
                route.nodes[j].node_id,
                route.nodes[j+1].node_id
            )
            remaining_distance += distance

        remaining_energy = (remaining_distance / 1000.0) * energy_config.consumption_rate
        return remaining_energy
