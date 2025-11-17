"""Simulate executing a planned route to produce visit level telemetry.

``RouteExecutor`` replays a route with the configured distance, energy, and time
models, applies the active charging strategy whenever a charging node is
encountered, and records the resulting timeline in ``RouteNodeVisit`` entries.
The generated trace is used by feasibility checks, debugging tools, and the
charging strategy benchmarks.
"""

from typing import List, Optional, Tuple
from core.route import Route, RouteNodeVisit
from core.node import Node
from core.vehicle import Vehicle
from core.charging_context import (
    ChargingContext,
    ChargingContextDiscretizer,
    ChargingStateLevels,
)
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
        self.context_discretizer = ChargingContextDiscretizer(self.energy_config)

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
                energy_consumed = self.energy_config.consumption_rate * travel_time

                arrival_time = current_time + travel_time
                battery_after_travel = current_battery - energy_consumed

            # 服务节点（更新负载和电量）
            battery_after_service = battery_after_travel
            load_after_service = current_load

            # 如果是充电站，充电
            if node.is_charging_station():
                if charging_strategy:
                    # 计算剩余路径能量需求
                    remaining_demand, demand_to_next_cs, has_next_cs = self._calculate_remaining_demand(
                        executed_route, i, self.energy_config
                    )

                    context_raw = self._build_charging_context(
                        route=executed_route,
                        current_index=i,
                        current_battery=battery_after_travel,
                        arrival_time=arrival_time,
                        remaining_demand=remaining_demand,
                        demand_to_next_cs=demand_to_next_cs,
                        has_next_station=has_next_cs,
                        vehicle=vehicle,
                    )
                    context_levels = self.context_discretizer.discretize(context_raw)

                    target_demand = self._select_demand_target(
                        remaining_demand=remaining_demand,
                        demand_to_next_cs=demand_to_next_cs,
                        has_next_station=has_next_cs,
                        context_levels=context_levels,
                    )

                    context = ChargingContext(
                        battery_ratio=context_raw.battery_ratio,
                        demand_ratio=0.0 if vehicle.battery_capacity <= 0 else min(
                            1.0, target_demand / vehicle.battery_capacity
                        ),
                        time_slack_ratio=context_raw.time_slack_ratio,
                        station_density=context_raw.station_density,
                    )

                    # 使用充电策略决定充电量
                    charge_amount = charging_strategy.determine_charging_amount(
                        current_battery=battery_after_travel,
                        remaining_demand=target_demand,
                        battery_capacity=vehicle.battery_capacity,
                        context=context,
                        context_levels=context_levels,
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
                                    energy_config: EnergyConfig) -> Tuple[float, float, bool]:
        """
        计算从当前位置到终点的剩余能量需求

        参数:
            route: 路径
            current_index: 当前节点索引
            energy_config: 能量配置

        返回:
            Tuple[float, float, bool]: (剩余能量需求, 到下一充电站的能量需求, 是否存在后续充电站)
        """
        total_remaining_energy = 0.0
        energy_to_next_station = 0.0
        next_station_found = False

        for j in range(current_index, len(route.nodes) - 1):
            distance = self.distance.get_distance(
                route.nodes[j].node_id,
                route.nodes[j+1].node_id
            )
            travel_time = distance / self.time_config.vehicle_speed
            segment_energy = energy_config.consumption_rate * travel_time

            total_remaining_energy += segment_energy

            if not next_station_found:
                energy_to_next_station += segment_energy

            if route.nodes[j+1].is_charging_station() and j >= current_index:
                next_station_found = True
                break

        if not next_station_found:
            energy_to_next_station = total_remaining_energy

        return total_remaining_energy, energy_to_next_station, next_station_found

    # ------------------------------------------------------------------
    def _build_charging_context(self,
                                route: Route,
                                current_index: int,
                                current_battery: float,
                                arrival_time: float,
                                remaining_demand: float,
                                demand_to_next_cs: float,
                                has_next_station: bool,
                                vehicle: Vehicle) -> ChargingContext:
        battery_ratio = 0.0 if vehicle.battery_capacity <= 0 else current_battery / vehicle.battery_capacity
        battery_ratio = max(0.0, min(1.0, battery_ratio))

        target_demand = demand_to_next_cs if has_next_station else remaining_demand
        demand_ratio = 0.0 if vehicle.battery_capacity <= 0 else target_demand / vehicle.battery_capacity
        demand_ratio = max(0.0, min(1.0, demand_ratio))

        time_slack_ratio = self._estimate_time_slack_ratio(route, current_index, arrival_time)

        remaining_nodes = route.nodes[current_index + 1:]
        if not remaining_nodes:
            station_density = 0.0
        else:
            remaining_charging = sum(1 for n in remaining_nodes if n.is_charging_station())
            remaining_customers = sum(1 for n in remaining_nodes if not n.is_charging_station()) or 1
            station_density = remaining_charging / remaining_customers

        return ChargingContext(
            battery_ratio=battery_ratio,
            demand_ratio=demand_ratio,
            time_slack_ratio=time_slack_ratio,
            station_density=station_density,
        )

    def _estimate_time_slack_ratio(self,
                                   route: Route,
                                   current_index: int,
                                   arrival_time: float) -> float:
        for node in route.nodes[current_index + 1:]:
            time_window = getattr(node, "time_window", None)
            if time_window is None:
                continue

            width = max(1.0, time_window.latest - time_window.earliest)
            slack = (time_window.latest - arrival_time) / width
            return max(-1.0, min(1.0, slack))

        # 没有时间窗约束时视为宽松
        return 1.0

    def _select_demand_target(self,
                              remaining_demand: float,
                              demand_to_next_cs: float,
                              has_next_station: bool,
                              context_levels: ChargingStateLevels) -> float:
        # 密度高：优先只补到下一站；密度低：补到终点
        if context_levels.density_level >= 2 and has_next_station:
            return demand_to_next_cs
        return remaining_demand
