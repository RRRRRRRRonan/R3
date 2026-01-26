"""Simulate executing a planned route to produce visit level telemetry.

``RouteExecutor`` replays a route with the configured distance, energy, and time
models, applies the active charging strategy whenever a charging node is
encountered, and records the resulting timeline in ``RouteNodeVisit`` entries.
The generated trace is used by feasibility checks, debugging tools, and the
charging strategy benchmarks.
"""

from typing import Optional, Tuple
from core.route import Route
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
        from strategy.charging_strategies import get_default_charging_strategy

        executed_route = route.copy()
        strategy = charging_strategy or get_default_charging_strategy()
        executed_route.compute_schedule(
            distance_matrix=self.distance,
            vehicle_capacity=vehicle.capacity,
            vehicle_battery_capacity=vehicle.battery_capacity,
            initial_battery=vehicle.initial_battery,
            time_config=self.time_config,
            energy_config=self.energy_config,
            charging_strategy=strategy,
            conflict_waiting_times=executed_route.conflict_waiting_times,
            standby_times=executed_route.standby_times,
        )
        if start_time != 0.0 and executed_route.visits:
            # Apply a uniform time shift when a non-zero start is requested.
            for visit in executed_route.visits:
                visit.arrival_time += start_time
                visit.start_service_time += start_time
                visit.departure_time += start_time
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
