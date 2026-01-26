"""Regression tests for the execution layer + traffic manager."""

from core.node import create_task_node_pair
from core.task import Task, TaskPool
from core.vehicle import create_vehicle
from physics.time import TimeWindow, TimeWindowType
from coordinator.traffic_manager import TrafficManager
from strategy.execution_layer import ExecutionLayer
from strategy.rules import AtomicAction
from strategy.simulator import EventDrivenSimulator


def _make_task(task_id: int, pickup_id: int, delivery_id: int, pickup_xy, delivery_xy) -> Task:
    pickup, delivery = create_task_node_pair(
        task_id=task_id,
        pickup_id=pickup_id,
        delivery_id=delivery_id,
        pickup_coords=pickup_xy,
        delivery_coords=delivery_xy,
        demand=1.0,
        service_time=5.0,
        pickup_time_window=TimeWindow(0.0, 10_000.0, TimeWindowType.SOFT),
        delivery_time_window=TimeWindow(0.0, 10_000.0, TimeWindowType.SOFT),
    )
    return Task(task_id=task_id, pickup_node=pickup, delivery_node=delivery, demand=1.0, arrival_time=0.0)


def test_conflict_waiting_and_headway_enforced():
    pool = TaskPool()
    # Two tasks share the same pickup/delivery nodes to force contention on the same edge.
    task1 = _make_task(1, 1, 2, (10.0, 0.0), (20.0, 0.0))
    task2 = _make_task(2, 1, 2, (10.0, 0.0), (20.0, 0.0))
    pool.add_tasks([task1, task2])

    vehicle1 = create_vehicle(vehicle_id=1, initial_location=(0.0, 0.0), battery_capacity=100.0, initial_battery=100.0)
    vehicle2 = create_vehicle(vehicle_id=2, initial_location=(0.0, 0.0), battery_capacity=100.0, initial_battery=100.0)

    traffic = TrafficManager(headway_s=2.0)
    simulator = EventDrivenSimulator(task_pool=pool, vehicles=[vehicle1, vehicle2], traffic_manager=traffic)
    executor = ExecutionLayer(task_pool=pool, simulator=simulator, traffic_manager=traffic)

    state = simulator.build_state()
    action1 = AtomicAction(kind="DISPATCH", payload={"robot_id": 1, "task_id": 1, "mode": "dispatch"})
    executor.execute(action1, state, None)

    state = simulator.build_state()
    action2 = AtomicAction(kind="DISPATCH", payload={"robot_id": 2, "task_id": 2, "mode": "dispatch"})
    executor.execute(action2, state, None)

    assert state.metrics.total_conflict_waiting > 0.0

    edge_key = (0, task1.pickup_node.node_id)
    reservations = traffic._edge_table.get(edge_key, [])
    assert len(reservations) >= 2
    assert reservations[1].start >= reservations[0].end + traffic.headway_s - 1e-6
