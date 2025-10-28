"""Tests for the fleet-level ALNS coordinator."""

from core.node import DepotNode, create_task_node_pair
from core.task import Task, TaskPool, TaskStatus
from core.vehicle import create_vehicle
from physics.distance import DistanceMatrix
from physics.energy import EnergyConfig
from planner.fleet import FleetPlanner


def _build_tasks(num_tasks: int):
    depot = DepotNode(coordinates=(0.0, 0.0))
    tasks = []
    coordinates = {0: depot.coordinates}
    next_node_id = 1

    for idx in range(num_tasks):
        x_offset = 50.0 * (idx + 1)
        pickup, delivery = create_task_node_pair(
            task_id=idx + 1,
            pickup_id=next_node_id,
            delivery_id=next_node_id + 1,
            pickup_coords=(x_offset, 0.0),
            delivery_coords=(x_offset, 40.0),
            demand=10.0,
        )
        next_node_id += 2
        tasks.append(
            Task(
                task_id=idx + 1,
                pickup_node=pickup,
                delivery_node=delivery,
                demand=10.0,
            )
        )
        coordinates[pickup.node_id] = pickup.coordinates
        coordinates[delivery.node_id] = delivery.coordinates

    distance = DistanceMatrix(
        coordinates=coordinates,
        num_tasks=num_tasks,
        num_charging_stations=0,
    )

    return depot, tasks, distance


def test_fleet_planner_balances_tasks_across_vehicles():
    depot, tasks, distance = _build_tasks(num_tasks=4)
    task_pool = TaskPool()
    task_pool.add_tasks(tasks)

    vehicles = [
        create_vehicle(1, capacity=100.0, battery_capacity=150.0),
        create_vehicle(2, capacity=100.0, battery_capacity=150.0),
    ]

    planner = FleetPlanner(
        distance_matrix=distance,
        depot=depot,
        vehicles=vehicles,
        task_pool=task_pool,
        energy_config=EnergyConfig(consumption_rate=0.2, charging_rate=10.0, battery_capacity=150.0),
        repair_mode="greedy",
        use_adaptive=False,
        verbose=False,
    )

    result = planner.plan_routes(max_iterations=10)

    assert set(result.routes.keys()) == {1, 2}
    assert set(result.initial_routes.keys()) == {1, 2}
    assert result.initial_cost >= result.optimised_cost

    total_pickups = 0
    for vehicle in vehicles:
        route = result.routes[vehicle.vehicle_id]
        total_pickups += len(route.get_pickup_nodes())
        feasible, _ = route.check_capacity_feasibility(vehicle.capacity)
        assert feasible

    assert total_pickups == len(tasks)

    assigned_statuses = {
        tracker.assigned_vehicle_id
        for tracker in task_pool.trackers.values()
        if tracker.status == TaskStatus.ASSIGNED
    }
    assert assigned_statuses == {1, 2}


def test_fleet_planner_handles_more_vehicles_than_tasks():
    depot, tasks, distance = _build_tasks(num_tasks=2)
    task_pool = TaskPool()
    task_pool.add_tasks(tasks)

    vehicles = [
        create_vehicle(1, capacity=100.0, battery_capacity=120.0),
        create_vehicle(2, capacity=100.0, battery_capacity=120.0),
        create_vehicle(3, capacity=100.0, battery_capacity=120.0),
    ]

    planner = FleetPlanner(
        distance_matrix=distance,
        depot=depot,
        vehicles=vehicles,
        task_pool=task_pool,
        energy_config=EnergyConfig(consumption_rate=0.2, charging_rate=10.0, battery_capacity=120.0),
        repair_mode="greedy",
        use_adaptive=False,
        verbose=False,
    )

    result = planner.plan_routes(max_iterations=5)

    assert set(result.routes.keys()) == {1, 2, 3}
    assert set(result.initial_routes.keys()) == {1, 2, 3}
    assert len(result.unassigned_tasks) == 0
    assert result.initial_cost >= result.optimised_cost

    empty_routes = [route for route in result.routes.values() if route.is_empty()]
    assert len(empty_routes) == 1  # one vehicle should remain idle
