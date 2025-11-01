import math

from core.node import DepotNode, TaskNode, NodeType
from core.route import Route, RouteNodeVisit


def build_sample_route() -> Route:
    depot_start = DepotNode((0.0, 0.0))
    pickup = TaskNode(
        node_id=1,
        coordinates=(1.0, 0.0),
        node_type=NodeType.PICKUP,
        task_id=1,
        service_time=5.0,
        demand=2.0,
    )
    delivery = TaskNode(
        node_id=2,
        coordinates=(2.0, 0.0),
        node_type=NodeType.DELIVERY,
        task_id=1,
        service_time=4.0,
        demand=2.0,
    )
    depot_end = DepotNode((0.0, 0.0))

    route = Route(
        vehicle_id=42,
        nodes=[depot_start, pickup, delivery, depot_end],
        visits=[
            RouteNodeVisit(node=depot_start, departure_time=0.0),
            RouteNodeVisit(node=pickup, arrival_time=1.0, departure_time=6.0, load_after_service=2.0),
            RouteNodeVisit(node=delivery, arrival_time=7.0, departure_time=11.0),
            RouteNodeVisit(node=depot_end, arrival_time=12.5, departure_time=12.5),
        ],
        is_feasible=True,
    )
    return route


def test_copy_without_visits_only_duplicates_node_sequence():
    original = build_sample_route()

    clone = original.copy()

    assert clone is not original
    assert clone.nodes is not original.nodes
    assert clone.nodes == original.nodes
    assert clone.visits is None

    # Nodes are immutable so the copy reuses the instances safely.
    for original_node, cloned_node in zip(original.nodes, clone.nodes):
        assert cloned_node is original_node


def test_copy_with_visits_duplicates_visit_entries():
    original = build_sample_route()

    clone = original.copy(include_visits=True)

    assert clone.visits is not None
    assert clone.visits is not original.visits
    assert len(clone.visits) == len(original.visits)

    for source_visit, cloned_visit in zip(original.visits, clone.visits):
        assert cloned_visit is not source_visit
        assert cloned_visit.node is source_visit.node
        assert cloned_visit.arrival_time == source_visit.arrival_time
        assert cloned_visit.departure_time == source_visit.departure_time

    # Mutating the cloned visits must not affect the original route.
    clone.visits[0].departure_time = math.pi
    assert not math.isclose(original.visits[0].departure_time, math.pi)
