"""Reward delta calculation tests."""

from strategy.reward import compute_delta_cost
from strategy.state import EpisodeMetrics
from config import CostParameters


def test_compute_delta_cost_with_metrics_snapshot():
    prev = EpisodeMetrics(
        total_distance=10.0,
        total_charging=2.0,
        total_delay=1.0,
        total_conflict_waiting=0.5,
        total_standby=0.2,
        rejected_tasks=0,
    )
    new = EpisodeMetrics(
        total_distance=15.0,
        total_charging=3.0,
        total_delay=1.5,
        total_conflict_waiting=1.0,
        total_standby=0.4,
        rejected_tasks=1,
    )
    cost_params = CostParameters(
        C_tr=1.0,
        C_ch=2.0,
        C_delay=3.0,
        C_conflict=4.0,
        C_standby=5.0,
        C_missing_task=100.0,
    )

    result = compute_delta_cost(
        prev_state=None,
        new_state=None,
        action=None,
        dt=0.0,
        prev_metrics=prev,
        new_metrics=new,
        cost_params=cost_params,
    )

    assert result.travel == 5.0
    assert result.charging == 2.0
    assert result.tardiness == 1.5 * 3.0
    assert result.conflict_wait == 0.5 * 4.0
    assert result.standby == 0.2 * 5.0
    assert result.rejection == 100.0
    assert result.total == (
        result.travel
        + result.charging
        + result.tardiness
        + result.conflict_wait
        + result.standby
        + result.rejection
    )
