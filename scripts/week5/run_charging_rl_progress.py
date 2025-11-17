"""Show the Week 5 charging RL optimisation process for each scenario scale."""

from __future__ import annotations

import argparse
from typing import Iterable, List, Sequence, Set

from planner.q_learning import ChargingQLearningAgent
from tests.optimization.charging_rl.utils import run_contextual_charging_trial

SCALES: Sequence[str] = ("small", "medium", "large")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise the contextual charging Q-learning behaviour (Week 5 §2)."
    )
    parser.add_argument(
        "--scale",
        choices=(*SCALES, "all"),
        default="small",
        help="Scenario scale to run. Use 'all' to iterate over small/medium/large.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of repeated executions per scale (higher = longer progress log).",
    )
    parser.add_argument(
        "--show-q-table",
        action="store_true",
        help="Print the top actions per visited state after each episode.",
    )
    return parser.parse_args()


def _format_ratio(value: float) -> str:
    return f"{value:.3f}"


def _print_episode_header(scale: str, episode: int) -> None:
    print(f"\n=== Scale: {scale} | Episode {episode} ===")


def _print_decision_log(entries: List[dict]) -> Set[str]:
    visited_states: Set[str] = set()
    if not entries:
        print("  (No charging decisions were required in this episode.)")
        return visited_states

    for idx, entry in enumerate(entries, start=1):
        visited_states.add(entry["state"])
        action_ratio = entry["action_level"]
        print(
            f"  Decision {idx}: state={entry['state']} action={action_ratio:>4.0%}"
            f" base={entry['base_charge']:.2f}kWh final={entry['final_amount']:.2f}kWh"
        )
        print(
            "    context="
            f"battery {_format_ratio(entry['battery_ratio'])} | "
            f"slack {_format_ratio(entry['time_slack_ratio'])} | "
            f"density {_format_ratio(entry['station_density'])}"
        )
        print(
            "    reward="
            f"{entry['reward']:+.3f} | updated_Q={entry['updated_q']:+.3f}"
            f" | epsilon {entry['epsilon_before']:.3f}→{entry['epsilon_after']:.3f}"
        )
    return visited_states


def _print_state_stats(agent: ChargingQLearningAgent, states: Iterable[str]) -> None:
    stats = agent.statistics()
    for state in states:
        if state not in stats:
            continue
        print(f"  Q-table snapshot for {state}:")
        for action_index, q_value, usage in stats[state][:3]:
            ratio = ChargingQLearningAgent.ACTION_LEVELS[action_index]
            print(
                f"    action {action_index} ({ratio:>4.0%}) -> Q={q_value:+.3f} | usage={usage}"
            )


def run_progress_demo(scale: str, episodes: int, show_q_table: bool) -> None:
    agent: ChargingQLearningAgent | None = None
    for episode in range(1, episodes + 1):
        result = run_contextual_charging_trial(scale, agent=agent)
        agent = result.agent
        _print_episode_header(scale, episode)
        states = _print_decision_log(result.strategy.consume_debug_log())
        if show_q_table and agent is not None:
            _print_state_stats(agent, states)


def main() -> None:
    args = _parse_args()
    scales = SCALES if args.scale == "all" else (args.scale,)
    for scale in scales:
        run_progress_demo(scale, args.episodes, args.show_q_table)


if __name__ == "__main__":
    main()
