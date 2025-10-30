"""Reinforcement-learning helpers for the ALNS planner."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from config import QLearningParams


State = str
Action = Tuple[str, str]


@dataclass
class QLearningActionStats:
    """Expose usage and value statistics for an operator pair."""

    action: Action
    total_usage: int
    average_q_value: float


class QLearningOperatorAgent:
    """Tabular Q-learning agent that selects destroy/repair operator pairs."""

    def __init__(
        self,
        destroy_operators: Iterable[str],
        repair_operators: Sequence[str],
        params: QLearningParams,
    ) -> None:
        self.params = params
        self.destroy_operators: List[str] = list(destroy_operators)
        self.repair_operators: List[str] = list(repair_operators)
        if not self.destroy_operators or not self.repair_operators:
            raise ValueError("destroy and repair operator sets must not be empty")

        self.actions: List[Action] = [
            (destroy, repair)
            for destroy in self.destroy_operators
            for repair in self.repair_operators
        ]

        self.states: Tuple[State, State, State] = (
            "explore",
            "stuck",
            "deep_stuck",
        )

        self.q_table: Dict[State, Dict[Action, float]] = {
            state: {action: 0.0 for action in self.actions} for state in self.states
        }
        self._epsilon = params.initial_epsilon
        self._action_usage: Dict[Action, int] = defaultdict(int)

    @property
    def epsilon(self) -> float:
        """Current exploration rate."""

        return self._epsilon

    def select_action(self, state: State) -> Action:
        """Select an operator pair using an epsilon-greedy policy."""

        if state not in self.q_table:
            raise KeyError(f"Unknown state '{state}' for Q-learning agent")

        if random.random() < self._epsilon:
            action = random.choice(self.actions)
        else:
            action = self._best_action(state)

        self._action_usage[action] += 1
        return action

    def update(self, state: State, action: Action, reward: float, next_state: State) -> None:
        """Apply the Bellman update for the executed action."""

        if state not in self.q_table or next_state not in self.q_table:
            raise KeyError("Attempted to update Q-table with unknown state")

        old_q = self.q_table[state][action]
        max_future_q = max(self.q_table[next_state].values())
        self.q_table[state][action] = old_q + self.params.alpha * (
            reward + self.params.gamma * max_future_q - old_q
        )

    def decay_epsilon(self) -> None:
        """Decay the exploration rate after each iteration."""

        if self._epsilon > self.params.epsilon_min:
            self._epsilon = max(
                self.params.epsilon_min, self._epsilon * self.params.epsilon_decay
            )

    def statistics(self) -> Dict[str, List[QLearningActionStats]]:
        """Aggregate usage and Q-value statistics per state."""

        stats: Dict[str, List[QLearningActionStats]] = {}
        for state, q_values in self.q_table.items():
            state_stats: List[QLearningActionStats] = []
            for action, q_value in q_values.items():
                usage = self._action_usage[action]
                state_stats.append(
                    QLearningActionStats(
                        action=action,
                        total_usage=usage,
                        average_q_value=q_value,
                    )
                )
            stats[state] = sorted(
                state_stats,
                key=lambda stat: stat.average_q_value,
                reverse=True,
            )
        return stats

    def format_statistics(self) -> str:
        """Format statistics into a human-readable string."""

        rows = [f"epsilon={self._epsilon:.3f}"]
        stats = self.statistics()
        for state in self.states:
            rows.append(f"状态 {state}:")
            for stat in stats[state][:3]:
                destroy, repair = stat.action
                rows.append(
                    f"  ({destroy}, {repair}) -> 使用 {stat.total_usage} 次, Q={stat.average_q_value:>7.3f}"
                )
        return "\n".join(rows)

    def _best_action(self, state: State) -> Action:
        """Return the greedy action for a given state."""

        q_values = self.q_table[state]
        max_q = max(q_values.values())
        best_actions = [action for action, value in q_values.items() if value == max_q]
        return random.choice(best_actions)
