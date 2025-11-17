"""Reinforcement-learning helpers for the ALNS planner."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from config import QLearningParams
from planner.epsilon_strategy import EpsilonStrategy
from planner.q_learning_init import QInitStrategy, initialize_q_table
from core.charging_context import ChargingStateLevels


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
        *,
        initial_q_values: Optional[Dict[State, Dict[Action, float]]] = None,
        state_labels: Optional[Sequence[State]] = None,
        init_strategy: QInitStrategy = QInitStrategy.ZERO,
        epsilon_strategy: Optional[EpsilonStrategy] = None,
    ) -> None:
        """Initialize Q-learning agent with configurable initialization strategy.

        Args:
            destroy_operators: Available destroy operators
            repair_operators: Available repair operators
            params: Q-learning parameters
            initial_q_values: Optional pre-defined Q-values (overrides init_strategy)
            state_labels: Optional custom state labels
            init_strategy: Q-table initialization strategy (default: ZERO)
            epsilon_strategy: Optional epsilon strategy (default: use params.initial_epsilon)
        """
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

        if state_labels:
            states = tuple(dict.fromkeys(state_labels))
        else:
            states = (
                "explore",
                "stuck",
                "deep_stuck",
            )

        if not states:
            raise ValueError("Q-learning agent requires at least one state label")

        self.states: Tuple[State, ...] = states
        self.init_strategy = init_strategy

        # Initialize Q-table using specified strategy
        if initial_q_values:
            # Use provided Q-values (e.g., for transfer learning)
            self.q_table: Dict[State, Dict[Action, float]] = {
                state: {action: 0.0 for action in self.actions} for state in self.states
            }
            for state, action_map in initial_q_values.items():
                if state not in self.q_table:
                    continue
                for action, value in action_map.items():
                    if action in self.q_table[state]:
                        self.q_table[state][action] = float(value)
        else:
            # Use initialization strategy
            self.q_table = initialize_q_table(
                states=self.states,
                actions=self.actions,
                strategy=init_strategy,
            )

        # Initialize epsilon (use strategy if provided, otherwise use params)
        self.epsilon_strategy = epsilon_strategy
        if epsilon_strategy:
            self._epsilon = epsilon_strategy.initial_epsilon
            self._epsilon_decay = epsilon_strategy.decay_rate
            self._epsilon_min = epsilon_strategy.min_epsilon
        else:
            self._epsilon = params.initial_epsilon
            self._epsilon_decay = params.epsilon_decay
            self._epsilon_min = 0.01  # Default min
        self._action_usage: Dict[Action, int] = defaultdict(int)
        self._experience_buffer: List[Tuple[State, Action, float, State]] = []

    @property
    def epsilon(self) -> float:
        """Current exploration rate."""

        return self._epsilon

    def select_action(
        self, state: State, mask: Optional[Sequence[bool]] = None
    ) -> Action:
        """Select an operator pair using an epsilon-greedy policy."""

        if state not in self.q_table:
            raise KeyError(f"Unknown state '{state}' for Q-learning agent")

        allowed_actions = self._masked_actions(mask)
        if not allowed_actions:
            raise ValueError("No valid actions available for the provided mask")

        if random.random() < self._epsilon:
            action = random.choice(allowed_actions)
        else:
            action = self._best_action(state, mask=mask)

        self._action_usage[action] += 1
        return action

    def update(self, state: State, action: Action, reward: float, next_state: State) -> None:
        """Apply the Bellman update for the executed action."""

        if state not in self.q_table or next_state not in self.q_table:
            raise KeyError("Attempted to update Q-table with unknown state")

        self._experience_buffer.append((state, action, reward, next_state))
        if not self.params.enable_online_updates:
            return

        old_q = self.q_table[state][action]
        max_future_q = max(self.q_table[next_state].values())
        self.q_table[state][action] = old_q + self.params.alpha * (
            reward + self.params.gamma * max_future_q - old_q
        )

    def decay_epsilon(self) -> None:
        """Decay the exploration rate after each iteration."""

        if self._epsilon > self._epsilon_min:
            self._epsilon = max(
                self._epsilon_min, self._epsilon * self._epsilon_decay
            )

    def set_epsilon(self, value: float) -> None:
        """Manually adjust the exploration rate (used for offline policies)."""

        self._epsilon = max(self.params.epsilon_min, min(1.0, float(value)))

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


class ChargingQLearningAgent:
    """Tabular policy for partial-charging actions (Week5 §2.1)."""

    ACTION_LEVELS: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.4, 0.6, 1.0)

    def __init__(
        self,
        params: QLearningParams,
        *,
        epsilon_strategy: Optional[EpsilonStrategy] = None,
        initial_q_values: Optional[Dict[str, Sequence[float]]] = None,
    ) -> None:
        self.params = params
        self._epsilon_strategy = epsilon_strategy
        if epsilon_strategy:
            self._epsilon = epsilon_strategy.initial_epsilon
            self._epsilon_decay = epsilon_strategy.decay_rate
            self._epsilon_min = epsilon_strategy.min_epsilon
        else:
            self._epsilon = params.initial_epsilon
            self._epsilon_decay = params.epsilon_decay
            self._epsilon_min = params.epsilon_min

        action_count = len(self.ACTION_LEVELS)
        self._q_table: Dict[str, List[float]] = {}
        if initial_q_values:
            for state, values in initial_q_values.items():
                padded = list(values)[:action_count]
                while len(padded) < action_count:
                    padded.append(0.0)
                self._q_table[state] = padded

        self._usage: Dict[int, int] = defaultdict(int)
        self._experience_buffer: List[Tuple[str, int, float, str]] = []

    @property
    def epsilon(self) -> float:
        """Current exploration rate for progress visualisation."""

        return self._epsilon

    @staticmethod
    def encode_state(levels: ChargingStateLevels) -> str:
        """Encode the 4×3×3 grid into a stable label (Week5 §2.1)."""

        return f"b{levels.battery_level}|s{levels.slack_level}|d{levels.density_level}"

    def select_action(
        self, state: str, mask: Optional[Sequence[bool]] = None
    ) -> Tuple[int, float]:
        """Sample an action index + ratio using epsilon-greedy (Week5 §2.1)."""

        self._ensure_state(state)
        valid_indices = self._mask_indices(mask)
        if not valid_indices:
            raise ValueError("ChargingQLearningAgent requires at least one valid action")

        if random.random() < self._epsilon:
            action_index = random.choice(valid_indices)
        else:
            q_values = self._q_table[state]
            best_value = max(q_values[i] for i in valid_indices)
            best_candidates = [i for i in valid_indices if q_values[i] == best_value]
            action_index = random.choice(best_candidates)

        self._usage[action_index] += 1
        return action_index, self.ACTION_LEVELS[action_index]

    def update(self, state: str, action_index: int, reward: float, next_state: str) -> None:
        """Apply Bellman update tailored to charging rewards (Week5 §2.3)."""

        self._ensure_state(state)
        self._ensure_state(next_state)
        self._experience_buffer.append((state, action_index, reward, next_state))
        if not self.params.enable_online_updates:
            return

        old_q = self._q_table[state][action_index]
        max_future = max(self._q_table[next_state])
        self._q_table[state][action_index] = old_q + self.params.alpha * (
            reward + self.params.gamma * max_future - old_q
        )

    def decay_epsilon(self) -> None:
        if self._epsilon > self._epsilon_min:
            self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)

    def statistics(self) -> Dict[str, List[Tuple[int, float, int]]]:
        """Return per-state action usage for debugging Week5 runs."""

        summary: Dict[str, List[Tuple[int, float, int]]] = {}
        for state, q_values in self._q_table.items():
            state_rows: List[Tuple[int, float, int]] = []
            for idx, q_value in enumerate(q_values):
                state_rows.append((idx, q_value, self._usage.get(idx, 0)))
            summary[state] = sorted(state_rows, key=lambda item: item[1], reverse=True)
        return summary

    def get_action_value(self, state: str, action_index: int) -> float:
        """Expose the learned value for logging/visualisation commands."""

        self._ensure_state(state)
        if action_index < 0 or action_index >= len(self.ACTION_LEVELS):
            raise IndexError("Action index out of range for charging agent")
        return self._q_table[state][action_index]

    def get_q_values(self, state: str) -> List[float]:
        """Return a copy of the Q-values for a given state label."""

        self._ensure_state(state)
        return list(self._q_table[state])

    # ------------------------------------------------------------------
    def _ensure_state(self, state: str) -> None:
        if state not in self._q_table:
            self._q_table[state] = [0.0 for _ in self.ACTION_LEVELS]

    @staticmethod
    def _mask_indices(mask: Optional[Sequence[bool]]) -> List[int]:
        if not mask:
            return list(range(len(ChargingQLearningAgent.ACTION_LEVELS)))
        return [i for i, allowed in enumerate(mask) if allowed]

    def consume_experiences(self) -> List[Tuple[str, int, float, str]]:
        """Return and clear the accumulated experience buffer."""

        experiences = list(self._experience_buffer)
        self._experience_buffer.clear()
        return experiences

    def _best_action(
        self, state: State, *, mask: Optional[Sequence[bool]] = None
    ) -> Action:
        """Return the greedy action for a given state."""

        q_values = self.q_table[state]
        if mask is None:
            candidate_items = q_values.items()
        else:
            candidate_items = [
                (action, value)
                for action, value, allowed in self._iter_masked_q_values(q_values, mask)
                if allowed
            ]
        if not candidate_items:
            raise ValueError("No valid actions when evaluating greedy policy")

        max_q = max(value for _, value in candidate_items)
        best_actions = [action for action, value in candidate_items if value == max_q]
        # Fallback: if no best actions found (e.g., due to NaN values), pick randomly from candidates
        if not best_actions:
            best_actions = [action for action, _ in candidate_items]
        return random.choice(best_actions)

    def _masked_actions(self, mask: Optional[Sequence[bool]]) -> List[Action]:
        if mask is None:
            return list(self.actions)
        if len(mask) != len(self.actions):
            raise ValueError("Action mask length mismatch")
        return [action for action, allowed in zip(self.actions, mask) if allowed]

    def _iter_masked_q_values(
        self, q_values: Dict[Action, float], mask: Sequence[bool]
    ):
        if len(mask) != len(self.actions):
            raise ValueError("Action mask length mismatch")
        for action, allowed in zip(self.actions, mask):
            yield action, q_values[action], allowed
