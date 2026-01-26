"""Gymnasium wrapper for the rule-selection environment."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "gymnasium is required for RuleSelectionGymEnv. Install via `pip install gymnasium`."
    ) from exc

from strategy.rule_env import RuleSelectionEnv, StepResult
from strategy.state import EVENT_TYPES


class RuleSelectionGymEnv(gym.Env):
    """Wrap RuleSelectionEnv to a Gymnasium-compatible API."""

    metadata = {"render_modes": []}

    def __init__(self, core_env: RuleSelectionEnv):
        super().__init__()
        self.core_env = core_env
        self.num_vehicles = len(core_env.simulator.vehicles)
        self.top_k_tasks = core_env.top_k_tasks
        self.top_k_chargers = core_env.top_k_chargers

        self.action_space = spaces.Discrete(13)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim(),),
            dtype=np.float32,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        obs, info = self.core_env.reset()
        return self._flatten_obs(obs), info

    def step(self, action: int):
        result: StepResult = self.core_env.step(action)
        obs = self._flatten_obs(result.obs)
        return obs, result.reward, result.terminated, result.truncated, result.info

    def action_masks(self) -> Sequence[bool]:
        return self.core_env.action_masks()

    def _flatten_obs(self, obs: Dict[str, List[float]]) -> np.ndarray:
        values = obs.get("vehicles", []) + obs.get("tasks", []) + obs.get("chargers", []) + obs.get("meta", [])
        return np.asarray(values, dtype=np.float32)

    def _obs_dim(self) -> int:
        vehicle_dim = 6 * self.num_vehicles
        task_dim = 4 * self.top_k_tasks
        charger_dim = 3 * self.top_k_chargers
        meta_dim = 1 + len(EVENT_TYPES)
        return vehicle_dim + task_dim + charger_dim + meta_dim


__all__ = ["RuleSelectionGymEnv"]
