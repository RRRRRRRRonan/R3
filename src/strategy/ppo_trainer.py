"""PPO training utilities for rule-selection RL."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from strategy.rule_env import RuleSelectionEnv


def _default_ppo_kwargs() -> dict[str, object]:
    # Tuned defaults for MaskablePPO. Returned via a factory to avoid sharing
    # mutable nested objects (e.g., policy network architecture).
    return {
        "gamma": 1.0,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "learning_rate": 3e-4,
        "batch_size": 256,
        "n_epochs": 4,
        "ent_coef": 0.01,
        "n_steps": 2048,
        "policy_kwargs": {"net_arch": [256, 128]},
    }


@dataclass(frozen=True)
class PPOTrainingConfig:
    """Configuration for MaskablePPO training."""

    total_timesteps: int = 50_000
    seed: int = 42
    policy: str = "MlpPolicy"
    log_dir: Optional[str] = None
    eval_freq: int = 2_000
    eval_episodes: int = 5
    deterministic_eval: bool = True
    ppo_kwargs: dict[str, object] = field(default_factory=_default_ppo_kwargs)


def make_masked_env(core_env: RuleSelectionEnv):
    """Wrap the rule-selection env with Gymnasium + action masking."""

    _, _, ActionMasker = _require_maskable_ppo()
    from strategy.gym_env import RuleSelectionGymEnv

    gym_env = RuleSelectionGymEnv(core_env)
    return ActionMasker(gym_env, lambda env: env.action_masks())


def train_maskable_ppo(
    train_env,
    eval_env=None,
    *,
    config: Optional[PPOTrainingConfig] = None,
):
    """Train a MaskablePPO agent and optionally run periodic evaluation."""

    MaskablePPO, MaskableEvalCallback, _ = _require_maskable_ppo()
    config = config or PPOTrainingConfig()

    callback = None
    save_dir = None
    if config.log_dir:
        save_dir = Path(config.log_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    if eval_env is not None and config.eval_freq > 0:
        callback = MaskableEvalCallback(
            eval_env,
            best_model_save_path=str(save_dir / "best_model") if save_dir else None,
            log_path=str(save_dir / "eval_logs") if save_dir else None,
            eval_freq=config.eval_freq,
            n_eval_episodes=config.eval_episodes,
            deterministic=config.deterministic_eval,
        )

    model = MaskablePPO(
        config.policy,
        train_env,
        verbose=1,
        seed=config.seed,
        **config.ppo_kwargs,
    )
    model.learn(total_timesteps=config.total_timesteps, callback=callback)
    if save_dir:
        model.save(str(save_dir / "final_model"))
    return model


def _require_maskable_ppo():
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "torch is required for PPO training. Install via `python3 -m pip install torch`."
        ) from exc

    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
        from sb3_contrib.common.maskable.wrappers import ActionMasker
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "sb3-contrib (and stable-baselines3) is required. "
            "Install via `python3 -m pip install sb3-contrib stable-baselines3`."
        ) from exc

    return MaskablePPO, MaskableEvalCallback, ActionMasker


__all__ = [
    "PPOTrainingConfig",
    "make_masked_env",
    "train_maskable_ppo",
]
