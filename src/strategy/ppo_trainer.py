"""PPO training utilities for rule-selection RL."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from strategy.rule_env import RuleSelectionEnv


def _default_ppo_kwargs() -> dict[str, object]:
    # Tuned defaults for MaskablePPO. Returned via a factory to avoid sharing
    # mutable nested objects (e.g., policy network architecture).
    return {
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "learning_rate": 3e-4,
        "batch_size": 256,
        "n_epochs": 4,
        "ent_coef": 0.05,
        "n_steps": 4096,
        "policy_kwargs": {"net_arch": [256, 128]},
    }


@dataclass(frozen=True)
class PPOTrainingConfig:
    """Configuration for MaskablePPO training."""

    total_timesteps: int = 1_000_000
    seed: int = 42
    policy: str = "MlpPolicy"
    log_dir: Optional[str] = None
    eval_freq: int = 50_000
    eval_episodes: int = 3
    deterministic_eval: bool = False
    use_vec_normalize: bool = True
    vec_norm_obs: bool = True
    vec_norm_reward: bool = True
    vec_clip_obs: float = 10.0
    vec_clip_reward: float = 10.0
    vec_norm_epsilon: float = 1e-8
    ppo_kwargs: dict[str, object] = field(default_factory=_default_ppo_kwargs)
    load_model: Optional[str] = None
    load_vecnormalize: Optional[str] = None


def make_masked_env(core_env: RuleSelectionEnv):
    """Wrap the rule-selection env with Gymnasium + action masking."""

    _, _, ActionMasker = _require_maskable_ppo()
    from strategy.gym_env import RuleSelectionGymEnv

    gym_env = RuleSelectionGymEnv(core_env)
    return ActionMasker(gym_env, lambda env: env.action_masks())

def _make_vecnorm_saving_callback(
    base_callback_cls,
    vec_normalize_env,
    best_model_save_path: str | None,
):
    """Create a callback subclass that saves VecNormalize stats alongside best_model.

    This fixes a critical bug where best_model.zip was saved mid-training but
    vecnormalize.pkl was only saved at the END of training, causing observation
    distribution mismatch during evaluation.
    """
    if vec_normalize_env is None or best_model_save_path is None:
        return base_callback_cls

    class _VecNormSavingEvalCallback(base_callback_cls):
        """MaskableEvalCallback that also persists VecNormalize running stats."""

        def __init__(self, *args, _vec_normalize_env=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._vec_normalize_env = _vec_normalize_env
            self._best_model_dir = kwargs.get("best_model_save_path") or (
                args[1] if len(args) > 1 else None
            )

        def _on_step(self) -> bool:
            result = super()._on_step()
            # After parent's _on_step, check if a new best model was saved.
            # MaskableEvalCallback sets self.best_mean_reward when saving.
            if (
                self._vec_normalize_env is not None
                and self._best_model_dir is not None
            ):
                best_model_path = Path(self._best_model_dir) / "best_model.zip"
                vecnorm_path = Path(self._best_model_dir) / "vecnormalize.pkl"
                # Save vecnorm if best_model is newer than vecnorm (or vecnorm doesn't exist)
                if best_model_path.exists():
                    save_needed = not vecnorm_path.exists() or (
                        best_model_path.stat().st_mtime > vecnorm_path.stat().st_mtime
                    )
                    if save_needed:
                        self._vec_normalize_env.save(str(vecnorm_path))
            return result

    return _VecNormSavingEvalCallback

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

    train_vec_env, eval_vec_env, vec_normalize_env = _prepare_vec_envs(
        train_env=train_env,
        eval_env=eval_env,
        config=config,
        load_vecnormalize=config.load_vecnormalize,
    )

    if eval_vec_env is not None and config.eval_freq > 0:
        best_model_path = str(save_dir / "best_model") if save_dir else None
        CallbackCls = _make_vecnorm_saving_callback(
            MaskableEvalCallback, vec_normalize_env, best_model_path,
        )
        callback = CallbackCls(
            eval_vec_env,
            best_model_save_path=best_model_path,
            log_path=str(save_dir / "eval_logs") if save_dir else None,
            eval_freq=config.eval_freq,
            n_eval_episodes=config.eval_episodes,
            deterministic=config.deterministic_eval,
            _vec_normalize_env=vec_normalize_env,
        )

    if config.load_model:
        print(f"[fine-tune] Loading pretrained model from {config.load_model}")
        model = MaskablePPO.load(
            config.load_model,
            env=train_vec_env,
            verbose=1,
            seed=config.seed,
            **config.ppo_kwargs,
        )
    else:
        model = MaskablePPO(
            config.policy,
            train_vec_env,
            verbose=1,
            seed=config.seed,
            **config.ppo_kwargs,
        )
    model.learn(total_timesteps=config.total_timesteps, callback=callback)
    if save_dir:
        model.save(str(save_dir / "final_model"))
        if vec_normalize_env is not None:
            vec_normalize_env.save(str(save_dir / "vecnormalize.pkl"))
    return model


def _prepare_vec_envs(
    *,
    train_env,
    eval_env,
    config: PPOTrainingConfig,
    load_vecnormalize: Optional[str] = None,
):
    from stable_baselines3.common.vec_env import VecNormalize

    train_vec_env = _ensure_vec_env(train_env)
    eval_vec_env = _ensure_vec_env(eval_env) if eval_env is not None else None

    if not config.use_vec_normalize:
        return train_vec_env, eval_vec_env, None

    gamma = float(config.ppo_kwargs.get("gamma", 0.99))

    if load_vecnormalize:
        print(f"[fine-tune] Loading VecNormalize stats from {load_vecnormalize}")
        train_vec_env = VecNormalize.load(load_vecnormalize, train_vec_env)
        train_vec_env.training = True
        train_vec_env.gamma = gamma
    else:
        train_vec_env = VecNormalize(
            train_vec_env,
            training=True,
            norm_obs=config.vec_norm_obs,
            norm_reward=config.vec_norm_reward,
            clip_obs=float(config.vec_clip_obs),
            clip_reward=float(config.vec_clip_reward),
            gamma=gamma,
            epsilon=float(config.vec_norm_epsilon),
        )
    if eval_vec_env is not None:
        eval_vec_env = VecNormalize(
            eval_vec_env,
            training=False,
            norm_obs=config.vec_norm_obs,
            norm_reward=False,
            clip_obs=float(config.vec_clip_obs),
            clip_reward=float(config.vec_clip_reward),
            gamma=gamma,
            epsilon=float(config.vec_norm_epsilon),
        )
        eval_vec_env.obs_rms = train_vec_env.obs_rms
    return train_vec_env, eval_vec_env, train_vec_env


def _ensure_vec_env(env):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

    if isinstance(env, VecEnv):
        return env
    return DummyVecEnv([lambda: env])


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
        try:
            # Newer sb3-contrib versions expose ActionMasker here.
            from sb3_contrib.common.wrappers.action_masker import ActionMasker
        except ImportError:
            # Backward compatibility for older layouts.
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
