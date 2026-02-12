"""Robustness tests for unified evaluator edge cases."""

from __future__ import annotations

import sys
import types

from scripts import evaluate_all as eval_mod


def test_load_maskable_ppo_model_retries_on_numpy_core_mismatch(monkeypatch):
    calls = {"load": 0, "alias": 0, "ctor": 0}

    class FakeMaskablePPO:
        @staticmethod
        def load(path, **kwargs):
            calls["load"] += 1
            if calls["load"] == 1:
                raise ModuleNotFoundError("No module named 'numpy._core.numeric'")
            return {"loaded_path": path}

    fake_module = types.ModuleType("sb3_contrib")
    fake_module.MaskablePPO = FakeMaskablePPO
    monkeypatch.setitem(sys.modules, "sb3_contrib", fake_module)

    def fake_install_aliases():
        calls["alias"] += 1

    def fake_install_ctor():
        calls["ctor"] += 1

    monkeypatch.setattr(eval_mod, "_install_numpy_pickle_compat_aliases", fake_install_aliases)
    monkeypatch.setattr(eval_mod, "_install_numpy_random_pickle_ctor_compat", fake_install_ctor)

    model = eval_mod._load_maskable_ppo_model("model.zip")
    assert model == {"loaded_path": "model.zip"}
    assert calls["load"] == 2
    assert calls["alias"] == 1
    assert calls["ctor"] == 1


def test_load_maskable_ppo_model_reraises_non_numpy_error(monkeypatch):
    class FakeMaskablePPO:
        @staticmethod
        def load(path, **kwargs):
            raise ModuleNotFoundError("No module named 'torch'")

    fake_module = types.ModuleType("sb3_contrib")
    fake_module.MaskablePPO = FakeMaskablePPO
    monkeypatch.setitem(sys.modules, "sb3_contrib", fake_module)

    try:
        eval_mod._load_maskable_ppo_model("model.zip")
    except ModuleNotFoundError as exc:
        assert "torch" in str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected ModuleNotFoundError to be re-raised")


def test_load_maskable_ppo_model_injects_spaces_when_dims_provided(monkeypatch):
    captured = {}

    class FakeMaskablePPO:
        @staticmethod
        def load(path, **kwargs):
            captured["path"] = path
            captured["custom_objects"] = kwargs.get("custom_objects")
            return {"loaded_path": path}

    fake_module = types.ModuleType("sb3_contrib")
    fake_module.MaskablePPO = FakeMaskablePPO
    monkeypatch.setitem(sys.modules, "sb3_contrib", fake_module)

    model = eval_mod._load_maskable_ppo_model(
        "model.zip",
        observation_dim=17,
        action_dim=13,
    )
    assert model == {"loaded_path": "model.zip"}
    assert captured["path"] == "model.zip"
    custom = captured["custom_objects"]
    assert custom is not None
    assert custom["observation_space"].shape == (17,)
    assert custom["action_space"].n == 13


def test_install_numpy_random_pickle_ctor_compat_accepts_class_like_name(monkeypatch):
    class FakePCG64:
        pass

    calls = {"value": None}

    def original_ctor(name="MT19937"):
        calls["value"] = name
        return {"name": name}

    fake_pickle_module = types.ModuleType("numpy.random._pickle")
    fake_pickle_module.__bit_generator_ctor = original_ctor
    monkeypatch.setitem(sys.modules, "numpy.random._pickle", fake_pickle_module)

    eval_mod._install_numpy_random_pickle_ctor_compat()
    out = fake_pickle_module.__bit_generator_ctor(FakePCG64)
    assert out == {"name": "FakePCG64"}
    assert calls["value"] == "FakePCG64"


def test_get_cached_rl_model_reuses_loaded_instance(monkeypatch):
    calls = {"count": 0}

    def fake_loader(path, *, observation_dim=None, action_dim=None):
        calls["count"] += 1
        return {
            "path": path,
            "obs_dim": observation_dim,
            "act_dim": action_dim,
        }

    monkeypatch.setattr(eval_mod, "_load_maskable_ppo_model", fake_loader)
    eval_mod._RL_MODEL_CACHE.clear()

    first = eval_mod._get_cached_rl_model("model.zip", observation_dim=17, action_dim=13)
    second = eval_mod._get_cached_rl_model("model.zip", observation_dim=17, action_dim=13)
    assert first == second
    assert calls["count"] == 1
