"""Tests for NHPP arrival model defaults and samplers."""

from __future__ import annotations

import random

from config.arrival_model import build_default_nhpp_model


def test_default_segment_order_is_peak_then_normal_then_offpeak():
    model = build_default_nhpp_model(
        episode_length_s=300.0,
        base_rate_per_s=1.0,
        peak_multiplier=3.0,
        normal_multiplier=1.0,
        offpeak_multiplier=0.5,
    )
    assert model.rate(10.0) == 3.0
    assert model.rate(150.0) == 1.0
    assert model.rate(290.0) == 0.5


def test_fixed_count_sampler_returns_sorted_exact_count():
    model = build_default_nhpp_model(episode_length_s=600.0)
    times = model.sample_arrivals_fixed_count(
        20,
        random.Random(7),
        horizon_s=600.0,
    )
    assert len(times) == 20
    assert times == sorted(times)
    assert all(0.0 <= value <= 600.0 for value in times)


def test_thinning_sampler_returns_times_within_horizon():
    model = build_default_nhpp_model(
        episode_length_s=600.0,
        base_rate_per_s=0.02,
    )
    times = model.sample_arrivals_thinning(
        random.Random(11),
        horizon_s=600.0,
    )
    assert times == sorted(times)
    assert all(0.0 <= value < 600.0 for value in times)
