"""Tests for simple RuleSelectionEnv baseline policy wrappers."""

from __future__ import annotations

from strategy.baseline_policies import (
    FixedRulePolicy,
    GREEDY_FR_RULE_ID,
    GREEDY_PR_RULE_ID,
    RandomMaskedRulePolicy,
)


def test_fixed_policy_returns_locked_rule():
    fr = FixedRulePolicy(rule_id=GREEDY_FR_RULE_ID)
    pr = FixedRulePolicy(rule_id=GREEDY_PR_RULE_ID)
    mask = [False] * 13
    assert fr.select_action(mask) == GREEDY_FR_RULE_ID
    assert pr.select_action(mask) == GREEDY_PR_RULE_ID


def test_random_masked_policy_selects_allowed_action_index():
    policy = RandomMaskedRulePolicy(seed=7)
    mask = [False, True, False, True]
    samples = {policy.select_action(mask) for _ in range(20)}
    assert samples.issubset({1, 3})
