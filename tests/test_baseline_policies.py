"""Tests for simple RuleSelectionEnv baseline policy wrappers."""

from __future__ import annotations

from strategy.baseline_policies import (
    FixedRulePolicy,
    GREEDY_FR_RULE_ID,
    GREEDY_PR_RULE_ID,
    RandomMaskedRulePolicy,
)
from strategy.rule_gating import RULE_CHARGE_OPPORTUNITY, RULE_CHARGE_TARGET


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


def test_greedy_rule_constants_match_rule_gating_ids():
    assert GREEDY_FR_RULE_ID == RULE_CHARGE_OPPORTUNITY
    assert GREEDY_PR_RULE_ID == RULE_CHARGE_TARGET


def test_random_masked_policy_fallbacks_to_zero_when_no_action_allowed():
    policy = RandomMaskedRulePolicy(seed=3)
    assert policy.select_action([False, False, False]) == 0
