"""Simple baseline policies for RuleSelectionEnv evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from strategy.rule_gating import RULE_CHARGE_OPPORTUNITY, RULE_CHARGE_TARGET_HIGH


# Screenshot Section 4 baselines:
# - Greedy-FR: lock to opportunity charging
# - Greedy-PR: lock to HIGH target charge depth (80%)
GREEDY_FR_RULE_ID = RULE_CHARGE_OPPORTUNITY
GREEDY_PR_RULE_ID = RULE_CHARGE_TARGET_HIGH


@dataclass(frozen=True)
class FixedRulePolicy:
    """Policy that always emits the same rule id.

    RuleSelectionEnv handles mask conflicts internally and falls back to an
    allowed rule when needed.
    """

    rule_id: int

    def select_action(self, mask: Sequence[bool]) -> int:
        _ = mask
        return int(self.rule_id)


class RandomMaskedRulePolicy:
    """Policy that samples uniformly from currently allowed action indices."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def select_action(self, mask: Sequence[bool]) -> int:
        allowed = [idx for idx, is_allowed in enumerate(mask) if is_allowed]
        if not allowed:
            return 0
        return int(self._rng.choice(allowed))


__all__ = [
    "FixedRulePolicy",
    "GREEDY_FR_RULE_ID",
    "GREEDY_PR_RULE_ID",
    "RandomMaskedRulePolicy",
]
