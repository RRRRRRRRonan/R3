"""Q-table initialization strategies for reinforcement learning.

This module provides different initialization strategies for Q-tables
in Q-learning agents. Different initialization strategies can significantly
affect the exploration-exploitation balance and convergence speed.
"""

from enum import Enum
from typing import Callable, Dict, List, Tuple

# Type aliases
State = str
Action = Tuple[str, str]  # (destroy_operator, repair_operator)


class QInitStrategy(Enum):
    """Q-table initialization strategies.

    Attributes:
        ZERO: Initialize all Q-values to 0.0 (current baseline)
        UNIFORM: Initialize all Q-values to a uniform positive bias
        ACTION_SPECIFIC: Initialize based on operator type (higher for matheuristic)
        STATE_SPECIFIC: Initialize based on state (higher for stuck states)
    """

    ZERO = "zero"
    UNIFORM = "uniform"
    ACTION_SPECIFIC = "action_specific"
    STATE_SPECIFIC = "state_specific"


def init_zero(
    state: State,
    action: Action,
    states: Tuple[State, ...],
    actions: List[Action],
) -> float:
    """Strategy A: Zero initialization (current baseline method).

    Initializes all Q-values to 0.0. This is the current implementation
    but provides no exploration bias.

    Args:
        state: Current state label
        action: Action tuple (destroy_op, repair_op)
        states: All possible states
        actions: All possible actions

    Returns:
        Q-value of 0.0
    """
    return 0.0


def init_uniform(
    state: State,
    action: Action,
    states: Tuple[State, ...],
    actions: List[Action],
    bias: float = 50.0,
) -> float:
    """Strategy B: Uniform positive bias.

    Initializes all Q-values to a uniform positive value, encouraging
    exploration of all actions equally in the early phase.

    Args:
        state: Current state label
        action: Action tuple (destroy_op, repair_op)
        states: All possible states
        actions: All possible actions
        bias: Initial Q-value for all state-action pairs (default: 50.0)

    Returns:
        Uniform bias value

    Theory:
        Positive bias encourages exploration because all actions appear
        promising initially, preventing premature convergence to suboptimal
        policies.
    """
    return bias


def init_action_specific(
    state: State,
    action: Action,
    states: Tuple[State, ...],
    actions: List[Action],
) -> float:
    """Strategy C: Action-specific initialization.

    Gives higher initial Q-values to matheuristic repair operators
    (lp) since they are known to be generally effective.

    Args:
        state: Current state label
        action: Action tuple (destroy_op, repair_op)
        states: All possible states
        actions: All possible actions

    Returns:
        100.0 for matheuristic repairs (lp), 50.0 for others

    Theory:
        Incorporating domain knowledge by biasing towards known good
        operators can accelerate learning, especially in early iterations.
    """
    destroy_op, repair_op = action

    # Matheuristic repair operators get higher initial values
    # 'lp' uses LP solver for optimization (matheuristic component)
    repair_key = repair_op.lower()
    matheuristic_repairs = {"lp", "greedy_lp", "segments", "segment"}

    if repair_key in matheuristic_repairs or "lp" in repair_key or "segment" in repair_key:
        return 100.0
    else:
        return 50.0


def init_state_specific(
    state: State,
    action: Action,
    states: Tuple[State, ...],
    actions: List[Action],
) -> float:
    """Strategy D: State-specific initialization.

    Different states require different levels of aggressiveness.
    Stuck states get higher initial Q-values to encourage more
    aggressive exploration.

    Args:
        state: Current state label
        action: Action tuple (destroy_op, repair_op)
        states: All possible states
        actions: All possible actions

    Returns:
        State-dependent Q-value (30.0 to 120.0)

    Theory:
        When stuck, more aggressive operators are needed. Higher Q-values
        for stuck states encourage trying more disruptive operators.
    """
    # State-dependent bias mapping
    state_bias = {
        "explore": 30.0,       # Early exploration, lower urgency
        "stuck": 70.0,         # Stuck, need more aggressive actions
        "deep_stuck": 120.0,   # Deeply stuck, maximum aggressiveness
        # Defaults for 7-state space (if used)
        "early_explore": 30.0,
        "active_improve": 50.0,
        "slow_progress": 70.0,
        "plateau": 90.0,
        "intensive_search": 110.0,
        "final_polish": 40.0,
        "emergency": 120.0,
    }

    return state_bias.get(state, 50.0)  # Default to 50.0 for unknown states


# Strategy function mapping
INIT_STRATEGIES: Dict[QInitStrategy, Callable] = {
    QInitStrategy.ZERO: init_zero,
    QInitStrategy.UNIFORM: init_uniform,
    QInitStrategy.ACTION_SPECIFIC: init_action_specific,
    QInitStrategy.STATE_SPECIFIC: init_state_specific,
}


def initialize_q_table(
    states: Tuple[State, ...],
    actions: List[Action],
    strategy: QInitStrategy = QInitStrategy.ZERO,
    **kwargs,
) -> Dict[State, Dict[Action, float]]:
    """Initialize a Q-table using the specified strategy.

    Args:
        states: All possible state labels
        actions: All possible action tuples
        strategy: Initialization strategy to use
        **kwargs: Additional parameters for the strategy function

    Returns:
        Initialized Q-table as nested dictionary {state: {action: q_value}}

    Example:
        >>> states = ("explore", "stuck", "deep_stuck")
        >>> actions = [("random", "greedy"), ("worst", "greedy_lp")]
        >>> q_table = initialize_q_table(states, actions, QInitStrategy.UNIFORM, bias=50.0)
    """
    init_func = INIT_STRATEGIES[strategy]

    q_table: Dict[State, Dict[Action, float]] = {}

    for state in states:
        q_table[state] = {}
        for action in actions:
            # Call initialization function
            q_value = init_func(
                state=state,
                action=action,
                states=states,
                actions=actions,
                **kwargs,
            )
            q_table[state][action] = float(q_value)

    return q_table
