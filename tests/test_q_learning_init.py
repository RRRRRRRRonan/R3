"""Unit tests for Q-learning initialization strategies."""

import pytest
from planner.q_learning import QLearningOperatorAgent
from planner.q_learning_init import QInitStrategy, initialize_q_table
from config import QLearningParams


class TestQInitializationStrategies:
    """Test Q-table initialization strategies."""

    def test_zero_initialization(self):
        """Test that zero initialization creates all zeros."""
        states = ("explore", "stuck", "deep_stuck")
        actions = [("random", "greedy"), ("worst", "regret")]

        q_table = initialize_q_table(
            states=states,
            actions=actions,
            strategy=QInitStrategy.ZERO
        )

        # All values should be 0.0
        for state in states:
            for action in actions:
                assert q_table[state][action] == 0.0

    def test_uniform_initialization(self):
        """Test uniform positive bias initialization."""
        states = ("explore", "stuck", "deep_stuck")
        actions = [("random", "greedy"), ("worst", "regret")]

        q_table = initialize_q_table(
            states=states,
            actions=actions,
            strategy=QInitStrategy.UNIFORM,
            bias=50.0
        )

        # All values should be 50.0
        for state in states:
            for action in actions:
                assert q_table[state][action] == 50.0

    def test_action_specific_initialization(self):
        """Test action-specific initialization."""
        states = ("explore",)
        actions = [
            ("random", "greedy"),      # Non-matheuristic
            ("worst", "greedy_lp"),    # Matheuristic
            ("random", "segments"),    # Matheuristic
        ]

        q_table = initialize_q_table(
            states=states,
            actions=actions,
            strategy=QInitStrategy.ACTION_SPECIFIC
        )

        # Check values
        assert q_table["explore"][("random", "greedy")] == 50.0
        assert q_table["explore"][("worst", "greedy_lp")] == 100.0
        assert q_table["explore"][("random", "segments")] == 100.0

    def test_state_specific_initialization(self):
        """Test state-specific initialization."""
        states = ("explore", "stuck", "deep_stuck")
        actions = [("random", "greedy")]

        q_table = initialize_q_table(
            states=states,
            actions=actions,
            strategy=QInitStrategy.STATE_SPECIFIC
        )

        # Check state-dependent values
        assert q_table["explore"][("random", "greedy")] == 30.0
        assert q_table["stuck"][("random", "greedy")] == 70.0
        assert q_table["deep_stuck"][("random", "greedy")] == 120.0

    def test_q_learning_agent_with_init_strategy(self):
        """Test QLearningOperatorAgent with initialization strategy."""
        destroy_ops = ["random", "worst"]
        repair_ops = ["greedy", "regret", "greedy_lp"]
        params = QLearningParams()

        # Create agent with uniform initialization
        agent = QLearningOperatorAgent(
            destroy_operators=destroy_ops,
            repair_operators=repair_ops,
            params=params,
            init_strategy=QInitStrategy.UNIFORM,
        )

        # Check that Q-table has positive values
        for state in agent.states:
            for action in agent.actions:
                assert agent.q_table[state][action] == 50.0

    def test_default_init_strategy_is_zero(self):
        """Test that default initialization is ZERO (backward compatible)."""
        destroy_ops = ["random"]
        repair_ops = ["greedy"]
        params = QLearningParams()

        # Create agent without specifying init_strategy
        agent = QLearningOperatorAgent(
            destroy_operators=destroy_ops,
            repair_operators=repair_ops,
            params=params,
        )

        # Should default to zero initialization
        for state in agent.states:
            for action in agent.actions:
                assert agent.q_table[state][action] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
