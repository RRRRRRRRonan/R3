"""Verify Week 1 installation and functionality.

This script tests that all Week 1 code is properly installed and working.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from planner.q_learning_init import (
            QInitStrategy,
            initialize_q_table,
            init_zero,
            init_uniform,
            init_action_specific,
            init_state_specific,
        )
        print("  ✓ q_learning_init module imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import q_learning_init: {e}")
        return False

    try:
        from planner.q_learning import QLearningOperatorAgent
        print("  ✓ q_learning module imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import q_learning: {e}")
        return False

    try:
        from config import QLearningParams
        print("  ✓ config module imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import config: {e}")
        return False

    return True


def test_q_table_initialization():
    """Test Q-table initialization strategies."""
    print("\nTesting Q-table initialization...")

    from planner.q_learning_init import QInitStrategy, initialize_q_table

    states = ("explore", "stuck", "deep_stuck")
    actions = [("random", "greedy"), ("worst", "regret")]

    # Test ZERO
    q_table = initialize_q_table(states, actions, QInitStrategy.ZERO)
    if all(q_table[s][a] == 0.0 for s in states for a in actions):
        print("  ✓ ZERO initialization works correctly")
    else:
        print("  ✗ ZERO initialization failed")
        return False

    # Test UNIFORM
    q_table = initialize_q_table(states, actions, QInitStrategy.UNIFORM, bias=50.0)
    if all(q_table[s][a] == 50.0 for s in states for a in actions):
        print("  ✓ UNIFORM initialization works correctly")
    else:
        print("  ✗ UNIFORM initialization failed")
        return False

    # Test ACTION_SPECIFIC
    actions_with_math = [("random", "greedy"), ("worst", "greedy_lp")]
    q_table = initialize_q_table(states, actions_with_math, QInitStrategy.ACTION_SPECIFIC)
    if q_table["explore"][("random", "greedy")] == 50.0 and \
       q_table["explore"][("worst", "greedy_lp")] == 100.0:
        print("  ✓ ACTION_SPECIFIC initialization works correctly")
    else:
        print("  ✗ ACTION_SPECIFIC initialization failed")
        return False

    # Test STATE_SPECIFIC
    q_table = initialize_q_table(states, actions, QInitStrategy.STATE_SPECIFIC)
    if q_table["explore"][actions[0]] == 30.0 and \
       q_table["stuck"][actions[0]] == 70.0 and \
       q_table["deep_stuck"][actions[0]] == 120.0:
        print("  ✓ STATE_SPECIFIC initialization works correctly")
    else:
        print("  ✗ STATE_SPECIFIC initialization failed")
        return False

    return True


def test_q_learning_agent():
    """Test QLearningOperatorAgent with init strategies."""
    print("\nTesting Q-learning agent integration...")

    from planner.q_learning import QLearningOperatorAgent
    from planner.q_learning_init import QInitStrategy
    from config import QLearningParams

    destroy_ops = ["random", "worst"]
    repair_ops = ["greedy", "regret"]
    params = QLearningParams()

    # Test with ZERO (default)
    agent = QLearningOperatorAgent(
        destroy_operators=destroy_ops,
        repair_operators=repair_ops,
        params=params,
    )
    if all(agent.q_table[s][a] == 0.0 for s in agent.states for a in agent.actions):
        print("  ✓ Agent with default (ZERO) initialization works")
    else:
        print("  ✗ Agent with default initialization failed")
        return False

    # Test with UNIFORM
    agent = QLearningOperatorAgent(
        destroy_operators=destroy_ops,
        repair_operators=repair_ops,
        params=params,
        init_strategy=QInitStrategy.UNIFORM,
    )
    if all(agent.q_table[s][a] == 50.0 for s in agent.states for a in agent.actions):
        print("  ✓ Agent with UNIFORM initialization works")
    else:
        print("  ✗ Agent with UNIFORM initialization failed")
        return False

    # Test agent properties
    assert agent.epsilon == params.initial_epsilon
    assert len(agent.states) == 3  # Default 3 states
    assert len(agent.actions) == len(destroy_ops) * len(repair_ops)
    print("  ✓ Agent properties correct")

    return True


def test_scripts_exist():
    """Test that all required scripts exist."""
    print("\nChecking script files...")

    scripts_dir = Path(__file__).parent
    required_files = [
        "run_experiment.py",
        "01_baseline_collection.sh",
        "02_init_experiments.sh",
        "analyze_baseline.py",
        "analyze_init_strategies.py",
        "README.md",
    ]

    all_exist = True
    for filename in required_files:
        filepath = scripts_dir / filename
        if filepath.exists():
            print(f"  ✓ {filename} exists")
        else:
            print(f"  ✗ {filename} NOT FOUND")
            all_exist = False

    return all_exist


def main():
    """Run all tests."""
    print("=" * 70)
    print("Week 1 Installation Verification")
    print("=" * 70)
    print()

    tests = [
        ("Module Imports", test_imports),
        ("Q-table Initialization", test_q_table_initialization),
        ("Q-learning Agent Integration", test_q_learning_agent),
        ("Script Files", test_scripts_exist),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n  ✗ Test '{name}' raised exception: {e}")
            results.append((name, False))
        print()

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")

    print()
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! Week 1 is ready to use.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
