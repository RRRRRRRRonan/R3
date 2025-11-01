"""Quick test to check performance with different seeds."""

import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

from tests.optimization.q_learning.utils import run_q_learning_trial
from tests.optimization.common import get_scale_config, run_matheuristic_trial
from tests.optimization.presets import get_scale_preset


def test_seeds(scale='large', seeds=[2025, 2026, 2027]):
    """Test multiple seeds on a given scale."""
    print(f"\n{'='*60}")
    print(f"Testing {scale.upper()} scale with different seeds")
    print(f"{'='*60}\n")

    preset = get_scale_preset(scale)
    config = get_scale_config(scale)

    results = {'q_learning': [], 'matheuristic': []}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")

        # Test Q-learning
        _, baseline_q, optimised_q = run_q_learning_trial(
            config,
            iterations=preset.iterations.q_learning,
            seed=seed
        )
        improvement_q = (baseline_q - optimised_q) / baseline_q
        results['q_learning'].append(improvement_q)
        print(f"Q-learning:    {improvement_q*100:.2f}%")

        # Test Matheuristic
        baseline_m, optimised_m = run_matheuristic_trial(
            config,
            iterations=preset.iterations.matheuristic,
            seed=seed
        )
        improvement_m = (baseline_m - optimised_m) / baseline_m
        results['matheuristic'].append(improvement_m)
        print(f"Matheuristic:  {improvement_m*100:.2f}%")
        print(f"Difference:    {(improvement_q - improvement_m)*100:+.2f}%")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Q-learning:   Mean={sum(results['q_learning'])/len(seeds)*100:.2f}%")
    print(f"Matheuristic: Mean={sum(results['matheuristic'])/len(seeds)*100:.2f}%")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', default='large', choices=['small', 'medium', 'large'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[2025, 2026, 2027])
    args = parser.parse_args()

    test_seeds(args.scale, args.seeds)
