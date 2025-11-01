"""Quick test to check performance with different seeds - WITH PROGRESS OUTPUT."""

import sys
from pathlib import Path
import time

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

from tests.optimization.q_learning.utils import run_q_learning_trial
from tests.optimization.common import get_scale_config, run_matheuristic_trial
from tests.optimization.presets import get_scale_preset


def test_seeds(scale='large', seeds=[2025, 2026, 2027]):
    """Test multiple seeds on a given scale."""
    print(f"\n{'='*60}", flush=True)
    print(f"Testing {scale.upper()} scale with different seeds", flush=True)
    print(f"{'='*60}\n", flush=True)

    preset = get_scale_preset(scale)
    config = get_scale_config(scale)

    print(f"Configuration:", flush=True)
    print(f"  Scale: {scale}", flush=True)
    print(f"  Seeds: {seeds}", flush=True)
    print(f"  Iterations: Q-learning={preset.iterations.q_learning}, Matheuristic={preset.iterations.matheuristic}", flush=True)
    print(f"  Estimated time per seed: 6-10 minutes\n", flush=True)

    results = {'q_learning': [], 'matheuristic': []}

    for idx, seed in enumerate(seeds, 1):
        print(f"\n{'='*60}", flush=True)
        print(f"Seed {seed} ({idx}/{len(seeds)})", flush=True)
        print(f"{'='*60}", flush=True)

        # Test Q-learning
        print(f"\n[{time.strftime('%H:%M:%S')}] Running Q-learning (seed={seed})...", flush=True)
        print(f"  This will take ~3-5 minutes with no output...", flush=True)
        print(f"  Please wait patiently...", flush=True)

        start_time = time.time()
        _, baseline_q, optimised_q = run_q_learning_trial(
            config,
            iterations=preset.iterations.q_learning,
            seed=seed
        )
        q_time = time.time() - start_time

        improvement_q = (baseline_q - optimised_q) / baseline_q
        results['q_learning'].append(improvement_q)

        print(f"  ✓ Completed in {q_time:.1f}s", flush=True)
        print(f"  Baseline:    {baseline_q:.2f}", flush=True)
        print(f"  Optimised:   {optimised_q:.2f}", flush=True)
        print(f"  Improvement: {improvement_q*100:.2f}%", flush=True)

        # Test Matheuristic
        print(f"\n[{time.strftime('%H:%M:%S')}] Running Matheuristic (seed={seed})...", flush=True)
        print(f"  This will take ~3-5 minutes with no output...", flush=True)
        print(f"  Please wait patiently...", flush=True)

        start_time = time.time()
        baseline_m, optimised_m = run_matheuristic_trial(
            config,
            iterations=preset.iterations.matheuristic,
            seed=seed
        )
        m_time = time.time() - start_time

        improvement_m = (baseline_m - optimised_m) / baseline_m
        results['matheuristic'].append(improvement_m)

        print(f"  ✓ Completed in {m_time:.1f}s", flush=True)
        print(f"  Baseline:    {baseline_m:.2f}", flush=True)
        print(f"  Optimised:   {optimised_m:.2f}", flush=True)
        print(f"  Improvement: {improvement_m*100:.2f}%", flush=True)

        # Comparison
        diff = (improvement_q - improvement_m) * 100
        print(f"\n  Comparison:", flush=True)
        print(f"    Q-learning:   {improvement_q*100:.2f}%", flush=True)
        print(f"    Matheuristic: {improvement_m*100:.2f}%", flush=True)
        print(f"    Difference:   {diff:+.2f}%", flush=True)

        if diff > 0:
            print(f"    → Q-learning is BETTER ✓", flush=True)
        elif diff < -2:
            print(f"    → Q-learning is WORSE ✗", flush=True)
        else:
            print(f"    → Similar performance ≈", flush=True)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print(f"SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)

    q_mean = sum(results['q_learning'])/len(seeds)*100
    m_mean = sum(results['matheuristic'])/len(seeds)*100

    print(f"Q-learning:   Mean={q_mean:.2f}%", flush=True)
    print(f"Matheuristic: Mean={m_mean:.2f}%", flush=True)
    print(f"Difference:   {q_mean - m_mean:+.2f}%", flush=True)

    if len(seeds) > 1:
        import statistics
        q_std = statistics.stdev(results['q_learning']) * 100
        m_std = statistics.stdev(results['matheuristic']) * 100
        print(f"\nStandard Deviation:", flush=True)
        print(f"Q-learning:   {q_std:.2f}%", flush=True)
        print(f"Matheuristic: {m_std:.2f}%", flush=True)

    print(f"{'='*60}\n", flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', default='large', choices=['small', 'medium', 'large'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[2025, 2026, 2027])
    args = parser.parse_args()

    print(f"\n🚀 Starting seed comparison test", flush=True)
    print(f"⏱️  Expected total time: {len(args.seeds) * 6}-{len(args.seeds) * 10} minutes\n", flush=True)

    test_seeds(args.scale, args.seeds)

    print(f"✅ All tests completed!", flush=True)
