"""Ultra-simple single seed test with guaranteed output."""

import sys
from pathlib import Path
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

print("="*60, flush=True)
print("SIMPLE SEED TEST", flush=True)
print("="*60, flush=True)

from tests.optimization.q_learning.utils import run_q_learning_trial
from tests.optimization.common import get_scale_config, run_matheuristic_trial
from tests.optimization.presets import get_scale_preset

# Configuration
scale = sys.argv[1] if len(sys.argv) > 1 else 'large'
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 2026

preset = get_scale_preset(scale)
config = get_scale_config(scale)

print(f"\nScale: {scale}", flush=True)
print(f"Seed: {seed}", flush=True)
print(f"Q-learning iterations: {preset.iterations.q_learning}", flush=True)
print(f"Matheuristic iterations: {preset.iterations.matheuristic}", flush=True)
print(f"\nEstimated time: 6-10 minutes total", flush=True)
print("="*60, flush=True)

# Q-learning
print(f"\n[{time.strftime('%H:%M:%S')}] Starting Q-learning...", flush=True)
sys.stdout.flush()
sys.stderr.flush()

start = time.time()
alns, baseline_q, optimised_q = run_q_learning_trial(
    config,
    iterations=preset.iterations.q_learning,
    seed=seed
)
elapsed_q = time.time() - start

improvement_q = (baseline_q - optimised_q) / baseline_q * 100

print(f"\n[{time.strftime('%H:%M:%S')}] Q-learning COMPLETED", flush=True)
print(f"  Time: {elapsed_q:.1f}s ({elapsed_q/60:.1f} min)", flush=True)
print(f"  Baseline: {baseline_q:.2f}", flush=True)
print(f"  Optimised: {optimised_q:.2f}", flush=True)
print(f"  Improvement: {improvement_q:.2f}%", flush=True)
sys.stdout.flush()

# Matheuristic
print(f"\n[{time.strftime('%H:%M:%S')}] Starting Matheuristic...", flush=True)
sys.stdout.flush()
sys.stderr.flush()

start = time.time()
baseline_m, optimised_m = run_matheuristic_trial(
    config,
    iterations=preset.iterations.matheuristic,
    seed=seed
)
elapsed_m = time.time() - start

improvement_m = (baseline_m - optimised_m) / baseline_m * 100

print(f"\n[{time.strftime('%H:%M:%S')}] Matheuristic COMPLETED", flush=True)
print(f"  Time: {elapsed_m:.1f}s ({elapsed_m/60:.1f} min)", flush=True)
print(f"  Baseline: {baseline_m:.2f}", flush=True)
print(f"  Optimised: {optimised_m:.2f}", flush=True)
print(f"  Improvement: {improvement_m:.2f}%", flush=True)
sys.stdout.flush()

# Summary
print("\n" + "="*60, flush=True)
print("FINAL RESULTS", flush=True)
print("="*60, flush=True)
print(f"Q-learning:   {improvement_q:.2f}%", flush=True)
print(f"Matheuristic: {improvement_m:.2f}%", flush=True)
print(f"Difference:   {improvement_q - improvement_m:+.2f}%", flush=True)

if improvement_q > improvement_m:
    print("\n→ Q-learning is BETTER ✅", flush=True)
elif improvement_q < improvement_m - 2:
    print("\n→ Q-learning is WORSE ❌", flush=True)
else:
    print("\n→ Similar performance ≈", flush=True)

print(f"\nTotal time: {(elapsed_q + elapsed_m)/60:.1f} minutes", flush=True)
print("="*60, flush=True)
