"""Minimal test to diagnose ALNS running issue."""

import sys
from pathlib import Path
import time

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

print("🔍 Starting diagnostic test...", flush=True)
print(f"Project root: {project_root}", flush=True)

# Test 1: Imports
print("\n[1/5] Testing imports...", flush=True)
try:
    from tests.optimization.q_learning.utils import run_q_learning_trial
    from tests.optimization.common import get_scale_config
    from tests.optimization.presets import get_scale_preset
    print("  ✅ Imports successful", flush=True)
except Exception as e:
    print(f"  ❌ Import failed: {e}", flush=True)
    sys.exit(1)

# Test 2: Get config
print("\n[2/5] Getting small scale config...", flush=True)
try:
    config = get_scale_config('small')
    preset = get_scale_preset('small')
    print(f"  ✅ Config loaded: {preset.iterations.q_learning} iterations", flush=True)
except Exception as e:
    print(f"  ❌ Config failed: {e}", flush=True)
    sys.exit(1)

# Test 3: Run with VERY FEW iterations
print("\n[3/5] Testing Q-learning with ONLY 2 iterations...", flush=True)
print("  (This should take ~10-30 seconds)", flush=True)
try:
    start = time.time()
    alns, baseline, optimised = run_q_learning_trial(
        config,
        iterations=2,  # Only 2 iterations!
        seed=2025
    )
    elapsed = time.time() - start

    improvement = (baseline - optimised) / baseline * 100
    print(f"  ✅ Completed in {elapsed:.1f}s", flush=True)
    print(f"     Baseline: {baseline:.2f}", flush=True)
    print(f"     Optimised: {optimised:.2f}", flush=True)
    print(f"     Improvement: {improvement:.2f}%", flush=True)
except Exception as e:
    print(f"  ❌ Q-learning failed: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Check if verbose works
print("\n[4/5] Checking ALNS verbose output...", flush=True)
try:
    if hasattr(alns, 'verbose'):
        print(f"  ALNS verbose setting: {alns.verbose}", flush=True)
    else:
        print(f"  No verbose attribute found", flush=True)
except Exception as e:
    print(f"  ❌ Check failed: {e}", flush=True)

# Test 5: Time estimate
print("\n[5/5] Time estimates...", flush=True)
iterations_per_second = 2 / elapsed
print(f"  Iterations per second: {iterations_per_second:.3f}", flush=True)
print(f"  Expected time for 40 iterations: {40/iterations_per_second:.1f}s", flush=True)
print(f"  Expected time for 44 iterations: {44/iterations_per_second:.1f}s", flush=True)

print("\n✅ Diagnostic complete!", flush=True)
print("\nConclusion:", flush=True)
print(f"  - ALNS can run successfully", flush=True)
print(f"  - With current performance, large scale (44 iterations) will take ~{44/iterations_per_second:.0f}s ({44/iterations_per_second/60:.1f} minutes)", flush=True)
