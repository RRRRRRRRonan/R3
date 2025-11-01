"""Step-by-step diagnostic for Large scale issue.

This script progressively tests larger configurations to find where it hangs.
"""

import sys
from pathlib import Path
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

from tests.optimization.q_learning.utils import run_q_learning_trial
from tests.optimization.common import get_scale_config
from tests.optimization.presets import get_scale_preset

print("="*60, flush=True)
print("PROGRESSIVE DIAGNOSTIC - Finding where Large scale hangs", flush=True)
print("="*60, flush=True)

config = get_scale_config('large')
seed = 2026

# Test 1: Large scale, 2 iterations
print(f"\n[Test 1] Large scale, 2 iterations, seed={seed}", flush=True)
print("Expected: ~20 seconds", flush=True)
try:
    start = time.time()
    alns, baseline, optimised = run_q_learning_trial(config, iterations=2, seed=seed)
    elapsed = time.time() - start
    improvement = (baseline - optimised) / baseline * 100
    print(f"✅ PASSED in {elapsed:.1f}s", flush=True)
    print(f"   Improvement: {improvement:.2f}%", flush=True)
except Exception as e:
    print(f"❌ FAILED: {e}", flush=True)
    sys.exit(1)

# Test 2: Large scale, 5 iterations
print(f"\n[Test 2] Large scale, 5 iterations, seed={seed}", flush=True)
print("Expected: ~50 seconds", flush=True)
print("(If this hangs, the problem is in iterations 3-5)", flush=True)
try:
    start = time.time()
    alns, baseline, optimised = run_q_learning_trial(config, iterations=5, seed=seed)
    elapsed = time.time() - start
    improvement = (baseline - optimised) / baseline * 100
    print(f"✅ PASSED in {elapsed:.1f}s", flush=True)
    print(f"   Improvement: {improvement:.2f}%", flush=True)
except Exception as e:
    print(f"❌ FAILED: {e}", flush=True)
    sys.exit(1)

# Test 3: Large scale, 10 iterations
print(f"\n[Test 3] Large scale, 10 iterations, seed={seed}", flush=True)
print("Expected: ~100 seconds", flush=True)
print("(If this hangs, the problem is in iterations 6-10)", flush=True)
try:
    start = time.time()
    alns, baseline, optimised = run_q_learning_trial(config, iterations=10, seed=seed)
    elapsed = time.time() - start
    improvement = (baseline - optimised) / baseline * 100
    print(f"✅ PASSED in {elapsed:.1f}s", flush=True)
    print(f"   Improvement: {improvement:.2f}%", flush=True)
except Exception as e:
    print(f"❌ FAILED: {e}", flush=True)
    sys.exit(1)

# Test 4: Large scale, 20 iterations
print(f"\n[Test 4] Large scale, 20 iterations, seed={seed}", flush=True)
print("Expected: ~200 seconds (~3.3 minutes)", flush=True)
print("(If this hangs, the problem is in iterations 11-20)", flush=True)
print("Starting at:", time.strftime('%H:%M:%S'), flush=True)
try:
    start = time.time()
    alns, baseline, optimised = run_q_learning_trial(config, iterations=20, seed=seed)
    elapsed = time.time() - start
    improvement = (baseline - optimised) / baseline * 100
    print(f"✅ PASSED in {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)
    print(f"   Improvement: {improvement:.2f}%", flush=True)
except Exception as e:
    print(f"❌ FAILED: {e}", flush=True)
    sys.exit(1)

print("\n" + "="*60, flush=True)
print("ALL TESTS PASSED!", flush=True)
print("The problem is likely in iterations 21-44", flush=True)
print("This suggests the issue may be:", flush=True)
print("  1. State transition (stuck/deep_stuck) triggering expensive operations", flush=True)
print("  2. LP solver taking too long in later iterations", flush=True)
print("  3. Elite pool operations becoming slow", flush=True)
print("="*60, flush=True)
