"""Verify Phase 1.5 recalibrated parameters."""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from planner.adaptive_params import get_adaptive_params

print("=" * 60)
print("Phase 1.5 Recalibrated Parameters")
print("=" * 60)
print()

for scale in ['small', 'medium', 'large']:
    params = get_adaptive_params(scale)
    print(f"{scale.upper()}:")
    print(f"  alpha:                  {params.alpha}")
    print(f"  epsilon_min:            {params.epsilon_min}")
    print(f"  stagnation_ratio:       {params.stagnation_ratio}")
    print(f"  deep_stagnation_ratio:  {params.deep_stagnation_ratio}")
    print()

print("=" * 60)
print("Key Changes from Phase 1:")
print("=" * 60)
print("Small:  α 0.30→0.35, ε_min 0.05→0.08, stag 0.15→0.12")
print("Medium: α 0.20→0.30, ε_min 0.10→0.12, stag 0.25→0.18  ← Critical")
print("Large:  α 0.15→0.25, ε_min 0.15→0.15, stag 0.35→0.22  ← Critical")
print()
