"""Run the minimal MIP baseline instance with OR-Tools."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (SRC_ROOT, PROJECT_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from baselines.mip import MIPBaselineSolverConfig, solve_minimal_instance


def main() -> int:
    parser = argparse.ArgumentParser(description="Solve the minimal MIP baseline instance.")
    parser.add_argument("--time-limit", type=float, default=30.0, help="Time limit (seconds).")
    parser.add_argument("--mip-gap", type=float, default=0.0, help="Relative MIP gap.")
    args = parser.parse_args()

    solver_config = MIPBaselineSolverConfig(
        time_limit_s=args.time_limit,
        mip_gap=args.mip_gap,
    )
    result = solve_minimal_instance(solver_config=solver_config)

    print(f"status: {result.status}")
    if result.objective_value is not None:
        print(f"objective: {result.objective_value:.4f}")
    if result.details:
        for key in sorted(result.details.keys()):
            print(f"{key}: {result.details[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

