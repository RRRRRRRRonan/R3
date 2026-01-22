"""Solver hooks for the MIP baseline model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from baselines.mip.config import MIPBaselineSolverConfig
from baselines.mip.model import MIPBaselineInstance, MIPBaselineResult, build_model_spec


@dataclass
class SolverStatus:
    """Minimal solver status tracking."""

    status: str
    objective: Optional[float] = None


class MIPBaselineSolver:
    """Abstract solver interface for the baseline model."""

    name: str = "abstract"

    def solve(
        self,
        instance: MIPBaselineInstance,
        config: Optional[MIPBaselineSolverConfig] = None,
    ) -> MIPBaselineResult:
        raise NotImplementedError("Solver integration is not implemented yet.")


class PulpCBCSolver(MIPBaselineSolver):
    """Placeholder for PuLP + CBC integration."""

    name = "pulp_cbc"

    def solve(
        self,
        instance: MIPBaselineInstance,
        config: Optional[MIPBaselineSolverConfig] = None,
    ) -> MIPBaselineResult:
        try:
            import pulp  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "PuLP is required for the CBC baseline. "
                "Install with: pip install pulp"
            ) from exc

        spec = build_model_spec(instance)
        return MIPBaselineResult(
            status="not_implemented",
            objective_value=None,
            details={"info": float(len(spec.constraints))},
        )


def get_default_solver(config: Optional[MIPBaselineSolverConfig] = None) -> MIPBaselineSolver:
    """Return the baseline solver configured for the project."""

    config = config or MIPBaselineSolverConfig()
    if config.solver_name == "pulp_cbc":
        return PulpCBCSolver()
    raise ValueError(f"Unknown baseline solver: {config.solver_name}")
