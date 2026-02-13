"""Tests for MIP solver backend selection."""

from __future__ import annotations

import pytest

from baselines.mip import MIPBaselineSolverConfig, get_default_solver
from baselines.mip.solver import ORToolsSolver


def test_get_default_solver_accepts_ortools_backend():
    solver = get_default_solver(MIPBaselineSolverConfig(solver_name="ortools"))
    assert isinstance(solver, ORToolsSolver)


def test_get_default_solver_accepts_gurobi_backend():
    solver = get_default_solver(MIPBaselineSolverConfig(solver_name="gurobi"))
    assert isinstance(solver, ORToolsSolver)


def test_get_default_solver_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported MIP baseline solver"):
        get_default_solver(MIPBaselineSolverConfig(solver_name="unknown_solver"))
