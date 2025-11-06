"""Linear-programming backed repair operator for matheuristic ALNS.

The implementation mirrors the "Repair_LP" operator described by Singh et al.
(2022).  It enumerates high quality insertion plans for the tasks removed by a
Destroy operator, formulates a small linear program that selects exactly one
plan per task (with the option of skipping at a high penalty), and solves it via
an internal simplex routine.  The resulting assignment dictates how the tasks
are reinserted into the incumbent partial schedule before the ALNS loop resumes.

The linear model focuses on the weighted tardiness + travel objective while
respecting time-window and battery feasibility for each candidate plan.  Because
only a handful of tasks are removed at a time, the generated LPs remain small
and can be solved quickly without relying on external optimisation libraries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from config import LPRepairParams
from core.route import Route
from core.task import Task


@dataclass
class _Plan:
    """Candidate reinsertion plan for a single task."""

    task_id: int
    pickup_pos: int
    delivery_pos: int
    pickup_anchor_id: int
    delivery_anchor_id: int
    incremental_cost: float
    variable_index: int


class SimplexSolver:
    """Two-phase simplex solver for small linear programs.

    The solver handles problems of the form::

        minimise    c^T x
        subject to  A x = b
                    x >= 0

    which covers the assignment-style models produced by :class:`LPBasedRepair`.
    The implementation is intentionally lightweight so that we do not rely on
    external optimisation libraries that may be unavailable in the execution
    environment.
    """

    def __init__(self, A: Sequence[Sequence[float]], b: Sequence[float], c: Sequence[float], *, epsilon: float = 1e-9) -> None:
        if not A:
            raise ValueError("Constraint matrix A must be non-empty")
        self.m = len(A)
        self.n = len(A[0])
        self.A = [list(row) for row in A]
        self.b = list(b)
        self.c = list(c)
        self.epsilon = epsilon

    def solve(self) -> Optional[List[float]]:
        """Solve the LP and return the optimal variable values if feasible."""

        self._normalise_rhs()
        tableau, basis = self._build_phase_one_tableau()
        if not self._run_simplex(tableau, basis):
            return None

        # Phase 1 optimal value should be zero; otherwise infeasible.
        if tableau[-1][-1] < -self.epsilon:
            return None

        self._remove_artificial_columns(tableau, basis)
        self._initialise_phase_two_objective(tableau, basis)
        if not self._run_simplex(tableau, basis):
            return None

        solution = [0.0] * self.n
        for row_index, var_index in enumerate(basis):
            if var_index < self.n:
                solution[var_index] = tableau[row_index][-1]
        return solution

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalise_rhs(self) -> None:
        for idx, value in enumerate(self.b):
            if value < 0:
                self.b[idx] = -value
                self.A[idx] = [-coeff for coeff in self.A[idx]]

    def _build_phase_one_tableau(self) -> tuple[List[List[float]], List[int]]:
        total_vars = self.n + self.m
        tableau = [[0.0 for _ in range(total_vars + 1)] for _ in range(self.m + 1)]
        basis = []

        for i in range(self.m):
            row = tableau[i]
            for j in range(self.n):
                row[j] = self.A[i][j]
            # artificial variable
            row[self.n + i] = 1.0
            row[-1] = self.b[i]
            basis.append(self.n + i)

        obj = tableau[-1]
        for j in range(self.n):
            obj[j] = 0.0
        for i in range(self.m):
            obj[self.n + i] = -1.0
        obj[-1] = -sum(self.b)

        # Adjust for artificial variables in the initial basis.
        for i in range(self.m):
            row = tableau[i]
            for j in range(total_vars + 1):
                obj[j] += row[j]
        return tableau, basis

    def _remove_artificial_columns(self, tableau: List[List[float]], basis: List[int]) -> None:
        total_vars = self.n + self.m
        cols_to_delete = list(range(self.n, total_vars))

        # Pivot artificial variables out of the basis when possible.
        for row_index, var_index in enumerate(basis):
            if var_index >= self.n:
                pivot_col = self._find_pivot_column(tableau[row_index], limit=self.n)
                if pivot_col is not None:
                    self._pivot(tableau, basis, row_index, pivot_col)
                else:
                    # Row is redundant (all zeros); keep artificial with zero value.
                    pass

        # Delete artificial columns.
        for row in tableau:
            for offset, col_index in enumerate(cols_to_delete):
                del row[self.n]

        # Update basis indices to reflect removed columns.
        for idx, var_index in enumerate(basis):
            if var_index >= self.n:
                basis[idx] = -1  # mark redundant constraint

    def _initialise_phase_two_objective(self, tableau: List[List[float]], basis: List[int]) -> None:
        total_cols = self.n + 1
        obj = tableau[-1]
        for j in range(total_cols):
            obj[j] = 0.0
        for j in range(self.n):
            obj[j] = -self.c[j]
        obj[-1] = 0.0

        for row_index, var_index in enumerate(basis):
            if 0 <= var_index < self.n:
                coeff = obj[var_index]
                if abs(coeff) > self.epsilon:
                    for col in range(total_cols):
                        obj[col] -= coeff * tableau[row_index][col]

    def _run_simplex(self, tableau: List[List[float]], basis: List[int]) -> bool:
        num_rows = len(tableau) - 1
        num_cols = len(tableau[0]) - 1

        while True:
            entering_col = self._choose_entering_variable(tableau[-1][:-1])
            if entering_col is None:
                break

            pivot_row = self._choose_pivot_row(tableau, entering_col)
            if pivot_row is None:
                return False

            self._pivot(tableau, basis, pivot_row, entering_col)

        return True

    def _choose_entering_variable(self, objective_row: Sequence[float]) -> Optional[int]:
        entering_col = None
        best_value = self.epsilon
        for idx, value in enumerate(objective_row):
            if value > best_value:
                best_value = value
                entering_col = idx
        return entering_col

    def _choose_pivot_row(self, tableau: List[List[float]], entering_col: int) -> Optional[int]:
        best_ratio = float("inf")
        pivot_row = None
        for idx in range(len(tableau) - 1):
            coeff = tableau[idx][entering_col]
            if coeff > self.epsilon:
                ratio = tableau[idx][-1] / coeff
                if ratio < best_ratio - self.epsilon:
                    best_ratio = ratio
                    pivot_row = idx
        return pivot_row

    def _pivot(self, tableau: List[List[float]], basis: List[int], pivot_row: int, pivot_col: int) -> None:
        pivot_value = tableau[pivot_row][pivot_col]
        row = tableau[pivot_row]
        for col in range(len(row)):
            row[col] /= pivot_value

        for r_idx, current_row in enumerate(tableau):
            if r_idx == pivot_row:
                continue
            factor = current_row[pivot_col]
            if abs(factor) <= self.epsilon:
                continue
            for col in range(len(current_row)):
                current_row[col] -= factor * row[col]

        basis[pivot_row] = pivot_col

    def _find_pivot_column(self, row: Sequence[float], *, limit: int) -> Optional[int]:
        for idx, value in enumerate(row[:limit]):
            if abs(value) > self.epsilon:
                return idx
        return None


class LPBasedRepair:
    """Reinsert removed tasks using an LP-guided selection of insertion plans."""

    def __init__(self, planner, params: LPRepairParams) -> None:
        self.planner = planner
        self.params = params

    def rebuild(self, partial_route: Route, removed_task_ids: Sequence[int]) -> Route:
        if not removed_task_ids:
            return partial_route.copy()

        base_route = partial_route.copy()
        base_cost = self.planner.evaluate_cost(base_route)
        tasks: List[Task] = []
        for task_id in removed_task_ids:
            task = self.planner.task_pool.get_task(task_id)
            if task is not None:
                tasks.append(task)

        plans: List[_Plan] = []
        plan_by_task: Dict[int, List[_Plan]] = {}
        column_costs: List[float] = []

        for task in tasks:
            task_plans = self._enumerate_plans(base_route, base_cost, task)
            if not task_plans:
                # Allow the LP to mark the task as skipped with a high penalty.
                skip_index = len(column_costs)
                column_costs.append(self.params.skip_penalty)
                skip_plan = _Plan(
                    task_id=task.task_id,
                    pickup_pos=-1,
                    delivery_pos=-1,
                    pickup_anchor_id=-1,
                    delivery_anchor_id=-1,
                    incremental_cost=self.params.skip_penalty,
                    variable_index=skip_index,
                )
                plans.append(skip_plan)
                plan_by_task.setdefault(task.task_id, []).append(skip_plan)
                continue

            limited = task_plans[: self.params.max_plans_per_task]
            for plan in limited:
                plan.variable_index = len(column_costs)
                column_costs.append(plan.incremental_cost)
                plans.append(plan)
                plan_by_task.setdefault(task.task_id, []).append(plan)

        if not plans:
            return self.planner.regret2_insertion(partial_route.copy(), [task.task_id for task in tasks])

        A, b = self._build_constraints(plan_by_task, plans)
        solver = SimplexSolver(A, b, column_costs)
        solution = solver.solve()
        if solution is None:
            return self.planner.regret2_insertion(partial_route.copy(), [task.task_id for task in tasks])

        assignment: Dict[int, _Plan] = {}
        for task_id, candidate_plans in plan_by_task.items():
            best_plan = max(candidate_plans, key=lambda plan: solution[plan.variable_index])
            if solution[best_plan.variable_index] < self.params.fractional_threshold:
                continue
            assignment[task_id] = best_plan

        rebuilt = base_route.copy()
        for task in tasks:
            selected = assignment.get(task.task_id)
            if selected is None or selected.pickup_pos < 0:
                # Fallback to regret insertion for unassigned tasks.
                rebuilt = self.planner.regret2_insertion(rebuilt, [task.task_id])
                continue
            self._apply_plan(rebuilt, task, selected)

        rebuilt_cost = self.planner.evaluate_cost(rebuilt)
        if rebuilt_cost + self.params.improvement_tolerance >= base_cost:
            return self.planner.regret2_insertion(partial_route.copy(), [task.task_id for task in tasks])
        return rebuilt

    # ------------------------------------------------------------------
    # Plan generation and application helpers
    # ------------------------------------------------------------------

    def _enumerate_plans(self, base_route: Route, base_cost: float, task: Task) -> List[_Plan]:
        """Enumerate feasible insertion plans for a task.

        Phase 0 Fix: When a plan is battery-infeasible, try inserting charging stations
        to make it feasible instead of immediately discarding it. This is critical for
        large-scale instances where battery constraints are tight.
        """
        plans: List[_Plan] = []
        max_pickup = len(base_route.nodes)
        for pickup_pos in range(1, max_pickup):
            for delivery_pos in range(pickup_pos + 1, max_pickup + 1):
                candidate = base_route.copy()
                try:
                    candidate.insert_task(task, (pickup_pos, delivery_pos))
                except ValueError:
                    continue

                # PHASE 0 FIX: Try to make infeasible routes feasible by adding charging
                if not self.planner.ensure_route_schedule(candidate):
                    # Battery infeasible - try inserting charging stations
                    if hasattr(self.planner, '_insert_necessary_charging_stations'):
                        candidate = self.planner._insert_necessary_charging_stations(candidate)
                        # Re-check feasibility after charging station insertion
                        if not self.planner.ensure_route_schedule(candidate):
                            continue  # Still infeasible, discard this plan
                    else:
                        continue  # No charging insertion available, discard

                candidate_cost = self.planner.evaluate_cost(candidate)
                if candidate_cost == float("inf"):
                    continue
                incremental = candidate_cost - base_cost
                plan = _Plan(
                    task_id=task.task_id,
                    pickup_pos=pickup_pos,
                    delivery_pos=delivery_pos,
                    pickup_anchor_id=candidate.nodes[pickup_pos - 1].node_id,
                    delivery_anchor_id=candidate.nodes[delivery_pos - 1].node_id,
                    incremental_cost=incremental,
                    variable_index=-1,
                )
                plans.append(plan)
        plans.sort(key=lambda plan: plan.incremental_cost)
        return plans

    def _build_constraints(self, plan_by_task: Dict[int, List[_Plan]], plans: List[_Plan]) -> tuple[List[List[float]], List[float]]:
        task_ids = list(plan_by_task.keys())
        num_vars = len(plans)
        A: List[List[float]] = []
        b: List[float] = []

        for task_id in task_ids:
            row = [0.0] * num_vars
            for plan in plan_by_task[task_id]:
                row[plan.variable_index] = 1.0
            A.append(row)
            b.append(1.0)

        return A, b

    def _apply_plan(self, route: Route, task: Task, plan: _Plan) -> None:
        pickup_anchor_index = route.get_node_position(plan.pickup_anchor_id)
        if pickup_anchor_index is None:
            pickup_anchor_index = len(route.nodes) - 1
        pickup_position = pickup_anchor_index + 1
        route.nodes.insert(pickup_position, task.pickup_node)

        delivery_anchor_index = route.get_node_position(plan.delivery_anchor_id)
        if delivery_anchor_index is None:
            delivery_anchor_index = len(route.nodes) - 1
        delivery_position = delivery_anchor_index + 1
        route.nodes.insert(delivery_position, task.delivery_node)
        route.visits = []
        route.is_feasible = None
        route.infeasibility_info = None
