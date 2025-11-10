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

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from config import LPRepairParams
from core.route import Route
from core.task import Task

logger = logging.getLogger(__name__)


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

        logger.debug(f"[SIMPLEX] Starting solve: {self.m} constraints, {self.n} variables")

        self._normalise_rhs()
        tableau, basis = self._build_phase_one_tableau()

        # CRITICAL DIAGNOSTIC: Print initial Phase 1 tableau
        logger.debug(f"[SIMPLEX] Initial Phase 1 objective row (last 10 entries): {tableau[-1][-10:]}")
        logger.debug(f"[SIMPLEX] Initial Phase 1 objective value: {tableau[-1][-1]}")
        logger.debug(f"[SIMPLEX] Initial basis: {basis}")

        if not self._run_simplex(tableau, basis):
            logger.warning(f"[SIMPLEX] Phase 1 simplex FAILED (unbounded or cycling)")
            return None

        # Phase 1 optimal value should be zero; otherwise infeasible.
        phase1_value = tableau[-1][-1]
        logger.debug(f"[SIMPLEX] Phase 1 complete: objective={phase1_value:.6f}")
        if phase1_value < -self.epsilon:
            logger.warning(f"[SIMPLEX] LP INFEASIBLE: phase 1 objective={phase1_value:.6f} < {-self.epsilon}")
            return None

        self._remove_artificial_columns(tableau, basis)
        self._initialise_phase_two_objective(tableau, basis)
        if not self._run_simplex(tableau, basis):
            logger.warning(f"[SIMPLEX] Phase 2 simplex FAILED (unbounded or cycling)")
            return None

        solution = [0.0] * self.n
        for row_index, var_index in enumerate(basis):
            if var_index < self.n:
                solution[var_index] = tableau[row_index][-1]

        logger.debug(f"[SIMPLEX] SUCCESS: solution found")
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
        iteration = 0

        while True:
            entering_col = self._choose_entering_variable(tableau[-1][:-1])
            if entering_col is None:
                logger.debug(f"[SIMPLEX] Optimal reached after {iteration} iterations")
                break

            pivot_row = self._choose_pivot_row(tableau, entering_col)
            if pivot_row is None:
                logger.warning(f"[SIMPLEX] Unbounded at iteration {iteration}")
                return False

            logger.debug(f"[SIMPLEX] Iteration {iteration}: pivot row={pivot_row}, col={entering_col}, "
                        f"leaving={basis[pivot_row] if pivot_row < len(basis) else 'N/A'}, entering={entering_col}")
            self._pivot(tableau, basis, pivot_row, entering_col)
            iteration += 1

            if iteration > 1000:
                logger.error(f"[SIMPLEX] Max iterations exceeded (cycling suspected)")
                return False

        return True

    def _choose_entering_variable(self, objective_row: Sequence[float]) -> Optional[int]:
        """Choose entering variable for MINIMIZATION LP.

        CRITICAL FIX: We are solving a minimization problem (minimize cost).
        The entering variable should have the most NEGATIVE reduced cost.
        """
        entering_col = None
        best_value = -self.epsilon  # For minimization: most negative
        for idx, value in enumerate(objective_row):
            if value < best_value:  # Choose most negative (minimization!)
                best_value = value
                entering_col = idx

        if entering_col is None:
            # Log why no entering variable found
            min_rc = min(objective_row) if objective_row else 999
            logger.debug(f"[SIMPLEX] No entering variable (min reduced cost = {min_rc:.6f}, threshold = {-self.epsilon})")
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
        logger.info(f"[LP REPAIR] rebuild() called with {len(removed_task_ids)} removed tasks: {removed_task_ids}")

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
        tasks_with_zero_plans = 0

        for task in tasks:
            task_plans = self._enumerate_plans(base_route, base_cost, task)
            if not task_plans:
                tasks_with_zero_plans += 1
                logger.warning(f"[LP REPAIR] Task {task.task_id} has ZERO feasible plans - adding skip penalty")
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

        logger.info(f"[LP REPAIR] Generated {len(plans)} total plans for {len(tasks)} tasks "
                   f"(tasks_with_zero_plans={tasks_with_zero_plans})")

        if not plans:
            logger.warning(f"[LP REPAIR] NO PLANS generated - falling back to regret2")
            return self.planner.regret2_insertion(partial_route.copy(), [task.task_id for task in tasks])

        A, b = self._build_constraints(plan_by_task, plans)

        # CRITICAL DIAGNOSTIC: Verify constraint matrix and test a simple solution
        logger.debug(f"[LP VERIFY] Constraint matrix A: {len(A)} rows x {len(A[0]) if A else 0} cols")
        logger.debug(f"[LP VERIFY] RHS vector b: {b}")

        # Test if a simple solution (select first plan for each task) satisfies constraints
        test_solution = [0.0] * len(plans)
        for task_id, task_plans in plan_by_task.items():
            if task_plans:
                first_plan_idx = task_plans[0].variable_index
                test_solution[first_plan_idx] = 1.0

        # Verify test solution
        for row_idx, (row, rhs) in enumerate(zip(A, b)):
            row_result = sum(a * x for a, x in zip(row, test_solution))
            task_id = list(plan_by_task.keys())[row_idx] if row_idx < len(plan_by_task) else -1
            if abs(row_result - rhs) > 1e-6:
                logger.error(f"[LP VERIFY] CONSTRAINT VIOLATION! Task {task_id} row {row_idx}: "
                           f"Ax={row_result:.6f}, b={rhs:.6f}, diff={abs(row_result - rhs):.6f}")
            else:
                logger.debug(f"[LP VERIFY] Task {task_id} row {row_idx}: Ax={row_result:.6f}, b={rhs:.6f} ✓")

        # Check for zero rows
        for row_idx, row in enumerate(A):
            row_sum = sum(row)
            if row_sum < 1e-9:
                task_id = list(plan_by_task.keys())[row_idx] if row_idx < len(plan_by_task) else -1
                logger.error(f"[LP VERIFY] ZERO ROW! Task {task_id} row {row_idx} has sum={row_sum}")

        solver = SimplexSolver(A, b, column_costs)
        solution = solver.solve()
        if solution is None:
            logger.warning(f"[LP REPAIR] Simplex solver FAILED - falling back to regret2")
            return self.planner.regret2_insertion(partial_route.copy(), [task.task_id for task in tasks])

        logger.info(f"[LP REPAIR] Simplex solver SUCCESS - analyzing solution")

        assignment: Dict[int, _Plan] = {}
        skipped_tasks = 0
        for task_id, candidate_plans in plan_by_task.items():
            best_plan = max(candidate_plans, key=lambda plan: solution[plan.variable_index])
            if solution[best_plan.variable_index] < self.params.fractional_threshold:
                skipped_tasks += 1
                continue
            assignment[task_id] = best_plan
            logger.info(f"[LP REPAIR] Task {task_id} assigned: pickup_pos={best_plan.pickup_pos}, "
                       f"delivery_pos={best_plan.delivery_pos}, cost={best_plan.incremental_cost:.2f}")

        logger.info(f"[LP REPAIR] LP assigned {len(assignment)}/{len(tasks)} tasks (skipped={skipped_tasks})")

        rebuilt = base_route.copy()
        regret_fallback_count = 0
        for task in tasks:
            selected = assignment.get(task.task_id)
            if selected is None or selected.pickup_pos < 0:
                # Fallback to regret insertion for unassigned tasks.
                regret_fallback_count += 1
                rebuilt = self.planner.regret2_insertion(rebuilt, [task.task_id])
                continue
            self._apply_plan(rebuilt, task, selected)

        rebuilt_cost = self.planner.evaluate_cost(rebuilt)
        cost_improvement = base_cost - rebuilt_cost
        logger.info(f"[LP REPAIR] Rebuilt route: base_cost={base_cost:.2f}, "
                   f"rebuilt_cost={rebuilt_cost:.2f}, improvement={cost_improvement:.2f}, "
                   f"regret_fallbacks={regret_fallback_count}")

        if rebuilt_cost + self.params.improvement_tolerance >= base_cost:
            logger.warning(f"[LP REPAIR] NO IMPROVEMENT - falling back to full regret2 rebuild")
            return self.planner.regret2_insertion(partial_route.copy(), [task.task_id for task in tasks])

        logger.info(f"[LP REPAIR] SUCCESS - returning improved route")
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

        # Diagnostic counters
        total_positions_tried = 0
        battery_infeasible_count = 0
        charging_insertion_attempts = 0
        charging_insertion_successes = 0
        insertion_errors = 0

        for pickup_pos in range(1, max_pickup):
            for delivery_pos in range(pickup_pos + 1, max_pickup + 1):
                total_positions_tried += 1
                candidate = base_route.copy()
                try:
                    candidate.insert_task(task, (pickup_pos, delivery_pos))
                except ValueError:
                    insertion_errors += 1
                    continue

                # PHASE 0 FIX: Try to make infeasible routes feasible by adding charging
                if not self.planner.ensure_route_schedule(candidate):
                    battery_infeasible_count += 1
                    # Battery infeasible - try inserting charging stations
                    if hasattr(self.planner, '_insert_necessary_charging_stations'):
                        charging_insertion_attempts += 1
                        original_node_count = len(candidate.nodes)
                        candidate = self.planner._insert_necessary_charging_stations(candidate)
                        nodes_added = len(candidate.nodes) - original_node_count

                        # Re-check feasibility after charging station insertion
                        if not self.planner.ensure_route_schedule(candidate):
                            continue  # Still infeasible, discard this plan
                        else:
                            charging_insertion_successes += 1
                            logger.info(f"[LP REPAIR] Task {task.task_id}: Charging insertion SUCCESS - "
                                      f"added {nodes_added} nodes, plan now feasible")
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

        # Log diagnostic summary
        logger.info(f"[LP REPAIR] Task {task.task_id} plan enumeration: "
                   f"positions_tried={total_positions_tried}, "
                   f"battery_infeasible={battery_infeasible_count}, "
                   f"charging_attempts={charging_insertion_attempts}, "
                   f"charging_successes={charging_insertion_successes}, "
                   f"insertion_errors={insertion_errors}, "
                   f"feasible_plans={len(plans)}")

        plans.sort(key=lambda plan: plan.incremental_cost)
        return plans

    def _build_constraints(self, plan_by_task: Dict[int, List[_Plan]], plans: List[_Plan]) -> tuple[List[List[float]], List[float]]:
        task_ids = list(plan_by_task.keys())
        num_vars = len(plans)
        A: List[List[float]] = []
        b: List[float] = []

        logger.debug(f"[LP CONSTRAINTS] Building constraints for {len(task_ids)} tasks, {num_vars} variables")

        for task_id in task_ids:
            row = [0.0] * num_vars
            task_plan_count = 0
            for plan in plan_by_task[task_id]:
                if 0 <= plan.variable_index < num_vars:
                    row[plan.variable_index] = 1.0
                    task_plan_count += 1
                else:
                    logger.error(f"[LP CONSTRAINTS] INVALID variable_index={plan.variable_index} for task {task_id}, num_vars={num_vars}")
            A.append(row)
            b.append(1.0)

            row_sum = sum(row)
            logger.debug(f"[LP CONSTRAINTS] Task {task_id}: {task_plan_count} plans, row_sum={row_sum}, "
                        f"variable_indices={[p.variable_index for p in plan_by_task[task_id]]}")

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
