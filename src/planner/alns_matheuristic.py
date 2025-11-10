"""Matheuristic ALNS variant inspired by Singh et al. (2022).

The implementation augments :class:`planner.alns.MinimalALNS` with the key
ideas from *"A matheuristic for AGV scheduling with battery constraints"*:

* **Elite solution memory.**  High-quality routes are retained so the search
  can periodically intensify around them rather than relying purely on the
  simulated annealing walk.
* **Segment re-optimisation.**  After accepted moves, battery-critical route
  segments are rebuilt through an exhaustive (but size-bounded) enumeration
  that mimics the paper's MILP subproblem for the removed requests.
* **Adaptive intensification.**  Iterations periodically jump to the best
  combination of the incumbent and elite routes, tightening the solution space
  while respecting the shared charging and time constraints.

The upgrade keeps the existing destroy/repair operators, simulated annealing
temperature control, and cost function so it can slot into the current planner
without altering the surrounding orchestration code.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, replace
from typing import Callable, List, Optional, Sequence, Tuple

from config import (
    ALNSHyperParameters,
    CostParameters,
    LPRepairParams,
    MatheuristicParams,
    SegmentOptimizationParams,
    DEFAULT_LP_REPAIR_PARAMS,
    DEFAULT_MATHEURISTIC_PARAMS,
)
from core.route import Route
from core.task import Task
from planner.alns import MinimalALNS
from planner.repair_lp import LPBasedRepair


@dataclass
class _EliteRoute:
    """Container storing the solutions maintained in the elite pool."""

    cost: float
    fingerprint: Tuple[int, ...]
    route: Route


class MatheuristicALNS(MinimalALNS):
    """ALNS variant with elite-pool intensification and MILP-inspired repair."""

    def __init__(
        self,
        distance_matrix,
        task_pool,
        repair_mode: str = "mixed",
        cost_params: Optional[CostParameters] = None,
        charging_strategy=None,
        use_adaptive: bool = True,
        *,
        verbose: bool = True,
        adaptation_mode: str = "q_learning",  # NEW: Allow override
        hyper_params: Optional[ALNSHyperParameters] = None,
        matheuristic_params: Optional[MatheuristicParams] = None,
        adapt_matheuristic_params: bool = True,
        ) -> None:
        """Initialise the matheuristic ALNS solver.

        Args:
            adapt_matheuristic_params: When ``True`` (default) the solver will
                expand the provided parameters to scale-aware defaults for the
                detected scenario size.  Setting this to ``False`` preserves the
                caller's values verbatim, which is useful for benchmarking or
                experimenting with lighter-weight configurations.
        """

        super().__init__(
            distance_matrix=distance_matrix,
            task_pool=task_pool,
            repair_mode=repair_mode,
            cost_params=cost_params,
            charging_strategy=charging_strategy,
            use_adaptive=use_adaptive,
            verbose=verbose,
            adaptation_mode=adaptation_mode,  # NEW: Pass through
            hyper_params=hyper_params,
            repair_operators=('greedy', 'regret2', 'random', 'lp'),
        )

        if matheuristic_params is None:
            if hasattr(self.hyper, "matheuristic"):
                matheuristic_params = self.hyper.matheuristic
            else:
                matheuristic_params = DEFAULT_MATHEURISTIC_PARAMS

        if adapt_matheuristic_params:
            tuned_params = self._adapt_matheuristic_params(matheuristic_params)
        else:
            tuned_params = matheuristic_params
        self.matheuristic_params = tuned_params
        self._segment_optimizer = _SegmentOptimizer(self, tuned_params.segment_optimization)
        self._lp_repair = LPBasedRepair(self, getattr(tuned_params, 'lp_repair', DEFAULT_LP_REPAIR_PARAMS))
        self._accepted_since_segment = 0
        self._elite_pool: List[_EliteRoute] = []

    def _adapt_matheuristic_params(self, params: MatheuristicParams) -> MatheuristicParams:
        """Produce a scenario-aware copy of the matheuristic parameters."""

        scale = self._scenario_scale
        segment = params.segment_optimization
        lp_params = getattr(params, "lp_repair", DEFAULT_LP_REPAIR_PARAMS)

        updates = {}
        segment_updates = {}
        lp_updates = {}

        if scale == "medium":
            updates["segment_frequency"] = max(3, params.segment_frequency - 1)
            updates["intensification_interval"] = max(32, params.intensification_interval - 8)
            segment_updates = {
                "max_segment_tasks": max(segment.max_segment_tasks, 5),
                "candidate_pool_size": max(segment.candidate_pool_size, 6),
                "max_permutations": max(segment.max_permutations, 60),
                "lookahead_window": max(segment.lookahead_window, 4),
            }
        elif scale == "large":
            updates["segment_frequency"] = max(3, params.segment_frequency - 2)
            updates["intensification_interval"] = max(28, params.intensification_interval - 12)
            updates["elite_pool_size"] = params.elite_pool_size + 2
            updates["max_elite_trials"] = params.max_elite_trials + 1
            segment_updates = {
                "max_segment_tasks": max(segment.max_segment_tasks, 6),
                "candidate_pool_size": max(segment.candidate_pool_size, 8),
                "max_permutations": max(segment.max_permutations, 96),
                "lookahead_window": max(segment.lookahead_window, 5),
            }
            lp_updates = {
                "time_limit_s": max(lp_params.time_limit_s, 0.9),
                "max_plans_per_task": max(lp_params.max_plans_per_task, 8),
            }

        if segment_updates:
            segment = replace(segment, **segment_updates)
        if lp_updates:
            lp_params = replace(lp_params, **lp_updates)
        if updates or segment_updates or lp_updates:
            params = replace(
                params,
                segment_optimization=segment,
                lp_repair=lp_params,
                **updates,
            )
        return params

    def optimize(
        self,
        initial_route: Route,
        max_iterations: int = 100,
        *,
        progress_callback: Optional[
            Callable[[int, int, float, str, bool], None]
        ] = None,
    ) -> Route:
        """Run ALNS with matheuristic intensification steps."""

        self._maybe_reconfigure_q_agent(initial_route)
        max_iterations = self._normalise_iteration_budget(max_iterations)

        current_route = initial_route.copy()
        best_route = initial_route.copy()
        self._segment_optimizer._ensure_schedule(current_route)
        self._segment_optimizer._ensure_schedule(best_route)
        best_cost = self._safe_evaluate(best_route)

        temperature = self.initial_temp

        repair_usage = {op: 0 for op in self.repair_operators}
        destroy_usage = {op: 0 for op in ("random_removal", "partial_removal")}
        consecutive_no_improve = 0

        self._prepare_stagnation_thresholds(max_iterations)

        self._log(f"初始成本: {best_cost:.2f}m")
        self._log(f"总迭代次数: {max_iterations}")
        if self.use_adaptive:
            if self._use_q_learning:
                self._log("使用Q-Learning算子选择 ✓ (Destroy + Repair)")
            elif self.adaptation_mode == "roulette":
                self._log("使用自适应算子选择 ✓ (Destroy + Repair)")

        for iteration in range(max_iterations):
            if progress_callback is not None:
                progress_callback(iteration, max_iterations, best_cost, "start", False)
            q_state = self._determine_state(consecutive_no_improve)
            (
                destroy_operator,
                repair_operator,
                destroyed_route,
                removed_task_ids,
                candidate_route,
                q_action,
                action_runtime,
            ) = self._execute_destroy_repair(current_route, state=q_state)

            self._segment_optimizer._ensure_schedule(candidate_route)
            self._segment_optimizer._ensure_schedule(current_route)
            previous_cost = self._safe_evaluate(current_route)
            candidate_cost = self._safe_evaluate(candidate_route)
            original_candidate_cost = candidate_cost
            original_repair_operator = repair_operator

            fallback_used = False

            if (
                self._use_q_learning
                and q_action is not None
            ):
                fallback_operator = self._fallback_repair_operator(
                    state=q_state,
                    chosen_repair=repair_operator,
                )
                if fallback_operator and fallback_operator != repair_operator:
                    fallback_route, fallback_runtime = self._build_fallback_candidate(
                        destroyed_route=destroyed_route,
                        removed_task_ids=removed_task_ids,
                        fallback_operator=fallback_operator,
                        postprocess=self._segment_optimizer._ensure_schedule,
                    )
                    fallback_cost = self._safe_evaluate(fallback_route)
                    if fallback_cost + 1e-6 < candidate_cost:
                        candidate_route = fallback_route
                        candidate_cost = fallback_cost
                        repair_operator = fallback_operator
                        action_runtime = fallback_runtime
                        fallback_used = True
                        self._log_fallback_usage(
                            iteration=iteration,
                            state=q_state,
                            chosen=original_repair_operator,
                            fallback=fallback_operator,
                            original_cost=original_candidate_cost,
                            fallback_cost=fallback_cost,
                        )

            destroy_usage[destroy_operator] += 1
            repair_usage[repair_operator] += 1

            improvement = previous_cost - candidate_cost

            is_accepted = self.accept_solution(candidate_cost, previous_cost, temperature)
            is_new_best = False

            if is_accepted:
                current_route = candidate_route
                current_cost = candidate_cost
                self._accepted_since_segment += 1

                if (
                    self.matheuristic_params.segment_frequency > 0
                    and self._accepted_since_segment >= self.matheuristic_params.segment_frequency
                ):
                    improved_route = self._segment_optimizer.improve(current_route)
                    improved_cost = self._safe_evaluate(improved_route)
                    if improved_cost + self._segment_optimizer.params.improvement_tolerance < current_cost:
                        current_route = improved_route
                        current_cost = improved_cost
                        candidate_cost = improved_cost
                        improvement = max(improvement, previous_cost - candidate_cost)
                        self._log(
                            "  ↳ Matheuristic segment re-optimisation accepted"
                        )
                    self._accepted_since_segment = 0

                if current_cost < best_cost:
                    best_route = current_route.copy()
                    self._segment_optimizer._ensure_schedule(best_route)
                    best_cost = current_cost
                    is_new_best = True
                    self._log(f"迭代 {iteration+1}: 新最优成本 {best_cost:.2f}m")

            if self.use_adaptive and self.adaptation_mode == "roulette":
                self._update_adaptive_weights(
                    repair_operator=repair_operator,
                    destroy_operator=destroy_operator,
                    improvement=improvement,
                    is_new_best=is_new_best,
                    is_accepted=is_accepted,
                )

            if is_new_best:
                consecutive_no_improve = 0
            else:
                consecutive_no_improve += 1

            if (
                self._use_q_learning
                and q_state is not None
                and q_action is not None
            ):
                reward = self._compute_q_reward(
                    improvement=improvement,
                    is_new_best=is_new_best,
                    is_accepted=is_accepted,
                    action_cost=action_runtime,
                    repair_operator=repair_operator,
                    previous_cost=previous_cost,
                )
                if fallback_used:
                    reward = min(reward, -self._fallback_penalty_value())
                next_state = self._determine_state(consecutive_no_improve)
                self._q_agent.update(q_state, q_action, reward, next_state)
                self._q_agent.decay_epsilon()

            self._update_elite_pool(current_route)

            if (
                self.matheuristic_params.intensification_interval > 0
                and (iteration + 1) % self.matheuristic_params.intensification_interval == 0
            ):
                intensified_route = self._intensify(current_route)
                intensified_cost = self._safe_evaluate(intensified_route)
                if intensified_cost + self._segment_optimizer.params.improvement_tolerance < self._safe_evaluate(current_route):
                    current_route = intensified_route
                    if intensified_cost < best_cost:
                        best_cost = intensified_cost
                        best_route = intensified_route.copy()
                        self._segment_optimizer._ensure_schedule(best_route)
                        is_new_best = True
                        consecutive_no_improve = 0
                        self._log(
                            f"  ↳ Elite intensification improved cost to {best_cost:.2f}m"
                        )

            if progress_callback is not None:
                progress_callback(
                    iteration,
                    max_iterations,
                    best_cost,
                    "end",
                    is_new_best,
                )

            temperature *= self.cooling_rate

            if (iteration + 1) % 50 == 0:
                self._log(
                    f"  [进度] 已完成 {iteration+1}/{max_iterations} 次迭代, 当前最优: {best_cost:.2f}m"
                )

        if progress_callback is not None:
            progress_callback(max_iterations, max_iterations, best_cost, "complete", False)

        self._log("\n算子使用统计:")
        self._log(
            "  Repair: "
            + ", ".join(f"{op}={repair_usage[op]}" for op in self.repair_operators)
        )
        self._log(
            "  Destroy: "
            + ", ".join(f"{op}={destroy_usage[op]}" for op in ("random_removal", "partial_removal"))
        )
        self._log(
            f"最终最优成本: {best_cost:.2f}m (改进 {self._safe_evaluate(initial_route)-best_cost:.2f}m)"
        )

        if self.use_adaptive and self.adaptation_mode == "roulette" and self.verbose:
            self._log("\n" + "=" * 70)
            self._log("Repair算子自适应统计")
            self._log("=" * 70)
            self._log(self.adaptive_repair_selector.format_statistics())

            if self.adaptive_destroy_selector is not None:
                self._log("\n" + "=" * 70)
                self._log("Destroy算子自适应统计")
                self._log("=" * 70)
                self._log(self.adaptive_destroy_selector.format_statistics())

        if self._use_q_learning:
            self._log_q_learning_statistics()

        return best_route

    def _update_elite_pool(self, route: Route) -> None:
        """Store the best solutions found so far in an elite memory."""

        if not route.nodes:
            return

        self._segment_optimizer._ensure_schedule(route)
        fingerprint = tuple(node.node_id for node in route.nodes)
        cost = self._safe_evaluate(route)

        tolerance = self._segment_optimizer.params.improvement_tolerance
        for elite in self._elite_pool:
            if elite.fingerprint == fingerprint:
                if cost + tolerance < elite.cost:
                    elite.cost = cost
                    elite.route = route.copy()
                return

        self._elite_pool.append(_EliteRoute(cost=cost, fingerprint=fingerprint, route=route.copy()))
        self._elite_pool.sort(key=lambda entry: entry.cost)
        if len(self._elite_pool) > self.matheuristic_params.elite_pool_size:
            self._elite_pool = self._elite_pool[: self.matheuristic_params.elite_pool_size]

    def _intensify(self, current_route: Route) -> Route:
        """Return a promising elite solution for intensification."""

        if not self._elite_pool:
            return current_route

        trials = min(len(self._elite_pool), self.matheuristic_params.max_elite_trials)
        elite_choice = random.choice(self._elite_pool[:trials])
        elite_route = elite_choice.route.copy()

        improved_elite = self._segment_optimizer.improve(elite_route)
        elite_cost = self._safe_evaluate(improved_elite)
        current_cost = self._safe_evaluate(current_route)

        if elite_cost + self._segment_optimizer.params.improvement_tolerance < current_cost:
            return improved_elite
        return current_route

    def lp_insertion(self, route: Route, removed_task_ids: List[int]) -> Route:
        """Use the LP-based repair operator to rebuild removed tasks."""

        return self._lp_repair.rebuild(route, removed_task_ids)

    def _safe_evaluate(self, route: Route) -> float:
        """Evaluate the route cost after ensuring the schedule is available."""

        if not self._segment_optimizer._ensure_schedule(route):
            return float("inf")
        return super().evaluate_cost(route)


class _SegmentOptimizer:
    """Enumerative optimiser for battery-critical segments."""

    def __init__(self, planner: MatheuristicALNS, params: SegmentOptimizationParams) -> None:
        self.planner = planner
        self.params = params

    def improve(self, route: Route) -> Route:
        """Return a potentially improved copy of the given route."""

        segments = self._identify_segments(route)
        if not segments:
            return route

        incumbent_cost = self.planner._safe_evaluate(route)
        best_route = route
        best_cost = incumbent_cost

        for segment in segments:
            improved = self._optimise_segment(route, segment)
            if improved is None:
                continue
            improved_cost = self.planner._safe_evaluate(improved)
            if improved_cost + self.params.improvement_tolerance < best_cost:
                best_cost = improved_cost
                best_route = improved

        return best_route

    def _identify_segments(self, route: Route) -> List[List[int]]:
        """Detect segments with low battery slack for re-optimisation."""

        if route.get_num_nodes() <= 2:
            return []

        if not self._ensure_schedule(route):
            return []

        vehicle = getattr(self.planner, "vehicle", None)
        energy_config = getattr(self.planner, "energy_config", None)
        if vehicle is None or energy_config is None:
            return self._fallback_segments(route)

        warning_level = energy_config.warning_threshold * vehicle.battery_capacity

        if not route.visits:
            return self._fallback_segments(route)

        hotspots: List[int] = [
            idx
            for idx, visit in enumerate(route.visits)
            if visit.battery_after_service < warning_level and hasattr(visit.node, "task_id")
        ]

        candidate_segments: List[List[int]] = []
        seen = set()
        for hotspot in hotspots:
            task_ids = self._collect_tasks_around(route, hotspot)
            if not task_ids:
                continue
            signature = tuple(task_ids)
            if signature in seen:
                continue
            candidate_segments.append(task_ids)
            seen.add(signature)

        if not candidate_segments:
            return self._fallback_segments(route)

        return candidate_segments[: self.params.candidate_pool_size]

    def _collect_tasks_around(self, route: Route, visit_idx: int) -> List[int]:
        """Collect neighbouring tasks around a hotspot index."""

        lookahead = self.params.lookahead_window
        start = max(1, visit_idx - lookahead)
        end = min(len(route.nodes) - 1, visit_idx + lookahead)

        task_ids: List[int] = []
        for node in route.nodes[start : end + 1]:
            if not hasattr(node, "task_id"):
                continue
            if node.task_id in task_ids:
                continue
            task_ids.append(node.task_id)
            if len(task_ids) >= self.params.max_segment_tasks:
                break

        return task_ids

    def _fallback_segments(self, route: Route) -> List[List[int]]:
        """Fallback enumeration when no energy hotspots are detected."""

        pickup_order = [
            node.task_id
            for node in route.nodes
            if hasattr(node, "task_id") and getattr(node, "is_pickup", lambda: False)()
        ]

        if not pickup_order:
            return []

        windows: List[List[int]] = []
        max_size = min(self.params.max_segment_tasks, len(pickup_order))
        for size in range(max_size, 0, -1):
            for start in range(0, len(pickup_order) - size + 1):
                segment = pickup_order[start : start + size]
                windows.append(segment)
                if len(windows) >= self.params.candidate_pool_size:
                    return windows

        return windows[: self.params.candidate_pool_size]

    def _optimise_segment(self, route: Route, task_ids: Sequence[int]) -> Optional[Route]:
        """Attempt to rebuild the segment using limited enumeration."""

        tasks: List[Task] = []
        for task_id in task_ids:
            task = self.planner.task_pool.get_task(task_id)
            if task is None:
                return None
            tasks.append(task)

        base_route = route.copy()
        for task in tasks:
            base_route.remove_task(task)

        best_candidate: Optional[Route] = None
        best_cost = math.inf

        for attempt, ordering in enumerate(itertools.permutations(tasks), start=1):
            if attempt > self.params.max_permutations:
                break

            candidate = base_route.copy()
            valid = True
            for task in ordering:
                candidate = self.planner.regret2_insertion(candidate, [task.task_id])
                if not self._is_route_feasible(candidate):
                    valid = False
                    break

            if not valid:
                continue

            cost = self.planner._safe_evaluate(candidate)
            if cost + self.params.improvement_tolerance < best_cost:
                best_cost = cost
                best_candidate = candidate

        return best_candidate

    def _is_route_feasible(self, route: Route) -> bool:
        """Check capacity, time-window, and battery feasibility."""

        vehicle = getattr(self.planner, "vehicle", None)
        if vehicle is None:
            return True

        capacity_feasible, _ = route.check_capacity_feasibility(vehicle.capacity, debug=False)
        if not capacity_feasible:
            return False

        time_feasible, _ = self.planner._check_time_window_feasibility_fast(route)
        if not time_feasible:
            return False

        if hasattr(self.planner, "_check_battery_feasibility"):
            return self.planner._check_battery_feasibility(route)
        return True

    def _ensure_schedule(self, route: Route) -> bool:
        """Populate ``route.visits`` if the planner is fully configured."""

        return self.planner.ensure_route_schedule(route)

