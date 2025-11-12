"""Charging-aware Adaptive Large Neighbourhood Search implementation.

The module bundles the end-to-end ALNS workflow that powers the single-AMR
planner: destroy/repair operators, adaptive operator scoring, simulated annealing
acceptance, and cost evaluation that blends distance, time, lateness and
charging penalties.  It also embeds feasibility checks for capacity, time
windows, and the three-tier battery threshold model so that candidate solutions
stay compatible with the vehicle energy configuration and charging strategy.
"""

from collections import Counter
import math
import random
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from core.route import Route
from core.task import Task
from core.vehicle import Vehicle, create_vehicle
from physics.distance import DistanceMatrix
from physics.energy import EnergyConfig
from physics.time import TimeConfig
from planner.operators import AdaptiveOperatorSelector
from planner.epsilon_strategy import EpsilonStrategy
from planner.q_learning import Action, QLearningOperatorAgent
from config import (
    ALNSHyperParameters,
    CostParameters,
    DEFAULT_ALNS_HYPERPARAMETERS,
    DEFAULT_COST_PARAMETERS,
    DEFAULT_Q_LEARNING_PARAMS,
    QLearningParams,
)

class MinimalALNS:
    """Single-vehicle ALNS solver with adaptive destroy/repair operators.

    The implementation keeps track of operator usage, evaluates candidate routes
    through the weighted cost function, and coordinates simulated annealing
    acceptance with temperature cooling.  Destroy and repair choices can be
    deterministic, randomly mixed, or steered by adaptive scoring.  Charging
    feasibility is safeguarded via vehicle-aware battery simulation combined with
    the pluggable charging strategy.
    """

    def __init__(
        self,
        distance_matrix: DistanceMatrix,
        task_pool,
        repair_mode: str = 'mixed',
        cost_params: Optional[CostParameters] = None,
        charging_strategy=None,
        use_adaptive: bool = True,
        *,
        verbose: bool = True,
        adaptation_mode: str = "q_learning",
        hyper_params: Optional[ALNSHyperParameters] = None,
        repair_operators: Optional[Sequence[str]] = None,
        q_learning_initial_q: Optional[Dict[str, Dict[Action, float]]] = None,
        epsilon_strategy: Optional[EpsilonStrategy] = None,
    ):
        """Initialise the ALNS planner with distance data and candidate tasks."""
        self.distance = distance_matrix
        self.task_pool = task_pool
        self.repair_mode = repair_mode
        self.verbose = verbose

        self.cost_params = cost_params or DEFAULT_COST_PARAMETERS
        self.hyper = hyper_params or DEFAULT_ALNS_HYPERPARAMETERS
        self.charging_strategy = charging_strategy
        self.use_adaptive = use_adaptive or repair_mode == 'adaptive'
        self.repair_operators = list(repair_operators or ['greedy', 'regret2', 'random'])

        self._destroy_operators = ('random_removal', 'partial_removal')
        self.adaptation_mode = adaptation_mode
        self._use_q_learning = self.use_adaptive and adaptation_mode == 'q_learning'
        self._stagnation_threshold: Optional[int] = None
        self._deep_stagnation_threshold: Optional[int] = None

        self._scenario_task_count = self._estimate_task_count(task_pool)
        self._scenario_scale = self._infer_scenario_scale(self._scenario_task_count)
        self._q_state_labels = self._build_state_labels()

        self._q_params: QLearningParams = getattr(
            self.hyper, 'q_learning', DEFAULT_Q_LEARNING_PARAMS
        )

        self._matheuristic_repairs = {
            operator
            for operator in self.repair_operators
            if operator in {'lp'}
        }

        normalised_initial_q = self._normalise_initial_q_values(q_learning_initial_q)
        self._user_provided_initial_q = normalised_initial_q
        self._epsilon_strategy = epsilon_strategy
        if self._use_q_learning:
            initial_q_values = (
                normalised_initial_q or self._default_q_learning_initial_q()
            )
            self._q_agent = QLearningOperatorAgent(
                destroy_operators=self._destroy_operators,
                repair_operators=self.repair_operators,
                params=self._q_params,
                initial_q_values=initial_q_values,
                state_labels=self._q_state_labels,
                epsilon_strategy=epsilon_strategy,
            )
            if not self._q_params.enable_online_updates:
                self._q_agent.set_epsilon(self._q_params.epsilon_min)
            self.adaptive_repair_selector = None
            self.adaptive_destroy_selector = None
        elif self.use_adaptive and adaptation_mode == 'roulette':
            self.adaptive_repair_selector = AdaptiveOperatorSelector(
                operators=self.repair_operators,
                params=self.hyper.adaptive_selector
            )
            self.adaptive_destroy_selector = AdaptiveOperatorSelector(
                operators=list(self._destroy_operators),
                params=self.hyper.adaptive_selector
            )
            self._q_agent = None
        else:
            self.adaptive_repair_selector = None
            self.adaptive_destroy_selector = None
            self._q_agent = None
        self.initial_temp = self.hyper.simulated_annealing.initial_temperature
        self.cooling_rate = self.hyper.simulated_annealing.cooling_rate

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    # ------------------------------------------------------------------
    # Scenario-aware helpers
    # ------------------------------------------------------------------
    def _estimate_task_count(self, task_pool) -> int:
        """Best-effort estimation of the optimisation task count."""

        if hasattr(task_pool, "get_all_tasks"):
            try:
                return len(task_pool.get_all_tasks())
            except TypeError:
                pass
        if hasattr(task_pool, "tasks"):
            try:
                return len(task_pool.tasks)  # type: ignore[attr-defined]
            except TypeError:
                pass
        if hasattr(task_pool, "__len__"):
            try:
                return int(len(task_pool))  # type: ignore[arg-type]
            except TypeError:
                pass
        return 0

    def _infer_scenario_scale(self, task_count: int) -> str:
        """Map the task count to a coarse optimisation scale label."""

        if task_count >= 40:
            return "large"
        if task_count >= 22:
            return "medium"
        return "small"

    def _build_state_labels(self) -> Tuple[str, ...]:
        """Return the Q-learning state labels (kept stable for compatibility)."""

        return ("explore", "stuck", "deep_stuck")

    def _compose_state_label(self, phase: str) -> str:
        return phase

    def _extract_phase(self, state: Optional[str]) -> str:
        if not state:
            return "explore"
        if ":" in state:
            return state.split(":", 1)[1]
        return state

    def _normalise_iteration_budget(self, requested_iterations: int) -> int:
        """Scale the iteration budget when callers rely on the default."""

        if requested_iterations != 100:
            return requested_iterations

        task_count = self._scenario_task_count
        if task_count <= 0:
            # Fall back to coarse defaults when the pool size is unknown.
            recommendations = {"small": 150, "medium": 220, "large": 320}
            return recommendations.get(self._scenario_scale, 180)

        if task_count >= 48:
            return 430
        if task_count >= 36:
            return 360
        if task_count >= 26:
            return 280
        if task_count >= 18:
            return 220
        return 170

    def _maybe_reconfigure_q_agent(self, initial_route: Route) -> None:
        """Rebuild the Q-agent if the scenario scale becomes known later."""

        if not self._use_q_learning:
            return

        if self._scenario_task_count > 0:
            return

        task_count = len(initial_route.get_served_tasks())
        if task_count <= 0:
            return

        self._scenario_task_count = task_count
        new_scale = self._infer_scenario_scale(task_count)
        if new_scale == self._scenario_scale:
            return

        self._scenario_scale = new_scale
        self._q_state_labels = self._build_state_labels()
        initial_q_values = (
            self._user_provided_initial_q or self._default_q_learning_initial_q()
        )
        self._q_agent = QLearningOperatorAgent(
            destroy_operators=self._destroy_operators,
            repair_operators=self.repair_operators,
            params=self._q_params,
            initial_q_values=initial_q_values,
            state_labels=self._q_state_labels,
        )
        if not self._q_params.enable_online_updates:
            self._q_agent.set_epsilon(self._q_params.epsilon_min)

    def _destroy_fraction_config(
        self, task_count: int
    ) -> Tuple[Dict[str, float], Dict[str, int]]:
        """Return destroy fractions/minimums tailored to the active route size."""

        base_fraction = {"explore": 0.12, "stuck": 0.22, "deep_stuck": 0.35}
        base_min_remove = {"explore": 2, "stuck": 3, "deep_stuck": 4}

        # Gradually request stronger destroys on larger routes while capping the
        # aggressiveness to avoid destabilising the repair phase.
        if task_count >= 45:
            size_boost = 0.10
            min_increment = 3
        elif task_count >= 28:
            size_boost = 0.05
            min_increment = 1
        else:
            size_boost = 0.0
            min_increment = 0

        phase_bias = {"explore": 0.7, "stuck": 1.0, "deep_stuck": 1.3}
        fraction_map = {
            phase: min(0.55, base_fraction[phase] + size_boost * phase_bias[phase])
            for phase in base_fraction
        }

        min_remove = {
            phase: base_min_remove[phase] + min_increment
            for phase in base_min_remove
        }

        return fraction_map, min_remove

    def _normalise_initial_q_values(
        self, initial_q: Optional[Dict[str, Dict[Action, float]]]
    ) -> Optional[Dict[str, Dict[Action, float]]]:
        if not initial_q:
            return None

        normalised: Dict[str, Dict[Action, float]] = {}
        for state_key, action_map in initial_q.items():
            phase = self._extract_phase(state_key)
            label = self._compose_state_label(phase)
            state_values: Dict[Action, float] = {}
            for action, value in action_map.items():
                state_values[action] = float(value)
            normalised[label] = state_values
        return normalised

    def _scenario_time_penalty_threshold(self, base_threshold: float) -> float:
        """Relax runtime penalties for expensive operators on smaller cases."""

        scale = self._scenario_scale
        if scale == "small":
            return base_threshold * 1.75
        if scale == "medium":
            return base_threshold * 1.45
        return base_threshold

    def _scenario_penalty_factor(self, *, is_matheuristic: bool) -> float:
        if not is_matheuristic:
            return 1.0
        scale = self._scenario_scale
        if scale == "small":
            return 0.55
        if scale == "medium":
            return 0.72
        return 1.0

    def _fallback_repair_operator(
        self,
        *,
        state: Optional[str],
        chosen_repair: str,
    ) -> Optional[str]:
        """Return a safeguard repair operator when Q-learning underperforms."""

        if not self._use_q_learning:
            return None

        phase = self._extract_phase(state)
        scale = self._scenario_scale

        prefers_lp = (
            "lp" in self.repair_operators
            and chosen_repair != "lp"
            and (phase in {"stuck", "deep_stuck"} or scale == "large")
        )
        if prefers_lp:
            return "lp"

        if "regret2" in self.repair_operators and chosen_repair != "regret2":
            return "regret2"
        if "greedy" in self.repair_operators and chosen_repair != "greedy":
            return "greedy"
        return None

    def _build_fallback_candidate(
        self,
        *,
        destroyed_route: Route,
        removed_task_ids: List[int],
        fallback_operator: str,
        postprocess: Optional[Callable[[Route], None]] = None,
    ) -> Tuple[Route, float]:
        """Construct a fallback candidate and return it with the runtime."""

        route_copy = destroyed_route.copy()
        start = time.perf_counter()
        candidate = self._run_repair_operator(
            fallback_operator,
            route_copy,
            list(removed_task_ids),
        )
        if postprocess is not None:
            postprocess(candidate)
        runtime = time.perf_counter() - start
        return candidate, runtime

    def _fallback_penalty_value(self) -> float:
        params = self._q_params or DEFAULT_Q_LEARNING_PARAMS
        return abs(params.reward_rejected) + params.reward_improvement

    def _log_fallback_usage(
        self,
        *,
        iteration: int,
        state: Optional[str],
        chosen: str,
        fallback: str,
        original_cost: float,
        fallback_cost: float,
    ) -> None:
        if not self.verbose:
            return

        phase = self._extract_phase(state)
        self._log(
            "  [Q-Guard] 迭代 "
            f"{iteration + 1}: {phase}阶段将修复算子 {chosen} → {fallback} "
            f"(成本 {original_cost:.2f} → {fallback_cost:.2f})"
        )

    def _scenario_roi_multiplier(self, improvement: float) -> float:
        if improvement <= 0:
            return 1.0
        scale = self._scenario_scale
        if scale == "small":
            return 1.45
        if scale == "medium":
            return 1.25
        return 1.0

    def _apply_destroy_operator(
        self,
        route: Route,
        destroy_params: Optional[Tuple[int, float]] = None,
    ) -> Tuple[str, Route, List[int]]:
        if self.use_adaptive and self.adaptive_destroy_selector is not None:
            operator = self.adaptive_destroy_selector.select()
        else:
            operator = 'random_removal'

        q, remove_prob = self._resolve_destroy_params(route, destroy_params)

        if operator == 'partial_removal':
            destroyed_route, removed_tasks = self.partial_removal(route, q=q)
        else:
            operator = 'random_removal'
            destroyed_route, removed_tasks = self.random_removal(
                route, q=q, remove_cs_prob=remove_prob
            )

        return operator, destroyed_route, removed_tasks

    def _apply_repair_operator(self, route: Route, removed_task_ids: List[int]) -> Tuple[str, Route]:
        if self.use_adaptive and self.adaptive_repair_selector is not None:
            operator = self.adaptive_repair_selector.select()
        else:
            operator = self._deterministic_repair_choice()

        repaired_route = self._execute_repair_operator(operator, route, removed_task_ids)
        return operator, repaired_route

    def _execute_destroy_repair(
        self,
        current_route: Route,
        *,
        state: Optional[str] = None,
    ) -> Tuple[
        str,
        str,
        Route,
        List[int],
        Route,
        Optional[Tuple[str, str]],
        float,
    ]:
        destroy_params = self._select_destroy_parameters(current_route, state)

        if self._use_q_learning:
            if state is None:
                raise ValueError("state must be provided when using Q-learning adaptation")
            mask = self._build_action_mask(state)
            action = self._q_agent.select_action(state, mask=mask)
            destroy_operator, repair_operator = action
            start_time = time.perf_counter()
            destroyed_route, removed_task_ids = self._run_destroy_operator(
                destroy_operator, current_route, destroy_params
            )
            candidate_route = self._run_repair_operator(
                repair_operator, destroyed_route, removed_task_ids
            )
            elapsed = time.perf_counter() - start_time
            return (
                destroy_operator,
                repair_operator,
                destroyed_route,
                removed_task_ids,
                candidate_route,
                action,
                elapsed,
            )

        destroy_operator, destroyed_route, removed_task_ids = self._apply_destroy_operator(
            current_route, destroy_params
        )
        repair_operator, candidate_route = self._apply_repair_operator(
            destroyed_route, removed_task_ids
        )
        return (
            destroy_operator,
            repair_operator,
            destroyed_route,
            removed_task_ids,
            candidate_route,
            None,
            0.0,
        )

    def _deterministic_repair_choice(self) -> str:
        if self.repair_mode in self.repair_operators:
            return self.repair_mode
        if self.repair_mode == 'mixed':
            return random.choice(self.repair_operators)
        return self.repair_operators[0]

    def _execute_repair_operator(self, operator: str, route: Route, removed_task_ids: List[int]) -> Route:
        if operator == 'greedy':
            return self.greedy_insertion(route, removed_task_ids)
        if operator == 'regret2':
            return self.regret2_insertion(route, removed_task_ids)
        if operator == 'lp':
            return self.lp_insertion(route, removed_task_ids)
        return self.random_insertion(route, removed_task_ids)

    def _run_destroy_operator(
        self,
        operator: str,
        route: Route,
        destroy_params: Optional[Tuple[int, float]] = None,
    ) -> Tuple[Route, List[int]]:
        q, remove_prob = self._resolve_destroy_params(route, destroy_params)
        if operator == 'partial_removal':
            return self.partial_removal(route, q=q)
        if operator == 'random_removal':
            return self.random_removal(route, q=q, remove_cs_prob=remove_prob)
        raise ValueError(f"未知的Destroy算子: {operator}")

    def _run_repair_operator(
        self,
        operator: str,
        route: Route,
        removed_task_ids: List[int],
    ) -> Route:
        return self._execute_repair_operator(operator, route, removed_task_ids)

    def _prepare_stagnation_thresholds(self, max_iterations: int) -> None:
        """Scale stagnation thresholds to the upcoming iteration budget."""

        params = self._q_params or DEFAULT_Q_LEARNING_PARAMS
        scaled_stuck = max(
            1,
            int(round(max_iterations * params.stagnation_ratio)),
        )
        scaled_deep = max(
            scaled_stuck + 1,
            int(round(max_iterations * params.deep_stagnation_ratio)),
        )

        self._stagnation_threshold = min(params.stagnation_threshold, scaled_stuck)
        self._deep_stagnation_threshold = min(
            params.deep_stagnation_threshold,
            scaled_deep,
        )

    def _determine_state(self, consecutive_no_improve: int) -> str:
        stuck_threshold = self._stagnation_threshold
        deep_threshold = self._deep_stagnation_threshold

        if stuck_threshold is None or deep_threshold is None:
            params = self._q_params or DEFAULT_Q_LEARNING_PARAMS
            stuck_threshold = params.stagnation_threshold
            deep_threshold = params.deep_stagnation_threshold

        if consecutive_no_improve >= deep_threshold:
            return self._compose_state_label('deep_stuck')
        if consecutive_no_improve >= stuck_threshold:
            return self._compose_state_label('stuck')
        return self._compose_state_label('explore')

    def _default_q_learning_initial_q(self) -> Dict[str, Dict[Action, float]]:
        """Return expert-informed initial Q-values for destroy/repair pairs."""

        default_value = 8.5
        scale = self._scenario_scale

        common = {
            'explore': {'regret2': 15.0, 'greedy': 13.0, 'lp': 12.0, 'random': 6.0},
            'stuck': {'lp': 26.0, 'regret2': 16.0, 'greedy': 11.0, 'random': 6.0},
            'deep_stuck': {'lp': 34.0, 'regret2': 14.0, 'greedy': 10.0, 'random': 6.0},
        }

        scale_bias = {
            'small': {
                'explore': {'lp': 6.0, 'regret2': 1.0},
                'stuck': {'lp': 8.0, 'regret2': 1.0},
                'deep_stuck': {'lp': 10.0},
            },
            'medium': {
                'explore': {'lp': 4.0},
                'stuck': {'lp': 6.0},
                'deep_stuck': {'lp': 8.0},
            },
            'large': {
                'explore': {'lp': 9.0, 'regret2': -1.0},
                'stuck': {'lp': 12.0},
                'deep_stuck': {'lp': 14.0},
            },
        }

        adjustments = scale_bias.get(scale, {})

        initial_values: Dict[str, Dict[Action, float]] = {}
        for phase, repair_map in common.items():
            label = self._compose_state_label(phase)
            phase_adjust = adjustments.get(phase, {})
            state_values: Dict[Action, float] = {}
            for destroy in self._destroy_operators:
                for repair in self.repair_operators:
                    base = repair_map.get(repair, default_value)
                    bonus = phase_adjust.get(repair, 0.0)
                    state_values[(destroy, repair)] = base + bonus
            initial_values[label] = state_values
        return initial_values

    def _default_q_learning_initial_q(self) -> Dict[str, Dict[Action, float]]:
        """Return expert-informed initial Q-values for destroy/repair pairs."""

        base_values = {
            'explore': {
                'lp': 15.0,
                'regret2': 12.0,
                'greedy': 10.0,
                'random': 5.0,
            },
            'stuck': {
                'lp': 30.0,
                'regret2': 12.0,
                'greedy': 10.0,
                'random': 5.0,
            },
            'deep_stuck': {
                'lp': 35.0,
                'regret2': 12.0,
                'greedy': 10.0,
                'random': 5.0,
            },
        }
        default_value = 8.0

        initial_values: Dict[str, Dict[Action, float]] = {}
        for state, repair_map in base_values.items():
            state_values: Dict[Action, float] = {}
            for destroy in self._destroy_operators:
                for repair in self.repair_operators:
                    value = repair_map.get(repair, default_value)
                    state_values[(destroy, repair)] = value
            initial_values[state] = state_values
        return initial_values

    def _compute_q_reward(
        self,
        *,
        improvement: float,
        is_new_best: bool,
        is_accepted: bool,
        action_cost: float,
        repair_operator: str,
        previous_cost: float,
    ) -> float:
        """Compute Q-learning reward with ROI-aware time penalty.

        The reward function now implements a "Return on Investment" concept:
        - Expensive operators (matheuristic) MUST deliver high quality to justify cost
        - Cheap operators can be used more liberally even with modest returns
        - Time penalty scales based on both action cost AND quality outcome

        This addresses the core issue where Q-learning couldn't learn the value
        of expensive operators because they were penalized equally regardless of
        their contribution.
        """
        params = self._q_params or DEFAULT_Q_LEARNING_PARAMS

        # Step 1: Determine quality-based reward
        if is_new_best:
            quality = params.reward_new_best
        elif improvement > 0:
            quality = params.reward_improvement
        elif is_accepted:
            quality = params.reward_accepted
        else:
            quality = params.reward_rejected

        # Step 2: Incorporate relative quality change (ROI boost)
        baseline_cost = max(previous_cost, 1.0)
        relative_gain = improvement / baseline_cost
        if improvement > 0:
            quality += (
                relative_gain
                * params.roi_positive_scale
                * self._scenario_roi_multiplier(improvement)
            )
        elif improvement < 0:
            quality += relative_gain * params.roi_negative_scale

        # Step 3: Apply ROI-aware time penalty
        penalty = 0.0
        threshold = self._scenario_time_penalty_threshold(
            params.time_penalty_threshold
        )
        if action_cost > threshold:
            is_matheuristic = repair_operator in self._current_matheuristic_repairs()

            if is_matheuristic:
                # Matheuristic operators: Demand high returns
                if quality >= params.reward_new_best:
                    # Best solution found: minimal penalty, this is exactly what we want
                    scale = 1.0
                elif quality >= params.reward_improvement:
                    # Improvement: moderate penalty, good but could be better
                    scale = params.time_penalty_positive_scale
                else:
                    # No improvement: heavy penalty, expensive operator wasted
                    scale = params.time_penalty_negative_scale
            else:
                # Standard (cheap) operators: lenient penalty
                scale = params.standard_time_penalty_scale

            penalty = action_cost * scale
            penalty *= self._scenario_penalty_factor(
                is_matheuristic=is_matheuristic
            )

        return quality - penalty

    def _select_destroy_parameters(
        self, route: Route, state: Optional[str]
    ) -> Optional[Tuple[int, float]]:
        """Pick a destroy strength based on the current stagnation state."""

        phase = self._extract_phase(state)
        task_count = len(route.get_served_tasks())

        if task_count == 0:
            return None

        fraction_map, min_remove = self._destroy_fraction_config(task_count)

        fraction = fraction_map.get(phase, fraction_map['explore'])
        candidate_q = int(math.ceil(task_count * fraction))
        minimum = min_remove.get(phase, 2)

        if task_count <= minimum:
            q = task_count
        else:
            q = max(minimum, candidate_q)

        q = max(1, min(task_count, q))

        base_prob = self.hyper.destroy_repair.remove_cs_probability
        stagnation_boost = {"explore": 0.0, "stuck": 0.14, "deep_stuck": 0.28}
        scale_bonus = 0.05 if task_count >= 45 else 0.02 if task_count >= 28 else 0.0

        remove_prob = min(
            1.0,
            max(0.0, base_prob + stagnation_boost.get(phase, 0.0) + scale_bonus),
        )

        if task_count <= 1:
            remove_prob = 0.0

        return q, remove_prob

    def _build_action_mask(self, state: str) -> List[bool]:
        """Filter operator pairs that violate expert guidance for the state.

        CRITICAL INSIGHT: We REMOVED the explore-phase LP blocking that was
        preventing Q-learning from ever discovering LP's value. The previous
        implementation had a fatal flaw:
        - Explore phase: LP blocked by mask
        - By the time 'stuck' was reached, epsilon had decayed to ~0
        - Q-learning never got a chance to learn LP's ROI

        New strategy:
        1. Explore: ALLOW ALL operators - let Q-learning learn through ROI rewards
        2. Stuck: ALLOW ALL operators - Q-learning should prefer LP by now
        3. Deep_stuck: FORCE matheuristic - guaranteed escape from local optimum

        The ROI-aware reward function is sufficient to guide learning without
        heavy-handed action masking.
        """

        if not self._use_q_learning:
            return []

        matheuristic_repairs = self._current_matheuristic_repairs()
        if not matheuristic_repairs:
            # No matheuristic operators available, allow all
            return [True] * len(self._q_agent.actions)

        mask: List[bool] = []

        phase = self._extract_phase(state)

        for destroy, repair in self._q_agent.actions:
            allowed = True
            is_matheuristic_repair = repair in matheuristic_repairs

            # Rule 1: Explore phase - ALLOW ALL (removed LP blocking!)
            # Q-learning needs to try LP early to learn its ROI value
            # The ROI-aware reward will naturally discourage wasteful LP usage
            if phase == 'explore':
                # Allow everything - trust the ROI-aware rewards
                pass

            # Rule 2: Stuck phase - ALLOW ALL
            # Q-learning should have learned LP's value by now
            elif phase == 'stuck':
                # Allow everything - Q-learning decides based on learned Q-values
                pass

            # Rule 3: Deep stuck - FORCE matheuristic (only strict rule)
            # At this point we override Q-learning for guaranteed escape
            elif phase == 'deep_stuck':
                if not is_matheuristic_repair:
                    allowed = False

            mask.append(allowed)

        return mask

    def _current_matheuristic_repairs(self) -> set:
        """Return the set of expensive repair operators available."""

        candidates = {'lp'}
        available = {op for op in self.repair_operators if op in candidates}
        if available != self._matheuristic_repairs:
            self._matheuristic_repairs = available
        return available

    def _resolve_destroy_params(
        self, route: Route, destroy_params: Optional[Tuple[int, float]]
    ) -> Tuple[int, float]:
        """Normalise destroy parameters before invoking an operator."""

        task_count = len(route.get_served_tasks())
        default_prob = self.hyper.destroy_repair.remove_cs_probability

        if destroy_params is None or task_count == 0:
            q = 0 if task_count == 0 else (2 if task_count > 1 else 1)
            remove_prob = 0.0 if task_count <= 1 else default_prob
        else:
            q, remove_prob = destroy_params

        if task_count > 0:
            q = max(1, min(task_count, int(round(q))))
        else:
            q = 0

        remove_prob = float(remove_prob)
        remove_prob = max(0.0, min(1.0, remove_prob))

        if task_count <= 1:
            remove_prob = 0.0

        return q, remove_prob

    def _log_q_learning_statistics(self) -> None:
        if not self._use_q_learning or not self.verbose:
            return

        self._log("\n" + "=" * 70)
        self._log("Q-Learning算子统计")
        self._log("=" * 70)
        self._log(self._q_agent.format_statistics())

    def _update_adaptive_weights(
        self,
        *,
        repair_operator: str,
        destroy_operator: str,
        improvement: float,
        is_new_best: bool,
        is_accepted: bool,
    ) -> None:
        self.adaptive_repair_selector.update(
            operator=repair_operator,
            improvement=improvement,
            is_new_best=is_new_best,
            is_accepted=is_accepted,
        )

        if self.adaptive_destroy_selector is not None:
            self.adaptive_destroy_selector.update(
                operator=destroy_operator,
                improvement=improvement,
                is_new_best=is_new_best,
                is_accepted=is_accepted,
            )
    
    def optimize(self,
                 initial_route: Route,
                 max_iterations: int = 100) -> Route:
        """
        ALNS主循环
        
        参数：
            initial_route: 初始路径
            max_iterations: 迭代次数
        
        返回：
            优化后的最佳路径
        """
        self._maybe_reconfigure_q_agent(initial_route)
        max_iterations = self._normalise_iteration_budget(max_iterations)

        current_route = initial_route.copy()
        best_route = initial_route.copy()
        best_cost = self.evaluate_cost(best_route)

        temperature = self.initial_temp

        repair_usage: Counter[str] = Counter()
        destroy_usage: Counter[str] = Counter()
        consecutive_no_improve = 0

        self._log(f"初始成本: {best_cost:.2f}m")
        self._log(f"总迭代次数: {max_iterations}")
        if self.use_adaptive:
            if self._use_q_learning:
                self._log("使用Q-Learning算子选择 ✓ (Destroy + Repair)")
            elif self.adaptation_mode == 'roulette':
                self._log("使用自适应算子选择 ✓ (Destroy + Repair)")

        self._prepare_stagnation_thresholds(max_iterations)

        for iteration in range(max_iterations):
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

            # 评估成本
            previous_cost = self.evaluate_cost(current_route)
            candidate_cost = self.evaluate_cost(candidate_route)
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
                    )
                    fallback_cost = self.evaluate_cost(fallback_route)
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

            # 计算改进量
            improvement = previous_cost - candidate_cost

            # 接受准则
            is_accepted = self.accept_solution(candidate_cost, previous_cost, temperature)
            is_new_best = False

            if is_accepted:
                current_route = candidate_route
                if candidate_cost < best_cost:
                    best_route = candidate_route
                    best_cost = candidate_cost
                    is_new_best = True
                    self._log(f"迭代 {iteration+1}: 新最优成本 {best_cost:.2f}m")

            if self.use_adaptive and self.adaptation_mode == 'roulette':
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

            # 降温
            temperature *= self.cooling_rate

            # 进度报告
            if (iteration + 1) % 50 == 0:
                self._log(f"  [进度] 已完成 {iteration+1}/{max_iterations} 次迭代, 当前最优: {best_cost:.2f}m")

        repair_summary = ", ".join(
            f"{op}={repair_usage[op]}" for op in self.repair_operators
        )
        destroy_summary = ", ".join(
            f"{op}={destroy_usage[op]}" for op in ('random_removal', 'partial_removal')
        )

        self._log("\n算子使用统计:")
        self._log(f"  Repair: {repair_summary}")
        self._log(f"  Destroy: {destroy_summary}")
        self._log(f"最终最优成本: {best_cost:.2f}m (改进 {self.evaluate_cost(initial_route)-best_cost:.2f}m)")

        if self.adaptation_mode == 'roulette' and self.use_adaptive and self.verbose:
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
    
    def random_removal(
        self,
        route: Route,
        q: Optional[int] = None,
        remove_cs_prob: Optional[float] = None,
    ) -> Tuple[Route, List[int]]:
        """
        Destroy算子：随机移除q个任务 + 可选地移除充电站

        Week 2改进：支持充电站动态优化

        参数:
            route: 当前路径
            q: 移除的任务数量
            remove_cs_prob: 移除充电站的概率 (0.0-1.0)

        返回:
            (destroyed_route, removed_task_ids)
        """
        task_ids = route.get_served_tasks()

        if q is None:
            q = self.hyper.destroy_repair.random_removal_q
        if remove_cs_prob is None:
            remove_cs_prob = self.hyper.destroy_repair.remove_cs_probability

        if len(task_ids) < q:
            q = max(1, len(task_ids))

        if len(task_ids) == 0:
            return route.copy(), []

        removed_task_ids = random.sample(task_ids, q)

        destroyed_route = route.copy()

        # 移除任务
        for task_id in removed_task_ids:
            task = self.task_pool.get_task(task_id)
            destroyed_route.remove_task(task)

        if random.random() < remove_cs_prob:
            cs_nodes = [n for n in destroyed_route.nodes if n.is_charging_station()]

            if len(cs_nodes) > 0:
                # 随机决定移除多少个充电站（0-2个）
                num_to_remove = random.randint(0, min(2, len(cs_nodes)))

                if num_to_remove > 0:
                    removed_cs = random.sample(cs_nodes, num_to_remove)

                    # 移除充电站
                    for cs in removed_cs:
                        destroyed_route.nodes.remove(cs)

        return destroyed_route, removed_task_ids

    def partial_removal(
        self,
        route: Route,
        q: Optional[int] = None,
    ) -> Tuple[Route, List[int]]:
        """
        Destroy算子：只移除delivery节点（仓储场景迭代步骤2.2）

        功能:
            - 随机选择q个任务
            - 只移除这些任务的delivery节点
            - 保留pickup节点在路径中
            - 允许repair阶段重新选择delivery位置

        返回:
            (destroyed_route, removed_task_ids)
            其中removed_task_ids表示需要重新插入delivery的任务
        """
        task_ids = route.get_served_tasks()

        if q is None:
            q = self.hyper.destroy_repair.partial_removal_q

        if len(task_ids) < q:
            q = max(1, len(task_ids))

        if len(task_ids) == 0:
            return route.copy(), []

        # 随机选择要移除delivery的任务
        selected_task_ids = random.sample(task_ids, q)

        destroyed_route = route.copy()
        for task_id in selected_task_ids:
            task = self.task_pool.get_task(task_id)

            # 只移除delivery节点，保留pickup节点
            # 找到delivery节点并移除
            delivery_node_id = task.delivery_node.node_id
            destroyed_route.nodes = [
                node for node in destroyed_route.nodes
                if node.node_id != delivery_node_id
            ]

        return destroyed_route, selected_task_ids

    def pair_exchange(self, route: Route) -> Route:
        """
        Local search算子：交换两个任务的位置（仓储场景迭代步骤2.3）

        功能:
            - 随机选择两个任务
            - 交换它们在路径中的位置
            - 保持precedence约束（pickup在delivery之前）
            - 这是一个2-opt类型的local search

        返回:
            交换后的路径
        """
        task_ids = route.get_served_tasks()

        if len(task_ids) < 2:
            return route.copy()  # 少于2个任务，无法交换

        # 随机选择两个不同的任务
        task1_id, task2_id = random.sample(task_ids, 2)
        task1 = self.task_pool.get_task(task1_id)
        task2 = self.task_pool.get_task(task2_id)

        # 找到两个任务在路径中的位置
        task1_pickup_pos = None
        task1_delivery_pos = None
        task2_pickup_pos = None
        task2_delivery_pos = None

        for i, node in enumerate(route.nodes):
            if hasattr(node, 'task_id'):
                if node.task_id == task1_id:
                    if node.is_pickup():
                        task1_pickup_pos = i
                    elif node.is_delivery():
                        task1_delivery_pos = i
                elif node.task_id == task2_id:
                    if node.is_pickup():
                        task2_pickup_pos = i
                    elif node.is_delivery():
                        task2_delivery_pos = i

        # 检查是否找到所有节点
        if None in [task1_pickup_pos, task1_delivery_pos, task2_pickup_pos, task2_delivery_pos]:
            return route.copy()  # 无法找到完整任务，返回原路径

        # 创建新路径进行交换
        exchanged_route = route.copy()

        # 提取节点
        task1_pickup = task1.pickup_node
        task1_delivery = task1.delivery_node
        task2_pickup = task2.pickup_node
        task2_delivery = task2.delivery_node

        # 移除所有四个节点（从后往前移除，避免索引变化）
        positions = sorted([task1_pickup_pos, task1_delivery_pos, task2_pickup_pos, task2_delivery_pos], reverse=True)
        for pos in positions:
            exchanged_route.nodes.pop(pos)

        # 重新插入，交换两个任务的相对顺序
        # 策略：保持pickup-delivery的相对距离，但交换任务顺序

        # 简化策略：将task2插入到task1原来的位置，task1插入到task2原来的位置
        # 需要确保precedence约束

        # 找到最小和最大位置
        min_pos = min(task1_pickup_pos, task2_pickup_pos)

        # 按照新顺序插入
        if task1_pickup_pos < task2_pickup_pos:
            # 原顺序：task1在前，task2在后
            # 交换后：task2在前，task1在后
            # 在min_pos位置插入task2
            exchanged_route.nodes.insert(min_pos, task2_pickup)
            exchanged_route.nodes.insert(min_pos + 1, task2_delivery)
            exchanged_route.nodes.insert(min_pos + 2, task1_pickup)
            exchanged_route.nodes.insert(min_pos + 3, task1_delivery)
        else:
            # 原顺序：task2在前，task1在后
            # 交换后：task1在前，task2在后
            exchanged_route.nodes.insert(min_pos, task1_pickup)
            exchanged_route.nodes.insert(min_pos + 1, task1_delivery)
            exchanged_route.nodes.insert(min_pos + 2, task2_pickup)
            exchanged_route.nodes.insert(min_pos + 3, task2_delivery)

        return exchanged_route

    def greedy_insertion(self, route: Route, removed_task_ids: List[int]) -> Route:
        """
        贪心插入算子 + 充电支持

        仓储场景迭代改进（步骤2.1）：
        - 支持pickup/delivery分离插入（可以先集中取货，再集中送货）
        - 增加容量约束检查，防止超载

        策略：
        1. 对每个任务，找到成本最小的插入位置
        2. 仓储场景迭代新增：检查容量可行性（避免超载）
        3. 如果需要充电，在总成本中加入充电惩罚
        4. 插入成本 = 距离增量 + 充电惩罚
        """
        from core.vehicle import create_vehicle
        from physics.energy import EnergyConfig

        repaired_route = route.copy()

        if not hasattr(self, 'vehicle') or self.vehicle is None:
            raise ValueError("必须设置vehicle属性才能进行充电约束规划")
        if not hasattr(self, 'energy_config') or self.energy_config is None:
            raise ValueError("必须设置energy_config属性才能进行充电约束规划")

        vehicle = self.vehicle
        energy_config = self.energy_config

        for task_id in removed_task_ids:
            task = self.task_pool.get_task(task_id)

            pickup_in_route = False
            pickup_position = None
            for i, node in enumerate(repaired_route.nodes):
                if hasattr(node, 'task_id') and node.task_id == task_id and node.is_pickup():
                    pickup_in_route = True
                    pickup_position = i
                    break

            best_cost = float('inf')
            best_position = None
            best_charging_plan = None

            if pickup_in_route:
                # 只需要插入delivery节点
                for delivery_pos in range(pickup_position + 1, len(repaired_route.nodes) + 1):
                    # 创建临时路径测试插入
                    temp_route = repaired_route.copy()
                    temp_route.nodes.insert(delivery_pos, task.delivery_node)

                    capacity_feasible, capacity_error = temp_route.check_capacity_feasibility(
                        vehicle.capacity,
                        debug=False
                    )

                    if not capacity_feasible:
                        continue

                    time_feasible, delay_cost = self._check_time_window_feasibility_fast(temp_route)
                    if not time_feasible:
                        # 违反硬时间窗，跳过
                        continue

                    # 计算插入delivery的成本增量
                    if delivery_pos == 0:
                        cost_delta = 0.0
                    else:
                        prev_node = repaired_route.nodes[delivery_pos - 1]
                        if delivery_pos < len(repaired_route.nodes):
                            next_node = repaired_route.nodes[delivery_pos]
                            # 移除原边，添加新边
                            old_dist = self.distance.get_distance(prev_node.node_id, next_node.node_id)
                            new_dist = (self.distance.get_distance(prev_node.node_id, task.delivery_node.node_id) +
                                       self.distance.get_distance(task.delivery_node.node_id, next_node.node_id))
                            cost_delta = new_dist - old_dist
                        else:
                            # 插入到末尾
                            cost_delta = self.distance.get_distance(prev_node.node_id, task.delivery_node.node_id)

                    if cost_delta < best_cost:
                        best_cost = cost_delta
                        best_position = ('delivery_only', delivery_pos)
            else:
                # 需要插入完整任务（pickup和delivery）
                # 允许pickup和delivery之间有间隔（支持分离插入）
                for pickup_pos in range(1, len(repaired_route.nodes)):
                    for delivery_pos in range(pickup_pos + 1, len(repaired_route.nodes) + 1):

                        # 创建临时路径测试插入
                        temp_route = repaired_route.copy()
                        temp_route.insert_task(task, (pickup_pos, delivery_pos))

                        capacity_feasible, capacity_error = temp_route.check_capacity_feasibility(
                            vehicle.capacity,
                            debug=False
                        )

                        if not capacity_feasible:
                            # 容量不可行，跳过此位置
                            continue

                        time_feasible, delay_cost = self._check_time_window_feasibility_fast(temp_route)
                        if not time_feasible:
                            # 违反硬时间窗，跳过
                            continue

                        cost_delta = repaired_route.calculate_insertion_cost_delta(
                            task,
                            (pickup_pos, delivery_pos),
                            self.distance
                        )

                        # 加入时间窗延迟成本
                        cost_delta += delay_cost

                        feasible, charging_plan = repaired_route.check_energy_feasibility_for_insertion(
                            task,
                            (pickup_pos, delivery_pos),
                            vehicle,
                            self.distance,
                            energy_config, 
                            charging_strategy=self.charging_strategy
                        )

                        if not feasible:
                            continue

                        if charging_plan:
                            charging_penalty_per_station = self.hyper.charging.penalty_per_station
                            total_charging_penalty = len(charging_plan) * charging_penalty_per_station
                            cost_delta += total_charging_penalty

                        if cost_delta < best_cost:
                            best_cost = cost_delta
                            best_position = (pickup_pos, delivery_pos)
                            best_charging_plan = charging_plan

            if best_position is not None:
                if isinstance(best_position, tuple) and len(best_position) == 2:
                    if best_position[0] == 'delivery_only':
                        # 只插入delivery节点
                        delivery_pos = best_position[1]
                        repaired_route.nodes.insert(delivery_pos, task.delivery_node)
                    else:
                        # 插入完整任务（pickup和delivery）
                        repaired_route.insert_task(task, best_position)

                        if best_charging_plan:
                            sorted_plans = sorted(best_charging_plan, key=lambda x: x['position'], reverse=True)
                            for plan in sorted_plans:
                                repaired_route.insert_charging_visit(
                                    station=plan['station_node'],
                                    position=plan['position'],
                                    charge_amount=plan['amount']
                                )

        return repaired_route

    def lp_insertion(self, route: Route, removed_task_ids: List[int]) -> Route:
        """Default LP repair fallback uses regret-2 insertion."""

        return self.regret2_insertion(route, removed_task_ids)

    def ensure_route_schedule(self, route: Route) -> bool:
        """Ensure that the route carries a valid schedule with energy trajectory."""

        if route.visits:
            return True

        vehicle = getattr(self, 'vehicle', None)
        energy_config = getattr(self, 'energy_config', None)
        if vehicle is None or energy_config is None:
            return False

        time_config = getattr(self, 'time_config', None)
        if time_config is None:
            time_config = TimeConfig(vehicle_speed=self.hyper.vehicle.cruise_speed_m_s)

        try:
            route.compute_schedule(
                distance_matrix=self.distance,
                vehicle_capacity=vehicle.capacity,
                vehicle_battery_capacity=vehicle.battery_capacity,
                initial_battery=vehicle.initial_battery,
                time_config=time_config,
                energy_config=energy_config,
            )
        except Exception:
            return False

        return bool(route.visits)

    def evaluate_cost(self, route: Route) -> float:
        """Evaluate the weighted tardiness + travel objective."""

        if not self.ensure_route_schedule(route):
            return float('inf')

        total_distance = route.calculate_total_distance(self.distance)
        distance_cost = total_distance * self.cost_params.C_tr

        tardiness_cost = 0.0
        waiting_cost = 0.0
        if route.visits:
            for visit in route.visits:
                if hasattr(visit.node, 'time_window') and visit.node.time_window:
                    tardiness = max(0.0, visit.start_service_time - visit.node.time_window.latest)
                    if tardiness > 0:
                        tardiness_cost += tardiness * self._tardiness_weight_for_visit(visit) * self.cost_params.C_delay
                    waiting = max(0.0, visit.start_service_time - visit.arrival_time)
                    waiting_cost += waiting * self.cost_params.C_wait

        served_tasks = set(route.get_served_tasks())
        expected_tasks = {task.task_id for task in self.task_pool.get_all_tasks()}
        missing_tasks = expected_tasks - served_tasks
        missing_penalty = len(missing_tasks) * self.cost_params.C_missing_task

        infeasible_penalty = self.cost_params.C_infeasible if route.is_feasible is False else 0.0

        battery_penalty = 0.0
        if hasattr(self, 'vehicle') and hasattr(self, 'energy_config'):
            if not self._check_battery_feasibility(route):
                battery_penalty = self.cost_params.C_infeasible * 10.0

        return distance_cost + tardiness_cost + waiting_cost + missing_penalty + infeasible_penalty + battery_penalty

    def _tardiness_weight_for_visit(self, visit) -> float:
        node = visit.node
        if hasattr(node, 'task_id'):
            task = self.task_pool.get_task(node.task_id)
            if task is not None and getattr(task, 'priority', 0) >= 0:
                return 1.0 + float(task.priority)
        return 1.0

    def _check_battery_feasibility(self, route: Route, debug=False) -> bool:
        """
        检查路径的电池可行性 (Week 2新增)

        模拟整个路径的电池消耗，检查是否会出现电量不足

        返回:
            bool: True表示可行，False表示不可行
        """
        if not hasattr(self, 'vehicle') or not hasattr(self, 'energy_config'):
            return True  # 没有约束则认为可行

        vehicle = self.vehicle
        energy_config = self.energy_config

        # 模拟电池消耗
        current_battery = vehicle.battery_capacity  # 满电出发

        for i in range(len(route.nodes) - 1):
            current_node = route.nodes[i]
            next_node = route.nodes[i + 1]
            vehicle_speed = self.hyper.vehicle.cruise_speed_m_s
            distance = self.distance.get_distance(
                current_node.node_id,
                next_node.node_id
            )
            travel_time = distance / vehicle_speed
            energy_consumed = energy_config.consumption_rate * travel_time
            safety_threshold_value = energy_config.safety_threshold * vehicle.battery_capacity
            # 如果当前节点是充电站，先充电
            if current_node.is_charging_station():
                # 使用充电策略决定充电量
                if self.charging_strategy:
                    # 计算剩余路径能耗（包括当前到下一节点的距离）
                    # 正确公式：energy = consumption_rate(kWh/s) * time(s)
                    remaining_energy_demand = 0.0
                    energy_to_next_stop = 0.0
                    next_stop_is_cs = False
                    for j in range(i, len(route.nodes) - 1):
                        seg_distance = self.distance.get_distance(
                            route.nodes[j].node_id,
                            route.nodes[j + 1].node_id
                        )
                        travel_time = seg_distance / vehicle_speed
                        seg_energy = energy_config.consumption_rate * travel_time
                        remaining_energy_demand += seg_energy
                        energy_to_next_stop += seg_energy

                        if route.nodes[j + 1].is_charging_station() and j >= i:
                            next_stop_is_cs = True
                            break

                    if not next_stop_is_cs:
                        # 继续累加剩余路径以便返回仓库
                        energy_to_next_stop = remaining_energy_demand

                    target_energy_demand = (energy_to_next_stop
                                            if next_stop_is_cs
                                            else remaining_energy_demand)

                    charge_amount = self.charging_strategy.determine_charging_amount(
                        current_battery=current_battery,
                        remaining_demand=target_energy_demand,
                        battery_capacity=vehicle.battery_capacity
                    )
                    current_battery = min(
                        vehicle.battery_capacity,
                        current_battery + max(0.0, charge_amount)
                    )
                else:
                    # 没有充电策略，默认充满
                    current_battery = vehicle.battery_capacity
                # 根据下一段行程的能量需求，确保最低出发电量
                min_departure_energy = energy_consumed

                # 确保具备到达下一充电站或终点的能量
                if self.charging_strategy and current_node.is_charging_station():
                    required_for_next_stop = energy_to_next_stop
                    if not next_stop_is_cs:
                        required_for_next_stop += safety_threshold_value
                    min_departure_energy = max(min_departure_energy, required_for_next_stop)
                elif not next_node.is_charging_station():
                    min_departure_energy += safety_threshold_value

                if min_departure_energy > vehicle.battery_capacity:
                    return False

                if current_battery < min_departure_energy:
                    current_battery = min(vehicle.battery_capacity, min_departure_energy)

            # 计算到下一节点的距离和能耗
            # 消耗能量前往下一节点
            current_battery -= energy_consumed


            # 检查是否电量不足
            if current_battery < 0:
                return False  # 电量不足，不可行
            
            # 之前的逻辑会在到站前执行阈值检查，导致即使成功抵达充电站
            #（电量>=0）也会因为低于警告/安全阈值被判为不可行
            # 这会让大量候选解被提前否决，使优化率显著下降。
            if next_node.is_charging_station():
                # 到达充电站后，下一轮循环会触发充电逻辑
                # 因此这里无需再进行阈值检查
                continue

            # 安全层：绝对最低电量（5%），硬约束
            safety_threshold = energy_config.safety_threshold * vehicle.battery_capacity
            if current_battery < safety_threshold:
                if debug:
                    print(f"  ✗ Safety threshold violated at node {i+1}! ({current_battery:.1f} < {safety_threshold:.1f})")
                return False  # 低于安全层，绝对不可行

            # 警告层：策略感知的建议充电阈值（软约束+前瞻性检查）
            if self.charging_strategy:
                warning_threshold_ratio = self.charging_strategy.get_warning_threshold()
                warning_threshold = warning_threshold_ratio * vehicle.battery_capacity

                if current_battery < warning_threshold:
                    # 低于警告阈值，进行前瞻性检查
                    # 检查接下来是否有充电站，以及能否安全到达
                    next_cs_index = -1
                    energy_to_next_cs = 0.0

                    # 查找前方的第一个充电站（不能只看固定窗口，
                    # 否则真实存在但距离较远的充电站会被忽略，
                    # 导致错误地判定路径不可行）
                    for j in range(i + 2, len(route.nodes)):
                        if route.nodes[j].is_charging_station():
                            next_cs_index = j
                            break

                    if next_cs_index != -1:
                        # 计算到达下一个充电站需要的能量
                        for j in range(i + 1, next_cs_index):
                            seg_distance = self.distance.get_distance(
                                route.nodes[j].node_id,
                                route.nodes[j + 1].node_id
                            )
                            travel_time = seg_distance / vehicle_speed
                            energy_to_next_cs += energy_config.consumption_rate * travel_time

                        # 预估到达充电站时的电量
                        predicted_battery_at_cs = current_battery - energy_to_next_cs

                        # 如果预估电量为负，说明无法抵达下一个充电站
                        if predicted_battery_at_cs < 0:

                            return False  # 无法安全到达下一个充电站
                    else:
                        # 前方没有充电站，检查能否到达终点
                        remaining_energy_to_depot = 0.0
                        for j in range(i + 1, len(route.nodes) - 1):
                            seg_distance = self.distance.get_distance(
                                route.nodes[j].node_id,
                                route.nodes[j + 1].node_id
                            )
                            travel_time = seg_distance / vehicle_speed
                            remaining_energy_to_depot += energy_config.consumption_rate * travel_time

                        predicted_battery_at_depot = current_battery - remaining_energy_to_depot

                        # 如果无法安全到达终点，不可行
                        if predicted_battery_at_depot < 0:

                            return False  # 需要充电站但前方没有

        return True  # 整个路径可行

    def _check_time_window_feasibility_fast(
        self,
        temp_route: Route,
        vehicle_speed: Optional[float] = None,
    ) -> Tuple[bool, float]:
        """
        快速检查路径的时间窗可行性（仓储场景迭代新增）

        简化版：只检查硬时间窗违反，计算软时间窗延迟成本

        参数:
            temp_route: 待检查的路径
            vehicle_speed: 车辆速度 (m/s)，默认使用全局配置

        返回:
            Tuple[bool, float]: (是否满足硬时间窗, 延迟惩罚成本)
        """
        if vehicle_speed is None:
            vehicle_speed = self.hyper.vehicle.cruise_speed_m_s

        current_time = 0.0
        total_tardiness = 0.0

        for i in range(len(temp_route.nodes)):
            node = temp_route.nodes[i]

            # 1. 到达当前节点
            if i > 0:
                prev_node = temp_route.nodes[i - 1]
                distance = self.distance.get_distance(prev_node.node_id, node.node_id)
                travel_time = distance / vehicle_speed  # 秒
                current_time += travel_time

            # 2. 检查时间窗
            if hasattr(node, 'time_window') and node.time_window:
                tw = node.time_window

                if current_time < tw.earliest:
                    # 早到，等待
                    current_time = tw.earliest
                elif current_time > tw.latest:
                    # 晚到
                    tardiness = current_time - tw.latest

                    if tw.is_hard():
                        # 硬时间窗违反，不可行
                        return False, float('inf')
                    else:
                        # 软时间窗，累计延迟
                        total_tardiness += tardiness

            # 3. 服务时间
            service_time = node.service_time if hasattr(node, 'service_time') else 0.0
            current_time += service_time

            # 4. 充电时间（如果是充电站）
            if node.is_charging_station():
                # 简化：假设充电10秒（实际应根据充电量计算）
                if hasattr(node, 'charge_amount'):
                    charging_rate = self.energy_config.charging_rate if hasattr(self, 'energy_config') else 0.001
                    charge_time = (
                        node.charge_amount / charging_rate
                        if charging_rate > 0
                        else self.hyper.charging.fallback_duration_s
                    )
                    current_time += charge_time
                else:
                    current_time += self.hyper.charging.fallback_duration_s  # 默认充电时间

        # 计算延迟惩罚成本
        delay_cost = total_tardiness * self.cost_params.C_delay

        return True, delay_cost

    def _find_battery_depletion_position(self, route: Route) -> int:
        """
        找到电池耗尽的位置（Week 2新增 - 第1.2步）

        返回第一个电量不足的节点索引
        如果路径可行，返回-1
        """
        if not hasattr(self, 'vehicle') or not hasattr(self, 'energy_config'):
            return -1

        vehicle = self.vehicle
        energy_config = self.energy_config
        current_battery = vehicle.battery_capacity

        for i in range(len(route.nodes) - 1):
            current_node = route.nodes[i]
            next_node = route.nodes[i + 1]

            # 如果当前节点是充电站，先充电
            if current_node.is_charging_station():
                if self.charging_strategy:
                    # 正确公式：energy = consumption_rate(kWh/s) * time(s)
                    vehicle_speed = self.hyper.vehicle.cruise_speed_m_s
                    remaining_energy_demand = 0.0
                    energy_to_next_stop = 0.0
                    next_stop_is_cs = False
                    for j in range(i, len(route.nodes) - 1):
                        seg_distance = self.distance.get_distance(
                            route.nodes[j].node_id,
                            route.nodes[j + 1].node_id
                        )
                        travel_time = seg_distance / vehicle_speed
                        seg_energy = energy_config.consumption_rate * travel_time
                        remaining_energy_demand += seg_energy
                        energy_to_next_stop += seg_energy

                        if route.nodes[j + 1].is_charging_station() and j >= i:
                            next_stop_is_cs = True
                            break

                    if not next_stop_is_cs:
                        energy_to_next_stop = remaining_energy_demand

                    target_energy_demand = (energy_to_next_stop
                                            if next_stop_is_cs
                                            else remaining_energy_demand)

                    charge_amount = self.charging_strategy.determine_charging_amount(
                        current_battery=current_battery,
                        remaining_demand=target_energy_demand,
                        battery_capacity=vehicle.battery_capacity
                    )
                    current_battery = min(vehicle.battery_capacity, current_battery + charge_amount)
                else:
                    current_battery = vehicle.battery_capacity

            # 计算到下一节点的能耗
            # 正确公式：energy = consumption_rate(kWh/s) * time(s)
            distance = self.distance.get_distance(current_node.node_id, next_node.node_id)
            vehicle_speed = self.hyper.vehicle.cruise_speed_m_s
            travel_time = distance / vehicle_speed
            energy_consumed = energy_config.consumption_rate * travel_time
            current_battery -= energy_consumed

            # 检查是否电量不足（只检查耗尽，不检查临界值）
            if current_battery < 0:
                return i + 1  # 返回无法到达的节点位置

        return -1  # 路径可行

    def _get_available_charging_stations(self, route: Route):
        """
        获取可用的充电站列表（Week 2新增 - 第1.2步）

        从距离矩阵中获取所有充电站节点
        排除已经在路径中的充电站
        """
        # 获取路径中已有的充电站ID
        existing_cs_ids = set(n.node_id for n in route.nodes if n.is_charging_station())

        # 从距离矩阵的coordinates中找出所有充电站
        # 充电站节点ID通常 >= 100
        available_stations = []

        if hasattr(self.distance, 'coordinates'):
            for node_id, coords in self.distance.coordinates.items():
                # 假设充电站ID >= 100，任务节点ID < 100
                if node_id >= 100 and node_id not in existing_cs_ids:
                    # 创建充电站节点
                    from core.node import create_charging_node
                    cs_node = create_charging_node(node_id=node_id, coordinates=coords)
                    available_stations.append(cs_node)

        return available_stations

    def _find_best_charging_station(self, route: Route, position: int):
        """
        找到最优的充电站插入位置（Week 2新增 - 第1.2步）

        策略：选择绕路成本最小的充电站

        参数:
            route: 当前路径
            position: 需要插入充电站的位置

        返回:
            (best_station, best_insert_pos): 最优充电站和插入位置
        """
        available_stations = self._get_available_charging_stations(route)

        if not available_stations:
            return None, None

        best_detour_cost = float('inf')
        best_station = None
        best_insert_pos = None

        # 在position前后尝试插入充电站
        for insert_pos in range(max(1, position - 2), min(len(route.nodes), position + 2)):
            if insert_pos <= 0 or insert_pos >= len(route.nodes):
                continue

            prev_node = route.nodes[insert_pos - 1]
            next_node = route.nodes[insert_pos]

            # 原始距离
            original_distance = self.distance.get_distance(prev_node.node_id, next_node.node_id)

            # 尝试每个充电站
            for station in available_stations:
                # 绕路距离 = (prev -> station) + (station -> next) - (prev -> next)
                detour_distance = (
                    self.distance.get_distance(prev_node.node_id, station.node_id) +
                    self.distance.get_distance(station.node_id, next_node.node_id) -
                    original_distance
                )

                if detour_distance < best_detour_cost:
                    best_detour_cost = detour_distance
                    best_station = station
                    best_insert_pos = insert_pos

        return best_station, best_insert_pos

    def _insert_necessary_charging_stations(self, route: Route, max_attempts: int = 10) -> Route:
        """
        自动插入必要的充电站（Week 2新增 - 第1.2步）

        策略：
        1. 检查电池可行性（包括临界值）
        2. 如果不可行，找到电量耗尽或临界位置
        3. 在该位置前插入最优充电站
        4. 重复直到路径可行或达到最大尝试次数

        Week 2修复：增加max_attempts到10，更积极地修复

        参数:
            route: 当前路径
            max_attempts: 最大尝试次数（防止无限循环）

        返回:
            修复后的路径
        """
        attempts = 0

        while attempts < max_attempts:
            if self._check_battery_feasibility(route):
                return route  # 已经可行

            # 找到电量耗尽或临界位置
            depletion_pos = self._find_battery_depletion_position(route)

            if depletion_pos == -1:
                # 找不到耗尽位置，但可行性检查失败，可能是临界值问题
                # 尝试在路径末尾附近插入充电站
                depletion_pos = len(route.nodes) - 1

            # 找到最优充电站
            best_station, best_insert_pos = self._find_best_charging_station(route, depletion_pos)

            if best_station is None or best_insert_pos is None:
                # 找不到可用的充电站，无法修复
                return route

            # 插入充电站
            route.nodes.insert(best_insert_pos, best_station)

            attempts += 1

        return route

    def _estimate_charging_and_time(self, route: Route) -> tuple[float, float]:
        """
        估算路径的充电量和总时间（当visits不可用时）

        通过模拟电池消耗和充电过程，估算：
        1. 总充电量 (kWh)
        2. 总时间 (秒或分钟，取决于time_config)

        返回:
            tuple: (total_charging_amount, total_time)
        """
        if not hasattr(self, 'vehicle') or not hasattr(self, 'energy_config'):
            return 0.0, 0.0

        vehicle = self.vehicle
        energy_config = self.energy_config
        time_config = getattr(self, 'time_config', None)

        current_battery = vehicle.battery_capacity
        total_charging = 0.0
        total_time = 0.0

        for i in range(len(route.nodes) - 1):
            current_node = route.nodes[i]
            next_node = route.nodes[i + 1]

            # 如果当前节点是充电站，计算充电量和充电时间
            if current_node.is_charging_station():
                if self.charging_strategy:
                    # 计算剩余能耗
                    # 正确公式：energy = consumption_rate(kWh/s) * time(s)
                    vehicle_speed = self.hyper.vehicle.cruise_speed_m_s
                    remaining_energy_demand = 0.0
                    energy_to_next_stop = 0.0
                    next_stop_is_cs = False
                    for j in range(i, len(route.nodes) - 1):
                        seg_distance = self.distance.get_distance(
                            route.nodes[j].node_id,
                            route.nodes[j + 1].node_id
                        )
                        travel_time = seg_distance / vehicle_speed
                        seg_energy = energy_config.consumption_rate * travel_time
                        remaining_energy_demand += seg_energy
                        energy_to_next_stop += seg_energy

                        if route.nodes[j + 1].is_charging_station() and j >= i:
                            next_stop_is_cs = True
                            break

                    if not next_stop_is_cs:
                        energy_to_next_stop = remaining_energy_demand

                    target_energy_demand = (energy_to_next_stop
                                            if next_stop_is_cs
                                            else remaining_energy_demand)

                    # 决定充电量
                    charge_amount = self.charging_strategy.determine_charging_amount(
                        current_battery=current_battery,
                        remaining_demand=target_energy_demand,
                        battery_capacity=vehicle.battery_capacity
                    )

                    # 累计充电量
                    total_charging += charge_amount

                    # 计算充电时间
                    if charge_amount > 0:
                        charging_time = charge_amount / energy_config.charging_rate
                        total_time += charging_time

                    # 更新电池
                    current_battery = min(vehicle.battery_capacity, current_battery + charge_amount)

            # 计算行驶距离和时间
            distance = self.distance.get_distance(current_node.node_id, next_node.node_id)
            # 正确公式：energy = consumption_rate(kWh/s) * time(s)
            vehicle_speed = self.hyper.vehicle.cruise_speed_m_s
            travel_time = distance / vehicle_speed
            energy_consumed = energy_config.consumption_rate * travel_time

            # 行驶时间
            if time_config:
                travel_time = distance / time_config.speed
                total_time += travel_time

            # 服务时间（如果是任务节点）
            if hasattr(current_node, 'service_time') and current_node.service_time > 0:
                total_time += current_node.service_time

            # 更新电池
            current_battery -= energy_consumed

        return total_charging, total_time

    def get_cost_breakdown(self, route: Route) -> Dict:
        """
        获取成本分解（用于分析和调试）

        Week 2改进：
        - 使用RouteExecutor执行路径生成visits
        - 准确记录充电量和充电次数

        返回:
            Dict: 各项成本明细
        """
        from core.route_executor import RouteExecutor

        distance = route.calculate_total_distance(self.distance)

        if hasattr(self, 'vehicle') and hasattr(self, 'energy_config'):
            executor = RouteExecutor(
                distance_matrix=self.distance,
                energy_config=self.energy_config,
                time_config=getattr(self, 'time_config', None)
            )
            # 执行路径并生成visits
            executed_route = executor.execute(
                route=route,
                vehicle=self.vehicle,
                charging_strategy=self.charging_strategy
            )
            # 使用执行后的路径
            route_to_analyze = executed_route
        else:
            route_to_analyze = route

        # 充电量统计
        charging_amount = 0.0
        num_charging_stops = 0
        if route_to_analyze.visits:
            for visit in route_to_analyze.visits:
                if visit.node.is_charging_station():
                    charged = visit.battery_after_service - visit.battery_after_travel
                    if charged > 0.01:  # 只计算实际充电的
                        charging_amount += charged
                        num_charging_stops += 1

        # 时间统计
        total_time = 0.0
        if route_to_analyze.visits and len(route_to_analyze.visits) > 0:
            total_time = route_to_analyze.visits[-1].departure_time - route_to_analyze.visits[0].arrival_time

        # 延迟统计
        total_delay = 0.0
        if route_to_analyze.visits:
            for visit in route_to_analyze.visits:
                total_delay += visit.get_delay()

        # 计算总成本（使用executed route以确保包含visits）
        total_cost_value = self.evaluate_cost(route_to_analyze)

        return {
            'total_distance': distance,
            'total_charging': charging_amount,
            'total_time': total_time,
            'total_delay': total_delay,
            'num_charging_stops': num_charging_stops,
            'distance_cost': distance * self.cost_params.C_tr,
            'charging_cost': charging_amount * self.cost_params.C_ch,
            'time_cost': total_time * self.cost_params.C_time,
            'delay_cost': total_delay * self.cost_params.C_delay,
            'total_cost': total_cost_value,
            'cost_per_km': total_cost_value / (distance / 1000) if distance > 0 else 0
        }
    
    def accept_solution(self, 
                       new_cost: float, 
                       current_cost: float, 
                       temperature: float) -> bool:
        """
        模拟退火接受准则
        """
        if new_cost < current_cost:
            return True
        else:
            probability = math.exp(-(new_cost - current_cost) / temperature)
            return random.random() < probability
        
    def regret2_insertion(self,
                      route: Route,
                      removed_task_ids: List[int]) -> Route:
        """
        Regret-2插入算子+充电支持（仓储场景迭代步骤2.4改进）

        仓储场景迭代改进：
        - 支持容量约束检查
        - 支持partial delivery插入
        - 更智能的位置评估
        """
        repaired_route = route.copy()
        remaining_tasks = removed_task_ids.copy()

        if not hasattr(self, 'vehicle') or self.vehicle is None:
            raise ValueError("必须设置vehicle属性才能进行充电约束规划")
        if not hasattr(self, 'energy_config') or self.energy_config is None:
            raise ValueError("必须设置energy_config属性才能进行充电约束规划")

        vehicle = self.vehicle
        energy_config = self.energy_config

        while remaining_tasks:
            best_regret = -float('inf')
            best_task_id = None
            best_position = None
            best_charging_plan = None

            for task_id in remaining_tasks:
                task = self.task_pool.get_task(task_id)

                pickup_in_route = False
                pickup_position = None
                for i, node in enumerate(repaired_route.nodes):
                    if hasattr(node, 'task_id') and node.task_id == task_id and node.is_pickup():
                        pickup_in_route = True
                        pickup_position = i
                        break

                feasible_insertions = []

                if pickup_in_route:
                    # 只需要插入delivery节点
                    for delivery_pos in range(pickup_position + 1, len(repaired_route.nodes) + 1):
                        temp_route = repaired_route.copy()
                        temp_route.nodes.insert(delivery_pos, task.delivery_node)

                        capacity_feasible, _ = temp_route.check_capacity_feasibility(vehicle.capacity, debug=False)
                        if not capacity_feasible:
                            continue

                        time_feasible, delay_cost = self._check_time_window_feasibility_fast(temp_route)
                        if not time_feasible:
                            # 违反硬时间窗，跳过
                            continue

                        # 计算成本增量
                        prev_node = repaired_route.nodes[delivery_pos - 1]
                        if delivery_pos < len(repaired_route.nodes):
                            next_node = repaired_route.nodes[delivery_pos]
                            old_dist = self.distance.get_distance(prev_node.node_id, next_node.node_id)
                            new_dist = (self.distance.get_distance(prev_node.node_id, task.delivery_node.node_id) +
                                       self.distance.get_distance(task.delivery_node.node_id, next_node.node_id))
                            cost_delta = new_dist - old_dist
                        else:
                            cost_delta = self.distance.get_distance(prev_node.node_id, task.delivery_node.node_id)

                        # 加入时间窗延迟成本
                        cost_delta += delay_cost

                        feasible_insertions.append({
                            'cost': cost_delta,
                            'position': ('delivery_only', delivery_pos),
                            'charging_plan': None
                        })
                else:
                    # 需要插入完整任务
                    for pickup_pos in range(1, len(repaired_route.nodes)):
                        for delivery_pos in range(pickup_pos + 1, len(repaired_route.nodes) + 1):
                            temp_route = repaired_route.copy()
                            temp_route.insert_task(task, (pickup_pos, delivery_pos))

                            capacity_feasible, _ = temp_route.check_capacity_feasibility(vehicle.capacity, debug=False)
                            if not capacity_feasible:
                                continue

                            time_feasible, delay_cost = self._check_time_window_feasibility_fast(temp_route)
                            if not time_feasible:
                                # 违反硬时间窗，跳过
                                continue

                            cost_delta = repaired_route.calculate_insertion_cost_delta(
                                task,
                                (pickup_pos, delivery_pos),
                                self.distance
                            )

                            # 加入时间窗延迟成本
                            cost_delta += delay_cost

                            feasible, charging_plan = repaired_route.check_energy_feasibility_for_insertion(
                                task,
                                (pickup_pos, delivery_pos),
                                vehicle,
                                self.distance,
                                energy_config, 
                                charging_strategy=self.charging_strategy
                            )

                            if not feasible:
                                continue

                            if charging_plan:
                                cost_delta += len(charging_plan) * 50.0

                            feasible_insertions.append({
                                'cost': cost_delta,
                                'position': (pickup_pos, delivery_pos),
                                'charging_plan': charging_plan
                            })
                
                if len(feasible_insertions) >= 2:
                    feasible_insertions.sort(key=lambda x: x['cost'])
                    
                    best_cost = feasible_insertions[0]['cost']
                    second_best_cost = feasible_insertions[1]['cost']
                    
                    regret = second_best_cost - best_cost

                    if regret > best_regret:
                        best_regret = regret
                        best_task_id = task_id
                        best_position = feasible_insertions[0]['position']

                elif len(feasible_insertions) == 1:
                    # 只有一个可行位置，regret = inf
                    if best_regret < float('inf'):
                        best_regret = float('inf')
                        best_task_id = task_id
                        best_position = feasible_insertions[0]['position']

            # 插入regret最大的任务
            if best_task_id:
                task = self.task_pool.get_task(best_task_id)

                if isinstance(best_position, tuple) and len(best_position) == 2:
                    if best_position[0] == 'delivery_only':
                        # 只插入delivery节点
                        delivery_pos = best_position[1]
                        repaired_route.nodes.insert(delivery_pos, task.delivery_node)
                    else:
                        # 插入完整任务
                        repaired_route.insert_task(task, best_position)

                        if best_charging_plan:
                            sorted_plans = sorted(best_charging_plan,
                                                key=lambda x: x['position'],
                                                reverse=True)
                            for plan in sorted_plans:
                                repaired_route.insert_charging_visit(
                                    station=plan['station_node'],
                                    position=plan['position'],
                                    charge_amount=plan['amount']
                                )

                remaining_tasks.remove(best_task_id)
            else:
                break

        return repaired_route

    def random_insertion(self, route: Route, removed_task_ids: List[int]) -> Route:
        """
        随机插入算子 + 智能充电站插入 (Week 2改进 - 第1.2步)

        策略：
        1. 对每个任务，随机选择一个插入位置
        2. 插入任务后，自动插入必要的充电站
        """
        repaired_route = route.copy()

        if not hasattr(self, 'vehicle') or self.vehicle is None:
            raise ValueError("必须设置vehicle属性才能进行充电约束规划")
        if not hasattr(self, 'energy_config') or self.energy_config is None:
            raise ValueError("必须设置energy_config属性才能进行充电约束规划")

        for task_id in removed_task_ids:
            task = self.task_pool.get_task(task_id)

            # 收集所有可能的插入位置
            possible_positions = []

            for pickup_pos in range(1, len(repaired_route.nodes)):
                for delivery_pos in range(pickup_pos + 1, len(repaired_route.nodes) + 1):
                    possible_positions.append((pickup_pos, delivery_pos))

            # 随机选择一个位置
            if possible_positions:
                chosen_position = random.choice(possible_positions)
                repaired_route.insert_task(task, chosen_position)

                repaired_route = self._insert_necessary_charging_stations(repaired_route)

        return repaired_route
