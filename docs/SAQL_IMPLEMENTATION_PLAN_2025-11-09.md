# Scale-Aware Q-Learning + Dynamic E-VRP Implementation Plan

**Date**: 2025-11-09
**Project**: Electric Vehicle Routing Problem with Partial Recharging
**Research Direction**: Scale-Aware Q-Learning (SAQL) + Dynamic Online Optimization
**Target**: Q2+ Journal Publication

---

## Executive Summary

This document outlines a comprehensive 21-week implementation plan to enhance the current Q-learning-based ALNS framework with scale-aware adaptations and dynamic scenario capabilities. The plan addresses the identified instability issues in Q-learning performance across different problem scales and extends the framework to handle real-time dynamic scenarios.

**Current Performance Issues:**
- Large-scale Q-learning: 6.92% improvement (worse than Matheuristic's 27.05%)
- High seed variance: 6.92% to 38.31% across different seeds
- Q-table initialized to 0.0 (not zero-bias as previously claimed)
- Only 3 states (too coarse-grained)
- Initial epsilon too low (0.12) for sufficient exploration

**Target Improvements:**
- Small scale: Maintain 62%+ improvement
- Large scale: Improve from 7% to 25%+
- Reduce seed variance by 60%+
- Dynamic response time: < 1 second
- Statistical significance: p < 0.01 with Cohen's d > 0.8

---

## Phase 1: Scale-Aware Q-Learning Framework (Weeks 1-7)

### Week 1-2: Q-Learning Stability Analysis and Initialization Experiments

**Objectives:**
- Systematically analyze current Q-learning instability
- Test 4 different initialization strategies
- Establish baseline metrics for comparison

**Tasks:**

1. **Multi-Seed Baseline Collection** (Week 1, Days 1-3)
   ```bash
   # Run current implementation with 10 different seeds
   for seed in 2025 2026 2027 2028 2029 2030 2031 2032 2033 2034; do
       python scripts/run_alns_preset.py \
           --scenario small \
           --solver q_learning \
           --seed $seed \
           --output results/baseline_seed_${seed}.json
   done
   ```

   Expected outputs:
   - 10 result files in `results/baseline_seed_*.json`
   - Statistical summary: mean, std, min, max improvement ratios
   - Variance coefficient calculation

2. **Q-Table Initialization Experiments** (Week 1, Days 4-7)

   Test 4 strategies:

   **Strategy A: Current (Zero initialization)**
   ```python
   q_table[state][action] = 0.0
   ```

   **Strategy B: Uniform positive bias**
   ```python
   q_table[state][action] = 50.0  # Encourages exploration
   ```

   **Strategy C: Action-specific initialization**
   ```python
   # Boost matheuristic repairs
   if repair in ["greedy_lp", "segments"]:
       q_table[state][action] = 100.0
   else:
       q_table[state][action] = 50.0
   ```

   **Strategy D: State-specific initialization**
   ```python
   state_bias = {
       "explore": 30.0,
       "stuck": 70.0,
       "deep_stuck": 120.0
   }
   q_table[state][action] = state_bias[state]
   ```

   Implementation location: `src/planner/q_learning.py:64-66`

3. **Epsilon Strategy Analysis** (Week 2, Days 1-4)

   Test 3 epsilon configurations:

   | Config | Initial ε | Decay | Min ε |
   |--------|-----------|-------|-------|
   | Current | 0.12 | 0.995 | 0.01 |
   | High-exploration | 0.50 | 0.995 | 0.05 |
   | Adaptive | f(scale) | 0.998 | 0.02 |

   For adaptive epsilon:
   ```python
   def compute_initial_epsilon(num_requests: int) -> float:
       # Scale: small (8-10), medium (24), large (40)
       if num_requests <= 12:
           return 0.30
       elif num_requests <= 30:
           return 0.50
       else:
           return 0.70
   ```

4. **Statistical Validation** (Week 2, Days 5-7)
   - Wilcoxon signed-rank test for paired comparisons
   - Cohen's d for effect size measurement
   - 95% confidence intervals
   - Create summary table in `docs/experiments/q_init_analysis.md`

**Deliverables:**
- Experimental data for 4 × 3 × 10 = 120 runs
- Statistical analysis report
- Recommended initialization + epsilon strategy

---

### Week 3-4: Seven-State Space Design and Implementation

**Objectives:**
- Design enriched state space with 7 states
- Implement state transition logic
- Integrate with existing ALNS framework

**Current 3-State Design:**
```python
states = ("explore", "stuck", "deep_stuck")

# Transitions based on stagnation counter
if stagnation < 160:
    state = "explore"
elif stagnation < 560:
    state = "stuck"
else:
    state = "deep_stuck"
```

**Proposed 7-State Design:**

```python
@dataclass
class StateFeatures:
    stagnation: int          # Iterations since last improvement
    solution_quality: float  # Relative to initial cost
    time_remaining: float    # Fraction of budget remaining
    improvement_trend: str   # "improving", "stable", "degrading"

class SevenStateSpace:
    STATES = (
        "early_explore",      # Initial exploration phase
        "active_improve",     # Making consistent progress
        "slow_progress",      # Improvements slowing down
        "plateau",            # Stagnation but time remaining
        "intensive_search",   # Deep search needed
        "final_polish",       # End-game refinement
        "emergency"           # Critical stagnation
    )

    @staticmethod
    def classify_state(features: StateFeatures, scale: str) -> str:
        # Scale-dependent thresholds
        thresholds = {
            "small": {"stag_1": 80, "stag_2": 200, "stag_3": 400},
            "medium": {"stag_1": 120, "stag_2": 300, "stag_3": 600},
            "large": {"stag_1": 160, "stag_2": 400, "stag_3": 800}
        }
        t = thresholds[scale]

        stag = features.stagnation
        time_left = features.time_remaining
        trend = features.improvement_trend

        # Early phase (first 20% of budget)
        if time_left > 0.80:
            return "early_explore"

        # Active improvement
        if trend == "improving" and stag < t["stag_1"]:
            return "active_improve"

        # Slowing down
        if trend == "stable" and stag < t["stag_2"]:
            return "slow_progress"

        # Plateau with time
        if stag >= t["stag_2"] and time_left > 0.30:
            return "plateau"

        # Final phase
        if time_left < 0.15 and stag < t["stag_3"]:
            return "final_polish"

        # Deep search needed
        if stag >= t["stag_2"] and stag < t["stag_3"]:
            return "intensive_search"

        # Emergency
        return "emergency"
```

**Implementation Steps:**

1. **Create new module** `src/planner/state_classifier.py` (Week 3, Days 1-2)
   - Implement `SevenStateSpace` class
   - Add improvement trend tracking (5-iteration moving window)
   - Write unit tests in `tests/test_state_classifier.py`

2. **Update Q-learning agent** (Week 3, Days 3-5)

   File: `src/planner/q_learning.py`

   ```python
   class ScaleAwareQLearningAgent(QLearningOperatorAgent):
       def __init__(
           self,
           destroy_operators: Iterable[str],
           repair_operators: Sequence[str],
           params: QLearningParams,
           scale: str,  # NEW: "small", "medium", "large"
           *,
           state_classifier: Optional[SevenStateSpace] = None,
       ):
           self.scale = scale
           self.state_classifier = state_classifier or SevenStateSpace()

           # Override parent states
           states = SevenStateSpace.STATES

           super().__init__(
               destroy_operators,
               repair_operators,
               params,
               state_labels=states,
           )

           # Scale-aware epsilon
           self.set_epsilon(self._compute_scale_epsilon())

       def _compute_scale_epsilon(self) -> float:
           scale_map = {"small": 0.30, "medium": 0.50, "large": 0.70}
           return scale_map.get(self.scale, 0.50)
   ```

3. **Integrate with ALNS** (Week 3, Days 6-7)

   File: `src/planner/alns_matheuristic.py`

   Update state tracking in optimization loop:
   ```python
   # Track improvement trend
   recent_improvements = collections.deque(maxlen=5)

   for iteration in range(max_iterations):
       # ... existing code ...

       # Update trend
       recent_improvements.append(improvement)
       trend = self._classify_trend(recent_improvements)

       # Build state features
       features = StateFeatures(
           stagnation=stagnation_counter,
           solution_quality=current_cost / initial_cost,
           time_remaining=1.0 - (iteration / max_iterations),
           improvement_trend=trend
       )

       # Get state from classifier
       state = q_agent.state_classifier.classify_state(features, scale)
   ```

4. **Testing and validation** (Week 4)
   - Run 10-seed experiments with 7-state agent
   - Compare against 3-state baseline
   - Measure state transition patterns
   - Verify action distribution per state

**Deliverables:**
- `src/planner/state_classifier.py` (150 lines)
- `src/planner/q_learning.py` updated (20 lines added)
- `src/planner/alns_matheuristic.py` updated (40 lines added)
- Test suite with 95%+ coverage
- Experimental comparison report

---

### Week 5: Scale-Aware Reward Normalization

**Objectives:**
- Design scale-independent reward function
- Eliminate reward magnitude variance across problem sizes
- Maintain meaningful gradient for learning

**Problem Analysis:**

Current reward at `src/planner/alns.py:623-696`:
```python
def _compute_q_reward(...) -> float:
    baseline_cost = self._initial_solution_cost
    relative_gain = improvement / baseline_cost  # Scale-dependent!

    quality += relative_gain * params.roi_positive_scale  # 220.0
```

Issue: `baseline_cost` varies significantly:
- Small: ~35,000
- Medium: ~35,000
- Large: ~52,000

Same absolute improvement yields different rewards.

**Solution: Multi-Level Normalization**

```python
@dataclass
class ScaleAwareRewardParams:
    """Reward parameters that adapt to problem scale."""

    # Base rewards (scale-independent)
    reward_new_best_base: float = 100.0
    reward_improvement_base: float = 50.0
    reward_accepted_base: float = 10.0

    # Scale factors
    scale_factors: Dict[str, float] = field(default_factory=lambda: {
        "small": 1.0,
        "medium": 1.2,
        "large": 1.5
    })

    # ROI scaling (percentage-based)
    roi_scale: float = 1000.0  # Amplify small percentages

    # Time penalty (adaptive)
    time_penalty_scale: Dict[str, float] = field(default_factory=lambda: {
        "small": 1.0,
        "medium": 1.5,
        "large": 2.0
    })

def _compute_scale_aware_reward(
    self,
    *,
    improvement: float,
    is_new_best: bool,
    is_accepted: bool,
    action_cost: float,
    repair_operator: str,
    previous_cost: float,
    scale: str,
) -> float:
    params = self.sa_reward_params
    scale_factor = params.scale_factors[scale]

    # 1. Quality component (percentage-based, scale-independent)
    quality = 0.0
    if is_new_best:
        quality = params.reward_new_best_base * scale_factor
    elif improvement > 0:
        quality = params.reward_improvement_base * scale_factor
    elif is_accepted:
        quality = params.reward_accepted_base * scale_factor

    # 2. ROI component (normalized percentage)
    relative_improvement = improvement / previous_cost
    quality += relative_improvement * params.roi_scale * scale_factor

    # 3. Time penalty (scale-aware)
    is_matheuristic = repair_operator in ["greedy_lp", "segments"]
    if is_matheuristic and action_cost > 0:
        # Expected cost budget (scale-dependent)
        expected_cost = {"small": 0.5, "medium": 1.0, "large": 2.0}[scale]

        if action_cost > expected_cost:
            # Penalty only if benefit doesn't justify cost
            benefit_ratio = improvement / (previous_cost * 0.01)  # 1% threshold
            cost_ratio = action_cost / expected_cost

            if benefit_ratio < cost_ratio:
                penalty = (cost_ratio - benefit_ratio) * params.time_penalty_scale[scale]
                quality -= penalty * 10.0

    return quality
```

**Implementation Steps:**

1. Add `ScaleAwareRewardParams` to `src/config/defaults.py`
2. Update `src/planner/alns.py` with new reward function
3. Pass `scale` parameter through ALNS initialization chain
4. Run A/B test: old reward vs. new reward (10 seeds × 3 scales)

**Validation Metrics:**
- Reward variance across scales (should decrease by 50%+)
- Learning convergence speed (iterations to stability)
- Final solution quality (should improve on large scale)

**Deliverables:**
- Updated reward function implementation
- Configuration file updates
- A/B test results with statistical analysis
- Recommendation report

---

### Week 6-7: Complete SAQL Integration and Ablation Study

**Objectives:**
- Integrate all SAQL components
- Conduct comprehensive ablation study
- Identify critical components and optimal configuration

**Week 6: Integration**

1. **Create `ScaleAwareQLearningALNS` class** (Days 1-3)

   File: `src/planner/alns_saql.py` (new file)

   ```python
   class ScaleAwareQLearningALNS(MatheuristicALNS):
       """ALNS with scale-aware Q-learning operator selection."""

       def __init__(
           self,
           scenario: Scenario,
           preset: str = "medium",
           seed: Optional[int] = None,
       ):
           super().__init__(scenario, preset, seed)

           # Determine scale
           num_requests = len(scenario.requests)
           if num_requests <= 12:
               scale = "small"
           elif num_requests <= 30:
               scale = "medium"
           else:
               scale = "large"

           self.scale = scale

           # Initialize SAQL agent
           self.q_agent = ScaleAwareQLearningAgent(
               destroy_operators=self.destroy_operators,
               repair_operators=self.repair_operators,
               params=self.config.q_learning,
               scale=scale,
           )

           # Use scale-aware reward params
           self.sa_reward_params = ScaleAwareRewardParams()

       def optimize(self) -> Solution:
           # ... use scale-aware components ...
           pass
   ```

2. **Add preset configurations** (Days 4-5)

   File: `src/config/presets.py`

   ```python
   SAQL_PRESETS = {
       "small": {
           "max_iterations": 1000,
           "q_learning": QLearningParams(
               initial_epsilon=0.30,
               alpha=0.40,
               gamma=0.95,
               epsilon_decay=0.995,
           ),
           "stagnation_threshold": 80,
           "deep_stagnation_threshold": 200,
       },
       "medium": {
           "max_iterations": 2000,
           "q_learning": QLearningParams(
               initial_epsilon=0.50,
               alpha=0.35,
               gamma=0.95,
               epsilon_decay=0.997,
           ),
           "stagnation_threshold": 120,
           "deep_stagnation_threshold": 300,
       },
       "large": {
           "max_iterations": 4000,
           "q_learning": QLearningParams(
               initial_epsilon=0.70,
               alpha=0.30,
               gamma=0.95,
               epsilon_decay=0.998,
           ),
           "stagnation_threshold": 160,
           "deep_stagnation_threshold": 400,
       },
   }
   ```

3. **Update scripts and tests** (Days 6-7)
   - Add SAQL option to `scripts/run_alns_preset.py`
   - Create regression tests in `tests/test_saql.py`
   - Update documentation in `docs/README.md`

**Week 7: Ablation Study**

Test 5 configurations on each scale with 10 seeds:

| Config | Q-Init | State Space | Epsilon | Reward Norm |
|--------|--------|-------------|---------|-------------|
| A (Full SAQL) | Uniform(50) | 7-state | Adaptive | Scale-aware |
| B | **Zero** | 7-state | Adaptive | Scale-aware |
| C | Uniform(50) | **3-state** | Adaptive | Scale-aware |
| D | Uniform(50) | 7-state | **Fixed(0.12)** | Scale-aware |
| E | Uniform(50) | 7-state | Adaptive | **Original** |
| F (Baseline) | Zero | 3-state | Fixed(0.12) | Original |

Total runs: 5 configs × 3 scales × 10 seeds = 150 runs

**Analysis:**
1. Component contribution (ANOVA analysis)
2. Interaction effects
3. Pareto frontier (performance vs. complexity)
4. Seed variance reduction measurement

**Expected Results:**
- Full SAQL (Config A): 25-30% improvement on large scale
- Seed variance reduction: 60%+ compared to baseline
- Clear ranking of component importance

**Deliverables:**
- `src/planner/alns_saql.py` (300 lines)
- Ablation study results in `docs/experiments/saql_ablation.md`
- Configuration recommendations
- Publication-ready figures and tables

---

## Phase 2: Dynamic E-VRP Online Optimization (Weeks 8-13)

### Week 8-9: Dynamic Scenario Generator

**Objectives:**
- Design realistic dynamic scenario generator
- Implement three types of dynamic events
- Create benchmark suite for dynamic experiments

**Dynamic Event Types:**

1. **New Request Arrival** (Most common, 60% of events)
   ```python
   @dataclass
   class NewRequestEvent:
       timestamp: float        # Seconds since start
       request: Request        # New customer request
       priority: int          # 1 (normal) to 3 (urgent)
   ```

2. **Request Cancellation** (20% of events)
   ```python
   @dataclass
   class CancellationEvent:
       timestamp: float
       request_id: str
       compensation: float  # Penalty for cancellation
   ```

3. **Traffic Delay** (20% of events)
   ```python
   @dataclass
   class TrafficEvent:
       timestamp: float
       edge: Tuple[str, str]  # Affected road segment
       delay_multiplier: float  # 1.5 = 50% slower
       duration: float  # Seconds
   ```

**Generator Implementation:**

File: `src/scenario/dynamic_generator.py` (new file)

```python
class DynamicScenarioGenerator:
    """Generate dynamic E-VRP scenarios with realistic event distributions."""

    def __init__(
        self,
        base_scenario: Scenario,
        planning_horizon: float = 3600.0,  # 1 hour
        event_rate: float = 0.1,  # Events per minute
        seed: Optional[int] = None,
    ):
        self.base_scenario = base_scenario
        self.planning_horizon = planning_horizon
        self.event_rate = event_rate
        self.rng = random.Random(seed)

    def generate(self) -> DynamicScenario:
        """Generate a complete dynamic scenario."""

        # Split base requests into initial and dynamic
        all_requests = list(self.base_scenario.requests)
        self.rng.shuffle(all_requests)

        split_point = len(all_requests) // 2
        initial_requests = all_requests[:split_point]
        dynamic_pool = all_requests[split_point:]

        # Generate events
        events = self._generate_event_timeline(dynamic_pool)

        return DynamicScenario(
            initial_scenario=self.base_scenario.with_requests(initial_requests),
            events=events,
            planning_horizon=self.planning_horizon,
        )

    def _generate_event_timeline(
        self,
        request_pool: List[Request],
    ) -> List[DynamicEvent]:
        """Generate Poisson-distributed events."""

        events = []
        num_events = int(self.event_rate * self.planning_horizon / 60)

        for i in range(num_events):
            # Exponential inter-arrival times
            if i == 0:
                timestamp = self.rng.expovariate(self.event_rate / 60)
            else:
                timestamp = events[-1].timestamp + self.rng.expovariate(self.event_rate / 60)

            if timestamp > self.planning_horizon:
                break

            # Event type distribution
            event_type = self.rng.choices(
                ["new_request", "cancellation", "traffic"],
                weights=[0.60, 0.20, 0.20],
            )[0]

            if event_type == "new_request" and request_pool:
                request = request_pool.pop(0)
                priority = self.rng.choices([1, 2, 3], weights=[0.7, 0.2, 0.1])[0]
                events.append(NewRequestEvent(timestamp, request, priority))

            elif event_type == "cancellation" and events:
                # Cancel a previously added request
                request_events = [e for e in events if isinstance(e, NewRequestEvent)]
                if request_events:
                    to_cancel = self.rng.choice(request_events)
                    events.append(CancellationEvent(timestamp, to_cancel.request.id, 50.0))

            elif event_type == "traffic":
                edges = list(self.base_scenario.network.edges)
                edge = self.rng.choice(edges)
                delay = self.rng.uniform(1.2, 2.0)
                duration = self.rng.uniform(300, 1800)  # 5-30 minutes
                events.append(TrafficEvent(timestamp, edge, delay, duration))

        return sorted(events, key=lambda e: e.timestamp)
```

**Benchmark Suite Creation** (Week 9)

Create 15 dynamic scenarios:
- 5 based on small Schneider instances (c101, c102, etc.)
- 5 based on medium instances (r101, r102, etc.)
- 5 based on large instances (rc101, rc102, etc.)

Each with 3 event rates: low (0.05), medium (0.10), high (0.20) events/min

**Deliverables:**
- `src/scenario/dynamic_generator.py` (400 lines)
- 15 dynamic scenarios in `data/dynamic/`
- Scenario statistics and visualization
- Unit tests with edge case coverage

---

### Week 10-11: Anytime SAQL Implementation

**Objectives:**
- Implement anytime algorithm with quality guarantees
- Add solution quality monitoring and stopping criteria
- Optimize for 1-second response time constraint

**Anytime Framework:**

File: `src/planner/anytime_saql.py` (new file)

```python
@dataclass
class AnytimeConfig:
    """Configuration for anytime optimization."""

    min_iterations: int = 50          # Minimum before stopping
    max_time_budget: float = 1.0      # Seconds
    quality_threshold: float = 0.95   # Fraction of best-known
    stagnation_window: int = 20       # Stop if no improvement

    # Multi-fidelity settings
    enable_adaptive_operators: bool = True
    fast_mode_threshold: float = 0.5  # Switch to fast operators at 50% time

class AnytimeSAQL(ScaleAwareQLearningALNS):
    """Anytime variant with response time guarantees."""

    def __init__(
        self,
        scenario: Scenario,
        config: AnytimeConfig,
        seed: Optional[int] = None,
    ):
        super().__init__(scenario, preset="medium", seed=seed)
        self.anytime_config = config
        self._start_time = 0.0
        self._best_cost = float('inf')
        self._cost_history = []

    def optimize_with_timeout(self, timeout: float = 1.0) -> Tuple[Solution, Dict]:
        """Optimize with hard time limit and quality tracking."""

        self._start_time = time.time()
        max_time = min(timeout, self.anytime_config.max_time_budget)

        # Initialize with fast greedy heuristic
        current = self._fast_greedy_construct()
        best = current
        self._best_cost = current.cost

        iteration = 0
        stagnation = 0

        while True:
            elapsed = time.time() - self._start_time

            # Check stopping criteria
            if self._should_stop(iteration, stagnation, elapsed, max_time):
                break

            # Adaptive operator selection based on time remaining
            time_fraction = elapsed / max_time
            operators = self._select_operators(time_fraction)

            # ALNS iteration
            current = self._alns_iteration(current, operators)

            # Track progress
            if current.cost < self._best_cost:
                self._best_cost = current.cost
                best = current
                stagnation = 0
            else:
                stagnation += 1

            self._cost_history.append(current.cost)
            iteration += 1

        # Compile statistics
        stats = {
            "iterations": iteration,
            "runtime": time.time() - self._start_time,
            "final_cost": best.cost,
            "improvement_rate": self._compute_improvement_rate(),
            "quality_score": self._estimate_quality(best),
        }

        return best, stats

    def _should_stop(
        self,
        iteration: int,
        stagnation: int,
        elapsed: float,
        max_time: float,
    ) -> bool:
        """Determine if optimization should stop."""

        cfg = self.anytime_config

        # Hard time limit
        if elapsed >= max_time:
            return True

        # Minimum iterations not met
        if iteration < cfg.min_iterations:
            return False

        # Quality threshold reached
        if self._estimate_quality(self._best_cost) >= cfg.quality_threshold:
            return True

        # Stagnation
        if stagnation >= cfg.stagnation_window:
            return True

        return False

    def _select_operators(self, time_fraction: float) -> Dict[str, List[str]]:
        """Adaptively select operators based on time remaining."""

        if not self.anytime_config.enable_adaptive_operators:
            return {
                "destroy": self.destroy_operators,
                "repair": self.repair_operators,
            }

        # Early phase: use all operators
        if time_fraction < 0.3:
            return {
                "destroy": self.destroy_operators,
                "repair": self.repair_operators,
            }

        # Middle phase: balance speed and quality
        if time_fraction < 0.7:
            fast_repairs = ["greedy", "regret"]
            slow_repairs = ["greedy_lp", "segments"]
            return {
                "destroy": self.destroy_operators,
                "repair": fast_repairs + slow_repairs,
            }

        # Final phase: fast operators only
        fast_repairs = ["greedy", "regret"]
        fast_destroys = ["random", "worst"]
        return {
            "destroy": fast_destroys,
            "repair": fast_repairs,
        }

    def _fast_greedy_construct(self) -> Solution:
        """Quick initial solution using greedy insertion."""
        # Use existing greedy repair logic
        return self.repair_operators_impl["greedy"].repair(
            self.scenario,
            Solution.empty(self.scenario),
        )

    def _estimate_quality(self, cost: float) -> float:
        """Estimate solution quality relative to best-known."""
        # Use historical best from offline runs as reference
        reference_costs = {
            "small": 14000,
            "medium": 18000,
            "large": 36000,
        }
        reference = reference_costs.get(self.scale, cost)
        return max(0.0, 1.0 - (cost - reference) / reference)

    def _compute_improvement_rate(self) -> float:
        """Compute improvement rate over recent window."""
        if len(self._cost_history) < 10:
            return 0.0
        window = self._cost_history[-10:]
        return (window[0] - window[-1]) / window[0]
```

**Testing and Calibration** (Week 11)

1. **Response time benchmarking**
   - Measure iteration throughput per operator combination
   - Profile bottlenecks (LP solver, distance calculations)
   - Target: 100+ iterations per second on medium instances

2. **Quality-time tradeoff analysis**
   - Run with budgets: 0.1s, 0.5s, 1.0s, 2.0s, 5.0s
   - Measure solution quality at each time point
   - Plot diminishing returns curve

3. **Stopping criteria validation**
   - Verify early stopping doesn't sacrifice quality
   - Ensure minimum iteration threshold is met
   - Test stagnation detection effectiveness

**Deliverables:**
- `src/planner/anytime_saql.py` (500 lines)
- Performance benchmarks in `docs/experiments/anytime_performance.md`
- Calibrated `AnytimeConfig` defaults
- Response time guarantee validation

---

### Week 12-13: Transfer Learning and Multi-Fidelity Optimization

**Objectives:**
- Enable Q-table warm-starting from similar scenarios
- Implement adaptive fidelity control
- Demonstrate learning transfer benefits

**Transfer Learning Implementation:**

File: `src/planner/transfer_learning.py` (new file)

```python
class QTableTransferManager:
    """Manage Q-table transfer across scenarios."""

    def __init__(self, storage_path: str = "models/q_tables/"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def save_q_table(
        self,
        q_table: Dict[State, Dict[Action, float]],
        scenario_features: ScenarioFeatures,
        performance: float,
    ) -> str:
        """Save Q-table with metadata for future retrieval."""

        table_id = self._generate_id(scenario_features)
        metadata = {
            "features": asdict(scenario_features),
            "performance": performance,
            "timestamp": time.time(),
        }

        save_path = self.storage_path / f"{table_id}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump({"q_table": q_table, "metadata": metadata}, f)

        return table_id

    def find_similar_q_table(
        self,
        scenario_features: ScenarioFeatures,
        top_k: int = 5,
    ) -> List[Tuple[str, float, Dict]]:
        """Find most similar saved Q-tables."""

        candidates = []
        for table_file in self.storage_path.glob("*.pkl"):
            with open(table_file, "rb") as f:
                data = pickle.load(f)

            saved_features = ScenarioFeatures(**data["metadata"]["features"])
            similarity = self._compute_similarity(scenario_features, saved_features)

            candidates.append((
                table_file.stem,
                similarity,
                data["q_table"],
            ))

        # Return top-k by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    @staticmethod
    def _compute_similarity(f1: ScenarioFeatures, f2: ScenarioFeatures) -> float:
        """Compute scenario similarity score."""

        # Normalize features to [0, 1]
        size_diff = abs(f1.num_requests - f2.num_requests) / max(f1.num_requests, f2.num_requests)
        density_diff = abs(f1.spatial_density - f2.spatial_density)
        urgency_diff = abs(f1.avg_urgency - f2.avg_urgency)

        # Weighted average
        similarity = 1.0 - (
            0.4 * size_diff +
            0.3 * density_diff +
            0.3 * urgency_diff
        )

        return max(0.0, similarity)

    @staticmethod
    def merge_q_tables(
        tables: List[Dict[State, Dict[Action, float]]],
        weights: Optional[List[float]] = None,
    ) -> Dict[State, Dict[Action, float]]:
        """Merge multiple Q-tables via weighted averaging."""

        if not tables:
            raise ValueError("No Q-tables to merge")

        if weights is None:
            weights = [1.0 / len(tables)] * len(tables)

        merged = {}
        for state in tables[0].keys():
            merged[state] = {}
            for action in tables[0][state].keys():
                merged[state][action] = sum(
                    w * table[state][action]
                    for w, table in zip(weights, tables)
                )

        return merged

@dataclass
class ScenarioFeatures:
    """Features for scenario similarity comparison."""

    num_requests: int
    num_stations: int
    spatial_density: float  # Avg distance between requests
    avg_urgency: float      # Time window tightness
    fleet_size: int
```

**Multi-Fidelity Optimization:**

Add to `AnytimeSAQL`:

```python
class FidelityController:
    """Adaptively control optimization fidelity."""

    FIDELITY_LEVELS = {
        "low": {
            "lp_solver_time_limit": 0.1,
            "segment_opt_frequency": 0,  # Disabled
            "elite_pool_size": 5,
        },
        "medium": {
            "lp_solver_time_limit": 0.5,
            "segment_opt_frequency": 50,
            "elite_pool_size": 10,
        },
        "high": {
            "lp_solver_time_limit": 2.0,
            "segment_opt_frequency": 20,
            "elite_pool_size": 20,
        },
    }

    def select_fidelity(
        self,
        time_remaining: float,
        improvement_rate: float,
        current_quality: float,
    ) -> str:
        """Choose fidelity level based on context."""

        # Low time: use low fidelity
        if time_remaining < 0.2:
            return "low"

        # High quality + good progress: maintain high fidelity
        if current_quality > 0.90 and improvement_rate > 0.01:
            return "high"

        # Default: medium fidelity
        return "medium"
```

**Experimental Validation** (Week 13)

1. **Transfer learning benefits**
   - Train on 10 small scenarios
   - Test on 5 new small scenarios with/without transfer
   - Measure: convergence speed, final quality, cold-start gap

2. **Multi-fidelity effectiveness**
   - Compare fixed vs. adaptive fidelity
   - Measure: solution quality, runtime, iteration count
   - Validate fidelity switching decisions

**Expected Results:**
- Transfer learning: 20-30% faster convergence on similar scenarios
- Multi-fidelity: 15% runtime reduction with <5% quality loss

**Deliverables:**
- `src/planner/transfer_learning.py` (300 lines)
- Transfer learning experimental results
- Multi-fidelity controller integration
- Model repository with 20+ trained Q-tables

---

## Phase 3: Comprehensive Experiments and Paper Writing (Weeks 14-17)

### Week 14-15: Benchmark Experiments

**Objectives:**
- Run complete benchmark suite
- Generate publication-ready results
- Perform statistical validation

**Experiment Design:**

**Experiment 1: Static E-VRP Performance**

Test on Schneider instances (56 instances):
- 28 instances × 2 charging strategies (PR-Fixed, PR-Minimal)

Solvers:
1. MinimalALNS (baseline)
2. MatheuristicALNS (strong baseline)
3. SAQL (proposed)

Metrics per instance:
- Best cost (10 runs, best of 10)
- Average cost (10 runs)
- Standard deviation
- Runtime
- Iteration count

**Experiment 2: Dynamic E-VRP Performance**

Test on 15 dynamic scenarios:
- 5 small × 3 event rates
- 5 medium × 3 event rates
- 5 large × 3 event rates

Solvers:
1. Greedy reoptimization (baseline)
2. MatheuristicALNS with restart
3. AnytimeSAQL (proposed)

Metrics per scenario:
- Total cost (including cancellation penalties)
- Response time (per event)
- Service rate (% requests served)
- Solution quality score

**Experiment 3: Ablation Study** (From Week 7, extended)

Full factorial design:
- 4 components (Q-init, state space, epsilon, reward)
- 2 levels each (old vs. new)
- 16 configurations total

Run on 3 representative instances × 10 seeds = 480 runs

**Experiment 4: Transfer Learning**

Training pool: 20 Schneider instances
Test pool: 10 unseen instances

Compare:
1. Cold start (random Q-table)
2. Transfer (warm start from similar scenario)
3. Ensemble transfer (merge top-3 similar Q-tables)

**Statistical Analysis:**

For each experiment:
1. Wilcoxon signed-rank test (paired comparisons)
2. Friedman test (multiple solvers)
3. Effect size (Cohen's d)
4. 95% confidence intervals
5. Convergence curves
6. Performance profiles

**Computational Budget:**

Total runs: ~1,500
Estimated time: 100 CPU-hours
Parallel execution: 20 cores × 5 hours

**Deliverables:**
- Raw results in `results/benchmark_YYYY-MM-DD/`
- Processed data in CSV/JSON format
- Statistical analysis notebook
- Publication-ready figures (10+)
- Performance tables (LaTeX format)

---

### Week 16-17: Paper Writing

**Target Journals (Q2+):**

**Tier 1 (Q1 targets):**
1. **Transportation Research Part C** (IF: 8.3, Q1)
   - Focus: Methodological innovation in transport optimization
   - Fit: Strong for dynamic E-VRP with RL

2. **European Journal of Operational Research** (IF: 6.4, Q1)
   - Focus: Novel algorithms and applications
   - Fit: Good for SAQL framework

**Tier 2 (Q2 targets):**
3. **Computers & Operations Research** (IF: 4.6, Q1/Q2)
   - Focus: Computational methods
   - Fit: Excellent for our work

4. **Transportation Research Part E** (IF: 10.1, Q1)
   - Focus: Logistics and sustainability
   - Fit: Good if we emphasize electric vehicle sustainability

**Paper Structure:**

**Title (3 options):**
1. "Scale-Aware Q-Learning for Adaptive Large Neighborhood Search in Electric Vehicle Routing with Partial Recharging"
2. "Anytime Optimization for Dynamic Electric Vehicle Routing: A Reinforcement Learning Approach"
3. "Adaptive Operator Selection in ALNS via Scale-Aware Reinforcement Learning: Application to Electric Vehicle Routing"

**Abstract (250 words):**
- Problem: E-VRP with partial recharging, dynamic requests
- Gap: Existing RL-ALNS methods fail at large scale, lack online capability
- Contribution: SAQL framework with 7-state MDP, scale-aware rewards, anytime optimization
- Results: 25%+ improvement on large instances, <1s response for dynamic events
- Impact: Enables real-world deployment of RL-based metaheuristics

**1. Introduction (4 pages)**
- Motivation: E-VRP importance, partial charging benefits
- Literature gaps:
  1. Scale-dependent RL performance in metaheuristics
  2. Limited dynamic E-VRP research
  3. Lack of anytime algorithms for online VRP
- Contributions:
  1. Scale-aware Q-learning framework (SAQL)
  2. Anytime extension for dynamic scenarios
  3. Transfer learning for warm starts
  4. Comprehensive benchmark with 56 instances
- Paper organization

**2. Literature Review (5 pages)**

Subsections:
- 2.1 Electric Vehicle Routing Problems
  - Classic E-VRP formulations
  - Partial recharging strategies
  - Benchmark instances (Schneider et al.)

- 2.2 Adaptive Large Neighborhood Search
  - ALNS framework (Ropke & Pisinger, 2006)
  - Operator selection mechanisms
  - Matheuristic extensions

- 2.3 Reinforcement Learning for Metaheuristics
  - Q-learning for operator selection (Silva et al., 2019)
  - Deep RL approaches (Hottung et al., 2020)
  - State space design considerations

- 2.4 Dynamic Vehicle Routing
  - Online optimization strategies
  - Anytime algorithms
  - Real-time decision making

**3. Problem Formulation (3 pages)**
- E-VRP-PR mathematical model
- Notation table
- Assumptions and constraints
- Objective function
- Partial recharging policies

**4. Scale-Aware Q-Learning Framework (6 pages)**

- 4.1 Motivation: Scale-Dependent Performance Analysis
  - Empirical evidence of Q-learning degradation
  - Root cause analysis

- 4.2 Seven-State Markov Decision Process
  - State space design
  - State transition logic
  - Scale-dependent thresholds

- 4.3 Scale-Aware Reward Normalization
  - Problem: Reward magnitude variance
  - Solution: Multi-level normalization
  - ROI-based quality assessment

- 4.4 Adaptive Epsilon Strategy
  - Scale-specific exploration rates
  - Decay schedules
  - Theoretical justification

- 4.5 Q-Table Initialization
  - Uniform positive bias
  - Action-specific priors
  - Ablation analysis

**5. Anytime Dynamic E-VRP (4 pages)**

- 5.1 Dynamic Scenario Model
  - Event types (arrivals, cancellations, delays)
  - Poisson arrival process
  - Objective function with penalties

- 5.2 Anytime SAQL Algorithm
  - Quality-time tradeoff
  - Stopping criteria
  - Multi-fidelity operator selection

- 5.3 Transfer Learning
  - Scenario similarity metrics
  - Q-table merging strategies
  - Warm start benefits

**6. Computational Experiments (8 pages)**

- 6.1 Experimental Setup
  - Instances, hardware, parameters
  - Baseline solvers
  - Statistical methodology

- 6.2 Static E-VRP Results
  - Performance tables (by scale)
  - Statistical tests
  - Convergence analysis

- 6.3 Ablation Study
  - Component contribution
  - Interaction effects
  - Configuration recommendations

- 6.4 Dynamic E-VRP Results
  - Response time analysis
  - Solution quality comparison
  - Event handling effectiveness

- 6.5 Transfer Learning Results
  - Convergence speedup
  - Quality comparison
  - Scenario similarity validation

**7. Managerial Insights (2 pages)**
- When to use SAQL vs. matheuristic
- Dynamic scenario deployment guidelines
- Transfer learning repository benefits
- Scalability considerations

**8. Conclusion (2 pages)**
- Summary of contributions
- Practical impact
- Limitations
- Future work:
  - Deep RL extensions
  - Multi-objective optimization
  - Real-world deployment

**References:** 60-80 references

**Appendices:**
- A: Instance details
- B: Full parameter settings
- C: Additional statistical tests
- D: Supplementary figures

**Total Length:** 35-40 pages (double-spaced, EJOR format)

**Writing Timeline:**

| Days | Section | Word Count |
|------|---------|------------|
| 1-2 | Abstract + Introduction | 2,500 |
| 3-4 | Literature Review | 3,000 |
| 5-6 | Problem Formulation | 2,000 |
| 7-9 | SAQL Framework | 4,000 |
| 10-11 | Anytime Dynamic E-VRP | 2,500 |
| 12-15 | Experiments | 5,000 |
| 16-17 | Managerial Insights + Conclusion | 2,000 |
| 18-20 | Revision, references, formatting | - |
| 21 | Final proofreading | - |

**Deliverables:**
- Complete manuscript (35-40 pages)
- All figures (10+) in high-resolution
- All tables in LaTeX format
- Supplementary materials
- Cover letter draft

---

## Phase 4: Revision and Submission (Weeks 18-21)

### Week 18-19: Internal Review and Revision

**Objectives:**
- Conduct thorough self-review
- Address potential reviewer concerns
- Polish presentation

**Review Checklist:**

**Scientific Rigor:**
- [ ] All claims supported by evidence
- [ ] Statistical tests properly applied
- [ ] Assumptions clearly stated
- [ ] Limitations acknowledged
- [ ] Reproducibility ensured (code/data availability)

**Novelty:**
- [ ] Clear distinction from prior work (Silva et al., Hottung et al.)
- [ ] Contributions explicitly stated
- [ ] Technical depth sufficient for venue

**Experimental Validation:**
- [ ] Comprehensive benchmark coverage (56 instances)
- [ ] Appropriate baselines
- [ ] Fair parameter tuning for all methods
- [ ] Multiple seeds for statistical validity
- [ ] Performance profiles included

**Presentation:**
- [ ] Figures clear and informative
- [ ] Tables properly formatted
- [ ] Notation consistent
- [ ] Grammar and style polished
- [ ] Abstract compelling

**Anticipated Reviewer Concerns:**

1. **"Q-learning is not novel for ALNS"**

   Response: We acknowledge prior work (Silva et al., 2019) but our contributions are:
   - Scale-aware adaptations (addressing a known failure mode)
   - Seven-state MDP vs. simpler designs
   - Anytime extension for dynamic scenarios
   - Comprehensive ablation study

2. **"Why not deep RL?"**

   Response:
   - Tabular Q-learning offers transparency and interpretability
   - Lower computational overhead (suitable for real-time)
   - No need for large training datasets
   - Future work can extend to deep RL

3. **"Limited real-world validation"**

   Response:
   - Schneider instances are standard benchmarks
   - Dynamic scenario generator based on realistic assumptions
   - Future work: industrial pilot study

4. **"Transfer learning benefits seem modest"**

   Response:
   - 20-30% faster convergence is significant for online settings
   - Accumulating benefit over many scenarios
   - Scenario similarity metric can be refined (future work)

**Revision Tasks:**

1. Strengthen literature review (add 10+ recent papers)
2. Add algorithmic pseudo-code for all methods
3. Expand managerial insights section
4. Create supplementary material document
5. Prepare rebuttal preemptively

**Deliverables:**
- Revised manuscript (v2)
- Supplementary materials (20+ pages)
- Response to anticipated concerns
- Code repository (GitHub)

---

### Week 20: Submission Preparation

**Tasks:**

1. **Format Manuscript** (Days 1-2)
   - Apply journal template (EJOR LaTeX)
   - Check figure/table quality (300+ DPI)
   - Verify references format
   - Compile supplementary materials

2. **Prepare Code Repository** (Days 3-4)
   - Clean code, add documentation
   - Create README with reproduction instructions
   - Archive experiments, results
   - License selection (MIT or GPL)
   - GitHub release with DOI (Zenodo)

3. **Write Cover Letter** (Day 5)

   Template:
   ```
   Dear Editor,

   We submit our manuscript titled "Scale-Aware Q-Learning for Adaptive
   Large Neighborhood Search in Electric Vehicle Routing with Partial
   Recharging" for consideration in [Journal Name].

   This work addresses a critical gap in reinforcement learning-based
   metaheuristics: scale-dependent performance degradation. Our proposed
   SAQL framework demonstrates 25%+ improvement on large-scale instances
   where standard Q-learning fails, validated on 56 benchmark instances.

   We believe this work is of interest to [Journal] readers because:
   1. Novel scale-aware adaptations with theoretical justification
   2. Practical anytime algorithm for dynamic scenarios
   3. Comprehensive experimental validation
   4. Open-source implementation for reproducibility

   Suggested reviewers:
   - Dr. X (expert in E-VRP)
   - Dr. Y (expert in ALNS)
   - Dr. Z (expert in RL for optimization)

   All authors have approved the manuscript and declare no conflicts of interest.

   Sincerely,
   [Your Name]
   ```

4. **Final Quality Check** (Days 6-7)
   - Independent proofreader review
   - Plagiarism check (Turnitin/iThenticate)
   - Verify all figures/tables cited
   - Check supplementary material links

**Submission Checklist:**
- [ ] Manuscript PDF
- [ ] Supplementary materials PDF
- [ ] All figures (separate files)
- [ ] Cover letter
- [ ] Conflict of interest statement
- [ ] Code/data availability statement
- [ ] Suggested reviewers (3-5)
- [ ] Highlights (3-5 bullet points)
- [ ] Graphical abstract (if required)

**Deliverables:**
- Submission-ready package
- Code repository public (GitHub)
- Internal submission record

---

### Week 21: Submission and Next Steps

**Day 1-2: Submit to Target Journal**

Priority order:
1. Transportation Research Part C (first choice)
2. European Journal of Operational Research (if rejected by TRC)
3. Computers & Operations Research (backup)

**Day 3-5: Prepare Conference Submissions**

Target conferences (6-month lead time):
1. **INFORMS Annual Meeting** (October 2025)
   - 20-minute presentation slot
   - Submit abstract by May 2025

2. **VeRoLog** (Vehicle Routing and Logistics Optimization, June 2025)
   - 2-page extended abstract
   - Submit by February 2025

3. **EURO** (European Conference on Operational Research, July 2025)
   - Full paper or extended abstract
   - Submit by March 2025

**Day 6-7: Future Research Planning**

**Short-term (3 months):**
- Implement multi-objective Pareto optimization (original Direction 3)
- Explore deep RL extensions (policy networks)
- Real-world data collection (partner with logistics company)

**Medium-term (6-12 months):**
- Second paper: "Multi-Objective E-VRP with SAQL"
- Third paper: "Deep RL for Large-Scale Dynamic VRP"
- Workshop organization: "RL for Combinatorial Optimization"

**Long-term (1-2 years):**
- Book chapter on RL-based metaheuristics
- Industrial deployment case study
- Ph.D. thesis compilation

---

## Implementation Milestones and Success Criteria

### Key Milestones

| Week | Milestone | Success Criteria |
|------|-----------|------------------|
| 2 | Q-init experiments complete | 4 strategies tested, best identified |
| 4 | 7-state agent implemented | Tests pass, state transitions validated |
| 5 | Scale-aware rewards deployed | Variance reduction >50% across scales |
| 7 | SAQL fully integrated | Ablation study complete, 25%+ improvement on large |
| 9 | Dynamic generator ready | 15 scenarios generated, validated |
| 11 | Anytime SAQL working | Response time <1s, quality >90% |
| 13 | Transfer learning validated | 20%+ faster convergence demonstrated |
| 15 | All experiments complete | 1,500 runs finished, results analyzed |
| 17 | Paper draft complete | 35-40 pages, all sections written |
| 19 | Revision complete | Internal review passed |
| 21 | Paper submitted | Submission confirmed, code public |

### Success Metrics (Final Targets)

**Performance:**
- Small scale: ≥60% improvement (maintain current)
- Medium scale: ≥40% improvement (current: 29%)
- Large scale: ≥25% improvement (current: 7%) ← **Critical**
- Seed variance: CV < 0.15 (current: ~0.40)

**Dynamic Scenarios:**
- Response time: <1.0 seconds per event
- Service rate: >95% of requests served
- Quality score: >0.85 relative to offline optimal

**Publication:**
- Manuscript accepted at Q2+ journal within 12 months
- Conference presentations: 2+ in 2025
- Citations: 10+ within first year
- Code repository: 50+ stars on GitHub

---

## Resource Requirements

### Computational Resources

**Hardware:**
- CPU: 20+ cores recommended for parallel experiments
- RAM: 32GB minimum
- Storage: 100GB for results, models, code

**Software:**
- Python 3.9+
- Gurobi (academic license)
- Standard scientific stack (NumPy, SciPy, pandas, matplotlib)
- Git, GitHub Actions (for CI/CD)

### Time Commitment

**Weeks 1-7 (Phase 1):**
- Coding: 25 hours/week
- Experiments: 10 hours/week
- Literature: 5 hours/week
- **Total: 40 hours/week**

**Weeks 8-13 (Phase 2):**
- Coding: 20 hours/week
- Experiments: 15 hours/week
- Literature: 5 hours/week
- **Total: 40 hours/week**

**Weeks 14-17 (Phase 3):**
- Experiments: 10 hours/week
- Writing: 25 hours/week
- Analysis: 5 hours/week
- **Total: 40 hours/week**

**Weeks 18-21 (Phase 4):**
- Revision: 15 hours/week
- Formatting: 10 hours/week
- Preparation: 15 hours/week
- **Total: 40 hours/week**

**Grand Total: ~840 hours over 21 weeks**

---

## Risk Mitigation

### Risk 1: Large-Scale Performance Not Improving

**Probability:** Medium
**Impact:** High (undermines main claim)

**Mitigation:**
- Week 5 checkpoint: If reward normalization doesn't help, try alternative designs
- Week 7 checkpoint: If SAQL still underperforms, pivot to ensemble methods
- Fallback: Focus paper on dynamic E-VRP (anytime SAQL) instead of scale issues

### Risk 2: Transfer Learning Benefits Too Small

**Probability:** Medium
**Impact:** Low (nice-to-have, not core contribution)

**Mitigation:**
- Week 13 checkpoint: If benefit <10%, de-emphasize in paper
- Alternative: Use transfer learning only for dynamic scenarios (cold start problem)
- Can be removed from paper if results are weak

### Risk 3: Experiments Taking Too Long

**Probability:** Low
**Impact:** Medium (delays paper writing)

**Mitigation:**
- Parallelize aggressively (use cloud compute if needed)
- Reduce number of seeds from 10 to 5 if necessary
- Focus on representative instances rather than full 56

### Risk 4: Paper Rejected from Top Venues

**Probability:** Medium
**Impact:** Medium (delays publication timeline)

**Mitigation:**
- Have backup journals ready (3 tiers prepared)
- Preemptive rebuttal for common concerns
- Conference publications as fallback (faster review)

---

## Conclusion

This 21-week plan provides a structured roadmap to transform the current unstable Q-learning implementation into a publication-worthy Scale-Aware Q-Learning framework with dynamic optimization capabilities. The plan is designed to:

1. **Address root causes** of current Q-learning instability
2. **Deliver genuine innovations** suitable for Q2+ publication
3. **Maintain practicality** for real-world deployment
4. **Provide statistical rigor** meeting academic standards
5. **Manage risks** with clear checkpoints and fallbacks

By following this plan, we aim to achieve:
- **Technical Excellence:** 25%+ improvement on large-scale E-VRP instances
- **Scientific Rigor:** Comprehensive ablation study with statistical validation
- **Practical Impact:** Sub-second response times for dynamic scenarios
- **Publication Success:** Acceptance at a Q2+ journal (target: TRC or EJOR)

The combination of scale-aware adaptations and dynamic capabilities positions this work at the intersection of three active research areas (E-VRP, RL-metaheuristics, online optimization), maximizing its potential impact and citation prospects.

**Next Immediate Actions:**
1. Commit this plan to repository
2. Set up project management board (GitHub Projects)
3. Begin Week 1 experiments (Q-learning stability analysis)
4. Schedule weekly progress reviews

---

**Document Version:** 1.0
**Last Updated:** 2025-11-09
**Status:** Ready for Implementation
