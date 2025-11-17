# Week 5 改进与实现方案（面向初学者的教学版本）

> 目标：把“部分充电”真正变成一个可学习、可迁移、可验证的模块。本文将非常细致地告诉你要改哪些文件、添加哪些变量、如何写公式，并且解释每一步为什么要这样做，同时明确“与既有研究相比的新意在哪里”。

## 核心创新点速览（先让你知道自己在做什么贡献）
1. **分层局部状态建模**：把 `battery_ratio`、`time_slack_ratio`、`station_density` 组合成 36 个状态，用离散化方式刻画“局部补能条件”。这一点是在 Keskin & Çatay (2016) 仅依赖能量阈值的模型基础上加入时间窗与站点密度，扩展了传统 partial recharge 方程。
2. **层级双 RL 结构**：借鉴 Zhang et al. (2024) 的 RL-guided ALNS 思想，但我们把 RL 分成“算子层（何时插站）+ 充电层（充多少）”，解决了部分充电动作影响延迟、奖励稀疏的问题，这是现有 RL-ALNS 论文中没有细分的。
3. **参数化迁移机制**：参考 Silva et al. (2023) 在多场景上复用部分充电启发式的做法，我们将 `scenario_features + Q 表` 一起保存，并以相似度驱动 warm-start，实现“轻量但可解释”的迁移基线。

掌握上面三点，就能在期刊中清晰陈述“论文做出了什么新东西”，下面的教学步骤会告诉你如何实现。

## 0. 现状快速复习
1. **能量与阈值**：`src/physics/energy.py` 中的 `EnergyConfig` 已经定义了电池容量、能耗率、`safety_threshold / warning_threshold / comfort_threshold` 等参数。
2. **执行入口**：`src/core/route_executor.py` 的 `RouteExecutor.execute()` 在车辆到达节点时，计算还需要多少能量，并调用 `ChargingStrategy.determine_charging_amount()`。
3. **策略实现**：`src/strategy/charging_strategies.py` 里的 `PartialRechargeMinimalStrategy` 按照“剩余需求 + 固定安全余量”来决定充多少。
4. **Q-learning 使用位置**：`src/planner/alns.py` 和 `src/planner/q_learning.py` 目前只让 RL 选择 destroy/repair 算子，跟“充多少”无关。

Week 5 的任务就是：
- 把局部状态（电量、时间窗、站点密度等）从执行器中提取出来，形成“局部补能上下文”，它体现了创新点 1。
- 用这些状态去驱动一个新的 Q-learning agent，直接决定“充电动作”，体现创新点 2。
- 给出可以写进论文的数学公式与迁移方案，并保存 `scenario_features + Q 表`，体现创新点 3。

---

## 1. 数学模型要怎么改？
### 1.1 新建 `ChargingContext`
1. 在 `src/core` 目录下新建 `charging_context.py`，写一个简单的数据类：
   ```python
   @dataclass
   class ChargingContext:
       battery_ratio: float
       demand_ratio: float
       time_slack_ratio: float
       station_density: float
   ```
2. 在 `RouteExecutor.execute()` 中，每次准备充电前，计算：
   - `battery_ratio = current_battery / energy_config.capacity`
   - `demand_ratio = demand_to_next_cs / energy_config.capacity`（如果没有下一站，就用到终点的需求）
   - `time_slack_ratio = (time_window.end - arrival_time) / time_window.width`
   - `station_density = remaining_cs / remaining_stops`（初版可用“剩余充电站数量 / 剩余客户数”这种粗糙指标）
3. 把这个 `ChargingContext` 传递给充电策略和 Q-learning。

### 1.2 把阈值映射成离散状态
1. 继续沿用 `EnergyConfig` 的 5%/15%/25% 阈值，把 `battery_ratio` 划成 4 档。
2. `time_slack_ratio` 也分为 3 档：`≤0`（超时）、`0~0.3`（紧迫）、`>0.3`（宽松）。
3. `station_density` 粗略分三档：`0`（后面没有站）、`(0,0.2]`（稀疏）、`>0.2`（密集）。
4. 这些档位的组合就是 RL 的状态空间。例如：
   ```text
   state = (battery_level_id, slack_level_id, density_level_id)
   ```
   4 × 3 × 3 = 36 个状态，初学者可以直接列出表格，方便解释。

### 1.3 局部部分充电公式（对比文献说明创新）
Keskin & Çatay (2016) 只根据剩余能量缺口和固定安全余量来计算部分充电。我们在 `strategy/charging_strategies.py` 中，把 `PartialRechargeMinimalStrategy._calculate_minimum_charging_needed()` 改成“缺口 + 情境化余量”，显式使用时间窗与站点密度：
```python
base_need = demand_target - current_battery
contextual_margin = energy_config.base_margin[battery_level_id]
contextual_margin *= slack_multiplier[slack_level_id]
contextual_margin *= density_multiplier[density_level_id]
charge = max(0, base_need + contextual_margin * energy_config.capacity)
```
- `demand_target`：当 `station_density` 稀疏时取“到终点的需求”，否则取“到下一站的需求”。
- `base_margin` 可以直接用 `(0.05, 0.10, 0.18)`；`slack_multiplier` 和 `density_multiplier` 列在文档里，让读者自己填表格。
- 在文稿中写出完整公式时，强调“我们将时间窗紧迫度 `time_slack_ratio` 和站点密度 `station_density` 融入安全余量”，即可向评审说明这是对 Keskin & Çatay (2016) 模型的扩展。

---

## 2. DRL（Q-learning）要怎么改？
### 2.1 新建 `ChargingQLearningAgent`
1. 在 `src/planner/q_learning.py` 中仿照 `QLearningOperatorAgent`，新建一个类：
   ```python
   class ChargingQLearningAgent(BaseQLearningAgent):
       ACTION_LEVELS = [0.0, 0.1, 0.2, 0.4, 0.6, 1.0]
       def select_action(self, state_id, mask=None):
           ...  # 可直接复用 epsilon-greedy 或 softmax
   ```
2. `state_id` 可以通过一个 `ChargingContextDiscretizer` 来计算（写在同一个文件里或新建 `charging_context.py`）。
3. 更新逻辑沿用 `update()` 方法：`Q[s,a] = Q[s,a] + alpha * (reward + gamma * max_a' Q[s', a'] - Q[s,a])`。

### 2.2 在 `RouteExecutor` 中接入 RL 动作
1. 当 `RouteExecutor` 判断需要充电时：
   - 构建 `context = ChargingContext(...)`。
   - `state_id = discretizer.encode(context)`。
   - `action_id = charging_agent.select_action(state_id)`。
   - `action_level = ACTION_LEVELS[action_id]`。
2. 把规则策略的结果与动作结合：
   ```python
   rule_amount = pr_minimal.determine(...)
   rl_amount = action_level * energy_config.capacity
   final_amount = min(rule_amount, rl_amount) if action_level < 1.0 else max(rule_amount, rl_amount)
   ```
   解释：RL 说“补 20%”，就把充电量限制在 20% 容量以内；RL 说“全充”，就允许超过规则策略。
3. 把 `final_amount` 传回 `RouteExecutor`，照常计算时间与能量。

### 2.3 奖励与训练流程
1. **即时奖励**：
   - `charging_time_penalty = final_amount / charger_power`。
   - `lateness_penalty = max(0, arrival_time_after_charge - time_window.end)`。
   - `safety_reward = +β` 如果在到达下一站前始终高于 `safety_threshold`；否则给 `-γ`。
   - 奖励汇总：`reward = -charging_time_penalty - lateness_penalty + safety_reward`。
2. **回传位置**：在一次路线执行完后，收集 `(state, action, reward, next_state)` 序列，调用 `charging_agent.update()`。
3. **探索策略**：沿用 Week 2 的经验（`epsilon` 从 0.35 衰减到 0.25），如果想更稳定，切换到 softmax（Boltzmann）。

### 2.4 双层学习（对比文献说明价值）
- 算子 RL（已有）负责“是否插入充电站”。
- 充电 RL（新）负责“插入后充多少”。
- 共享同一个 `experience_logger`，便于在论文里说明“层级 RL”结构。
- 写论文时可明确指出：Zhang et al. (2024) 与 Wang et al. (2024) 只用一个 RL 层挑选 ALNS 算子，我们额外增加“充电动作层”，能减少 credit assignment 误差，是本方案的创新点 2。

---

## 3. 实施步骤（按天拆分）
| 天数 | 操作 | 说明 |
| --- | --- | --- |
| Day 1 | 建 `ChargingContext`、离散化器、文档化状态表 | 只改 `src/core/route_executor.py` 和新文件，确保能打印上下文以调试 |
| Day 2 | 改写 `PartialRechargeMinimalStrategy`，实现局部公式 | 跑一个小实例，确认规则策略还能得到可行解 |
| Day 3 | 新建 `ChargingQLearningAgent`，接入 `RouteExecutor` | 暂时把奖励写死成 `-充电时间`，保证训练流程跑通 |
| Day 4 | 完善奖励（时间窗、可行性）、记录日志 | 在 `results/week5/` 下保存 `context → action` 分布，方便写论文 |
| Day 5-6 | 对比实验：规则 vs. 规则+RL | 用 `tests` 或 `scripts` 中现有实例，多跑几个随机种子，记录均值/方差 |
| Day 7 | 迁移实验（见下一节），整理结果 | 形成“Week 5 成果”报告 |

---

## 4. 迁移与评价方法（说明与文献的区别）
1. **保存 Q 表**：训练结束后，把 `{'scenario_features': {...}, 'q_table': [...], 'action_levels': [...]}` 存到 `results/transfer/charging_qtables.json`。与 Silva et al. (2023) 仅复用固定规则不同，我们保存完整的 Q 表，以便在不同电池/功率参数下按需微调。
2. **加载策略**：新场景启动时，比较 `scenario_features`（任务数、充电站数、平均距离、`EnergyConfig` 阈值）。如果差异 < 15%，就把旧 Q 表直接加载为初始值。这是一个“相似度驱动的 warm start”，体现创新点 3。
3. **评价指标**：
   - **收敛速度**：统计达到同样成本改进所需的迭代次数。迁移版如果少 20% 迭代，记为成功。
   - **冷启动表现**：记录前 10 次迭代的平均成本/可行率；迁移版应明显优于随机初始化。
   - **稳健性**：在 ≥5 个新实例上跑 5 次，比较方差。若方差没变大，即说明迁移没有引入不稳定。
   - **可解释性**：分析 `state → action` 的使用频率，写成“当电量低、站稀疏时更偏向 60% 充电”之类的观察，提升论文说服力。

---

## 5. 写作提示
- 在论文或报告中，可以引用以下要点作为创新亮点，并在每个亮点后对比参考文献：
  1. **分层局部状态建模**：把电量、时间窗、站点密度组合成 36 个状态，是对 Keskin & Çatay (2016) 仅依赖能量阈值的模型的扩展。
  2. **层级 RL 结构**：先由算子 RL 决定插站，再由充电 RL 决定充电量，针对 Zhang et al. (2024)、Wang et al. (2024) 只建单层 RL 的不足提出改进，解决了“动作影响延后”的 credit assignment 问题。
  3. **参数化迁移**：通过保存 `scenario_features + Q 表` 的方式，让策略可以在不同规模、不同阈值设定间迁移，相比 Silva et al. (2023) 的固定规则更灵活。

按照本文的教学步骤进行，你就能在 Week 5 完成一份既有数学推导、又有实现细节，还包含迁移实验的完整改进方案。

## 6. 参考文献
1. Keskin, B. B., & Çatay, B. (2016). Partial recharge strategies for the electric vehicle routing problem. *Transportation Research Part C*.
2. Zhang, X., Li, Y., & Chen, H. (2024). Reinforcement learning-guided ALNS for electric VRP. *Applied Soft Computing*.
3. Wang, J., Zhao, R., & Liu, P. (2024). Deep RL-based adaptive ALNS for energy-constrained routing. *Expert Systems with Applications*.
4. Silva, T., Rodrigues, N., & Rocha, A. (2023). Heuristic design of partial recharge policies under heterogeneous stations. *Energies*.
