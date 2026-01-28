# MIP 基准模型（对照 RL 规则选择调度）

## 作用与定位

本模型用于小规模对照基准，和线上调度中的“RL 选择可解释规则（selection hyper-heuristic / rule-portfolio RL）”保持一致的建模假设：动态任务到达、部分充电、冲突规避（collision-aware）。该基准不包含固定成本项，目标仅保留行驶、充电、迟到、冲突等待、拒单与驻留惩罚。Scenario 与 Epoch 的定义与事件驱动调度保持一致。

默认求解器与规模由代码配置：

- 求解器：`ortools`
- 最小规模：`max_tasks=8`，`max_vehicles=2`，`max_charging_stations=2`
  （详见 `src/baselines/mip/config.py`）

## 与创新点的对应关系

- **规则库 + RL 选择**：通过规则选择变量 `pi_{e,h}` 表达“在决策时刻 e 选择规则 h”，RL 在线输出规则，MIP 可以作为离线最优或固定规则对照。
- **可学习的部分充电**：用离散充电深度集合 `S` 表达有限的充电目标（可后续扩展为连续决策）。
- **冲突规避安全屏蔽**：节点/边头距约束确保碰撞安全，冲突等待 `w` 进入成本；边级等待 `t` 仅提供下界，`w` 汇聚节点/边延迟。
- **Scenario 与 Epoch 对齐事件定义**：Scenario 记录任务释放时刻与可选扰动（时间窗/服务时间/需求/拥堵/充电站可用性）。Epoch 由事件触发，MIP 中以“任务到达事件时刻集合”形成离散决策时刻（可扩展到机器人空闲、充电完成、SOC 过低、死锁风险等）。

## 模型定义

### 索引与集合

- `A`: AMR 集合，`a,b ∈ A`
- `R`: 请求集合，`r ∈ R`
- `P/D`: 取/送节点集合
- `C`: 充电站集合
- `N`: 全部节点集合（含 depot）
- `E`: 有向边集合 `(i,j)`
- `W`: 场景集合（动态任务到达/需求不确定）
- `E^dec`: 决策时刻集合
- `H`: 规则库集合
- `S`: 充电目标等级集合（离散充电深度）

### 参数

- `d_ij`: 距离；`tau_ij`: 行驶时间
- `s_i`: 服务时间；`E_i`, `Ehat_i`: 时间窗
- `Q_a`: 载重上限；`B_max^a`, `B_0^a`: 电池上限/初始电量
- `rho`: 单位行驶时间能耗；`kappa`, `eta`: 充电速率与效率
- `T_ch_max`, `E_ch_max`: 每次充电上限
- `Delta_t_safe`: 最小安全时距
- `delta_r^w`: 场景 w 中请求 r 是否到达
- `tau_r^w`: 场景 w 中任务 r 的释放时间
- `gamma_w`: 场景 w 的行驶时间倍率（拥堵/不确定性）
- `a_i^w`: 场景 w 中充电站 i 是否可用
- `s_i^w / q_r^w`: 场景 w 中服务时间/需求量的可选扰动（若不提供则取基准任务属性）
- `q_i^w`: 需求量（未出现则为 0）
- `p_w`: 场景概率
- `alpha_2..alpha_9`: 成本权重（行驶/行驶时间/迟到/普通等待/充电/冲突等待/拒单/不可行/驻留）
- `M`: 大常数
- `SOC_min`: 仿真层硬约束的最低电量比例（来自 `EnergySystemDefaults.safety_threshold`）

### 决策变量

- `x_ij^{a,w} ∈ {0,1}`: 车辆 a 是否走边 (i,j)
- `y_i^{a,w} ∈ {0,1}`: 车辆 a 是否服务节点 i
- `z_r^w ∈ {0,1}`: 请求 r 是否被接受
- `T_i^{a,w} ≥ 0`: 服务开始时刻
- `F_i^{a,w} ≥ 0`: 离开时刻
- `L_i^{a,w} ≥ 0`: 迟到
- `u_i^{a,w} ≥ 0`: 普通等待（早到/空驶）
- `w_i^{a,w} ≥ 0`: 冲突等待（节点/边汇聚）
- `s_i^{a,w} ≥ 0`: 预测驻留（standby，仅用于驻留/DWELL）
- `b_arr,i^{a,w}`, `b_dep,i^{a,w} ≥ 0`: 到达/离开电量
- `q_i^{a,w} ≥ 0`: 充电量；`g_i^{a,w} ≥ 0`: 充电时间
- `l_i^{a,w} ≥ 0`: 载重
- `m_ij^{a,b,w}`, `n_k^{a,b,w} ∈ {0,1}`: 边/节点优先权
- `pi_{e,h}^w ∈ {0,1}`: 决策时刻 e 选择规则 h
- `delta_{i,s}^{a,w} ∈ {0,1}`: 充电目标等级选择

## 目标函数（无固定成本）

```
min  Σ_w p_w [
  α2 Σ_a Σ_(i,j) d_ij x_ij^{a,w}
  + α2t Σ_a Σ_(i,j) t_ij^{a,w}
  + α3 Σ_a Σ_i L_i^{a,w}
  + α3w Σ_a Σ_i u_i^{a,w}  # 普通等待（早到/排队），不包含 standby
  + α4 Σ_a Σ_{i∈C} q_i^{a,w}
  + α5 Σ_a Σ_i w_i^{a,w}
  + α6 Σ_r (1 - z_r^w)
  + α6i Σ_r (1 - z_r^w)
  + α7 Σ_a Σ_i s_i^{a,w}  # 驻留/预测等待
]
```

## 主要约束（对应项目约束 1–17）

1. **接受请求必须被服务一次 & 同车取送**
   - `Σ_a Σ_j x_{i(r),j}^{a,w} = z_r^w`
   - `Σ_a Σ_j x_{j(r),j}^{a,w} = z_r^w`
   - `Σ_j x_{i(r),j}^{a,w} - Σ_j x_{j(r),j}^{a,w} = 0`
   - `z_r^w ≤ delta_r^w`
2. **流守恒 / 起终点唯一**
   - `Σ_j x_{j,i}^{a,w} = Σ_j x_{i,j}^{a,w} = y_i^{a,w}`
   - `Σ_j x_{o_a,j}^{a,w} = 1`, `Σ_j x_{j,u_a}^{a,w} = 1`
3. **时间传播与离开时刻守恒**
   - `T_i^{a,w} = A_i^{a,w} + u_i^{a,w} + w_i^{a,w}`
   - `F_i^{a,w} = T_i^{a,w} + s_i + g_i^{a,w} + s_i^{a,w}`
   - `A_j^{a,w} ≥ F_i^{a,w} + tau_ij - M(1 - x_ij^{a,w})`
4. **取货先于送货**
   - `T_{j(r)}^{a,w} ≥ F_{i(r)}^{a,w} - M(1 - Σ_j x_{i(r),j}^{a,w})`
5. **时间窗与迟到**
   - `T_i^{a,w} ≥ E_i`
   - `L_i^{a,w} ≥ T_i^{a,w} - Ehat_i`
   - `T_{p(r)}^{a,w} ≥ tau_r^w`（取货节点释放时刻）
6. **电量初始化与传播**
   - `b_dep,o_a^{a,w} = B_0^a`
   - `b_arr,j^{a,w} ≥ b_dep,i^{a,w} - rho * tau_ij - M(1 - x_ij^{a,w})`
   - `b_dep,i^{a,w} = b_arr,i^{a,w} + q_i^{a,w}`
   - `0 ≤ b_arr,i^{a,w}, b_dep,i^{a,w} ≤ B_max^a`
7. **充电站限制与耦合**
   - `q_i^{a,w} ≤ E_ch_max * y_i^{a,w}`, `i ∈ C`
   - `q_i^{a,w} = eta * kappa * g_i^{a,w}`
   - `g_i^{a,w} ≤ T_ch_max * y_i^{a,w}`
   - `q_i^{a,w} = 0`, `i ∉ C`
8. **部分充电离散化**
   - `q_i^{a,w} = Σ_s q_s * delta_{i,s}^{a,w}`
   - `Σ_s delta_{i,s}^{a,w} = y_i^{a,w}`
9. **载重传播**
   - `l_j^{a,w} ≥ l_i^{a,w} + q_i^w - M(1 - x_ij^{a,w})`
   - `0 ≤ l_i^{a,w} ≤ Q_a`
10. **冲突规避（节点/边头距）**
   - `T_k^{b,w} ≥ F_k^{a,w} + Delta_t_safe - M(1 - n_k^{a,b,w})`
   - `T_k^{a,w} ≥ F_k^{b,w} + Delta_t_safe - M(n_k^{a,b,w})`
   - `F_i^{b,w} ≥ F_i^{a,w} + tau_ij + Delta_t_safe - M(1 - m_ij^{a,b,w})`
   - `F_i^{a,w} ≥ F_i^{b,w} + tau_ij + Delta_t_safe - M(m_ij^{a,b,w})`
   - `w_i^{a,w} ≥ Σ t_{i,j}^{a,b,w}`（边级等待为下界，节点/边延迟汇入 w）
11. **规则选择层（事件驱动 Epoch）**
   - `E(w)` 默认由任务释放时刻组成；如需包含“机器人空闲/充电完成/SOC 过低/死锁风险”等事件，可在场景中显式提供 `decision_epoch_times`
   - `Σ_h pi_{e,h}^w = 1`, `∀ e∈E(w), w`

## Scenario/epoch 文件化与回放

- 代码入口：`src/baselines/mip/scenario_io.py`
- 生成脚本：`scripts/generate_scenarios.py`
- 推荐格式：JSON（完整字段）+ CSV（任务表）

示例命令：

```bash
python scripts/generate_scenarios.py --epoch-mode release
python scripts/generate_scenarios.py --epoch-mode simulate --max-steps 200
```
   - 规则 h 的约束用大 M 与 `pi_{e,h}^w` 关联

## 基准使用方式

- **小规模对照**：固定 `W` 与 `R` 的规模（如 5–15 任务，1–3 车）求解最优，作为 RL 在线策略的上界对照。
- **下界版本**：放松二进制变量或移除冲突头距约束，得到可计算下界。
- **规则对照**：固定 `pi_{e,h}` 为某条规则（或轮换规则），与 RL 结果对比。
