# R3 项目结构说明

## 项目概述

R3 是一个电动车辆路径规划（E-VRP）项目，使用ALNS（Adaptive Large Neighborhood Search）元启发式算法，支持多种充电策略和动态充电站优化。

**当前版本**：Week 2 完成 ✅
**最后更新**：2025-10-23

---

## 目录结构

```
R3/
├── src/                    # 源代码
│   ├── core/              # 核心数据结构
│   ├── physics/           # 物理模型（能量、时间、距离）
│   ├── strategy/          # 充电策略
│   └── planner/           # ALNS优化器
├── tests/                 # 测试代码
│   ├── week1/            # Week 1功能测试
│   ├── week2/            # Week 2功能测试
│   ├── charging/         # 充电策略测试
│   ├── cost/             # 成本评估测试
│   └── debug/            # 调试工具
├── docs/                  # 文档
│   ├── planning/         # 计划和范围文档
│   ├── summaries/        # 工作总结
│   └── technical/        # 技术说明文档
├── .gitignore
├── README.md
└── PROJECT_STRUCTURE.md   # 本文档
```

---

## 源代码结构 (`src/`)

### `core/` - 核心数据结构

| 文件 | 说明 | 关键类/函数 |
|------|------|------------|
| `node.py` | 节点定义 | `Node`, `create_depot()`, `create_charging_node()` |
| `route.py` | 路径和访问记录 | `Route`, `RouteNodeVisit` |
| `task.py` | 任务定义 | `Task`, `TaskPool` |
| `vehicle.py` | 车辆定义 | `Vehicle`, `create_vehicle()` |
| `route_executor.py` | 路径执行器 | `RouteExecutor` - 模拟路径执行 |

**关键概念**：
- **Node**：depot（仓库）、pickup（取货）、delivery（送货）、charging_station（充电站）
- **Route**：节点序列 + 访问记录（visits）
- **Task**：一对pickup-delivery节点
- **RouteExecutor**：根据充电策略模拟路径执行，生成电池/时间轨迹

### `physics/` - 物理模型

| 文件 | 说明 | 关键函数 |
|------|------|---------|
| `distance.py` | 距离计算 | `DistanceMatrix`, `calculate_euclidean_distance()` |
| `energy.py` | 能量模型 | `EnergyConfig`, `calculate_minimum_charging_needed()` |
| `time.py` | 时间模型 | `TimeConfig`, `TimeWindow`, `calculate_travel_time()` |

**关键参数**：
- **能量**：`consumption_rate`（0.5 kWh/km）, `charging_rate`, `critical_battery_threshold`（0%禁用）
- **时间**：`vehicle_speed`（15 m/s = 54 km/h）
- **距离**：欧几里得距离（米）

### `strategy/` - 充电策略

| 文件 | 说明 |
|------|------|
| `charging_strategies.py` | 充电策略实现 |

**支持的策略**：
1. **FR (Full Recharge)**：完全充电到100%
2. **PR-Fixed**：充电到固定百分比（如30%）
3. **PR-Minimal**：最小充电策略（只充刚好够用的 + 10%安全余量）

### `planner/` - ALNS优化器

| 文件 | 说明 |
|------|------|
| `alns.py` | ALNS主算法 |

### `baselines/` - 对照基准（MIP）

| 文件 | 说明 |
|------|------|
| `baselines/mip/model.py` | MIP基准模型结构与约束说明 |
| `baselines/mip/solver.py` | 基准求解器接口（默认 OR-Tools） |
| `baselines/mip/config.py` | 基准规模与求解器配置 |

**核心组件**：
- **Destroy算子**：`random_removal()`（可移除充电站）
- **Repair算子**：`greedy_insertion()`, `regret2_insertion()`, `random_insertion()`
- **充电站管理**：
  - `_find_battery_depletion_position()`
  - `_get_available_charging_stations()`
  - `_find_best_charging_station()`
  - `_insert_necessary_charging_stations()`

**成本函数**：
```python
总成本 = 距离成本 + 充电成本 + 时间成本 + 延迟成本 + 惩罚
```

---

## 测试结构 (`tests/`)

### `week1/` - Week 1功能测试（基础功能）

| 文件 | 测试内容 |
|------|---------|
| `test_task.py` | 任务基本功能 |
| `test_task_comprehensive.py` | 任务综合测试 |
| `test_vehicle_route.py` | 车辆路径测试 |
| `test_minimal_alns.py` | ALNS基础功能 |
| `test_greedy_to_regret.py` | Greedy vs Regret-2对比 |
| `test_regret-2.py` | Regret-2算子测试 |

### `week2/` - Week 2功能测试（充电站动态优化）

| 文件 | 测试内容 |
|------|---------|
| `test_cs_removal.py` | 充电站移除功能（步骤1.1） |
| `test_cs_insertion.py` | 充电站智能插入（步骤1.2） |
| `test_critical_threshold.py` | 临界值机制（步骤1.3） |
| `test_alns_dynamic_charging.py` | **主测试**：完整ALNS优化 + FR vs PR-Minimal对比 |

### `charging/` - 充电策略测试

| 文件 | 测试内容 |
|------|---------|
| `test_alns_with_charging_strategies.py` | ALNS + 充电策略集成（9种组合） |
| `test_strategy_comparison.py` | 充电策略对比 |
| `test_route_executor.py` | 路径执行器测试 |
| `test_physics_integration.py` | 物理模型集成测试 |

### `cost/` - 成本评估测试

| 文件 | 测试内容 |
|------|---------|
| `test_cost_evaluation.py` | 多目标成本函数测试 |

### `debug/` - 调试工具

| 文件 | 用途 |
|------|------|
| `check_final_feasibility.py` | 检查最终解的电池可行性 |
| `debug_pr_minimal_initial.py` | 调试PR-Minimal初始解生成 |
| `test_charging_calculation.py` | 测试充电策略计算逻辑 |
| `test_battery_feasibility.py` | 测试电池可行性检查 |

---

## 文档结构 (`docs/`)

### 根目录文档

| 文件 | 说明 |
|------|------|
| `ARCHITECTURE.md` | 架构说明与整体能力概览 |
| `MIP_BASELINE_MODEL.md` | MIP基准模型（规则选择与冲突/充电对照） |
| `README.md` | 文档索引与快速导航 |

### `planning/` - 计划和范围文档

| 文件 | 说明 |
|------|------|
| `implementation_plan.md` | 完整实施计划（Week 1-4） |
| `alns_optimization_scope.md` | ALNS优化范围定义 |

### `summaries/` - 工作总结

| 文件 | 说明 |
|------|------|
| `week2_completion_summary.md` | **Week 2完成总结**（包含bug修复） |
| `charging_integration_summary.md` | 充电集成工作总结 |

### `technical/` - 技术说明文档

| 文件 | 说明 |
|------|------|
| `battery_infeasibility_explanation.md` | 电池不可行性解释 |

---

## Week 2 核心改进

### 1. 充电站动态优化
- **Destroy**：可随机移除0-2个充电站（概率30%）
- **Repair**：智能插入充电站（最小绕路成本）
- **自动化**：插入任务后自动检查并修复电池可行性

### 2. Bug修复
- **时间成本**：vehicle_speed从1m/s提升到15m/s（时间从54h降到3.6h）
- **PR-Minimal可行性**：禁用临界值（0%），max_attempts增加到10

### 3. 性能对比（8任务，60kWh电池）

| 指标 | FR策略 | PR-Minimal策略 | 差异 |
|------|--------|---------------|------|
| 充电站数量 | 1个 | 1个 | 相同 |
| 总充电量 | 44.57 kWh | 37.07 kWh | **-16.8%** ✓ |
| 充电成本 | 445.71 | 370.71 | **-75.00** ✓ |
| 总成本 | 207530.66 | 207455.66 | **-75.00** ✓ |

---

## 关键配置参数

### 能量配置
```python
EnergyConfig(
    consumption_rate=0.5,           # 0.5 kWh/km
    charging_rate=50.0/3600,        # 充电速率
    charging_efficiency=0.9,        # 90%充电效率
    critical_battery_threshold=0.0  # 临界值（暂时禁用）
)
```

### 时间配置
```python
TimeConfig(
    vehicle_speed=15.0,             # 15 m/s（54 km/h）
    default_service_time=30.0       # 30秒服务时间
)
```

### 成本配置
```python
CostParameters(
    C_tr=1.0,      # 距离成本权重
    C_ch=10.0,     # 充电成本权重（高权重突出策略差异）
    C_time=1.0,    # 时间成本权重
    C_delay=2.0    # 延迟惩罚权重
)
```

---

## 运行测试

### 快速验证
```bash
# Week 2主测试（推荐）
python tests/week2/test_alns_dynamic_charging.py

# 充电站移除测试
python tests/week2/test_cs_removal.py

# 充电站插入测试
python tests/week2/test_cs_insertion.py
```

### 充电策略对比
```bash
# 9种策略组合测试
python tests/charging/test_alns_with_charging_strategies.py

# 策略对比
python tests/charging/test_strategy_comparison.py
```

### 调试工具
```bash
# 检查解的可行性
python tests/debug/check_final_feasibility.py

# 调试PR-Minimal初始解
python tests/debug/debug_pr_minimal_initial.py
```

---

## 下一步计划

### 仓储场景迭代：取送货节点分离优化
- 允许在pickup和delivery之间插入其他任务
- 提高调度灵活性
- 降低距离成本

### Week 4：时间窗约束
- 引入软时间窗
- 实现延迟惩罚
- 集成到成本函数

### 未来改进
- 临界值机制改进（策略内置处理，5-10%）
- 多车辆路径规划
- 实例测试与性能优化

---

## 代码规范

### 命名约定
- **类名**：大驼峰（`EnergyConfig`, `RouteExecutor`）
- **函数名**：小写下划线（`calculate_distance`, `find_best_station`）
- **常量**：大写下划线（`C_TR`, `C_CH`）

### 注释规范
- **模块级**：文件开头说明功能和数学模型映射
- **函数级**：docstring说明参数、返回值、示例
- **Week标记**：重要修改标注`# Week 2新增`或`# Week 2修复`

### 测试规范
- **文件名**：`test_*.py`
- **结构**：场景创建 → 执行 → 验证 → 输出结果
- **断言**：使用`✓`和`✗`清晰标记

---

## 联系与贡献

**项目状态**：Week 2 已完成 ✅
**下一步**：仓储场景迭代（取送货分离优化）
**Git分支**：`claude/alns-charging-integration-011CUMKhsmT1H6wFBWgf8KxX`

---

**创建时间**：2025-10-23
**版本**：v1.0
**维护者**：R3 Development Team
