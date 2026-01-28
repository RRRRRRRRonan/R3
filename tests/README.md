# 测试目录说明

本目录包含 R3 框架的回归与策略测试，聚焦于当前维护的核心流程。

## 目录结构

```
tests/
├── warehouse_regression/ # 仓储回归阶段场景回归 (MinimalALNS)
├── charging/             # 充电策略对比与验证
└── optimization/         # ALNS 优化规模回归 (Minimal / Matheuristic / Q-learning)
```

## 仓储回归场景（`tests/warehouse_regression/`）

> 注：早期项目在第三周迭代中引入该批仓储场景回归测试，因此最初以阶段代号命名；现统一改为 "warehouse_regression" 以突显其仓储回归
含义。

| 文件 | 侧重点 |
|------|--------|
| `test_integrated_features.py` | 核心流程一键回归，验证 Destroy/Repair、时间窗与能量约束协同 |
| `test_regression_comprehensive.py` | 算子与多目标成本的细粒度断言 |
| `test_regression_small_scale.py` | 5-10 个任务的快速健康检查 |
| `test_regression_medium_scale.py` | 20-30 个任务的典型部署验证 |
| `test_regression_large_scale.py` | 50-100 个任务的压力测试 |
| `test_simple_capacity_check.py` | 载荷与能量可行性最小场景 |
| `test_execution_layer_regression.py` | 交通预约/冲突等待的执行层回归 |

### 推荐执行顺序

```bash
python3 tests/warehouse_regression/test_integrated_features.py
python3 tests/warehouse_regression/test_regression_small_scale.py
python3 tests/warehouse_regression/test_regression_medium_scale.py
```

## 充电策略验证（`tests/charging/`）

- `test_strategy_comparison.py` — 对比 FR、PR-Fixed、PR-Minimal 三种策略在同一场景下的成本与充电次数。

### 快速运行

```bash
python3 tests/charging/test_strategy_comparison.py
```

## ALNS 优化规模测试（`tests/optimization/`）

```
optimization/
├── common.py
├── minimal/
│   ├── test_minimal_small.py
│   ├── test_minimal_medium.py
│   └── test_minimal_large.py
├── matheuristic/
│   ├── test_matheuristic_small.py
│   ├── test_matheuristic_medium.py
│   └── test_matheuristic_large.py
├── presets.py
├── q_learning/
│   ├── test_q_learning_small.py
│   ├── test_q_learning_medium.py
│   ├── test_q_learning_large.py
│   └── utils.py
└── test_alns_matheuristic.py
```

### Minimal ALNS（大/中/小规模）

| 文件 | 场景规模 / 迭代 | 说明 |
|------|-----------------|------|
| `minimal/test_minimal_small.py` | 10 任务 / 16 迭代 | 验证基础版 ALNS 在统一小场景下优于贪心基线 |
| `minimal/test_minimal_medium.py` | 24 任务 / 32 迭代 | 中规模统一场景的回归，确保成本改进 |
| `minimal/test_minimal_large.py` | 30 任务 / 32 迭代 | 大规模统一场景，确认 Q-learning 模式下的稳健性 |

### Matheuristic ALNS（大/中/小规模）

| 文件 | 场景规模 / 迭代 | 说明 |
|------|-----------------|------|
| `matheuristic/test_matheuristic_small.py` | 10 任务 / 28 迭代 | 烟雾测试，三种充电策略均需优于贪心基线 |
| `matheuristic/test_matheuristic_medium.py` | 24 任务 / 44 迭代 | 周期性回归，验证统一中规模场景的成本改进 |
| `matheuristic/test_matheuristic_large.py` | 30 任务 / 44 迭代 | 压力测试，覆盖自适应算子在大规模下的表现 |

### Matheuristic + Q-learning（大/中/小规模）

| 文件 | 场景规模 / 迭代 | 说明 |
|------|-----------------|------|
| `q_learning/test_q_learning_small.py` | 10 任务 / 18 迭代 | 验证 Q-learning 算子在小规模下的学习与成本改进 |
| `q_learning/test_q_learning_medium.py` | 24 任务 / 36 迭代 | 确认 Q-learning 在中规模下仍能持续获得正奖励 |
| `q_learning/test_q_learning_large.py` | 30 任务 / 30 迭代 | 大规模压力测试，确保 Q-learning 统计和成本表现稳定 |

各测试均使用确定性的场景生成器，参数统一由 `optimization/presets.py` 管理：`minimal/` 套件验证基础 ALNS，`matheuristic/` 套件覆盖三种充电策略，而 `q_learning/` 套件专注于单一配置以验证强化学习信号（Q 值更新、epsilon 衰减、算子使用次数）。

运行示例：

```bash
python3 -m pytest tests/optimization/matheuristic/test_matheuristic_small.py -q
python3 -m pytest tests/optimization/q_learning/test_q_learning_small.py -q
```

> ⚠️ **命令不可用？**
>
> - 若提示 `pytest` 不是可执行文件，说明当前环境尚未安装该工具，请执行 `python3 -m pip install pytest`（或使用项目统一的依赖安装命令）。
> - Windows PowerShell 默认不会解析 `pytest` 入口，使用上面展示的 `python -m pytest …` 形式即可跨平台运行。

大规模测试可能耗时数分钟，可在调试阶段临时调整 `presets.py` 中的迭代次数以保持三套回归同步。

## 执行提示

1. 在仓库根目录运行测试，Python 会自动识别 `src/` 模块。
2. 大规模测试（warehouse regression large、optimization large）耗时较长，建议在 CI 中标记为慢测试或单独运行。
3. 若需新增测试，用以上结构为模板，保持命名清晰与输出可读。
