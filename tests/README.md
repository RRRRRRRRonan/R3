# 测试目录说明

本目录包含 R3 框架的回归与策略测试，聚焦于当前维护的核心流程。

## 目录结构

```
tests/
├── warehouse_regression/ # 仓储回归阶段场景回归
├── charging/     # 充电策略对比与验证
└── optimization/ # 小/中/大规模 ALNS 架构回归
```

## 仓储回归场景（`tests/warehouse_regression/`）

> 注：早期项目在第三周迭代中引入该批仓储场景回归测试，因此最初以阶段代号命名；现统一改为 "warehouse_regression" 以突显其仓储回归含义。

| 文件 | 侧重点 |
|------|--------|
| `test_integrated_features.py` | 核心流程一键回归，验证 Destroy/Repair、时间窗与能量约束协同 |
| `test_regression_comprehensive.py` | 算子与多目标成本的细粒度断言 |
| `test_regression_small_scale.py` | 5-10 个任务的快速健康检查 |
| `test_regression_medium_scale.py` | 20-30 个任务的典型部署验证 |
| `test_regression_large_scale.py` | 50-100 个任务的压力测试 |
| `test_simple_capacity_check.py` | 载荷与能量可行性最小场景 |

### 推荐执行顺序

```bash
python tests/warehouse_regression/test_integrated_features.py
python tests/warehouse_regression/test_regression_small_scale.py
python tests/warehouse_regression/test_regression_medium_scale.py
```

## 充电策略验证（`tests/charging/`）

- `test_strategy_comparison.py` — 对比 FR、PR-Fixed、PR-Minimal 三种策略在同一场景下的成本与充电次数。

### 快速运行

```bash
python tests/charging/test_strategy_comparison.py
```

## ALNS 优化规模测试（`tests/optimization/`）

| 文件 | 场景规模 | 说明 |
|------|----------|------|
| `test_alns_optimization_small.py` | 10 任务 / 1 站 | 烟雾测试，三种充电策略均需优于贪心基线 |
| `test_alns_optimization_medium.py` | 30 任务 / 2 站 | 周期性回归，验证主展示场景的成本改进 |
| `test_alns_optimization_large.py` | 50 任务 / 3 站 | 压力测试，覆盖自适应算子在大规模下的表现 |

每个测试都会生成固定随机场景，先通过贪心插入获得基线成本，再运行 ALNS 若干迭代，确保三种充电策略的优化解均优于基线。平均改进幅度也会被断言，以便快速捕获算法退化。

运行示例：

```bash
python -m pytest tests/optimization/test_alns_optimization_small.py -q
```

> ⚠️ **命令不可用？**
>
> - 若提示 `pytest` 不是可执行文件，说明当前环境尚未安装该工具，请执行 `python -m pip install pytest`（或使用项目统一的依赖安装命令）。
> - Windows PowerShell 默认不会解析 `pytest` 入口，使用上面展示的 `python -m pytest …` 形式即可跨平台运行。

大规模测试可能耗时数分钟，可在调试阶段临时降低测试文件内的迭代次数。

## 执行提示

1. 在仓库根目录运行测试，Python 会自动识别 `src/` 模块。
2. 大规模测试（warehouse regression large、optimization large）耗时较长，建议在 CI 中标记为慢测试或单独运行。
3. 若需新增测试，用以上结构为模板，保持命名清晰与输出可读。
