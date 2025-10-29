# R3 - 电动AMR路径规划与充电优化框架

## 📖 快速导航

### 🏗️ [架构文档 (ARCHITECTURE.md)](./ARCHITECTURE.md)
**← 请查看这里获取完整的架构说明和使用指南**

---

## 🎯 项目简介

R3是一个完整的**电动自主移动机器人(AMR)路径规划框架**，使用**ALNS元启发式算法**进行多目标优化，支持：

- ✅ **动态充电站优化**（局部充电 vs 完全充电）
- ✅ **时间窗约束**（硬约束 + 软约束）
- ✅ **容量和能量约束**
- ✅ **Pickup/Delivery分离优化**
- ✅ **小中大规模场景测试**（5-100任务）

---

## 🚀 快速开始

### 运行综合测试
```bash
# 推荐：一键验证核心流程
python tests/week3/test_integrated_features.py

# 依据规模拆分验证
python tests/week3/test_week3_small_scale.py
python tests/week3/test_week3_medium_scale.py
python tests/week3/test_week3_large_scale.py
```

### 基础使用示例
```python
from planner.alns import MinimalALNS, CostParameters
from strategy.charging_strategies import PartialRechargeMinimalStrategy

# 创建优化器
alns = MinimalALNS(
    distance_matrix=distance_matrix,
    task_pool=task_pool,
    repair_mode='regret2',
    cost_params=CostParameters(
        C_tr=1.0,      # 距离成本
        C_ch=0.6,      # 充电成本
        C_delay=2.0    # 时间窗延迟惩罚
    ),
    charging_strategy=PartialRechargeMinimalStrategy(safety_margin=0.1)
)
alns.vehicle = vehicle
alns.energy_config = energy_config

# 优化
optimized_route = alns.optimize(initial_route, max_iterations=100)
```

---

## 📁 项目结构

```
R3/
├── src/                    # 源代码
│   ├── core/              # 核心数据结构
│   ├── physics/           # 物理模型（距离/能量/时间）
│   ├── planner/           # ALNS算法
│   └── strategy/          # 充电策略
│
├── tests/                  # 测试套件
│   ├── week3/             # 核心流程测试
│   │   ├── test_integrated_features.py      ★ 综合测试
│   │   ├── test_week3_comprehensive.py
│   │   ├── test_week3_small_scale.py
│   │   ├── test_week3_medium_scale.py
│   │   └── test_week3_large_scale.py
│   └── charging/          # 充电策略验证
│       └── test_strategy_comparison.py
│
└── docs/
    ├── README.md          # 本文件
    └── ARCHITECTURE.md    # 架构详细说明 ★核心文档
```

---

## 🔧 核心功能

### 1. ALNS元启发式算法
- **Destroy算子**: random_removal, partial_removal
- **Repair算子**: greedy_insertion, regret2_insertion
- **Local Search**: pair_exchange

### 1.1 Matheuristic升级
- **MatheuristicALNS**: 在ALNS主循环之上加入精英解记忆和分段重优化，结合仿MILP的段内重构提升能量约束场景下的收敛质量。
- **设计说明**: 详见 [docs/summaries/matheuristic_alns.md](./summaries/matheuristic_alns.md)。

### 2. 充电策略
- **FR**: Full Recharge（完全充电）
- **PR-Fixed**: Partial Recharge Fixed（固定比例局部充电）
- **PR-Minimal**: Partial Recharge Minimal（最小充电）

### 3. 约束处理
- **容量约束**: 载重不超过车辆容量
- **时间窗约束**: 硬时间窗（拒绝）/ 软时间窗（惩罚）
- **能量约束**: 电池不耗尽，动态插入充电站
- **顺序约束**: Pickup必须先于Delivery

### 4. 多目标优化
```
最小化 = 距离成本 + 充电成本 + 时间成本 + 延迟惩罚
```

---

## 📊 测试覆盖

| 测试 | 规模 | 说明 |
|------|------|------|
| `test_week3_small_scale.py` | 小 | 5-10 个任务的快速健康检查 |
| `test_week3_medium_scale.py` | 中 | 20-30 个任务的典型部署 |
| `test_week3_large_scale.py` | 大 | 50-100 个任务的压力测试 |
| `test_integrated_features.py` | 综合 | 完整流程与约束联调 |

---

## 📖 详细文档

**完整的架构说明、API文档、使用示例和扩展开发指南，请查看：**

### 👉 [ARCHITECTURE.md](./ARCHITECTURE.md)

---

## 🗓️ 里程碑进度

| 阶段 | 重点 | 状态 |
|------|------|------|
| Phase 1 | 搭建ALNS核心与Destroy/Repair算子 | ✅ 完成 |
| Phase 2 | 集成充电策略与能量约束 | ✅ 完成 |
| Phase 3 | 时间窗、容量、多目标成本联调 | ✅ 完成 |
| 下一步 | 充电临界值机制与多车扩展 | 🚧 规划中 |

---

## 📞 问题反馈

如有问题或建议，请参考 [ARCHITECTURE.md](./ARCHITECTURE.md) 中的详细说明。

---

*版本: 1.0 (Week 3完成)*
*最后更新: 2024*
