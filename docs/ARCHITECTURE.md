# R3 框架架构说明

## 📋 核心功能

本框架实现**带充电站的电动AMR路径规划**，使用**ALNS元启发式算法**进行多目标优化。

### 优化目标
最小化：`距离成本 + 充电成本 + 时间成本 + 延迟惩罚`

### 核心约束
- ✅ **容量约束**：载重不超过车辆容量
- ✅ **时间窗约束**：硬时间窗（拒绝）/ 软时间窗（惩罚）
- ✅ **能量约束**：电池不耗尽
- ✅ **顺序约束**：Pickup先于Delivery
- ✅ **充电站约束**：动态插入/移除

---

## 🏗️ 模块结构

```
src/
├── core/               # 核心数据结构
│   ├── node.py        # 节点定义（Depot/Task/Charging）
│   ├── task.py        # 任务和任务池
│   ├── route.py       # 路径表示和可行性检查
│   └── vehicle.py     # 车辆属性
│
├── physics/           # 物理模型
│   ├── distance.py    # 距离计算
│   ├── energy.py      # 能耗和充电模型
│   └── time.py        # 时间窗和延迟计算
│
├── planner/           # 优化算法
│   └── alns.py        # ALNS核心算法
│
└── strategy/          # 充电策略
    └── charging_strategies.py  # FR/PR-Fixed/PR-Minimal
```

---

## 🔧 ALNS算法实现

### Destroy算子
- **random_removal**: 随机移除q个任务
- **partial_removal** (Week 3): 只移除delivery节点，保留pickup

### Repair算子
- **greedy_insertion**: 贪心插入（最小成本）
- **regret2_insertion**: Regret-2插入（最大遗憾值）
- **random_insertion**: 随机插入

### Local Search
- **pair_exchange** (Week 3): 交换两个任务位置

### 约束检查（Repair阶段）
```python
for 每个插入位置:
    ① 容量可行性检查           → 不可行则跳过
    ② 时间窗可行性检查         → 硬约束违反则跳过
    ③ 能量可行性检查           → 不可行则插入充电站
    ④ 计算成本（距离+充电+延迟）
    ⑤ 选择最优位置
```

---

## ⚡ 充电策略

### 1. Full Recharge (FR)
- **策略**: 每次充满100%
- **优点**: 充电次数少
- **缺点**: 充电时间长
- **适用**: 充电站稀疏场景

### 2. Partial Recharge Fixed (PR-Fixed)
- **策略**: 充到固定百分比（如50%）
- **优点**: 充电时间固定且短
- **缺点**: 需要更频繁充电
- **适用**: 充电站密集场景

### 3. Partial Recharge Minimal (PR-Minimal)
- **策略**: 只充够用的电量 + 安全余量
- **优点**: 充电时间最短
- **缺点**: 需要准确能量预测
- **适用**: 已知路径的静态规划

---

## ⏰ 时间窗约束

### 硬时间窗 (HARD)
```python
TimeWindow(earliest=100, latest=200, window_type=TimeWindowType.HARD)
```
- **违反**: 立即拒绝该插入位置
- **成本**: 无穷大（不可行）
- **适用**: 医疗紧急配送、法律截止时间

### 软时间窗 (SOFT)
```python
TimeWindow(earliest=100, latest=200, window_type=TimeWindowType.SOFT)
```
- **违反**: 允许但增加延迟成本
- **成本**: `延迟时间 × C_delay`
- **适用**: 普通快递、非紧急任务

---

## 📊 测试规模

| 规模 | 任务数 | 充电站 | 测试文件 |
|-----|-------|-------|---------|
| **小** | 5-10 | 0-1 | `test_week3_small_scale.py` |
| **中** | 20-30 | 2 | `test_week3_medium_scale.py` |
| **大** | 50-100 | 3-5 | `test_week3_large_scale.py` |

---

## 🎯 使用示例

### 基础示例
```python
from planner.alns import MinimalALNS, CostParameters
from strategy.charging_strategies import PartialRechargeMinimalStrategy

# 创建ALNS优化器
alns = MinimalALNS(
    distance_matrix=distance_matrix,
    task_pool=task_pool,
    repair_mode='regret2',
    cost_params=CostParameters(
        C_tr=1.0,      # 距离成本
        C_ch=0.6,      # 充电成本
        C_delay=2.0    # 延迟惩罚
    ),
    charging_strategy=PartialRechargeMinimalStrategy(safety_margin=0.1)
)
alns.vehicle = vehicle
alns.energy_config = energy_config

# 优化
initial_route = ... # 初始解
optimized_route = alns.optimize(initial_route, max_iterations=100)
```

### 添加时间窗
```python
from physics.time import TimeWindow, TimeWindowType

pickup, delivery = create_task_node_pair(
    task_id=1,
    pickup_id=1,
    delivery_id=2,
    pickup_coords=(10, 0),
    delivery_coords=(10, 10),
    demand=20.0,
    # 硬时间窗
    pickup_time_window=TimeWindow(100, 200, TimeWindowType.HARD),
    # 软时间窗
    delivery_time_window=TimeWindow(150, 250, TimeWindowType.SOFT)
)
```

---

## 📈 成本函数

```python
总成本 = 距离成本 + 充电成本 + 时间成本 + 延迟成本 + 惩罚项

其中:
  距离成本 = Σ距离 × C_tr
  充电成本 = Σ充电量 × C_ch
  时间成本 = 总时间 × C_time
  延迟成本 = Σ延迟 × C_delay  (时间窗违反)
  惩罚项 = 任务丢失惩罚 + 不可行解惩罚 + 电池耗尽惩罚
```

---

## ✅ 已实现功能

### Week 1
- ✅ 基础ALNS框架（Destroy + Repair）
- ✅ 多目标成本函数
- ✅ Greedy/Regret-2插入

### Week 2
- ✅ 充电站动态优化
- ✅ 三种充电策略（FR/PR-Fixed/PR-Minimal）
- ✅ 能量约束检查
- ✅ 充电站插入/移除算子

### Week 3
- ✅ Pickup/Delivery分离优化
- ✅ Partial removal算子
- ✅ Pair exchange算子
- ✅ 容量约束检查
- ✅ 时间窗约束集成

---

## 🚧 未实现/未启用

1. **充电临界值机制** (Week 4-5建议)
   - 端口已预留：`EnergyConfig.critical_battery_threshold`
   - 当前设置为0（禁用）
   - 建议在ALNS稳定后启用

2. **多车辆优化** (扩展功能)

3. **动态任务到达** (扩展功能)

---

## 📁 关键文件

### 核心算法
- `src/planner/alns.py` - ALNS主算法（1200行）

### 测试
- `tests/week3/test_integrated_features.py` - **综合功能测试**（推荐）
- `tests/week3/test_week3_comprehensive.py` - Week 3算子测试
- `tests/week3/test_week3_small_scale.py` - 小规模场景
- `tests/week3/test_week3_medium_scale.py` - 中规模场景
- `tests/week3/test_week3_large_scale.py` - 大规模场景

### 充电策略测试
- `tests/charging/test_strategy_comparison.py` - 策略对比
- `tests/charging/test_alns_with_charging_strategies.py` - ALNS+策略集成

---

## 🔬 运行测试

```bash
# 综合功能测试（推荐）
python tests/week3/test_integrated_features.py

# Week 3完整测试套件
python tests/week3/test_week3_comprehensive.py
python tests/week3/test_week3_small_scale.py
python tests/week3/test_week3_medium_scale.py
python tests/week3/test_week3_large_scale.py  # 注意：需要10-30分钟

# 充电策略对比
python tests/charging/test_strategy_comparison.py
```

---

## 📞 扩展开发

如需添加新功能，建议顺序：

1. **Week 4-5**: 启用充电临界值机制
2. **Week 6**: 性能优化（降低Regret-2复杂度）
3. **Week 7**: 多车辆扩展
4. **Week 8**: 动态任务

---

*最后更新：Week 3完成*
*版本：1.0*
