# Q-Learning参数调优修复分支

**分支名称**: `claude/fix-qlearning-failures-20251103-011CUhJ2dCiVnBt3HEiNW3oY`

---

## 📋 本次修改内容

### 修改的文件
- `src/config/defaults.py` - Q-learning参数配置

### 调优的参数

| 参数 | 原值 | 新值 | 目的 |
|:-----|-----:|-----:|:-----|
| **alpha** (学习率) | 0.35 | **0.1** | 避免Q值过度更新，提高稳定性 |
| **epsilon_min** (最小探索率) | 0.01 | **0.1** | 保持10%探索，防止过早收敛 |
| **stagnation_ratio** | 0.16 | **0.25** | 延迟stuck状态，给更多探索时间 |
| **deep_stagnation_ratio** | 0.4 | **0.45** | 与stagnation_ratio成比例调整 |

---

## 🎯 预期效果

### 目标失败案例
1. **Seed 2027 Medium**: 17.01% → 目标 >35%
2. **Seed 2031 Large**: 8.34% → 目标 >25%
3. **Seed 2029 全规模**: 整体失败 → 改善

### 改进机制

#### 1. 降低学习率 (alpha: 0.35→0.1)
- **问题**: 高学习率导致Q值剧烈波动
- **效果**: 更平滑的学习曲线，避免过度反应

#### 2. 增加最小探索率 (epsilon_min: 0.01→0.1)
- **问题**: 探索率降到1%后，几乎完全exploitation
- **效果**: 始终保持10%随机探索，避免算子严重失衡

#### 3. 延迟stuck状态触发 (stagnation_ratio: 0.16→0.25)
- **问题**: Medium规模44次迭代时，7次不改进就进入stuck
- **效果**: 现在需要11次不改进才进入stuck，给Q-learning更多学习时间

---

## 🚀 如何使用此分支

### 第1步：切换到修复分支

```bash
# 在您的项目目录
cd F:\simulation3

# 拉取最新分支
git fetch origin

# 切换到修复分支
git checkout claude/fix-qlearning-failures-20251103-011CUhJ2dCiVnBt3HEiNW3oY

# 确认切换成功
git branch
```

### 第2步：验证单个失败案例

```bash
# 测试之前最严重的失败案例
python scripts/generate_alns_visualization.py --seed 2027
```

**查看输出**，检查Medium规模的Q-learning结果：
- 原来: 17.01%
- 期待: >35%

**成功标准**:
- ✅ Seed 2027 Medium提升到35%+
- ✅ 其他规模不受负面影响

### 第3步：如果验证成功，运行完整10-seed测试

#### PowerShell批量运行（推荐）
```powershell
# 运行所有seeds
for ($seed=2025; $seed -le 2034; $seed++) {
    Write-Host "Running seed $seed..."
    python scripts/generate_alns_visualization.py --seed $seed
}
```

#### 或手动逐个运行
```bash
python scripts/generate_alns_visualization.py --seed 2025
python scripts/generate_alns_visualization.py --seed 2026
python scripts/generate_alns_visualization.py --seed 2027
# ... 继续到2034
```

### 第4步：收集结果并重新分析

运行完所有seeds后，收集JSON数据，然后：

```bash
python scripts/analyze_10seeds_results.py
```

**检查关键指标**:
- ✅ t统计量 > 2.045 (p<0.05)
- ✅ 平均差异 > 5%
- ✅ 失败案例数量 < 3
- ✅ Large规模CV < 25%

---

## 📊 预期的统计改进

| 指标 | 修复前 | 预期修复后 | 目标 |
|:-----|-------:|-----------:|:----:|
| **平均差异** | +3.80% | **+6~8%** | >5% |
| **t统计量** | 1.516 | **>2.2** | >2.045 |
| **Q-learning胜率** | 60% | **70%+** | >65% |
| **灾难级失败** | 3个 | **0个** | 0 |
| **Large CV** | 31.23% | **<23%** | <25% |

---

## 🔄 如果效果不理想怎么办？

### 情况A：Seed 2027仍然<30%
**原因**: 参数调整不够激进
**解决**: 进一步增加epsilon_min到0.15

### 情况B：整体性能下降
**原因**: 参数调整过度保守
**解决**: alpha可能需要0.15而不是0.1

### 情况C：某些seeds变好但某些变差
**原因**: 参数对不同seeds敏感性不同
**解决**: 这是正常的，只要整体统计显著即可

---

## 📝 如果需要微调参数

修改文件: `src/config/defaults.py` 第180-203行

```python
@dataclass(frozen=True)
class QLearningParams:
    alpha: float = 0.1  # 可调整：0.05-0.2
    epsilon_min: float = 0.1  # 可调整：0.05-0.15
    stagnation_ratio: float = 0.25  # 可调整：0.2-0.3
    deep_stagnation_ratio: float = 0.45  # 可调整：0.35-0.5
```

---

## ⏰ 预计时间线

| 步骤 | 时间 | 说明 |
|:-----|:----:|:-----|
| 第1步：切换分支 | 1分钟 | git操作 |
| 第2步：验证单个case | 5分钟 | 运行seed 2027 |
| 第3步：完整10-seed | 3-4小时 | 运行10个seeds |
| 第4步：统计分析 | 5分钟 | 重新计算显著性 |

**今天可完成**: 第1-2步（验证修复有效性）
**明天可完成**: 第3-4步（完整测试）

---

## ✅ 成功检查清单

验证修复成功前，确保：

- [ ] Seed 2027 Medium从17.01%提升到>35%
- [ ] Seed 2031 Large从8.34%提升到>25%
- [ ] Seed 2029整体改善
- [ ] 无新的失败案例出现
- [ ] t统计量>2.045
- [ ] 平均差异>5%

---

## 🆘 需要帮助？

如果遇到问题：
1. 分享Seed 2027的新结果数值
2. 告诉我是否有改善
3. 如果没改善，我们可以进一步调整参数

---

**祝实验顺利！** 🚀

这次参数调优应该能显著改善失败案例！
