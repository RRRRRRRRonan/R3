# Large Scale 卡住问题 - 完整诊断报告

**问题**: Large scale seed=2026运行44次迭代时卡住超过10分钟

**日期**: 2025-11-01

---

## 📊 **症状总结**

| 测试 | 迭代次数 | 结果 | 时间 |
|-----|---------|------|------|
| Small诊断 | 2次 | ✅ 成功 | 11.4秒 |
| Large正式运行 | 44次 | ❌ 卡住 | >10分钟无响应 |

**预期时间**: 44次迭代应该只需~4.2分钟（根据诊断估算）

**实际情况**: 卡在"Starting Q-learning..."后超过10分钟

---

## 🔍 **问题分析**

### **已排除的问题**

✅ **不是import问题** - 诊断脚本能正常导入
✅ **不是ALNS基础问题** - 2次迭代能正常运行
✅ **不是输出缓冲** - 开头能正常显示

### **锁定的问题范围**

❌ **问题在Large scale的44次完整迭代中**

可能的具体原因：

**1. LP求解器超时/卡住** 🔴 最可能
```
Large scale: 30个任务
→ LP问题规模更大
→ 某次迭代的LP求解可能陷入困境
→ 超时设置可能不够（当前0.3秒）
```

**2. 状态转换触发昂贵操作**
```
后期迭代可能进入stuck/deep_stuck状态
→ 触发elite pool intensification
→ 触发segment optimization
→ 这些操作在Large scale可能非常慢
```

**3. 特定迭代的死循环**
```
某个特定的场景+种子组合
→ 触发了算法中的边界情况
→ 进入无限循环或极长的计算
```

**4. Windows内存问题**
```
Large scale搜索树更大
→ 内存占用高
→ 可能触发内存交换（swap）
→ 导致极慢但不是真正卡死
```

---

## 🛠️ **诊断工具**

我创建了两个工具来定位具体问题：

### **工具1: progressive_diagnostic.py** - 渐进式测试

**作用**: 逐步增加迭代次数，找出问题出现在哪个区间

**使用**:
```bash
python scripts/progressive_diagnostic.py
```

**测试流程**:
```
Test 1: Large scale, 2次迭代  → 预计20秒
Test 2: Large scale, 5次迭代  → 预计50秒
Test 3: Large scale, 10次迭代 → 预计100秒
Test 4: Large scale, 20次迭代 → 预计3-4分钟
```

**判断规则**:
- 如果Test 1-2通过，Test 3卡住 → 问题在第3-10次迭代
- 如果Test 1-3通过，Test 4卡住 → 问题在第11-20次迭代
- 如果全部通过 → 问题在第21-44次迭代

---

### **工具2: test_verbose_alns.py** - 详细输出模式

**作用**: 开启ALNS的verbose模式，看到每次迭代的详细输出

**使用**:
```bash
python scripts/test_verbose_alns.py
```

**输出示例**:
```
Iteration 1/10
  Destroy: random_removal, Repair: greedy
  Cost: 60000 → 58000 (improved)
  Temperature: 1000

Iteration 2/10
  Destroy: partial_removal, Repair: lp
  LP solving... (this might be slow)
  Cost: 58000 → 57500 (improved)

Iteration 3/10
  ... (如果卡在这里，就能看出来)
```

**优点**:
- 能准确看到卡在第几次迭代
- 能看到当时使用的算子
- 能看到是否在LP求解时卡住

---

## 🚀 **立即行动方案**

### **Step 1: 停止卡住的进程**

```bash
# 在PowerShell中按 Ctrl+C
# 或者找到进程并杀掉
```

### **Step 2: 运行渐进式诊断**

```bash
python scripts/progressive_diagnostic.py
```

**预期时间**: 4-8分钟（如果不卡的话）

**观察点**:
- 哪个Test开始卡住？
- 在那之前的Test用了多长时间？

### **Step 3: 根据结果采取行动**

**情况A: 渐进式测试全部通过**
```
说明: 问题在21-44次迭代中

下一步: 运行verbose测试
python scripts/test_verbose_alns.py

观察: 在verbose模式下也会卡吗？
      能看到卡在第几次迭代吗？
```

**情况B: 某个Test卡住**
```
说明: 问题出现在那个迭代区间

记录:
- 卡在哪个Test？（例如Test 3）
- 之前的Test用了多长时间？
- CPU使用率是否是100%？（任务管理器查看）
```

**情况C: 第一个Test(2次迭代)就卡住**
```
说明: 与之前的诊断结果矛盾

可能:
- 进程冲突
- 内存不足
- 其他系统问题

尝试: 重启电脑后再测试
```

---

## 💡 **临时解决方案**

如果确实无法运行44次迭代，可以：

### **方案A: 减少迭代次数**

修改代码临时使用20次迭代（如果20次能通过）：

```python
# 在 tests/optimization/presets.py
"large": ScalePreset(
    scenario_overrides={"num_tasks": 30, "num_charging": 3, "seed": 17},
    iterations=IterationPreset(
        minimal=32,
        matheuristic=20,  # 从44改为20
        q_learning=20     # 从44改为20
    ),
)
```

**优点**: 能快速验证算法工作
**缺点**: 结果不完整，性能可能偏低

### **方案B: 禁用LP求解器**

如果是LP导致的，可以临时禁用：

```python
# 在 tests/optimization/q_learning/utils.py
# 修改 LPRepairParams
lp_repair=LPRepairParams(
    time_limit_s=0.1,  # 从0.3改为0.1（更短超时）
    # 或者干脆设为0禁用
)
```

### **方案C: 使用Small/Medium scale**

先在Small和Medium上验证seed=2026的性能：

```bash
python scripts/test_single_seed.py small 2026
python scripts/test_single_seed.py medium 2026
```

看看seed=2026在其他规模上的表现。

---

## 🔬 **深度诊断（如果上述方法都失败）**

### **检查系统资源**

**内存**:
```bash
# 任务管理器 → 性能 → 内存
# 查看是否接近满载
```

**CPU**:
```bash
# 任务管理器 → 性能 → CPU
# 如果是100%: 在计算（虽然慢但没死）
# 如果是0%: 可能死锁了
```

**磁盘**:
```bash
# 任务管理器 → 性能 → 磁盘
# 如果磁盘使用率高: 可能在内存交换
```

### **Python profiling**

如果想看具体卡在哪个函数：

```bash
# 使用cProfile
python -m cProfile -o profile.stats scripts/test_single_seed.py large 2026

# Ctrl+C停止后，分析结果
python -c "import pstats; p=pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"
```

---

## 📋 **需要您提供的信息**

运行完诊断后，请告诉我：

1. **渐进式测试结果**:
   - Test 1 (2次): 通过/卡住？
   - Test 2 (5次): 通过/卡住？
   - Test 3 (10次): 通过/卡住？
   - Test 4 (20次): 通过/卡住？

2. **如果某个Test卡住**:
   - 卡在哪个Test？
   - 卡住时CPU使用率？
   - 等了多久？

3. **系统信息**:
   - 内存总量和已用量？
   - Python版本？
   - Windows版本？

4. **Verbose测试（如果能运行）**:
   - 能看到迭代输出吗？
   - 卡在第几次迭代？
   - 最后一次输出是什么？

---

## ✅ **预期结果**

**最好的情况**:
- 渐进式测试全通过
- Verbose测试能看到详细输出
- 找到具体卡在哪一步

**可接受的情况**:
- 渐进式某个Test卡住
- 至少知道问题范围（例如10-20次迭代）
- 可以针对性修复

**最坏的情况**:
- Test 1就卡住
- 说明环境有问题
- 需要检查系统配置

---

## 🎯 **下一步计划**

根据诊断结果：

**如果找到了具体卡点**:
1. 分析那一步在做什么
2. 针对性优化（调整超时、禁用某功能等）
3. 重新测试

**如果是LP求解器问题**:
1. 调整LP超时时间（0.3s → 0.1s或禁用）
2. 或者减少LP使用频率
3. 使用更简单的repair算子

**如果是系统资源问题**:
1. 关闭其他程序释放内存
2. 减少迭代次数
3. 或者在更强大的机器上运行

---

**现在请运行**: `python scripts/progressive_diagnostic.py`

然后告诉我结果！🔍
