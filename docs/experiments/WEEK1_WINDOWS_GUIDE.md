# Week 1 Windows 运行指南

**适用系统**: Windows 10/11 with PowerShell
**前提条件**: Python 3.9+, Git

---

## 🚀 快速开始（Windows）

### Step 1: 测试安装

```powershell
# 在项目根目录（例如 F:\simulation3）

# 设置环境变量（PowerShell）
$env:PYTHONPATH = ".\src;$env:PYTHONPATH"

# 运行测试
python scripts\week1\test_installation.py
```

**或者使用批处理文件**:

```cmd
# 双击运行或命令行执行
scripts\week1\test_installation.bat
```

**预期输出**:
```
✓ PASS: Module Imports
✓ PASS: Q-table Initialization
✓ PASS: Q-learning Agent Integration
✓ PASS: Script Files

Total: 4/4 tests passed
✓ All tests passed! Week 1 is ready to use.
```

---

## 📊 运行实验

### Day 1-3: 基线收集

#### PowerShell 方式

```powershell
# 设置Python路径
$env:PYTHONPATH = ".\src;$env:PYTHONPATH"

# 创建输出目录
New-Item -ItemType Directory -Force -Path results\week1\baseline

# 运行基线收集（30次运行）
foreach ($scenario in @("small", "medium", "large")) {
    foreach ($seed in 2025..2034) {
        python scripts\week1\run_experiment.py `
            --scenario $scenario `
            --init_strategy zero `
            --seed $seed `
            --output "results\week1\baseline\baseline_${scenario}_seed${seed}.json"
    }
}

# 分析结果
python scripts\week1\analyze_baseline.py
```

#### 批处理文件方式（推荐）

```cmd
# 双击运行或命令行执行
scripts\week1\01_baseline_collection.bat

# 完成后运行分析
python scripts\week1\analyze_baseline.py
```

**预期时间**: ~30分钟
**输出**: 30个JSON文件 + baseline_summary.json

---

### Day 4-7: 初始化实验

#### PowerShell 方式

```powershell
# 设置Python路径
$env:PYTHONPATH = ".\src;$env:PYTHONPATH"

# 创建输出目录
New-Item -ItemType Directory -Force -Path results\week1\init_experiments

# 运行完整实验（120次运行）
$strategies = @("zero", "uniform", "action_specific", "state_specific")
$scenarios = @("small", "medium", "large")

foreach ($strategy in $strategies) {
    foreach ($scenario in $scenarios) {
        foreach ($seed in 2025..2034) {
            python scripts\week1\run_experiment.py `
                --scenario $scenario `
                --init_strategy $strategy `
                --seed $seed `
                --output "results\week1\init_experiments\init_${strategy}_${scenario}_seed${seed}.json"
        }
    }
}

# 分析结果
python scripts\week1\analyze_init_strategies.py
```

#### 批处理文件方式（推荐）

```cmd
# 双击运行或命令行执行
scripts\week1\02_init_experiments.bat

# 完成后运行分析
python scripts\week1\analyze_init_strategies.py
```

**预期时间**: ~2小时
**输出**: 120个JSON文件 + 统计报告 + 图表

---

## 🧪 单次测试运行

测试单个实验是否工作：

```powershell
# PowerShell
$env:PYTHONPATH = ".\src;$env:PYTHONPATH"

python scripts\week1\run_experiment.py `
    --scenario small `
    --init_strategy uniform `
    --seed 2025 `
    --output test.json `
    --verbose
```

**检查输出**:
```powershell
# 查看JSON内容（PowerShell）
Get-Content test.json | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

或使用Python:
```powershell
python -c "import json; print(json.dumps(json.load(open('test.json')), indent=2))"
```

---

## 📁 文件路径说明

Windows使用反斜杠 `\` 而不是正斜杠 `/`:

| Linux风格 | Windows风格 |
|-----------|-------------|
| `results/week1/baseline` | `results\week1\baseline` |
| `scripts/week1/run_experiment.py` | `scripts\week1\run_experiment.py` |

Python脚本会自动处理路径，但在批处理文件和PowerShell中需要使用 `\`。

---

## 🔧 故障排查

### 问题1: "No module named 'planner'"

**原因**: PYTHONPATH未设置

**解决方案**:

```powershell
# PowerShell - 临时设置
$env:PYTHONPATH = ".\src;$env:PYTHONPATH"

# 或使用批处理文件（已自动设置）
scripts\week1\test_installation.bat
```

**永久设置（可选）**:
```powershell
# PowerShell（需要管理员权限）
[System.Environment]::SetEnvironmentVariable("PYTHONPATH", "C:\path\to\your\project\src", "User")
```

### 问题2: PowerShell执行策略错误

**错误信息**: "无法加载文件，因为在此系统上禁止运行脚本"

**解决方案**:
```powershell
# 设置执行策略（管理员PowerShell）
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

或直接使用 `.bat` 批处理文件，不会有执行策略问题。

### 问题3: 路径中有空格

如果项目路径包含空格（如 `C:\My Projects\R3`），使用引号：

```powershell
$env:PYTHONPATH = "C:\My Projects\R3\src;$env:PYTHONPATH"
cd "C:\My Projects\R3"
```

### 问题4: Python未找到

**检查Python安装**:
```cmd
python --version
```

应输出 `Python 3.9.x` 或更高版本。

如未安装或版本过低，从 [python.org](https://www.python.org/) 下载安装。

---

## 📊 检查结果

### 验证基线数据

```powershell
# PowerShell - 检查文件数量
(Get-ChildItem results\week1\baseline\*.json).Count
# 应输出: 30

# 查看汇总
Get-Content results\week1\baseline\baseline_summary.json | ConvertFrom-Json | ConvertTo-Json
```

### 验证实验数据

```powershell
# PowerShell - 检查文件数量
(Get-ChildItem results\week1\init_experiments\*.json).Count
# 应输出: 120

# 查看推荐策略
Get-Content results\week1\init_experiments\recommendations.json | ConvertFrom-Json | ConvertTo-Json
```

---

## 🎨 查看可视化结果

生成的PNG图表在:
```
results\week1\init_experiments\init_strategies_comparison.png
```

可以用Windows照片查看器或任何图片查看器打开。

---

## 💡 推荐工作流（Windows）

### 使用批处理文件（最简单）

1. **测试安装**
   ```cmd
   scripts\week1\test_installation.bat
   ```

2. **基线收集**
   ```cmd
   scripts\week1\01_baseline_collection.bat
   python scripts\week1\analyze_baseline.py
   ```

3. **初始化实验**
   ```cmd
   scripts\week1\02_init_experiments.bat
   python scripts\week1\analyze_init_strategies.py
   ```

### 使用PowerShell（更灵活）

在PowerShell中设置一次环境变量，然后运行所有命令：

```powershell
# 在项目根目录打开PowerShell

# 设置Python路径（只需一次）
$env:PYTHONPATH = ".\src;$env:PYTHONPATH"

# 运行实验
scripts\week1\01_baseline_collection.bat
python scripts\week1\analyze_baseline.py

scripts\week1\02_init_experiments.bat
python scripts\week1\analyze_init_strategies.py
```

---

## 📖 相关文档

- **详细测试计划**: `docs\experiments\WEEK1_TEST_PLAN.md`
- **快速参考**: `scripts\week1\README.md`
- **准备就绪**: `docs\experiments\WEEK1_READY.md`

---

## ✅ Windows快速检查清单

- [ ] Python 3.9+ 已安装
- [ ] Git已安装（用于克隆代码）
- [ ] 在项目根目录（例如 `F:\simulation3`）
- [ ] 运行 `scripts\week1\test_installation.bat` 通过
- [ ] 准备好~2.5小时运行完整实验

---

**Windows提示**:
- 使用 `\` 而不是 `/` 表示路径
- 批处理文件 `.bat` 最简单（双击即可）
- PowerShell命令用反引号 `` ` `` 换行
- 遇到问题优先使用批处理文件

**下一步**: 运行 `scripts\week1\test_installation.bat` 验证安装
