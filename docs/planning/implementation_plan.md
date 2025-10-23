# ALNS充电站优化 - 实现计划

## 用户需求总结

### 核心需求
1. **充电站动态优化**：ALNS应该优化充电站的位置和数量，只在需要时插入
2. **充电临界值**：设置电池阈值，低于某个百分比必须充电
3. **取送货分离优化**：取货和送货可以分开，不必严格先取后送
4. **时间窗约束**：考虑取货和送货的时间窗

---

## 实现目标分解

### 目标1：充电站动态插入和移除（高优先级）
**问题**：当前充电站在初始解中固定，无法优化

**目标**：
- ALNS可以动态决定在哪里、何时插入充电站
- 可以移除不必要的充电站
- 不同充电策略产生不同的充电站布局

**预期效果**：
- FR策略：少量充电站（3个），每次充满，距离更短
- PR-Minimal策略：适量充电站（4-5个），精确充电，总成本更低

---

### 目标2：充电临界值机制（高优先级）
**问题**：当前只检查电量是否<0，可能过于激进

**目标**：
- 设置充电临界值（例如20%）
- 电量低于临界值时必须尽快充电
- 在成本评估中体现"低电量风险"

**预期效果**：
- 路径更安全（不会到最后一刻才充电）
- 更符合实际运营（司机不会等到完全没电）

---

### 目标3：取送货分离优化（中优先级）
**问题**：当前pickup和delivery必须连续（先取后送）

**目标**：
- 允许pickup和delivery之间插入其他节点
- 可以先集中取货，再集中送货
- 仍然保证每个任务的pickup在delivery之前

**预期效果**：
- 更灵活的路径规划
- 可能减少总距离（例如顺路取货）

---

### 目标4：时间窗约束集成（低优先级）
**问题**：当前没有考虑时间窗

**目标**：
- 取货和送货都有时间窗 [earliest, latest]
- 路径必须满足时间窗约束
- 违反时间窗的路径被惩罚

**预期效果**：
- 更贴近实际场景
- 时间成本更重要

---

## 实现步骤（按优先级排序）

### 第一步：充电站动态优化（Week 2 关键功能）

#### 1.1 修改Destroy操作
**任务**：让random_removal可以移除充电站

**实现**：
```python
def random_removal(self, route: Route, q: int = 2, remove_cs: bool = True):
    """
    移除任务，同时可选地移除部分充电站

    策略：
    - 移除q个任务（已有）
    - 移除r个充电站（新增），r = random(0, min(2, num_cs))
    """
    # 移除任务（已有逻辑）
    removed_task_ids = random.sample(task_ids, q)

    # 新增：移除充电站
    if remove_cs:
        cs_nodes = [n for n in route.nodes if n.is_charging_station()]
        if len(cs_nodes) > 0:
            num_to_remove = random.randint(0, min(2, len(cs_nodes)))
            removed_cs = random.sample(cs_nodes, num_to_remove)
            for cs in removed_cs:
                destroyed_route.nodes.remove(cs)

    return destroyed_route, removed_task_ids
```

**测试**：
- 初始解有4个CS，destroy后应该有2-4个CS
- 确保移除CS后路径仍然有效

---

#### 1.2 修改Repair操作 - 智能充电站插入
**任务**：greedy_insertion在插入任务后，自动检测并插入必要的充电站

**实现**：
```python
def greedy_insertion_with_charging(self, route: Route, removed_task_ids: List[int]):
    """
    贪心插入任务 + 智能充电站插入

    策略：
    1. 对每个任务，找到成本最小的插入位置
    2. 插入任务后，检查电池可行性
    3. 如果不可行，自动插入必要的充电站
    """
    repaired_route = route.copy()

    for task_id in removed_task_ids:
        # 找到最优插入位置（已有逻辑）
        best_position = find_best_insertion_position(task)
        repaired_route.insert_task(task, best_position)

        # 新增：检查电池可行性
        if not self._check_battery_feasibility(repaired_route):
            # 不可行！需要插入充电站
            repaired_route = self._insert_necessary_charging_stations(repaired_route)

    return repaired_route
```

**关键子函数**：
```python
def _insert_necessary_charging_stations(self, route: Route):
    """
    自动插入必要的充电站

    策略：
    1. 模拟电池消耗，找到电量不足的位置
    2. 在该位置前插入最近的充电站
    3. 重复直到路径可行
    """
    while not self._check_battery_feasibility(route):
        # 找到电量耗尽的位置
        depletion_position = self._find_battery_depletion_position(route)

        # 找到最近的充电站
        nearest_cs = self._find_nearest_charging_station(
            route.nodes[depletion_position-1]
        )

        # 插入充电站
        route.nodes.insert(depletion_position, nearest_cs)

    return route
```

**测试**：
- 初始解只有1个CS，插入任务后应该自动增加到3-4个CS
- 最终路径应该是电池可行的

---

#### 1.3 充电临界值机制
**任务**：添加充电临界值，不等到完全没电才充

**实现**：
```python
class EnergyConfig:
    def __init__(self,
                 consumption_rate: float,
                 charging_rate: float,
                 charging_efficiency: float = 0.9,
                 critical_battery_threshold: float = 0.2):  # 新增：临界值20%
        self.critical_battery_threshold = critical_battery_threshold

def _check_battery_feasibility(self, route: Route, debug=False):
    """检查电池可行性（改进版）"""
    critical_threshold = self.energy_config.critical_battery_threshold * vehicle.battery_capacity

    for i in range(len(route.nodes) - 1):
        # ... 移动逻辑

        # 新增：检查是否低于临界值
        if current_battery < critical_threshold:
            # 查找附近是否有充电站
            has_nearby_cs = self._check_upcoming_charging_station(route, i, max_distance=3)
            if not has_nearby_cs:
                if debug:
                    print(f"  ⚠️  Battery critical at node {i}! ({current_battery:.1f} < {critical_threshold:.1f})")
                return False  # 低电量且附近没有充电站，不可行

    return True
```

**成本惩罚**：
```python
def evaluate_cost(self, route: Route):
    # ... 现有逻辑

    # 新增：低电量风险成本
    low_battery_penalty = self._calculate_low_battery_risk(route)

    total_cost = (distance_cost + charging_cost + time_cost +
                 delay_cost + missing_penalty + infeasible_penalty +
                 battery_penalty + low_battery_penalty)

    return total_cost

def _calculate_low_battery_risk(self, route: Route):
    """
    计算低电量风险成本

    如果路径中有段落电量低于临界值，给予风险惩罚
    """
    risk_penalty = 0.0
    critical_threshold = self.energy_config.critical_battery_threshold * vehicle.battery_capacity

    # 模拟电池
    for i, battery_level in enumerate(battery_levels):
        if battery_level < critical_threshold:
            # 低于临界值，增加风险惩罚
            deficit = critical_threshold - battery_level
            risk_penalty += deficit * 0.5  # 每kWh低于临界值惩罚0.5

    return risk_penalty
```

**测试**：
- 临界值20%，70kWh电池 → 14kWh临界值
- 路径中电量低于14kWh时应该触发惩罚或不可行判定

---

### 第二步：取送货分离优化（Week 3 可选功能）

#### 2.1 修改Task模型
**任务**：允许pickup和delivery之间有其他节点

**当前约束**：
```python
# 当前：pickup和delivery必须连续
route: depot → pickup1 → delivery1 → pickup2 → delivery2 → depot
```

**新约束**：
```python
# 新：只要pickup在delivery前即可
route: depot → pickup1 → pickup2 → delivery1 → delivery2 → depot
                        ^^^^^^^^   ^^^^^^^^^^
                     可以先集中取货，再集中送货
```

**实现**：
```python
def check_task_precedence(self, route: Route):
    """
    检查任务优先级约束

    约束：每个任务的pickup必须在delivery之前
    """
    for task in route.get_served_tasks():
        pickup_index = route.find_node_index(task.pickup_node)
        delivery_index = route.find_node_index(task.delivery_node)

        if pickup_index >= delivery_index:
            return False  # 违反优先级

    return True
```

**修改insertion逻辑**：
```python
def greedy_insertion(self, route: Route, removed_task_ids: List[int]):
    for task_id in removed_task_ids:
        # 新：pickup和delivery可以分开插入
        best_cost = float('inf')
        best_pickup_pos = None
        best_delivery_pos = None

        # 遍历所有可能的pickup位置
        for pickup_pos in range(1, len(route.nodes)):
            # 遍历所有在pickup之后的delivery位置
            for delivery_pos in range(pickup_pos + 1, len(route.nodes) + 1):
                # 可以有很大的间隔！
                cost = calculate_insertion_cost(pickup_pos, delivery_pos)
                if cost < best_cost:
                    best_cost = cost
                    best_pickup_pos = pickup_pos
                    best_delivery_pos = delivery_pos

        # 插入（先插入delivery，再插入pickup，避免索引变化）
        route.nodes.insert(best_delivery_pos, task.delivery_node)
        route.nodes.insert(best_pickup_pos, task.pickup_node)
```

**测试**：
- 2个任务，允许路径：depot → p1 → p2 → d1 → d2 → depot
- 验证p1在d1前，p2在d2前

---

#### 2.2 容量约束检查
**问题**：如果先集中取货，车辆容量可能超载

**实现**：
```python
def check_capacity_feasibility(self, route: Route):
    """
    检查容量可行性

    模拟货物装载，确保不超过容量
    """
    current_load = 0

    for node in route.nodes:
        if node.type == 'pickup':
            current_load += node.demand
        elif node.type == 'delivery':
            current_load -= node.demand

        if current_load > vehicle.capacity:
            return False  # 超载！
        if current_load < 0:
            return False  # 送货多于取货，逻辑错误

    return True
```

**测试**：
- 车辆容量100，3个任务各需求40
- 路径depot → p1 → p2 → p3 → d1 → d2 → d3应该不可行（120 > 100）
- 路径depot → p1 → p2 → d1 → p3 → d2 → d3应该可行

---

### 第三步：时间窗约束（Week 3-4 可选）

#### 3.1 添加时间窗到Task模型
**实现**：
```python
class Task:
    def __init__(self,
                 task_id: int,
                 pickup_loc: Tuple[float, float],
                 delivery_loc: Tuple[float, float],
                 demand: int,
                 pickup_time_window: Tuple[float, float] = None,  # 新增
                 delivery_time_window: Tuple[float, float] = None):  # 新增
        self.pickup_time_window = pickup_time_window or (0, float('inf'))
        self.delivery_time_window = delivery_time_window or (0, float('inf'))
```

#### 3.2 时间窗可行性检查
**实现**：
```python
def check_time_window_feasibility(self, route: Route):
    """
    检查时间窗可行性

    确保所有节点在时间窗内访问
    """
    current_time = 0

    for i, node in enumerate(route.nodes):
        # 到达节点
        if i > 0:
            travel_time = distance / speed
            current_time += travel_time

        # 检查时间窗
        if hasattr(node, 'time_window'):
            earliest, latest = node.time_window

            if current_time < earliest:
                # 早到，等待
                current_time = earliest
            elif current_time > latest:
                # 晚到，违反时间窗
                return False

        # 服务时间
        current_time += node.service_time

    return True
```

---

## 实施计划

### Week 2 (当前周) - 充电站动态优化
- [x] 第1.1步：修改Destroy操作，可移除充电站
- [ ] 第1.2步：修改Repair操作，智能插入充电站
- [ ] 第1.3步：实现充电临界值机制
- [ ] 测试：对比FR和PR-Minimal的充电站配置差异

**预期成果**：
- FR策略：3个充电站，每次充满，总成本约235000
- PR-Minimal策略：5个充电站，精确充电，总成本约230000
- 成本差异提升到2-3%

---

### Week 3 (可选) - 取送货分离优化
- [ ] 第2.1步：修改insertion逻辑，允许pickup/delivery分离
- [ ] 第2.2步：实现容量约束检查
- [ ] 测试：对比连续取送vs分离取送的成本差异

**预期成果**：
- 路径更灵活，可能减少5-10%距离

---

### Week 4 (可选) - 时间窗约束
- [ ] 第3.1步：添加时间窗到Task模型
- [ ] 第3.2步：实现时间窗可行性检查
- [ ] 测试：带时间窗的场景

**预期成果**：
- 更贴近实际场景

---

## 测试场景设计

### 场景1：充电站优化测试
```
任务：12个，分布80km×80km
充电站候选：10个，均匀分布
车辆：70kWh电池，消耗0.5kWh/km
临界值：20%（14kWh）

测试：
- FR策略应该选择3-4个远距离充电站
- PR-Minimal策略应该选择4-6个近距离充电站
- 总成本差异应该>2%
```

### 场景2：取送货分离测试
```
任务：3个，每个需求40
车辆：容量100

测试：
- 连续取送：depot → p1 → d1 → p2 → d2 → p3 → d3
- 分离取送：depot → p1 → p2 → d1 → p3 → d2 → d3
- 对比距离和容量可行性
```

### 场景3：时间窗测试
```
任务：5个，带时间窗
- task1: pickup[0-100], delivery[50-150]
- task2: pickup[100-200], delivery[150-250]
- ...

测试：
- 早到应该等待
- 晚到应该被惩罚
- 验证所有任务满足时间窗
```

---

## 关键决策点

### 决策1：充电临界值设置多少？
**选项**：
- 20%（保守）：更安全，但可能过度充电
- 10%（适中）：平衡安全和效率
- 5%（激进）：最少充电，但风险高

**建议**：10%，可配置

---

### 决策2：充电站选择策略？
**选项**：
- 最近充电站：简单，但可能绕路
- 路径上充电站：顺路，但可能不是最优
- 综合评估：距离+绕路成本

**建议**：综合评估

---

### 决策3：取送货分离的优先级？
**选项**：
- 高优先级：先实现，可能带来显著改进
- 低优先级：先优化充电站，再考虑取送货

**建议**：先充电站优化（Week 2），再取送货分离（Week 3）

---

## 成功标准

### 充电站优化成功标准：
- ✅ FR和PR-Minimal有不同的充电站数量（±1个以上）
- ✅ 总成本差异>2%（而不是当前的0.12%）
- ✅ 所有路径都是电池可行的
- ✅ 没有多余的充电站（充电量=0）

### 取送货分离成功标准：
- ✅ 允许p1 → p2 → d1 → d2的顺序
- ✅ 容量约束检查正确
- ✅ 距离减少5-10%

### 时间窗成功标准：
- ✅ 时间窗约束检查正确
- ✅ 早到会等待，晚到会惩罚
- ✅ 可行解满足所有时间窗
