"""
诊断ALNS是否正确集成了充电检查

这个脚本会检查：
1. ALNS类是否有vehicle和energy_config属性
2. repair方法中是否调用了充电检查
3. 提供修改建议
"""
import sys
from pathlib import Path
import inspect

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))


def diagnose_alns_integration():
    """诊断ALNS充电集成"""
    print("=" * 70)
    print("ALNS充电集成诊断")
    print("=" * 70)
    
    # 导入ALNS类
    try:
        from planner.alns import MinimalALNS
        print(f"\n✅ 成功导入 MinimalALNS")
    except ImportError as e:
        print(f"\n❌ 无法导入 MinimalALNS: {e}")
        return False
    
    # 检查类的属性和方法
    print(f"\n" + "=" * 70)
    print(f"步骤1：检查ALNS类结构")
    print("=" * 70)
    
    # 获取所有方法
    methods = [m for m in dir(MinimalALNS) if not m.startswith('_')]
    print(f"\n找到的公开方法：")
    for method in methods:
        print(f"  - {method}")
    
    # 查找repair相关方法
    repair_methods = [m for m in methods if 'repair' in m.lower()]
    print(f"\n找到的repair方法：")
    if repair_methods:
        for method in repair_methods:
            print(f"  ✅ {method}")
    else:
        print(f"  ❌ 没有找到repair方法")
        print(f"  这可能意味着repair逻辑在其他地方")
    
    # 检查optimize方法
    print(f"\n" + "=" * 70)
    print(f"步骤2：检查optimize方法")
    print("=" * 70)
    
    if hasattr(MinimalALNS, 'optimize'):
        print(f"\n✅ 找到 optimize 方法")
        
        # 获取源代码
        try:
            source = inspect.getsource(MinimalALNS.optimize)
            
            # 检查关键词
            has_repair = 'repair' in source.lower()
            has_energy = 'energy' in source.lower() or 'battery' in source.lower()
            has_charging = 'charging' in source.lower()
            has_feasibility = 'feasibility' in source.lower()
            
            print(f"\n代码分析：")
            print(f"  包含'repair'相关代码: {'✅' if has_repair else '❌'}")
            print(f"  包含'energy/battery'相关代码: {'✅' if has_energy else '❌'}")
            print(f"  包含'charging'相关代码: {'✅' if has_charging else '❌'}")
            print(f"  包含'feasibility'检查: {'✅' if has_feasibility else '❌'}")
            
            if not (has_energy or has_charging or has_feasibility):
                print(f"\n⚠️  警告：代码中似乎没有能量或充电相关的逻辑")
                print(f"  这可能就是为什么没有插入充电站的原因")
        except:
            print(f"\n⚠️  无法读取源代码")
    else:
        print(f"\n❌ 没有找到 optimize 方法")
    
    # 给出具体建议
    print(f"\n" + "=" * 70)
    print(f"诊断结论与修改建议")
    print("=" * 70)
    
    print(f"\n您需要在MinimalALNS中做以下修改：")
    print(f"\n1️⃣  在__init__方法中添加属性初始化：")
    print(f"""
    def __init__(self, distance_matrix, task_pool, repair_mode='greedy'):
        self.distance_matrix = distance_matrix
        self.task_pool = task_pool
        self.repair_mode = repair_mode
        
        # 添加这两行
        self.vehicle = None
        self.energy_config = None
    """)
    
    print(f"\n2️⃣  在repair方法中添加充电检查：")
    print(f"""
    # 在找到插入位置后，插入任务前，添加：
    
    is_feasible, charging_plan = route.check_energy_feasibility_for_insertion(
        task=task,
        insert_position=(pickup_pos, delivery_pos),
        vehicle=self.vehicle,
        distance_matrix=self.distance_matrix,
        energy_config=self.energy_config
    )
    
    if not is_feasible:
        continue  # 跳过这个不可行的位置
    
    # 然后在插入任务时：
    route.insert_task(task, best_position)
    
    if charging_plan:
        for plan in reversed(charging_plan):
            route.insert_charging_visit(
                station=plan['station_node'],
                position=plan['position'],
                charge_amount=plan['amount']
            )
    """)
    
    print(f"\n3️⃣  在使用ALNS时确保设置属性：")
    print(f"""
    alns = MinimalALNS(distance_matrix, task_pool)
    
    # 必须设置这两个属性
    alns.vehicle = vehicle
    alns.energy_config = energy_config
    
    # 然后才能调用optimize
    result = alns.optimize(initial_route)
    """)
    
    print(f"\n" + "=" * 70)
    print(f"下一步操作")
    print("=" * 70)
    print(f"\n请按照上述建议修改 src/planner/alns.py 文件")
    print(f"修改完成后，重新运行测试：")
    print(f"  python tests/test_alns_charging_fixed.py")
    
    return True


if __name__ == "__main__":
    print("\n")
    diagnose_alns_integration()
    print()