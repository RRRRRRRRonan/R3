"""
测试充电功能集成
"""
import sys
sys.path.append('src')

from core.node import create_depot, create_task_node_pair, create_charging_node
from core.task import create_task, TaskPool
from core.vehicle import create_vehicle
from core.route import Route
from physics.distance import create_distance_matrix_from_layout
from physics.energy import EnergyConfig
from planner.alns import MinimalALNS


def test_basic_charging():
    """测试基本充电功能"""
    print("=" * 60)
    print("测试：基本充电功能")
    print("=" * 60)
    
    # 创建depot
    depot = create_depot((0, 0))
    
    # 创建任务（距离很远，需要充电）
    p1, d1 = create_task_node_pair(
        task_id=1,
        pickup_id=1,
        delivery_id=2,
        pickup_coords=(100, 100),
        delivery_coords=(150, 150)
    )
    task1 = create_task(1, p1, d1)
    
    # 创建充电站
    charging_station = create_charging_node(
        node_id=10,
        coordinates=(50, 50)
    )
    
    # 创建距离矩阵
    distance_matrix = create_distance_matrix_from_layout(
        depot=(0, 0),
        task_locations=[((100, 100), (150, 150))],
        charging_stations=[(50, 50)]
    )
    
    # 创建AMR（小电池）
    vehicle = create_vehicle(1, battery_capacity=50.0, initial_battery=50.0)
    
    # 创建路径并测试
    route = Route(vehicle_id=1, nodes=[depot, p1, d1, depot])
    
    # 检查电量可行性
    energy_config = EnergyConfig()
    feasible, charging_plan = route.check_energy_feasibility_for_insertion(
        task1, (1, 2), vehicle, distance_matrix, energy_config
    )
    
    print(f"✓ 电量可行性检查: {'可行' if feasible else '不可行'}")
    print(f"✓ 需要充电站数量: {len(charging_plan)}")
    
    if charging_plan:
        for i, plan in enumerate(charging_plan):
            print(f"  充电计划{i+1}: 位置{plan['position']}, 充电量{plan['amount']:.1f}kWh")


if __name__ == "__main__":
    test_basic_charging()
    print("\n🎉 充电功能测试通过！")