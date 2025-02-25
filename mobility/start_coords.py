import random
from utils import config
from phy.large_scale_fading import maximum_communication_range


def get_random_start_point_3d(sim_seed):
    start_position = []
    for i in range(config.NUMBER_OF_DRONES):
        random.seed(sim_seed + i)
        position_x = random.uniform(50, config.MAP_LENGTH - 50)
        position_y = random.uniform(50, config.MAP_WIDTH - 50)
        position_z = random.uniform(50, config.MAP_HEIGHT - 50)

        start_position.append(tuple([position_x, position_y, position_z]))

    return start_position


def get_custom_start_point_3d(sim_seed):
    """自定义拓扑结构，确保节点在地图范围内"""
    max_range = maximum_communication_range()
    print(f"Maximum communication range: {max_range} meters")

    # 计算安全的链路距离（考虑地图大小）
    # 地图宽度是500，我们要放下4个节点的主链，所以要适当缩小距离
    link_distance = min(max_range * 0.8, config.MAP_LENGTH / 5)  # 确保主链能放入地图
    branch_distance = min(max_range * 0.9, config.MAP_WIDTH / 4)  # 确保分支不会超出地图

    # 起始位置（靠近地图边缘但留有余量）
    start_x = 50
    start_y = config.MAP_WIDTH / 2
    base_height = config.MAP_HEIGHT / 3

    custom_positions = [
        # 主链 (4个节点)
        (start_x, start_y, base_height),  # 节点0
        (start_x + link_distance, start_y, base_height),  # 节点1
        (start_x + 2 * link_distance, start_y, base_height),  # 节点2
        (start_x + 3 * link_distance, start_y, base_height),  # 节点3

        # 从节点1延伸的分支
        (start_x + link_distance, start_y + branch_distance, base_height),  # 节点4
        (start_x + link_distance, start_y + 2 * branch_distance / 3, base_height),  # 节点5

        # 从节点2延伸的分支
        (start_x + 2 * link_distance, start_y - branch_distance, base_height),  # 节点6
        (start_x + 2 * link_distance, start_y - 2 * branch_distance / 3, base_height),  # 节点7

        # 上层节点
        (start_x + 1.5 * link_distance, start_y, base_height + branch_distance / 2),  # 节点8
        (start_x + 1.5 * link_distance, start_y, base_height + branch_distance)  # 节点9
    ]

    # 打印坐标检查
    print("\nNode positions:")
    for i, pos in enumerate(custom_positions):
        print(f"Node {i}: {pos}")

    # 验证位置是否在地图范围内
    for pos in custom_positions:
        if not (0 <= pos[0] <= config.MAP_LENGTH and
                0 <= pos[1] <= config.MAP_WIDTH and
                0 <= pos[2] <= config.MAP_HEIGHT):
            raise ValueError(f"Position {pos} is outside map boundaries")

    # # 打印预期的连通性
    # print("\nExpected connectivity:")
    # for i in range(len(custom_positions)):
    #     connected_nodes = []
    #     for j in range(len(custom_positions)):
    #         if i != j:
    #             dist = euclidean_distance(custom_positions[i], custom_positions[j])
    #             if dist <= max_range:
    #                 connected_nodes.append(j)
    #     print(f"Node {i} can communicate with: {connected_nodes}")

    return custom_positions