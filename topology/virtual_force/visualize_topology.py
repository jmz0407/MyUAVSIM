import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import config
from utils.util_function import euclidean_distance
from phy.large_scale_fading import maximum_communication_range


def visualize_initial_topology(simulator):
    """可视化初始拓扑结构"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 提取坐标
    for drone in simulator.drones:
        # 绘制节点
        ax.scatter(drone.start_coords[0], drone.start_coords[1], drone.start_coords[2],
                   c='red', s=30)

        # 绘制通信链路
        for other_drone in simulator.drones:
            if drone.identifier != other_drone.identifier:
                distance = euclidean_distance(drone.start_coords, other_drone.start_coords)
                if distance <= maximum_communication_range():
                    x = [drone.start_coords[0], other_drone.start_coords[0]]
                    y = [drone.start_coords[1], other_drone.start_coords[1]]
                    z = [drone.start_coords[2], other_drone.start_coords[2]]
                    ax.plot(x, y, z, color='black', linestyle='dashed', linewidth=1)

    ax.set_xlim(0, config.MAP_LENGTH)
    ax.set_ylim(0, config.MAP_WIDTH)
    ax.set_zlim(0, config.MAP_HEIGHT)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Initial Network Topology')
    plt.figure()
    plt.show()


def visualize_current_topology(simulator):
    """可视化当前拓扑结构"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for drone in simulator.drones:
        # 绘制节点
        ax.scatter(drone.coords[0], drone.coords[1], drone.coords[2],
                   c='blue', s=30)

        # 绘制通信链路
        for other_drone in simulator.drones:
            if drone.identifier != other_drone.identifier:
                distance = euclidean_distance(drone.coords, other_drone.coords)
                if distance <= maximum_communication_range():
                    x = [drone.coords[0], other_drone.coords[0]]
                    y = [drone.coords[1], other_drone.coords[1]]
                    z = [drone.coords[2], other_drone.coords[2]]
                    ax.plot(x, y, z, color='black', linestyle='dashed', linewidth=1)

    ax.set_xlim(0, config.MAP_LENGTH)
    ax.set_ylim(0, config.MAP_WIDTH)
    ax.set_zlim(0, config.MAP_HEIGHT)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Current Network Topology at {simulator.env.now / 1e6}s')
    plt.figure()
    plt.show()