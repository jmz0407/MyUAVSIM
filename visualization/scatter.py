import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from utils import config
from utils.util_function import euclidean_distance
from phy.large_scale_fading import maximum_communication_range


def scatter_plot(simulator):
    plt.clf()  # 清理当前画布，避免重叠
    fig = plt.figure(figsize=(12, 8))  # 调整画布尺寸
    ax = fig.add_subplot(111, projection='3d')
    drawn_pairs = set()

    # 调整无人机颜色和大小
    colors = cm.get_cmap('viridis', len(simulator.drones))  # 渐变颜色
    for i, drone1 in enumerate(simulator.drones):
        ax.scatter(drone1.coords[0], drone1.coords[1], drone1.coords[2],
                   c=[colors(i)], s=60, label=f'UAV-{drone1.identifier}' if i == 0 else "")
        ax.text(drone1.coords[0], drone1.coords[1], drone1.coords[2],
                f'UAV-{drone1.identifier}', size=10, color='black')

        # 绘制连线
        for drone2 in simulator.drones:
            if drone1.identifier != drone2.identifier:
                pair = tuple(sorted((drone1.identifier, drone2.identifier)))
                if pair not in drawn_pairs:
                    distance = euclidean_distance(drone1.coords, drone2.coords)
                    if distance <= maximum_communication_range():
                        color_intensity = distance / maximum_communication_range()
                        ax.plot(
                            [drone1.coords[0], drone2.coords[0]],
                            [drone1.coords[1], drone2.coords[1]],
                            [drone1.coords[2], drone2.coords[2]],
                            color=cm.viridis(color_intensity),  # 根据距离改变颜色
                            alpha=0.6, linestyle='dashed', linewidth=1.5)
                        drawn_pairs.add(pair)

    ax.set_xlim(0, config.MAP_LENGTH)
    ax.set_ylim(0, config.MAP_WIDTH)
    ax.set_zlim(0, config.MAP_HEIGHT)

    # 设置更清晰的坐标轴
    ax.set_xlabel('X (m)', fontsize=12, labelpad=10)
    ax.set_ylabel('Y (m)', fontsize=12, labelpad=10)
    ax.set_zlabel('Z (m)', fontsize=12, labelpad=10)

    # 调整视角
    ax.view_init(elev=20, azim=30)  # 从特定角度查看

    # 添加图例
    ax.legend(loc='upper left', fontsize=10)

    # 保存图片并显示
    plt.tight_layout()
    plt.savefig('uav_topology_clear.png', dpi=300)  # 保存高分辨率图片
    plt.show()  # 确保只调用一次plt.show()
