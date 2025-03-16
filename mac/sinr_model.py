import math
import numpy as np
from utils import config


def calculate_path_loss(distance, frequency=None, path_loss_exponent=None):
    """
    计算路径损耗

    参数:
        distance: 距离 (米)
        frequency: 载波频率 (Hz)，默认使用配置文件中的值
        path_loss_exponent: 路径损耗指数，默认使用配置文件中的值

    返回:
        path_loss: 路径损耗值 (线性，非dB)
    """
    # 使用默认值
    if frequency is None:
        frequency = config.CARRIER_FREQUENCY

    if path_loss_exponent is None:
        path_loss_exponent = config.PATH_LOSS_EXPONENT

    # 预防零距离
    if distance <= 0.1:
        distance = 0.1

    # 计算波长
    wavelength = config.LIGHT_SPEED / frequency

    # 自由空间路径损耗公式
    path_loss = (4 * math.pi * distance / wavelength) ** path_loss_exponent

    return path_loss


def calculate_received_power(distance, transmit_power=None):
    """
    计算接收功率

    参数:
        distance: 发送方与接收方之间的距离 (米)
        transmit_power: 发送功率 (瓦特)，默认使用配置文件中的值

    返回:
        received_power: 接收功率 (瓦特)
    """
    if transmit_power is None:
        transmit_power = config.TRANSMITTING_POWER

    # 计算路径损耗
    path_loss = calculate_path_loss(distance)

    # 计算接收功率
    received_power = transmit_power / path_loss

    return received_power


def calculate_sinr(received_power, interference_powers, noise_power=None):
    """
    计算SINR (信号与干扰加噪声比)

    参数:
        received_power: 目标信号接收功率 (瓦特)
        interference_powers: 干扰信号功率列表 (瓦特)
        noise_power: 噪声功率 (瓦特)，默认使用配置文件中的值

    返回:
        sinr: 信号与干扰加噪声比 (线性，非dB)
    """
    if noise_power is None:
        noise_power = config.NOISE_POWER

    # 计算总干扰功率
    total_interference = sum(interference_powers) if interference_powers else 0

    # 计算SINR
    sinr = received_power / (total_interference + noise_power)

    return sinr


def sinr_to_db(sinr):
    """将线性SINR转换为dB单位"""
    return 10 * math.log10(sinr)


def db_to_sinr(sinr_db):
    """将dB单位的SINR转换为线性单位"""
    return 10 ** (sinr_db / 10)


def estimate_link_capacity(sinr, bandwidth=None):
    """
    使用Shannon公式估计链路容量

    参数:
        sinr: 信号与干扰加噪声比 (线性，非dB)
        bandwidth: 带宽 (Hz)，默认使用配置文件中的值

    返回:
        capacity: 链路容量 (比特/秒)
    """
    if bandwidth is None:
        bandwidth = config.BANDWIDTH

    # Shannon公式: C = B * log2(1 + SINR)
    capacity = bandwidth * math.log2(1 + sinr)

    return capacity


def check_interference(node1, node2, distance, sinr_threshold=None):
    """
    基于SINR模型检查两个节点之间是否存在干扰

    参数:
        node1: 第一个节点
        node2: 第二个节点
        distance: 两节点之间的距离
        sinr_threshold: SINR阈值 (dB)，默认使用配置文件中的值

    返回:
        interference: 是否存在干扰 (布尔值)
    """
    if sinr_threshold is None:
        sinr_threshold = config.SNR_THRESHOLD

    # 计算接收功率
    received_power = calculate_received_power(distance)

    # 计算SINR (假设没有其他干扰源)
    sinr = calculate_sinr(received_power, [])

    # 转换为dB
    sinr_db = sinr_to_db(sinr)

    # 如果SINR低于阈值，则存在干扰
    return sinr_db < sinr_threshold


def build_interference_matrix(drone_positions, interference_threshold=None):
    """
    构建干扰矩阵

    参数:
        drone_positions: 无人机位置列表，每个位置是(x,y,z)坐标
        interference_threshold: 干扰判定的SINR阈值 (dB)

    返回:
        interference_matrix: 干扰矩阵，matrix[i][j]=1表示i和j之间存在干扰
    """
    num_drones = len(drone_positions)
    interference_matrix = np.zeros((num_drones, num_drones), dtype=int)

    for i in range(num_drones):
        for j in range(i + 1, num_drones):
            # 计算距离
            pos1 = drone_positions[i]
            pos2 = drone_positions[j]
            distance = ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2 + (pos1[2] - pos2[2]) ** 2) ** 0.5

            # 检查干扰
            if check_interference(i, j, distance, interference_threshold):
                interference_matrix[i][j] = 1
                interference_matrix[j][i] = 1

    return interference_matrix


def calculate_minimum_slots_needed(interference_matrix):
    """
    基于干扰矩阵估计最小需要的时隙数

    参数:
        interference_matrix: 干扰矩阵，matrix[i][j]=1表示i和j之间存在干扰

    返回:
        min_slots: 最小需要的时隙数
    """
    # 实现一个简单的贪心图着色方法来估计
    n = interference_matrix.shape[0]

    # 节点颜色 (-1表示未着色)
    colors = [-1] * n

    # 为每个节点着色
    for node in range(n):
        # 邻居已使用的颜色
        used_colors = set()
        for neighbor in range(n):
            if interference_matrix[node][neighbor] == 1 and colors[neighbor] != -1:
                used_colors.add(colors[neighbor])

        # 找到最小可用颜色
        color = 0
        while color in used_colors:
            color += 1

        # 分配颜色
        colors[node] = color

    # 返回需要的最小颜色数 (从0开始计数，所以+1)
    return max(colors) + 1 if colors else 0