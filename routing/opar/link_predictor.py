import math
import numpy as np
from utils.util_function import euclidean_distance


class LinkPredictor:
    """链路生命期预测器"""

    @staticmethod
    def predict_lifetime(drone1, drone2, max_comm_range):
        """预测两个节点间链路的生命期"""
        coords1 = drone1.coords
        coords2 = drone2.coords
        velocity1 = drone1.velocity
        velocity2 = drone2.velocity

        # 计算相对运动参数
        rel_velocity = [v1 - v2 for v1, v2 in zip(velocity1, velocity2)]
        rel_position = [p1 - p2 for p1, p2 in zip(coords1, coords2)]

        # 计算二次方程系数
        a = sum(v * v for v in rel_velocity)  # v^2项系数
        b = 2 * sum(v * p for v, p in zip(rel_velocity, rel_position))  # v*p项系数
        c = sum(p * p for p in rel_position) - max_comm_range ** 2  # 常数项

        # 边界情况处理
        if a == 0:  # 相对速度为0
            if c <= 0:  # 当前在范围内
                return float('inf')
            else:
                return 0

        # 求解二次方程
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return 0

        t1 = (-b + math.sqrt(discriminant)) / (2 * a)
        t2 = (-b - math.sqrt(discriminant)) / (2 * a)

        # 取最大的正值作为生命期
        lifetime = max(t for t in (t1, t2) if t > 0) if max(t1, t2) > 0 else 0
        return lifetime

    @staticmethod
    def calculate_link_quality(drone1, drone2, max_comm_range):
        """计算链路质量指标"""
        # 计算距离
        distance = euclidean_distance(drone1.coords, drone2.coords)

        # 计算相对速度
        rel_velocity = [v1 - v2 for v1, v2 in zip(drone1.velocity, drone2.velocity)]
        speed = math.sqrt(sum(v * v for v in rel_velocity))

        # 归一化参数
        norm_distance = distance / max_comm_range
        norm_speed = min(speed / 50.0, 1.0)  # 假设最大相对速度为50m/s

        # 计算链路质量（0-1之间）
        quality = 1 - (0.7 * norm_distance + 0.3 * norm_speed)

        # 考虑可能的障碍物和干扰
        interference_factor = 1.0
        if hasattr(drone1, 'motion_controller') and hasattr(drone2, 'motion_controller'):
            # 检查是否有其他节点可能造成干扰
            for drone in drone1.simulator.drones:
                if drone.identifier not in [drone1.identifier, drone2.identifier]:
                    d1 = euclidean_distance(drone.coords, drone1.coords)
                    d2 = euclidean_distance(drone.coords, drone2.coords)
                    if d1 < max_comm_range and d2 < max_comm_range:
                        interference_factor *= 0.9  # 每个潜在干扰源降低10%质量

        quality *= interference_factor
        return max(0, min(1, quality))

    @staticmethod
    def predict_link_stability(drone1, drone2, history_window=10):
        """预测链路稳定性"""
        if not hasattr(drone1, 'link_history'):
            return 0.5  # 默认中等稳定性

        history = drone1.link_history.get(drone2.identifier, [])
        if len(history) < 2:
            return 0.5

        # 计算历史连接状态变化率
        changes = sum(1 for i in range(1, len(history)) if history[i] != history[i - 1])
        stability = 1 - (changes / (len(history) - 1))
        return stability