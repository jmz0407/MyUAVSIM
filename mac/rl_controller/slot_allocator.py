# mac/rl_controller/slot_allocator.py

import numpy as np
from stable_baselines3 import PPO
import torch
from phy.large_scale_fading import maximum_communication_range


class SlotAllocator:
    """
    时隙分配器类 - 负责使用训练好的模型生成时隙分配方案
    """

    def __init__(self, num_nodes, num_slots, model_path="stdma_ppo_model"):
        self.num_nodes = num_nodes
        self.num_slots = num_slots
        # 加载训练好的模型
        self.model = PPO.load(model_path)
        # 设置最大通信范围
        self.max_comm_range = maximum_communication_range()

    def get_slot_schedule(self, traffic_info, network_state):
        """
        根据业务需求和网络状态生成时隙分配方案

        Args:
            traffic_info: 业务需求信息字典
                {
                    'source_id': 源节点ID,
                    'dest_id': 目标节点ID,
                    'num_packets': 数据包数量,
                    'delay_requirement': 时延需求,
                    'qos_requirement': QoS需求
                }
            network_state: 网络状态信息
                {
                    'node_positions': 节点位置信息,
                    'link_qualities': 链路质量信息
                }

        Returns:
            schedule: 时隙分配方案字典 {slot_id: [node_ids]}
        """
        # 构建模型输入状态
        state = self._build_state_vector(traffic_info, network_state)

        # 使用模型预测动作
        action, _ = self.model.predict(state, deterministic=True)

        # 将动作转换为时隙分配方案
        schedule = self._action_to_schedule(action, traffic_info, network_state)

        return schedule

    def _build_state_vector(self, traffic_info, network_state):
        """
        构建模型输入的状态向量
        """
        # 初始化状态向量
        state = []

        # 1. 添加业务需求信息（归一化）
        state.extend([
            traffic_info['source_id'] / self.num_nodes,
            traffic_info['dest_id'] / self.num_nodes,
            traffic_info['num_packets'] / 100,  # 假设最大包数为100
            traffic_info['delay_requirement'] / 1000,  # 归一化延迟需求
            traffic_info['qos_requirement']
        ])

        # 2. 添加节点位置信息（归一化）
        positions = network_state['node_positions']
        for pos in positions.values():
            state.extend([
                pos[0] / 400,  # x坐标归一化
                pos[1] / 300,  # y坐标归一化
                pos[2] / 200  # z坐标归一化
            ])

        # 3. 添加链路质量信息
        link_qualities = network_state['link_qualities']
        quality_matrix = np.zeros((self.num_nodes, self.num_nodes))
        for (i, j), quality in link_qualities.items():
            quality_matrix[i][j] = quality
            quality_matrix[j][i] = quality
        state.extend(quality_matrix.flatten())

        return np.array(state, dtype=np.float32)

    def _action_to_schedule(self, action, traffic_info, network_state):
        """
        将模型输出的动作转换为实际的时隙分配方案
        同时确保分配方案满足物理约束
        """
        # 将连续动作值转换为二值化的分配矩阵
        allocation_matrix = np.round(action.reshape(self.num_slots, self.num_nodes))

        # 初始化分配方案
        schedule = {}

        # 检查并应用物理约束
        for slot in range(self.num_slots):
            schedule[slot] = []
            assigned_nodes = np.where(allocation_matrix[slot] > 0.5)[0]

            # 检查每个被分配的节点
            for node_id in assigned_nodes:
                # 检查是否与已分配节点存在干扰
                can_assign = True
                for existing_node in schedule[slot]:
                    if not self._check_interference(
                            node_id,
                            existing_node,
                            network_state['node_positions'],
                            network_state['link_qualities']
                    ):
                        can_assign = False
                        break

                if can_assign:
                    schedule[slot].append(int(node_id))

        return schedule

    def _check_interference(self, node1, node2, positions, link_qualities):
        """
        检查两个节点之间是否存在干扰
        """
        # 计算节点间距离
        pos1 = positions[node1]
        pos2 = positions[node2]
        distance = np.sqrt(np.sum((np.array(pos1) - np.array(pos2)) ** 2))

        # 获取链路质量
        quality = link_qualities.get((node1, node2), 0)

        # 检查干扰条件
        return distance >= self.max_comm_range and quality >= 0.7