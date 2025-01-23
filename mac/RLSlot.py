import numpy as np
from collections import defaultdict
from stable_baselines3 import PPO
from mac.stdma import Stdma
from entities.packet import DataPacket
from utils import config
from utils.util_function import euclidean_distance
import math
import  gym

class RLSlotController:
    def __init__(self, simulator):
        self.simulator = simulator
        self.env = simulator.env

        # RL模型参数
        self.state_dim = self._calculate_state_dim()
        self.action_dim = self._calculate_action_dim()

        # 初始化PPO模型
        self.model = PPO(
            "MlpPolicy",
            env=self,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            verbose=1
        )

    def get_state(self, traffic_info):
        """获取状态信息
        Args:
            traffic_info: 包含业务流信息的字典
                {
                    'source_id': 源节点ID,
                    'dest_id': 目标节点ID,
                    'num_packets': 数据包数量,
                    'delay_requirement': 时延需求,
                    'qos_requirement': QoS需求
                }
        """
        state = {
            # 1. 业务流信息
            'traffic': {
                'source': traffic_info['source_id'],
                'destination': traffic_info['dest_id'],
                'size': traffic_info['num_packets'],
                'delay_req': traffic_info['delay_requirement'],
                'qos_req': traffic_info['qos_requirement']
            },

            # 2. 网络拓扑信息
            'topology': self._get_topology_info(),

            # 3. 节点状态
            'node_states': self._get_node_states(),

            # 4. 链路状态
            'link_states': self._get_link_states()
        }

        return self._normalize_state(state)

    def _get_topology_info(self):
        """获取网络拓扑信息"""
        topology = {}

        # 获取节点位置
        for drone in self.simulator.drones:
            topology[drone.identifier] = {
                'position': drone.coords,
                'velocity': drone.velocity,
                'neighbors': list(drone.neighbors)
            }

        return topology

    def _get_node_states(self):
        """获取节点状态信息"""
        node_states = {}

        for drone in self.simulator.drones:
            node_states[drone.identifier] = {
                'queue_length': drone.transmitting_queue.qsize(),
                'energy_level': drone.residual_energy,
                'active_flows': len(drone.mac_protocol.flow_queue)
            }

        return node_states

    def _get_link_states(self):
        """获取链路状态信息"""
        link_states = {}

        for i in range(self.simulator.n_drones):
            for j in range(i + 1, self.simulator.n_drones):
                drone1 = self.simulator.drones[i]
                drone2 = self.simulator.drones[j]

                # 计算链路质量
                link_quality = drone1.mac_protocol.link_quality_manager.get_link_quality(i, j)

                # 预测链路生命周期
                link_lifetime = link_lifetime_predictor(drone1, drone2, drone1.routing_protocol.max_comm_range)

                link_states[(i, j)] = {
                    'quality': link_quality,
                    'lifetime': link_lifetime
                }

        return link_states

    def generate_slot_schedule(self, state):
        """根据状态生成时隙分配方案"""
        # 使用RL模型预测动作
        action, _ = self.model.predict(state)

        # 将动作转换为时隙分配方案
        schedule = self._action_to_schedule(action, state)

        return schedule

    def _action_to_schedule(self, action, state):
        """将RL动作转换为具体的时隙分配方案"""
        schedule = {}
        num_slots = self.simulator.drones[0].mac_protocol.num_slots

        # 获取业务流信息
        traffic = state['traffic']
        source_id = traffic['source']
        dest_id = traffic['destination']

        # 根据OPAR路由协议获取路由路径
        routing_path = self._get_routing_path(source_id, dest_id)

        # 为路径上的每个节点分配时隙
        for hop_idx, node_id in enumerate(routing_path):
            # 确定该节点需要的时隙数量
            if node_id == source_id:
                required_slots = math.ceil(traffic['size'] / num_slots)
            else:
                required_slots = math.ceil(traffic['size'] * 0.8 / num_slots)  # 考虑数据包丢失

            # 根据RL动作选择时隙
            node_slots = self._select_slots_for_node(
                node_id,
                required_slots,
                action[hop_idx],
                state
            )

            # 更新时隙分配表
            for slot in node_slots:
                if slot not in schedule:
                    schedule[slot] = []
                schedule[slot].append(node_id)

        return schedule


    def _select_slots_for_node(self, node_id, required_slots, action_part, state):
        """为节点选择合适的时隙"""
        selected_slots = []
        available_slots = list(range(self.simulator.drones[0].mac_protocol.num_slots))

        # 考虑干扰约束
        interference_matrix = self._calculate_interference_matrix(state)

        # 根据动作概率选择时隙
        action_probs = softmax(action_part)
        while len(selected_slots) < required_slots and available_slots:
            # 选择概率最高的时隙
            slot = available_slots[np.argmax(action_probs[available_slots])]

            # 检查干扰约束
            if self._check_interference_constraint(node_id, slot, selected_slots, interference_matrix):
                selected_slots.append(slot)

            available_slots.remove(slot)

        return selected_slots

    def _calculate_interference_matrix(self, state):
        """计算干扰矩阵"""
        n_drones = self.simulator.n_drones
        interference_matrix = np.zeros((n_drones, n_drones))

        for i in range(n_drones):
            for j in range(i + 1, n_drones):
                if (i, j) in state['link_states']:
                    interference = 1 - state['link_states'][(i, j)]['quality']
                    interference_matrix[i, j] = interference
                    interference_matrix[j, i] = interference

        return interference_matrix

    def _check_interference_constraint(self, node_id, slot, selected_slots, interference_matrix):
        """检查干扰约束"""
        for other_slot in selected_slots:
            for other_node in self.simulator.drones[0].mac_protocol.slot_schedule.get(other_slot, []):
                if interference_matrix[node_id, other_node] > 0.3:  # 干扰阈值
                    return False
        return True