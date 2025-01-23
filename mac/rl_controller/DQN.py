import numpy as np
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from utils import config


class DQNNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class RLSlotController:
    def __init__(self, simulator, num_nodes, num_slots):
        self.simulator = simulator
        self.num_nodes = num_nodes
        self.num_slots = num_slots

        # RL parameters
        self.state_size = self._calculate_state_size()
        self.action_size = self._calculate_action_size()
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        # Neural Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNNetwork(self.state_size, self.action_size).to(self.device)
        self.target_model = DQNNetwork(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # 统计信息
        self.training_info = {
            'episode_rewards': [],
            'losses': [],
            'epsilon_values': []
        }

    def _calculate_state_size(self):
        """计算状态空间大小"""
        # 状态包括：
        # 1. 每个节点的队列长度 (num_nodes)
        # 2. 每对节点间的链路质量 (num_nodes * num_nodes)
        # 3. 节点的3D位置信息 (num_nodes * 3)
        # 4. 当前时隙分配情况 (num_slots * num_nodes)
        # 5. 业务需求信息 (源节点、目标节点、包数量、延迟要求、QoS要求)
        return (self.num_nodes +
                self.num_nodes * self.num_nodes +
                self.num_nodes * 3 +
                self.num_slots * self.num_nodes +
                5)

    def _calculate_action_size(self):
        """计算动作空间大小"""
        # 动作空间为所有可能的时隙分配方案
        # 这里简化为每个时隙可以分配给哪些节点的组合
        return self.num_slots * self.num_nodes

    def get_state(self, traffic_req=None):
        """
        构建当前状态向量
        Args:
            traffic_req: 当前的业务需求
        """
        state = []

        # 1. 获取队列长度
        queue_lengths = []
        for drone in self.simulator.drones:
            queue_lengths.append(drone.transmitting_queue.qsize())
        state.extend(queue_lengths)

        # 2. 获取链路质量
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    link_quality = self.simulator.drones[i].mac_protocol.link_quality_manager.get_link_quality(i, j)
                    state.append(link_quality if link_quality is not None else -1)
                else:
                    state.append(0)

        # 3. 获取节点位置
        for drone in self.simulator.drones:
            state.extend(drone.coords)

        # 4. 当前时隙分配情况
        current_schedule = np.zeros((self.num_slots, self.num_nodes))
        if hasattr(self.simulator.drones[0].mac_protocol, 'slot_schedule'):
            schedule = self.simulator.drones[0].mac_protocol.slot_schedule
            for slot, nodes in schedule.items():
                for node in nodes:
                    current_schedule[slot][node] = 1
        state.extend(current_schedule.flatten())

        # 5. 业务需求信息
        if traffic_req:
            state.extend([
                traffic_req.source_id,
                traffic_req.dest_id,
                traffic_req.num_packets,
                traffic_req.delay_requirement,
                traffic_req.qos_requirement
            ])
        else:
            state.extend([0, 0, 0, 0, 0])

        return np.array(state)

    def get_action(self, state):
        """选择动作"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            return q_values.argmax().item()

    def calculate_reward(self, old_state, new_state, traffic_req):
        """计算奖励"""
        reward = 0

        # 1. 时隙利用率奖励
        old_schedule = old_state[-self.num_slots * self.num_nodes:].reshape(self.num_slots, self.num_nodes)
        new_schedule = new_state[-self.num_slots * self.num_nodes:].reshape(self.num_slots, self.num_nodes)

        old_utilization = np.sum(old_schedule) / (self.num_slots * self.num_nodes)
        new_utilization = np.sum(new_schedule) / (self.num_slots * self.num_nodes)

        reward += (new_utilization - old_utilization) * 10

        # 2. 延迟相关奖励
        if traffic_req:
            actual_delay = self._calculate_transmission_delay(new_schedule, traffic_req)
            if actual_delay <= traffic_req.delay_requirement:
                reward += 20
            else:
                reward -= 10

        # 3. 干扰惩罚
        interference_penalty = self._calculate_interference_penalty(new_schedule)
        reward -= interference_penalty

        return reward

    def _calculate_transmission_delay(self, schedule, traffic_req):
        """估算传输延迟"""
        src = traffic_req.source_id
        dst = traffic_req.dest_id

        # 找到源节点被分配的时隙
        src_slots = []
        for slot in range(self.num_slots):
            if schedule[slot][src] == 1:
                src_slots.append(slot)

        if not src_slots:
            return float('inf')

        # 简单估算：假设每个时隙可以传输一个数据包
        num_slots_needed = traffic_req.num_packets
        total_frame_time = self.num_slots * config.SLOT_DURATION

        return (num_slots_needed / len(src_slots)) * total_frame_time

    def _calculate_interference_penalty(self, schedule):
        """计算干扰惩罚"""
        penalty = 0
        for slot in range(self.num_slots):
            active_nodes = np.where(schedule[slot] == 1)[0]
            if len(active_nodes) > 1:
                for i in range(len(active_nodes)):
                    for j in range(i + 1, len(active_nodes)):
                        node1 = self.simulator.drones[active_nodes[i]]
                        node2 = self.simulator.drones[active_nodes[j]]
                        distance = np.sqrt(np.sum((np.array(node1.coords) - np.array(node2.coords)) ** 2))
                        if distance < config.SENSING_RANGE:
                            penalty += 1
        return penalty

    def update(self, state, action, reward, next_state, done):
        """更新经验回放内存和训练模型"""
        self.memory.append((state, action, reward, next_state, done))

        if len(self.memory) < 32:  # 最小批量大小
            return

        # 从经验回放中随机采样
        minibatch = random.sample(self.memory, 32)
        states = torch.FloatTensor([t[0] for t in minibatch]).to(self.device)
        actions = torch.LongTensor([t[1] for t in minibatch]).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in minibatch]).to(self.device)
        next_states = torch.FloatTensor([t[3] for t in minibatch]).to(self.device)
        dones = torch.FloatTensor([t[4] for t in minibatch]).to(self.device)

        # 当前Q值
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))

        # 下一状态的Q值
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # 计算损失并更新模型
        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 记录训练信息
        self.training_info['losses'].append(loss.item())
        self.training_info['epsilon_values'].append(self.epsilon)

    def update_target_model(self):
        """更新目标网络"""
        self.target_model.load_state_dict(self.model.state_dict())

    def decode_action(self, action):
        """将动作转换为具体的时隙分配方案"""
        slot = action // self.num_nodes
        node = action % self.num_nodes
        return slot, node

    def optimize_slot_allocation(self, traffic_req):
        """
        基于当前业务需求优化时隙分配
        Returns:
            dict: 优化后的时隙分配方案
        """
        current_state = self.get_state(traffic_req)
        action = self.get_action(current_state)
        slot, node = self.decode_action(action)

        # 获取当前调度方案
        current_schedule = {}
        if hasattr(self.simulator.drones[0].mac_protocol, 'slot_schedule'):
            current_schedule = self.simulator.drones[0].mac_protocol.slot_schedule.copy()

        # 更新调度方案
        if slot not in current_schedule:
            current_schedule[slot] = []
        if node not in current_schedule[slot]:
            current_schedule[slot].append(node)

        # 计算新状态和奖励
        new_state = self.get_state(traffic_req)
        reward = self.calculate_reward(current_state, new_state, traffic_req)

        # 更新模型
        self.update(current_state, action, reward, new_state, False)

        # 定期更新目标网络
        if len(self.memory) % 100 == 0:
            self.update_target_model()

        return current_schedule