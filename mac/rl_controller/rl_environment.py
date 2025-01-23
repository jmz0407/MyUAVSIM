import gym
from gym import spaces
import numpy as np
from utils import config
from utils.util_function import euclidean_distance

import numpy as np
from gym import spaces
import gym


class StdmaEnv(gym.Env):
    """
    Custom Environment for STDMA slot allocation
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, simulator, num_nodes, num_slots):
        super(StdmaEnv, self).__init__()

        self.simulator = simulator
        self.num_nodes = num_nodes
        self.num_slots = num_slots
        self.max_steps = 1000  # 每个episode的最大步数
        self.current_step = 0

        # 定义动作空间
        # 简化动作空间：每次只决定一个时隙的分配
        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(num_slots * num_nodes,),
            dtype=np.float32
        )

        # 定义观察空间
        # 1. 队列长度 (num_nodes,)
        # 2. 链路质量矩阵 (num_nodes * num_nodes,)
        # 3. 节点位置 (num_nodes * 3,)
        # 4. 当前时隙分配 (num_slots * num_nodes,)
        # 5. 业务需求信息 (5,) [源节点,目标节点,包数量,延迟要求,QoS要求]
        obs_size = (num_nodes +
                    num_nodes * num_nodes +
                    num_nodes * 3 +
                    num_slots * num_nodes +
                    5)

        self.observation_space = spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(obs_size,),
            dtype=np.float32
        )

        # 保存当前业务需求
        self.current_traffic_req = None

        # 用于评估性能的指标
        self.performance_metrics = {
            'throughput': [],
            'delay': [],
            'collision_rate': [],
            'slot_utilization': []
        }

    def _get_queue_lengths(self):
        """获取所有节点的队列长度"""
        return np.array([
            drone.transmitting_queue.qsize()
            for drone in self.simulator.drones
        ])

    def _get_link_quality_matrix(self):
        """获取链路质量矩阵"""
        quality_matrix = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    quality = self.simulator.drones[i].mac_protocol.link_quality_manager.get_link_quality(i, j)
                    quality_matrix[i][j] = quality if quality is not None else -1
        return quality_matrix.flatten()

    def _get_node_positions(self):
        """获取所有节点的3D位置"""
        positions = []
        for drone in self.simulator.drones:
            positions.extend(drone.coords)
        return np.array(positions)

    def _get_current_schedule(self):
        """获取当前时隙分配情况"""
        schedule = np.zeros((self.num_slots, self.num_nodes))

        # 优先使用模拟器中的时隙表（如果存在）
        if (hasattr(self.simulator.drones[0].mac_protocol, 'slot_schedule') and
                self.simulator.drones[0].mac_protocol.slot_schedule is not None):
            current_schedule = self.simulator.drones[0].mac_protocol.slot_schedule
        else:
            current_schedule = self.current_schedule

        for slot, nodes in current_schedule.items():
            for node in nodes:
                schedule[slot][node] = 1

        return schedule.flatten()

    def _calculate_interference(self, schedule):
        """计算给定调度方案的干扰程度"""
        interference_count = 0
        schedule = schedule.reshape((self.num_slots, self.num_nodes))

        for slot in range(self.num_slots):
            active_nodes = np.where(schedule[slot] == 1)[0]
            if len(active_nodes) > 1:
                for i in range(len(active_nodes)):
                    for j in range(i + 1, len(active_nodes)):
                        # 获取两个活跃节点
                        node1 = self.simulator.drones[active_nodes[i]]
                        node2 = self.simulator.drones[active_nodes[j]]

                        # 计算节点间距离
                        distance = euclidean_distance(node1.coords, node2.coords)

                        # 如果距离小于干扰范围，增加干扰计数
                        if distance < config.SENSING_RANGE:
                            interference_count += 1

        return interference_count

    def _calculate_transmission_delay(self, schedule, traffic_req):
        """估算传输延迟"""
        schedule = schedule.reshape((self.num_slots, self.num_nodes))
        src = traffic_req.source_id
        dst = traffic_req.dest_id

        # 计算源节点分配到的时隙数
        src_slots = np.where(schedule[:, src] == 1)[0]
        if len(src_slots) == 0:
            return float('inf')

        # 估算完成传输所需的帧数
        packets_per_slot = config.DATA_PACKET_LENGTH / (config.SLOT_DURATION * config.BIT_RATE)
        total_slots_needed = traffic_req.num_packets / packets_per_slot
        frames_needed = total_slots_needed / len(src_slots)

        # 计算总延迟
        frame_duration = self.num_slots * config.SLOT_DURATION
        total_delay = frames_needed * frame_duration

        return total_delay

    def _calculate_slot_utilization(self, schedule):
        """计算时隙利用率"""
        schedule = schedule.reshape((self.num_slots, self.num_nodes))
        return np.sum(schedule) / (self.num_slots * self.num_nodes)

    def _calculate_reward(self, schedule):
        """
        优化后的奖励函数：
        1. 更温和的奖励/惩罚值
        2. 更平滑的奖励曲线
        3. 更合理的权重分配
        """
        reward = 0.0
        schedule = schedule.reshape((self.num_slots, self.num_nodes))

        try:
            # 计算基础指标
            interference = self._calculate_interference(schedule)
            delay = self._calculate_transmission_delay(schedule, self.current_traffic_req)
            active_slots = np.any(schedule, axis=1).sum()
            slot_efficiency = 1.0 - (active_slots / self.num_slots)
            target_qos = self.current_traffic_req.qos_requirement if self.current_traffic_req else 0.9

            # 1. 基础可行性奖励
            feasible_allocation = True
            if interference > 10:  # 过高的干扰表示分配不合理
                feasible_allocation = False
                reward -= 10
            elif delay == float('inf'):  # 无法完成传输
                feasible_allocation = False
                reward -= 10
            elif active_slots == 0:  # 没有分配任何时隙
                feasible_allocation = False
                reward -= 10
            else:
                reward += 5  # 基础可行性奖励

            if feasible_allocation:
                # 2. 延迟相关奖励（更平滑的奖励曲线）
                if self.current_traffic_req:
                    delay_ratio = delay / self.current_traffic_req.delay_requirement
                    if delay_ratio <= 1.0:
                        # 使用对数函数使奖励更平滑
                        reward += 10 * (1 - np.log(1 + delay_ratio))
                    else:
                        # 使用对数函数使惩罚更平滑
                        reward -= 5 * np.log(1 + delay_ratio)

                # 3. 时隙效率奖励
                # 使用sigmoid函数使奖励更平滑
                efficiency_reward = 10 * (1 / (1 + np.exp(-10 * (slot_efficiency - 0.5))))
                reward += efficiency_reward

                # 4. 干扰控制奖励
                if interference == 0:
                    reward += 10
                else:
                    # 使用指数衰减的惩罚
                    reward -= 5 * (1 - np.exp(-interference / 5))

                # 5. 连续时隙分配奖励
                for node in range(self.num_nodes):
                    node_slots = schedule[:, node]
                    if np.sum(node_slots) > 0:
                        consecutive_slots = 0
                        max_consecutive = 0
                        for slot in node_slots:
                            if slot == 1:
                                consecutive_slots += 1
                                max_consecutive = max(max_consecutive, consecutive_slots)
                            else:
                                consecutive_slots = 0
                        reward += max_consecutive * 0.5

            # 确保奖励在合理范围内
            reward = np.clip(reward, -20.0, 20.0)

        except Exception as e:
            print(f"Error in reward calculation: {e}")
            reward = -5.0

        return float(reward)

    def step(self, action):
        """
        执行一步动作
        Args:
            action: 时隙分配方案,shape=(num_slots * num_nodes,)
        Returns:
            observation: 新的状态
            reward: 奖励值
            done: 是否结束
            info: 额外信息
        """
        # 将连续动作转换为离散动作（四舍五入）
        action = np.where(action > 0.5, 1, 0)
        self.current_step += 1

        # 将动作转换为时隙分配方案
        schedule = action.reshape((self.num_slots, self.num_nodes))

        # 更新模拟器中的时隙分配
        new_schedule = {}
        for slot in range(self.num_slots):
            new_schedule[slot] = list(np.where(schedule[slot] == 1)[0])

        if hasattr(self.simulator.drones[0].mac_protocol, 'slot_schedule'):
            self.simulator.drones[0].mac_protocol.slot_schedule = new_schedule

        # 计算奖励
        reward = self._calculate_reward(action)

        # 获取新的观察值
        observation = self._get_observation()

        # 检查是否结束
        done = self.current_step >= self.max_steps

        # 收集性能指标
        info = {
            'interference': self._calculate_interference(action),
            'utilization': self._calculate_slot_utilization(action),
            'delay': self._calculate_transmission_delay(schedule, self.current_traffic_req)
            if self.current_traffic_req else None
        }

        return observation, reward, done, info

    def reset(self):
        """重置环境"""
        self.current_step = 0

        # 重置时隙表
        self.current_schedule = {slot: [] for slot in range(self.num_slots)}

        # 随机生成新的业务需求
        # src = np.random.randint(0, self.num_nodes)
        # dst = np.random.randint(0, self.num_nodes)
        src = 9
        dst = 12
        while dst == src:
            dst = np.random.randint(0, self.num_nodes)

        class TrafficReq:
            def __init__(self):
                self.source_id = src
                self.dest_id = dst
                self.num_packets = np.random.randint(5, 20)
                self.delay_requirement = np.random.uniform(1000, 5000)
                self.qos_requirement = np.random.uniform(0.7, 0.95)

        self.current_traffic_req = TrafficReq()

        return self._get_observation()

    def _get_observation(self):
        """构建观察向量，并进行归一化"""
        # 获取各部分观察
        queue_lengths = self._get_queue_lengths()
        link_quality = self._get_link_quality_matrix()
        positions = self._get_node_positions()
        schedule = self._get_current_schedule()

        # 归一化队列长度
        max_queue_size = self.simulator.drones[0].max_queue_size
        queue_lengths = queue_lengths / max_queue_size

        # 归一化位置坐标
        positions = positions / max(config.MAP_LENGTH, config.MAP_WIDTH, config.MAP_HEIGHT)

        # 归一化业务需求信息
        traffic_info = np.array([
            self.current_traffic_req.source_id / self.num_nodes,
            self.current_traffic_req.dest_id / self.num_nodes,
            self.current_traffic_req.num_packets / 20.0,  # 假设最大包数为20
            self.current_traffic_req.delay_requirement / 5000.0,  # 归一化延迟要求
            self.current_traffic_req.qos_requirement  # QoS已经是0-1之间
        ])

        # 合并所有观察
        obs = np.concatenate([
            queue_lengths,
            link_quality,
            positions,
            schedule,
            traffic_info
        ])

        # 确保没有无穷大的值
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

        return obs.astype(np.float32)

    def render(self, mode='human'):
        """可视化环境状态"""
        pass  # 可以添加可视化代码

    def close(self):
        """关闭环境"""
        pass