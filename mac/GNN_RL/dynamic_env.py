import gymnasium as gym
import numpy as np
from gymnasium import spaces
import logging
import random
import math
from utils.util_function import euclidean_distance
from phy.large_scale_fading import maximum_communication_range


class DynamicStdmaEnv(gym.Env):
    """支持可变规模网络的STDMA环境"""

    # 修改 dynamic_env.py 中的 __init__ 方法
    def __init__(self, simulator=None, num_nodes=10, num_slots=None, seed=None, max_nodes=None):
        super().__init__()
        self.simulator = simulator
        self.num_nodes = num_nodes
        self.max_nodes = max_nodes if max_nodes is not None else max(30, num_nodes)
        self.base_num_slots = num_slots or num_nodes
        self.max_slots = self.base_num_slots * 2
        self.current_num_slots = self.base_num_slots
        self.max_comm_range = maximum_communication_range()

        # 初始化一个虚拟的 current_requirement
        from dataclasses import dataclass

        @dataclass
        class DummyRequirement:
            source_id: int = 0
            dest_id: int = 0
            routing_path: list = None
            delay_requirement: float = 100.0
            num_packets: int = 50
            qos_requirement: int = 1

        # 初始化 current_requirement
        self.current_requirement = DummyRequirement()

        # 创建随机数生成器
        self.np_random = np.random.RandomState(seed)

        # 动作空间和观察空间设置...
        self.action_space = spaces.Discrete(self.max_slots)
        self.max_obs_dim = self._calculate_obs_dim(self.max_nodes)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.max_obs_dim,),
            dtype=np.float32
        )

        # 初始化其他状态变量
        self.current_node = 0
        self.current_schedule = {}
        self.assigned_nodes = set()
        self.done = False
        self.retry_counts = {i: 0 for i in range(self.num_nodes)}
        self.episode_reward = 0
        self.last_reward = 0

        # 计算拓扑结构和初始位置
        if self.simulator:
            self.initial_positions = self.simulator.position[:self.num_nodes]
        else:
            # 随机生成位置
            self.initial_positions = self._generate_random_positions()

        # 计算拓扑矩阵
        self.topology_matrix = self._calculate_topology_matrix()

    def _calculate_obs_dim(self, n_nodes):
        """计算给定节点数量的观察空间维度"""
        topology_dim = n_nodes * n_nodes  # 拓扑矩阵
        position_dim = n_nodes * 3  # 3D位置
        routing_dim = n_nodes  # 路由路径
        link_lifetime_dim = n_nodes * n_nodes  # 链路生命期
        traffic_dim = 5  # 流量信息
        node_degrees_dim = n_nodes  # 节点度

        total_dim = (topology_dim + position_dim + routing_dim +
                     link_lifetime_dim + traffic_dim + node_degrees_dim)
        return total_dim

    def _pad_observation(self, obs, max_dim):
        """将观察填充到最大维度"""
        padded_obs = np.zeros(max_dim, dtype=np.float32)
        obs_len = min(len(obs), max_dim)  # 避免越界
        padded_obs[:obs_len] = obs[:obs_len]
        return padded_obs

    def seed(self, seed=None):
        """设置环境的随机种子"""
        # 明确使用RandomState对象而不是新版Generator
        self.np_random = np.random.RandomState(seed)
        return [seed]

    def reset(self, *, seed=None, options=None, requirement_data=None, num_nodes=None):
        """重置环境，可选择改变节点数量"""
        # 老式风格的seed处理
        if seed is not None:
            self.seed(seed)

        # 如果指定了新的节点数量，则更新
        if num_nodes is not None and num_nodes != self.num_nodes:
            self.num_nodes = num_nodes
            self.base_num_slots = num_nodes
            self.current_num_slots = self.base_num_slots

        # 重置状态变量
        self.current_node = 0
        self.current_schedule = {}
        self.assigned_nodes = set()
        self.done = False
        self.retry_counts = {i: 0 for i in range(self.num_nodes)}
        self.episode_reward = 0
        self.last_reward = 0

        # 重置路由路径完成标记
        if hasattr(self, 'max_slots_after_path'):
            delattr(self, 'max_slots_after_path')

        # 处理业务需求
        if requirement_data:
            self.current_requirement = requirement_data
        elif self.simulator:
            self._generate_requirement_from_simulator()
        else:
            self._generate_dummy_requirement()

        # 计算拓扑结构和初始位置
        if self.simulator:
            self.initial_positions = self.simulator.position[:self.num_nodes]
        else:
            # 随机生成位置
            self.initial_positions = self._generate_random_positions()

        # 计算拓扑矩阵
        self.topology_matrix = self._calculate_topology_matrix()

        # 获取路由路径
        if not hasattr(self.current_requirement, 'routing_path') or self.current_requirement.routing_path is None:
            self.current_requirement.routing_path = self._generate_routing_path()
            logging.info(f"Generated routing path: {self.current_requirement.routing_path}")

        # 设置初始节点为路由路径的起点
        if self.current_requirement.routing_path:
            self.current_node = int(self.current_requirement.routing_path[0])

        # 获取观察并填充到最大维度
        obs = self._get_observation()
        padded_obs = self._pad_observation(obs, self.max_obs_dim)

        return padded_obs, {}

    def step(self, action):
        """执行一步动作"""
        # 标准化动作
        if isinstance(action, np.ndarray):
            action = action.item()

        # 初始化返回值
        reward = 0
        terminated = False
        truncated = False
        info = {'schedule': self.current_schedule}

        # 检查是否为非路由节点且已完成路由路径分配
        is_non_route_node = (hasattr(self, 'max_slots_after_path') and
                             self.current_node not in self.current_requirement.routing_path)

        # 检查动作有效性
        if self._is_valid_assignment(self.current_node, action):
            # 更新时隙分配
            if action not in self.current_schedule:
                self.current_schedule[action] = []
            self.current_schedule[action].append(self.current_node)
            self.assigned_nodes.add(self.current_node)

            # 计算奖励
            reward = self._calculate_reward(action)

            # 更新状态
            self.current_node = self._get_next_node()
            self._remap_timeslots()  # 重新映射时隙
            print("当前时隙表：", self.current_schedule)

            # 检查是否完成所有分配
            if self.current_node is None:
                logging.info("All nodes assigned")
                terminated = True
        else:
            # 无效动作惩罚
            reward = -5
            if self.current_node is not None:
                self.retry_counts[self.current_node] += 1

                # 如果重试次数过多，强制分配
                if self.retry_counts[self.current_node] >= 4:
                    if is_non_route_node:
                        # 对于非路由节点，找到可以复用的时隙
                        assigned_slot = None
                        for slot in range(len(self.current_schedule)):
                            if self._is_valid_assignment(self.current_node, slot):
                                assigned_slot = slot
                                break

                        # 如果找到可复用时隙，则分配
                        if assigned_slot is not None:
                            self.current_schedule[assigned_slot].append(self.current_node)
                        else:
                            # 如果没有找到可复用时隙，分配到冲突最少的时隙
                            best_slot = self._find_least_conflict_slot(self.current_node)
                            if best_slot not in self.current_schedule:
                                self.current_schedule[best_slot] = []
                            self.current_schedule[best_slot].append(self.current_node)
                    else:
                        # 路由节点可以创建新时隙
                        new_slot = len(self.current_schedule)
                        self.current_schedule[new_slot] = [self.current_node]

                    self.assigned_nodes.add(self.current_node)
                    self.retry_counts[self.current_node] = 0
                    self.current_node = self._get_next_node()
                    self._remap_timeslots()

                if self.current_node is None:
                    logging.info("All nodes assigned")
                    terminated = True

        # 更新累积奖励
        self.episode_reward += reward
        self.last_reward = reward

        # 获取新的观察并填充
        observation = self._get_observation()
        padded_obs = self._pad_observation(observation, self.max_obs_dim)

        # 更新信息字典
        info.update({
            'schedule': self.current_schedule,
            'current_node': self.current_node,
            'assigned_nodes': list(self.assigned_nodes),
            'last_reward': reward,
            'episode_reward': self.episode_reward
        })
        return padded_obs, reward, terminated, truncated, info

    def _is_valid_assignment(self, node, action):
        """检查时隙分配是否有效"""
        if isinstance(action, np.ndarray):
            action = action.item()

        # 基本检查
        if action >= self.max_slots or node is None or node >= self.num_nodes:
            return False

        # 检查是否已完成路由路径分配且尝试创建新时隙
        if (hasattr(self, 'max_slots_after_path') and
                node not in self.current_requirement.routing_path and
                action >= len(self.current_schedule)):
            return False  # 非路由节点不允许创建新时隙

        # 计算两跳邻居
        two_hop_neighbors = set()
        # 一跳邻居
        for neighbor in range(self.num_nodes):
            if self.topology_matrix[node][neighbor] == 1:
                two_hop_neighbors.add(neighbor)
                # 二跳邻居
                for second_hop in range(self.num_nodes):
                    if self.topology_matrix[neighbor][second_hop] == 1 and second_hop != node:
                        two_hop_neighbors.add(second_hop)

        # 检查时隙内的节点
        if action in self.current_schedule:
            for existing_node in self.current_schedule[action]:
                if existing_node in two_hop_neighbors:
                    return False

        # 检查是否有更优的时隙选择
        for slot in range(action):
            if slot in self.current_schedule:
                can_reuse = True
                for existing_node in self.current_schedule[slot]:
                    if existing_node in two_hop_neighbors:
                        can_reuse = False
                        break
                if can_reuse and hasattr(self.current_requirement,
                                         'routing_path') and self.current_requirement.routing_path and self.current_node in self.current_requirement.routing_path:
                    index = self.current_requirement.routing_path.index(self.current_node)
                    # if index > 0:  # 只对路径中间节点强制最佳复用
                    #     return False

        return True

    def _find_least_conflict_slot(self, node):
        """找到冲突最少的时隙"""
        best_slot = 0
        min_conflicts = float('inf')

        # 计算两跳邻居
        two_hop_neighbors = set()
        for neighbor in range(self.num_nodes):
            if self.topology_matrix[node][neighbor] == 1:
                two_hop_neighbors.add(neighbor)
                for second_hop in range(self.num_nodes):
                    if self.topology_matrix[neighbor][second_hop] == 1 and second_hop != node:
                        two_hop_neighbors.add(second_hop)

        # 对每个现有时隙计算冲突数
        for slot in range(len(self.current_schedule)):
            conflicts = 0
            if slot in self.current_schedule:
                for existing_node in self.current_schedule[slot]:
                    if existing_node in two_hop_neighbors:
                        conflicts += 1

            if conflicts < min_conflicts:
                min_conflicts = conflicts
                best_slot = slot

        return best_slot

    def _generate_random_positions(self):
        """生成随机的节点位置"""
        positions = np.zeros((self.num_nodes, 3), dtype=np.float32)
        map_range = 500  # 仿真区域大小，与config.MAP_LENGTH保持一致

        for i in range(self.num_nodes):
            # 随机生成3D位置
            positions[i] = [
                self.np_random.uniform(0, map_range),  # x
                self.np_random.uniform(0, map_range),  # y
                self.np_random.uniform(0, map_range)  # z
            ]

        return positions

    def _calculate_topology_matrix(self):
        """计算拓扑连接矩阵"""
        matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if euclidean_distance(
                        self.initial_positions[i],
                        self.initial_positions[j]
                ) <= self.max_comm_range:
                    matrix[i][j] = matrix[j][i] = 1.0
        return matrix

    def _generate_dummy_requirement(self):
        """生成虚拟业务需求（用于测试）"""
        from dataclasses import dataclass

        @dataclass
        class DummyRequirement:
            source_id: int = 0
            dest_id: int = 0
            routing_path: list = None
            delay_requirement: float = 100.0
            num_packets: int = 50
            qos_requirement: int = 1

        # 随机选择源和目标节点
        source_id = self.np_random.randint(0, self.num_nodes)
        dest_id = self.np_random.randint(0, self.num_nodes)
        while dest_id == source_id:
            dest_id = self.np_random.randint(0, self.num_nodes)

        self.current_requirement = DummyRequirement(
            source_id=source_id,
            dest_id=dest_id,
            delay_requirement=self.np_random.uniform(50.0, 200.0),
            num_packets=self.np_random.randint(10, 100)
        )

    def _generate_requirement_from_simulator(self):
        """从模拟器生成业务需求"""
        pass  # 实际实现时补充此函数

    def _generate_routing_path(self):
        """生成路由路径"""
        if not hasattr(self.current_requirement, 'source_id') or not hasattr(self.current_requirement, 'dest_id'):
            return []

        source_id = self.current_requirement.source_id
        dest_id = self.current_requirement.dest_id

        # 使用Dijkstra算法找到最短路径
        def dijkstra(graph, start, end):
            distances = {node: float('infinity') for node in range(self.num_nodes)}
            distances[start] = 0
            unvisited = list(range(self.num_nodes))
            previous = {node: None for node in range(self.num_nodes)}

            while unvisited:
                current = min(unvisited, key=lambda node: distances[node])

                if current == end:
                    break

                if distances[current] == float('infinity'):
                    break

                unvisited.remove(current)

                for neighbor in range(self.num_nodes):
                    if graph[current][neighbor] == 1:  # 相邻节点
                        distance = distances[current] + 1
                        if distance < distances[neighbor]:
                            distances[neighbor] = distance
                            previous[neighbor] = current

            # 回溯路径
            path = []
            current = end
            while current is not None:
                path.append(current)
                current = previous[current]

            return path[::-1]  # 反转路径

        path = dijkstra(self.topology_matrix, source_id, dest_id)
        logging.info(f"Routing path: {path}")
        return path if path and path[0] == source_id else []

    def _remap_timeslots(self):
        """重新映射时隙编号"""
        used_slots = sorted(self.current_schedule.keys())
        if not used_slots:
            return

        slot_map = {old: new for new, old in enumerate(used_slots)}

        new_schedule = {}
        for old_slot, node_list in self.current_schedule.items():
            new_slot = slot_map[old_slot]
            new_schedule[new_slot] = node_list

        self.current_schedule = new_schedule

    def _get_next_node(self):
        """获取下一个要分配时隙的节点，先分配路由路径节点，再分配其他节点"""
        if not self.current_requirement or not self.current_requirement.routing_path:
            return None

        # 第一阶段：分配路由路径上的节点
        route_nodes = self.current_requirement.routing_path
        for node in route_nodes:
            if node not in self.assigned_nodes:
                return node

        # 第一阶段完成后，记录已使用的时隙数量
        if not hasattr(self, 'max_slots_after_path'):
            self.max_slots_after_path = len(self.current_schedule)
            logging.info(f"路由路径节点分配完成，固定时隙数量为: {self.max_slots_after_path}")

        # 第二阶段：分配非路由路径节点到现有时隙
        for node in range(self.num_nodes):
            if node not in self.assigned_nodes and node not in route_nodes:
                return node

        # 所有节点都已分配
        return None

    def _calculate_reward(self, action):
        """计算奖励"""
        reward = 0

        # 检查是否是路由路径节点
        is_route_node = False
        if self.current_requirement and self.current_requirement.routing_path:
            is_route_node = self.current_node in self.current_requirement.routing_path
            if not is_route_node:
                # 非路由节点基础奖励降低
                reward += 2
            else:
                # 路由节点基础奖励
                reward += 5

        # 时隙复用奖励
        if action in self.current_schedule and len(self.current_schedule[action]) > 1:
            # 时隙复用奖励与复用节点数成正比
            reuse_count = len(self.current_schedule[action]) - 1

            # 非路由节点的时隙复用给予更高奖励
            if not is_route_node:
                reward += 15 * reuse_count
            else:
                reward += 10 * reuse_count

        # 路由完成奖励
        if self.current_requirement and self.current_requirement.routing_path:
            if all(node in self.assigned_nodes for node in self.current_requirement.routing_path):
                reward += 20

                # 额外的复用效率奖励
                total_slots = len(self.current_schedule)
                total_assignments = sum(len(nodes) for nodes in self.current_schedule.values())
                efficiency = total_assignments / total_slots if total_slots > 0 else 0
                reward += 15 * max(0, efficiency - 1)

        return reward

    def _get_traffic_info(self):
        """获取业务信息"""
        if self.current_requirement:
            return np.array([
                float(self.current_requirement.source_id),
                float(self.current_requirement.dest_id),
                float(self.current_requirement.num_packets),
                float(self.current_requirement.delay_requirement),
                float(self.current_requirement.qos_requirement)
            ], dtype=np.float32)
        return np.zeros(5, dtype=np.float32)

    def _calculate_link_lifetime(self):
        """计算链路生命期"""
        lifetime = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if self.topology_matrix[i][j] == 1:
                    # 简化的链路生命期计算
                    distance = euclidean_distance(
                        self.initial_positions[i],
                        self.initial_positions[j]
                    )
                    lifetime[i][j] = lifetime[j][i] = 1.0 - (distance / self.max_comm_range)
        return lifetime

    def _get_routing_path_vector(self):
        """将路由路径转换为向量表示"""
        routing_path = np.full(self.num_nodes, -1, dtype=np.float32)

        if self.current_requirement and self.current_requirement.routing_path:
            path = self.current_requirement.routing_path
            for i, node in enumerate(path[:self.num_nodes]):
                routing_path[i] = float(node)

        return routing_path

    def _get_observation(self):
        """获取环境观察"""
        # 拓扑特征
        topology_features = np.asarray(self.topology_matrix, dtype=np.float32).flatten()

        # 节点位置
        node_positions = np.asarray(self.initial_positions, dtype=np.float32).flatten()

        # 路由路径
        routing_path = self._get_routing_path_vector().flatten()

        # 链路生命期
        link_lifetime = self._calculate_link_lifetime().flatten()

        # 业务信息
        traffic_info = self._get_traffic_info()

        # 节点度数
        node_degrees = np.sum(self.topology_matrix, axis=1, dtype=np.float32)

        # 合并所有特征
        observation = np.concatenate([
            topology_features,
            node_positions,
            routing_path,
            link_lifetime,
            traffic_info,
            node_degrees
        ])

        return observation