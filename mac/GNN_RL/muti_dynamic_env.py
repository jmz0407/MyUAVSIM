import gymnasium as gym
import numpy as np
from gymnasium import spaces
import logging
import random
import math
import traceback
from utils.util_function import euclidean_distance
from phy.large_scale_fading import maximum_communication_range


class MUtiDynamicStdmaEnv(gym.Env):
    """支持可变规模网络和多流的STDMA环境"""
    # 添加类变量标志表明支持多流
    supports_multiple_requirements = True
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
            is_active: bool = True
            flow_id: str = "flow_0_0"

        # 初始化 current_requirement 和活跃需求列表
        self.current_requirement = DummyRequirement()
        self.active_requirements = []
        self.all_routing_paths = []

        # 创建随机数生成器
        self.np_random = np.random.RandomState(seed)

        # 动作空间和观察空间设置
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
        self.node_priorities = {}  # 节点优先级字典

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

    def reset(self, *, seed=None, options=None, requirement_data=None, num_nodes=None, active_requirements=None):
        """重置环境，支持多个活跃流"""
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
        self.node_priorities = {}  # 重置节点优先级

        if active_requirements:
            # 如果传入的是列表，直接使用
            if isinstance(active_requirements, list):
                self.active_requirements = active_requirements
            # 如果传入的是单个需求，包装成列表
            elif hasattr(active_requirements, 'flow_id') or hasattr(active_requirements, 'source_id'):
                self.active_requirements = [active_requirements]
            else:
                # 未知类型，记录警告并初始化空列表
                logging.warning(f"收到未知类型的需求参数: {type(active_requirements)}")
                self.active_requirements = []
        elif requirement_data:
            # 兼容旧接口
            if isinstance(requirement_data, list):
                self.active_requirements = requirement_data
            else:
                self.active_requirements = [requirement_data]
        else:
            self.active_requirements = []
            self._generate_dummy_requirement()
            self.active_requirements = [self.current_requirement]

        # 如果没有活跃需求，生成一个虚拟需求
        if not self.active_requirements:
            self._generate_dummy_requirement()
            self.active_requirements = [self.current_requirement]

        # 更新当前需求为第一个活跃需求
        self.current_requirement = self.active_requirements[0]

        # 计算拓扑结构和初始位置
        if self.simulator:
            self.initial_positions = self.simulator.position[:self.num_nodes]
        else:
            # 随机生成位置
            self.initial_positions = self._generate_random_positions()

        # 计算拓扑矩阵
        self.topology_matrix = self._calculate_topology_matrix()

        # 收集所有活跃流的路由路径
        self.all_routing_paths = []
        for req in self.active_requirements:
            if hasattr(req, 'routing_path') and req.routing_path:
                # 确保每个路由路径都是完整的 - 如果没有路由路径则生成一个
                if not req.routing_path:
                    req.routing_path = self._generate_routing_path_for_requirement(req)

                # 记录路由路径
                self.all_routing_paths.append(req.routing_path)
                logging.info(
                    f"添加路由路径: {req.routing_path} 用于流 {req.flow_id if hasattr(req, 'flow_id') else 'unknown'}")

        # 计算节点优先级
        self._calculate_node_priorities()

        # 设置初始节点为需要分配的第一个节点
        self._set_next_node_to_assign()

        # 获取观察并填充到最大维度
        obs = self._get_observation()
        padded_obs = self._pad_observation(obs, self.max_obs_dim)

        return padded_obs, {}

    def _set_next_node_to_assign(self):
        """设置下一个需要分配时隙的节点"""
        # 首先从所有路由路径中收集所有需要分配的节点
        nodes_to_assign = set()
        for path in self.all_routing_paths:
            for node in path:
                if node not in self.assigned_nodes and 0 <= node < self.num_nodes:
                    nodes_to_assign.add(node)

        # 如果没有节点需要分配，返回None
        if not nodes_to_assign:
            self.current_node = None
            return

        # 根据优先级选择下一个节点
        highest_priority_node = None
        highest_priority = -float('inf')

        for node in nodes_to_assign:
            priority = self.node_priorities.get(node, 0)
            if priority > highest_priority:
                highest_priority = priority
                highest_priority_node = node

        if highest_priority_node is not None:
            self.current_node = highest_priority_node
            return

        # 如果没有优先级信息，使用原来的逻辑
        # 优先分配活跃流的源节点
        for req in self.active_requirements:
            src_id = req.source_id
            if src_id in nodes_to_assign:
                self.current_node = src_id
                return

        # 其次分配路由路径中的下一个节点
        for path in self.all_routing_paths:
            for node in path:
                if node in nodes_to_assign:
                    self.current_node = node
                    return

        # 如果上述都没有找到，选择任意一个未分配的节点
        self.current_node = next(iter(nodes_to_assign)) if nodes_to_assign else None

    def _calculate_node_priorities(self):
        """计算每个节点的优先级，用于时隙分配顺序"""
        self.node_priorities = {}

        # 计算每个节点在所有路径中出现的次数
        path_counts = {}
        for path in self.all_routing_paths:
            for node in path:
                if 0 <= node < self.num_nodes:
                    path_counts[node] = path_counts.get(node, 0) + 1

        # 计算路由关键度（节点是多少条路由的瓶颈）
        bottleneck_scores = {}
        for node in range(self.num_nodes):
            # 计算如果移除此节点，有多少条路径会受影响
            affected_paths = 0
            for path in self.all_routing_paths:
                if node in path:
                    affected_paths += 1

            bottleneck_scores[node] = affected_paths

        # 计算源节点和目标节点的加成
        endpoint_bonus = {}
        for req in self.active_requirements:
            if hasattr(req, 'source_id') and hasattr(req, 'dest_id'):
                # 源节点和目标节点获得额外优先级
                endpoint_bonus[req.source_id] = endpoint_bonus.get(req.source_id, 0) + 10
                endpoint_bonus[req.dest_id] = endpoint_bonus.get(req.dest_id, 0) + 5

        # 组合所有因素计算最终优先级
        for node in range(self.num_nodes):
            # 基础优先级
            priority = 0

            # 路径频率贡献
            priority += path_counts.get(node, 0) * 2

            # 瓶颈贡献
            priority += bottleneck_scores.get(node, 0) * 3

            # 端点贡献
            priority += endpoint_bonus.get(node, 0)

            # 已经分配的节点优先级降低
            if node in self.assigned_nodes:
                priority = -float('inf')

            self.node_priorities[node] = priority

    def _generate_routing_path_for_requirement(self, req):
        """为给定的需求生成路由路径"""
        source_id = req.source_id
        dest_id = req.dest_id

        # 使用Dijkstra算法找到最短路径
        path = self._find_shortest_path(source_id, dest_id)
        logging.info(f"为流 {req.flow_id if hasattr(req, 'flow_id') else f'{source_id}->{dest_id}'} 生成路径: {path}")
        return path

    def _find_shortest_path(self, source_id, dest_id):
        """找到从源节点到目标节点的最短路径"""
        # 确保节点ID有效
        if not (0 <= source_id < self.num_nodes and 0 <= dest_id < self.num_nodes):
            return []

        distances = {node: float('infinity') for node in range(self.num_nodes)}
        distances[source_id] = 0
        unvisited = list(range(self.num_nodes))
        previous = {node: None for node in range(self.num_nodes)}

        while unvisited:
            # 找到未访问节点中距离最小的
            current = min(unvisited, key=lambda node: distances[node])

            # 如果到达目标节点或者当前节点距离无限远（无法到达更多节点）
            if current == dest_id or distances[current] == float('infinity'):
                break

            unvisited.remove(current)

            # 更新邻居节点的距离
            for neighbor in range(self.num_nodes):
                if self.topology_matrix[current][neighbor] == 1:  # 相邻节点
                    # 计算通过当前节点到达neighbor的距离
                    distance = distances[current] + 1
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        previous[neighbor] = current

        # 回溯路径
        path = []
        current = dest_id
        while current is not None:
            path.append(current)
            current = previous[current]

        # 反转路径以获得从源到目标的顺序
        path = path[::-1]

        # 验证路径是否有效
        if not path or path[0] != source_id:
            return []

        return path

    def _original_step_logic(self, action):
        """执行一步动作，支持多个活跃流"""
        # 标准化动作
        if isinstance(action, np.ndarray):
            action = action.item()

        # 初始化返回值
        reward = 0
        terminated = False
        truncated = False
        info = {'schedule': self.current_schedule}

        # 检查当前节点是否有效
        if self.current_node is None:
            terminated = True
            info.update({
                'message': '所有节点已分配',
                'current_node': None,
                'assigned_nodes': list(self.assigned_nodes),
                'last_reward': 0,
                'episode_reward': self.episode_reward
            })
            return self._get_padded_observation(), 0, terminated, truncated, info

        # 检查动作有效性
        if self._is_valid_assignment(self.current_node, action):
            # 更新时隙分配
            if action not in self.current_schedule:
                self.current_schedule[action] = []
            self.current_schedule[action].append(self.current_node)
            self.assigned_nodes.add(self.current_node)

            # 计算奖励
            reward = self._calculate_reward(action)

            # 更新节点优先级
            self._calculate_node_priorities()

            # 更新状态
            self._set_next_node_to_assign()
            self._remap_timeslots()  # 重新映射时隙

            # 记录日志
            logging.info(f"节点 {self.current_node} 分配到时隙 {action}")
            logging.info(f"当前时隙表: {self.current_schedule}")

            # 检查是否完成所有分配
            if self.current_node is None:
                terminated = True

                # 额外奖励：如果所有活跃流的路由路径都已完全分配
                all_paths_assigned = True
                for path in self.all_routing_paths:
                    if not all(node in self.assigned_nodes for node in path):
                        all_paths_assigned = False
                        break

                if all_paths_assigned:
                    reward += 50  # 完成所有路由路径的额外奖励
        else:
            # 无效动作惩罚
            reward = -5

            if self.current_node is not None:
                self.retry_counts[self.current_node] += 1

                # 如果重试次数过多，强制分配
                if self.retry_counts[self.current_node] >= 3:
                    new_slot = len(self.current_schedule)
                    if new_slot not in self.current_schedule:
                        self.current_schedule[new_slot] = []
                    self.current_schedule[new_slot].append(self.current_node)
                    self.assigned_nodes.add(self.current_node)
                    self.retry_counts[self.current_node] = 0

                    # 更新节点优先级
                    self._calculate_node_priorities()

                    self._set_next_node_to_assign()
                    self._remap_timeslots()
                    if self.current_node is None:
                        terminated = True

        # 更新累积奖励
        self.episode_reward += reward
        self.last_reward = reward

        # 获取新的观察并填充
        observation = self._get_padded_observation()

        # 更新信息字典
        info.update({
            'schedule': self.current_schedule,
            'current_node': self.current_node,
            'assigned_nodes': list(self.assigned_nodes),
            'last_reward': reward,
            'episode_reward': self.episode_reward
        })

        return observation, reward, terminated, truncated, info

    def step(self, action):
        """执行一步动作，支持多个活跃流"""
        # 执行现有的动作处理逻辑
        observation, reward, terminated, truncated, info = self._original_step_logic(action)

        # 如果已经终止，验证时隙分配的完整性
        if terminated:
            # 检查是否所有路径节点都已分配
            is_complete = self.verify_schedule_completeness()

            if not is_complete:
                # 如果分配不完整，给予惩罚并提供信息
                reward -= 50  # 显著的惩罚
                info['schedule_complete'] = False

                # 记录缺失的节点
                info['missing_nodes'] = self.get_missing_nodes()

                # 可以选择不终止，允许继续分配
                if self.allow_continue_on_incomplete:
                    terminated = False
                    # 重新设置下一个节点为缺失节点之一
                    self._set_next_node_to_missing()
            else:
                # 如果分配完整，给予额外奖励
                reward += 20
                info['schedule_complete'] = True

        return observation, reward, terminated, truncated, info

    def verify_schedule_completeness(self):
        """验证时隙分配的完整性"""
        # 获取所有已分配节点
        assigned_nodes = set(self.assigned_nodes)

        # 获取应该被分配的节点
        expected_nodes = set()
        for path in self.all_routing_paths:
            for node in path:
                if 0 <= node < self.num_nodes:  # 确保节点ID有效
                    expected_nodes.add(node)

        # 检查是否有遗漏
        missing_nodes = expected_nodes - assigned_nodes
        return len(missing_nodes) == 0

    def get_missing_nodes(self):
        """获取缺失分配的节点"""
        assigned_nodes = set(self.assigned_nodes)

        expected_nodes = set()
        for path in self.all_routing_paths:
            for node in path:
                if 0 <= node < self.num_nodes:  # 确保节点ID有效
                    expected_nodes.add(node)

        return list(expected_nodes - assigned_nodes)

    def _set_next_node_to_missing(self):
        """设置下一个节点为缺失节点之一"""
        missing_nodes = self.get_missing_nodes()
        if missing_nodes:
            self.current_node = missing_nodes[0]
            # 重置此节点的重试计数
            self.retry_counts[self.current_node] = 0
    def _get_padded_observation(self):
        """获取填充后的观察"""
        observation = self._get_observation()
        return self._pad_observation(observation, self.max_obs_dim)

    def _is_valid_assignment(self, node, action):
        """检查时隙分配是否有效"""
        if isinstance(action, np.ndarray):
            action = action.item()

        # 基本检查
        if action >= self.max_slots or node is None or node >= self.num_nodes or action < 0:
            return False

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

        # 检查时隙内的节点是否会干扰
        if action in self.current_schedule:
            for existing_node in self.current_schedule[action]:
                # 检查是否是同一节点
                if existing_node == node:
                    return False

                # 检查是否在干扰范围内
                if existing_node in two_hop_neighbors:
                    return False

        # 检查是否有更优的时隙选择（可选，根据需要启用）
        # 对关键路径上的节点禁用此检查以提高效率
        is_critical_node = False
        for path in self.all_routing_paths:
            if node in path and len(path) <= 3:  # 路径短的节点视为关键节点
                is_critical_node = True
                break

        if not is_critical_node:
            for slot in range(action):
                if slot in self.current_schedule:
                    can_reuse = True
                    for existing_node in self.current_schedule[slot]:
                        # 检查是否是同一节点
                        if existing_node == node:
                            can_reuse = False
                            break

                        # 检查是否在干扰范围内
                        if existing_node in two_hop_neighbors:
                            can_reuse = False
                            break

                    if can_reuse:
                        # 找到更好的时隙，当前分配无效
                        return False

        return True

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

        # 确保我们有有效的位置
        if not hasattr(self, 'initial_positions') or self.initial_positions is None:
            # 如果没有位置信息，返回空矩阵
            return matrix

        if len(self.initial_positions) < self.num_nodes:
            # 如果位置不足，只计算可用的部分
            actual_nodes = len(self.initial_positions)
            logging.warning(f"位置信息不足: 需要 {self.num_nodes} 个节点位置，但只有 {actual_nodes} 个")
        else:
            actual_nodes = self.num_nodes

        try:
            for i in range(actual_nodes):
                for j in range(i + 1, actual_nodes):
                    if euclidean_distance(
                            self.initial_positions[i],
                            self.initial_positions[j]
                    ) <= self.max_comm_range:
                        matrix[i][j] = matrix[j][i] = 1.0
        except Exception as e:
            logging.error(f"计算拓扑矩阵时出错: {e}")
            traceback.print_exc()

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
            is_active: bool = True
            flow_id: str = "flow_0_0"

        # 随机选择源和目标节点
        source_id = self.np_random.randint(0, self.num_nodes)
        dest_id = self.np_random.randint(0, self.num_nodes)
        while dest_id == source_id and self.num_nodes > 1:
            dest_id = self.np_random.randint(0, self.num_nodes)

        dummy_req = DummyRequirement(
            source_id=source_id,
            dest_id=dest_id,
            delay_requirement=self.np_random.uniform(50.0, 200.0),
            num_packets=self.np_random.randint(10, 100),
            flow_id=f"flow_{source_id}_{dest_id}"
        )

        # 生成路由路径
        dummy_req.routing_path = self._find_shortest_path(source_id, dest_id)

        self.current_requirement = dummy_req
        return dummy_req

    def _remap_timeslots(self):
        """重新映射时隙编号，保持连续"""
        used_slots = sorted(self.current_schedule.keys())
        if not used_slots:
            return

        # 创建旧时隙到新时隙的映射
        slot_map = {old: new for new, old in enumerate(used_slots)}

        # 使用映射创建新的时隙表
        new_schedule = {}
        for old_slot, node_list in self.current_schedule.items():
            new_slot = slot_map[old_slot]
            new_schedule[new_slot] = node_list

        # 更新时隙表
        self.current_schedule = new_schedule

    def _calculate_reward(self, action):
        """计算奖励，考虑所有活跃流"""
        reward = 0

        # 基本奖励
        reward += 5

        # 时隙复用奖励
        if action in self.current_schedule and len(self.current_schedule[action]) > 1:
            # 时隙复用奖励与复用节点数成正比
            reuse_count = len(self.current_schedule[action]) - 1
            reward += 10 * reuse_count

        # 路由路径覆盖奖励
        if self.current_node is not None:
            # 检查当前节点是否在任何活跃流的路由路径上
            in_active_path = False
            for path in self.all_routing_paths:
                if self.current_node in path:
                    in_active_path = True
                    break

            if in_active_path:
                reward += 10  # 为活跃路径上的节点分配时隙

            # 检查是否有完整的路由路径被覆盖
            for path in self.all_routing_paths:
                if all(node in self.assigned_nodes for node in path):
                    reward += 20  # 完整路径奖励

        # 时隙效率奖励
        total_slots = len(self.current_schedule)
        total_assignments = sum(len(nodes) for nodes in self.current_schedule.values())
        efficiency = total_assignments / total_slots if total_slots > 0 else 0
        reward += 10 * max(0, efficiency - 1)  # 每个时隙平均分配超过1个节点的奖励

        # 节点优先级奖励
        if self.current_node is not None and self.current_node in self.node_priorities:
            # 根据节点优先级调整奖励
            node_priority = max(0, self.node_priorities[self.current_node])
            priority_reward = node_priority * 0.1  # 缩放因子可以调整
            reward += priority_reward

        return reward

    def _get_observation(self):
        """获取环境观察，包含多流信息"""
        try:
            # 基础拓扑特征
            topology_features = np.asarray(self.topology_matrix, dtype=np.float32).flatten()

            # 节点位置
            node_positions = np.asarray(self.initial_positions, dtype=np.float32).flatten()

            # 路由路径 - 现在考虑所有活跃流的路径
            routing_path = self._get_combined_routing_paths()

            # 链路生命期
            link_lifetime = self._calculate_link_lifetime().flatten()

            # 业务信息 - 结合所有活跃流
            traffic_info = self._get_combined_traffic_info()

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
        except Exception as e:
            logging.error(f"获取观察时出错: {e}")
            traceback.print_exc()
            # 返回默认观察
            return np.zeros(self.max_obs_dim, dtype=np.float32)

    def _get_combined_routing_paths(self):
        """合并所有活跃流的路由路径"""
        # 创建一个大小为节点数的向量，记录每个节点在多少条路径中出现
        routing_vector = np.zeros(self.num_nodes, dtype=np.float32)

        for path in self.all_routing_paths:
            for node in path:
                if 0 <= node < self.num_nodes:  # 确保节点ID有效
                    routing_vector[node] += 1

        return routing_vector

    def _get_combined_traffic_info(self):
        """合并所有活跃流的业务信息"""
        if not self.active_requirements:
            return np.zeros(5, dtype=np.float32)

        # 初始化合并特征
        num_flows = len(self.active_requirements)
        total_packets = 0
        avg_delay_req = 0
        max_qos_req = 0

        # 收集所有流的信息
        for req in self.active_requirements:
            total_packets += req.num_packets
            avg_delay_req += req.delay_requirement
            max_qos_req = max(max_qos_req, req.qos_requirement)

        # 计算平均值
        avg_delay_req = avg_delay_req / num_flows if num_flows > 0 else 0

        # 组合信息
        return np.array([
            float(num_flows),  # 活跃流数量
            float(total_packets),  # 总数据包数
            float(avg_delay_req),  # 平均延迟要求
            float(max_qos_req),  # 最高QoS要求
            float(len(self.assigned_nodes))  # 已分配节点数
        ], dtype=np.float32)

    def _calculate_link_lifetime(self):
        """计算链路生命期（考虑距离因素）"""
        lifetime = np.zeros((self.num_nodes, self.num_nodes))

        # 如果没有位置信息，返回零矩阵
        if not hasattr(self, 'initial_positions') or self.initial_positions is None:
            return lifetime

        if len(self.initial_positions) < self.num_nodes:
            actual_nodes = len(self.initial_positions)
        else:
            actual_nodes = self.num_nodes

        try:
            for i in range(actual_nodes):
                for j in range(i + 1, actual_nodes):
                    if self.topology_matrix[i][j] == 1:
                        # 计算距离
                        distance = euclidean_distance(
                            self.initial_positions[i],
                            self.initial_positions[j]
                        )
                        # 链路生命期与距离成反比
                        lifetime_value = 1.0 - (distance / self.max_comm_range)
                        lifetime[i][j] = lifetime[j][i] = max(0.1, lifetime_value)
        except Exception as e:
            logging.error(f"计算链路生命期时出错: {e}")
            traceback.print_exc()

        return lifetime

    def render(self, mode='human'):
        """渲染环境状态（用于可视化和调试）"""
        if mode != 'human':
            return

        # 打印当前状态
        print("\n============= 环境状态 =============")
        print(f"当前节点: {self.current_node}")
        print(f"已分配节点: {self.assigned_nodes}")
        print(f"当前时隙表: {self.current_schedule}")

        # 打印活跃流信息
        print("\n活跃流数量:", len(self.active_requirements))
        for i, req in enumerate(self.active_requirements):
            flow_id = req.flow_id if hasattr(req, 'flow_id') else f"flow_{i}"
            path = req.routing_path if hasattr(req, 'routing_path') else "无路径"
            print(f"流 {flow_id}: {req.source_id} -> {req.dest_id}, 路径: {path}")

        # 打印时隙分配可视化
        print("\n时隙分配:")
        for slot, nodes in sorted(self.current_schedule.items()):
            nodes_str = ", ".join(map(str, nodes))
            print(f"时隙 {slot}: [{nodes_str}]")

        print("=====================================\n")

    def close(self):
        """关闭环境并释放资源"""
        pass  # 目前不需要特殊的关闭操作

    def get_current_schedule(self):
        """获取当前时隙分配表"""
        return self.current_schedule.copy()

    def get_routing_paths(self):
        """获取所有活跃流的路由路径"""
        return [path.copy() for path in self.all_routing_paths]

    def analyze_schedule_efficiency(self):
        """分析当前时隙分配的效率"""
        if not self.current_schedule:
            return {
                'slot_count': 0,
                'node_count': 0,
                'reuse_factor': 0,
                'coverage': 0,
                'message': '无时隙分配'
            }

        total_slots = len(self.current_schedule)
        total_assignments = sum(len(nodes) for nodes in self.current_schedule.values())
        reuse_factor = total_assignments / total_slots if total_slots > 0 else 0

        # 计算路由路径覆盖率
        covered_paths = 0
        for path in self.all_routing_paths:
            if all(node in self.assigned_nodes for node in path):
                covered_paths += 1

        path_coverage = covered_paths / len(self.all_routing_paths) if self.all_routing_paths else 0

        return {
            'slot_count': total_slots,
            'node_count': len(self.assigned_nodes),
            'assigned_nodes': list(self.assigned_nodes),
            'total_assignments': total_assignments,
            'reuse_factor': reuse_factor,
            'path_coverage': path_coverage,
            'covered_paths': covered_paths,
            'total_paths': len(self.all_routing_paths)
        }