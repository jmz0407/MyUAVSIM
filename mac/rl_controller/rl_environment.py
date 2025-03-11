import logging

from routing.opar.opar import link_lifetime_predictor
from mobility import start_coords
from routing.opar.opar import Opar
from routing.opar.new_opar import NewOpar
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from utils import config
from utils.util_function import euclidean_distance
from phy.large_scale_fading import maximum_communication_range
import traceback
import copy
from simulator.TrafficGenerator import TrafficRequirement
from routing.opar.last_opar import LastOpar
from routing.multipath.amlb_opar import AMLB_OPAR
class StdmaEnv(gym.Env):
    def __init__(self, simulator, num_nodes=10, num_slots=10):
        super().__init__()
        self.simulator = simulator
        self.num_nodes = num_nodes
        self.base_num_slots = num_slots
        self.max_slots = num_slots
        self.current_num_slots = num_slots
        self.max_comm_range = maximum_communication_range()

        self.done = False

        # 动作空间
        self.action_space = spaces.Discrete(self.max_slots)

        # 计算观察空间维度
        topology_dim = num_nodes * num_nodes
        position_dim = num_nodes * 3
        routing_dim = num_nodes
        link_lifetime_dim = num_nodes * num_nodes
        traffic_dim = 5
        node_degrees_dim = num_nodes

        total_dim = (topology_dim + position_dim + routing_dim +
                     link_lifetime_dim + traffic_dim + node_degrees_dim)

        # 观察空间
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_dim,),
            dtype=np.float32
        )

        # 记录维度信息
        self.dims = {
            'topology': topology_dim,
            'position': position_dim,
            'routing': routing_dim,
            'link_lifetime': link_lifetime_dim,
            'traffic': traffic_dim,
            'node_degrees': node_degrees_dim
        }

        self.current_node = 0
        self.current_schedule = {}
        self.current_requirement = None
        self.assigned_nodes = set()

        # 添加奖励相关的初始化
        self.episode_reward = 0  # 当前episode的累积奖励
        self.last_reward = 0  # 上一步的奖励
        self.total_reward = 0  # 总体累积奖励
        self.episode_length = 0
        # 初始化重试计数器
        self.retry_counts = {}
        for i in range(num_nodes):
            self.retry_counts[i] = 0
        # 初始化拓扑矩阵
        if self.simulator:
            self.initial_positions = simulator.position
            self.topology_matrix = self._calculate_topology_matrix()
        else:
            self.initial_positions = np.zeros((num_nodes, 3))
            self.topology_matrix = np.zeros((num_nodes, num_nodes))

    def remap_timeslots(self):
        """
        对 self.current_schedule 中用到的时隙做一次重新编号。
        比如原先用到时隙 3, 7, 1, 9，会被重新映射成 0, 1, 2, 3。
        """
        # 1. 收集并排序所有已用到的时隙编号
        used_slots = sorted(self.current_schedule.keys())

        # 2. 建立旧时隙 -> 新时隙的映射
        slot_map = {}
        new_slot_index = 0
        for old_slot in used_slots:
            slot_map[old_slot] = new_slot_index
            new_slot_index += 1

        # 3. 根据映射生成新的调度表
        new_schedule = {}
        for old_slot, node_list in self.current_schedule.items():
            new_slot = slot_map[old_slot]
            new_schedule[new_slot] = node_list  # 直接用原先节点列表即可

        # 4. 替换旧的调度
        self.current_schedule = new_schedule
        # print(f"重新映射后的时隙调度: {self.current_schedule}")

    def step(self, action):
        """执行一步动作"""
        # print(f"Step执行: Node {self.current_node}, Action {action}")

        # 将numpy.ndarray转换为Python标量
        if isinstance(action, np.ndarray):
            action = action.item()  # 转换为Python标量

        # 初始化返回值
        terminated = False
        truncated = False
        info = {'schedule': self.current_schedule}

        try:
            # 检查动作有效性
            if self._is_valid_assignment(self.current_node, action):
                # print(f"动作有效，分配节点 {self.current_node} 到时隙 {action}")

                # 更新时隙分配
                if action not in self.current_schedule:
                    self.current_schedule[action] = []
                self.current_schedule[action].append(self.current_node)
                self.assigned_nodes.add(self.current_node)

                # 计算奖励
                reward = self._calculate_reward(action)

                # 更新状态
                self.current_node = self._get_next_node()
                # print(f"下一个节点: {self.current_node}")
                self.remap_timeslots()
                # 检查是否所有节点都已分配
                if self.current_node is None:
                    terminated = True
                    # print("所有节点都已分配完毕")
            else:
                # print(f"动作无效: 节点 {self.current_node} 不能分配到时隙 {action}")
                self.retry_counts[self.current_node] += 1

                # 如果尝试次数超过阈值，强制分配新时隙
                if self.retry_counts[self.current_node] >= 1:  # 这里的5是阈值，可以调整
                    # print(f"节点 {self.current_node} 尝试次数过多，分配新时隙")
                    new_slot = len(self.current_schedule)
                    self.current_schedule[new_slot] = [self.current_node]
                    self.assigned_nodes.add(self.current_node)
                    self.retry_counts[self.current_node] = 0  # 重置尝试次数
                    reward = 0  # 给予中性奖励
                    self.current_node = self._get_next_node()
                    self.remap_timeslots()
                    if self.current_node is None:
                        terminated = True
                else:
                    reward = -5

            # 获取新的观察
            observation = self._get_observation()

            info['schedule'] = self.current_schedule
            info['current_node'] = self.current_node
            info['assigned_nodes'] = list(self.assigned_nodes)

            # print(f"当前调度状态: {self.current_schedule}")
            return (observation, reward, terminated, truncated, info)
            # return (observation, reward, terminated, truncated, info, self.current_schedule)


        except Exception as e:
            print(f"Step执行出错: {str(e)}")
            traceback.print_exc()
            return self._get_observation(), -10, True, False, info

    # def _is_valid_assignment(self, node, action):
    #     """检查时隙分配是否有效"""
    #     # 确保操作的是Python标量
    #     if isinstance(action, np.ndarray):
    #         action = action.item()
    #
    #     print(f"检查时隙分配: 节点 {node}, 时隙 {action}")
    #
    #     # 1. 检查时隙范围
    #     if action >= self.current_num_slots:
    #         print(f"时隙 {action} 超出范围 ({self.current_num_slots})")
    #         return False
    #
    #     # 2. 检查节点有效性
    #     if node is None or node >= self.num_nodes:
    #         print(f"节点 {node} 无效")
    #         return False
    #
    #     # 3. 检查干扰
    #     if action in self.current_schedule:
    #         for other_node in self.current_schedule[action]:
    #             if self.topology_matrix[node][other_node] == 1:
    #                 print(f"存在干扰: 节点 {node} 和节点 {other_node}")
    #                 return False
    #
    #     print(f"时隙分配有效")
    #     return True

    # def _is_valid_assignment(self, node, action):
    #     """检查时隙分配是否有效，允许时隙复用"""
    #     if isinstance(action, np.ndarray):
    #         action = action.item()
    #
    #     print(f"检查时隙分配: 节点 {node}, 时隙 {action}")
    #
    #     # 1. 检查时隙范围
    #     if action >= self.current_num_slots:
    #         print(f"时隙 {action} 超出范围 ({self.current_num_slots})")
    #         return False
    #
    #     # 2. 检查节点有效性
    #     if node is None or node >= self.num_nodes:
    #         print(f"节点 {node} 无效")
    #         return False
    #
    #     # 3. 检查干扰和空间复用
    #     if action in self.current_schedule:
    #         # 遍历同一时隙的所有已分配节点
    #         for other_node in self.current_schedule[action]:
    #             # 检查是否存在直接连接（干扰）
    #             if self.topology_matrix[node][other_node] == 1:
    #                 print(f"存在干扰: 节点 {node} 和节点 {other_node}")
    #                 return False
    #
    #             # 检查二阶邻居（可选）
    #             second_order_neighbors = set()
    #             for n, connected in enumerate(self.topology_matrix[other_node]):
    #                 if connected == 1:
    #                     second_order_neighbors.update(
    #                         n2 for n2, c2 in enumerate(self.topology_matrix[n])
    #                         if c2 == 1
    #                     )
    #             if node in second_order_neighbors:
    #                 print(f"存在二阶干扰: 节点 {node} 和节点 {other_node}")
    #                 return False
    #
    #     print(f"时隙分配有效")
    #     return True
    def reset(self, seed=None, options=None, requirement_data=None):
        super().reset(seed=seed)
        """重置环境"""
        # print("重置环境...")  # 调试信息
        # 重置奖励相关变量
        self.episode_reward = 0
        self.last_reward = 0
        self.current_node = 0
        self.current_schedule = {}
        self.assigned_nodes = set()
        self.current_num_slots = self.base_num_slots
        self.done = False
        # 重置重试计数器
        self.retry_counts = {}
        for i in range(self.num_nodes):
            self.retry_counts[i] = 0
        if requirement_data:
            # copy_requirement_data = copy.deepcopy(requirement_data)
            self.current_requirement = requirement_data
            self.current_requirement.source_id = requirement_data.source_id
            self.current_requirement.dest_id = requirement_data.dest_id
            self.current_requirement.num_packets = requirement_data.num_packets
            self.current_requirement.delay_requirement = np.random.uniform(2000, 3000.0)  # 单位：微秒
            self.current_requirement.qos_requirement = np.random.choice([0, 1, 2])  # 示例QoS等级
            # self.current_requirement.routing_path = self._get_routing_path()

        elif self.simulator:
            # 使用模拟器生成实际业务需求
            source_id = np.random.randint(0, self.num_nodes)
            dest_id = np.random.randint(0, self.num_nodes)
            # source_id = 4
            # dest_id = 5
            while dest_id == source_id:
                dest_id = np.random.randint(0, self.num_nodes)
            num_packets = np.random.randint(10, 100)
            delay_req = np.random.uniform(2000, 3000.0)  # 单位：微秒
            qos_req = np.random.choice([0, 1, 2])  # 示例QoS等级

            # 调用模拟器的业务需求生成方法
            self.simulator.generate_traffic_requirement(
                source_id=source_id,
                dest_id=dest_id,
                num_packets=num_packets,
                delay_req=delay_req,
                qos_req=qos_req
            )
            # 确保需求传递到环境
            self.current_requirement = self.simulator.drones[source_id].transmitting_queue.get()
            self.current_requirement.routing_path = self._get_routing_path()
        else:
            # 训练时生成虚拟需求
            from dataclasses import dataclass
            @dataclass
            class DummyRequirement:
                source_id: int = 0
                dest_id: int = np.random.randint(0, self.num_nodes)
                routing_path: list = None
                delay_requirement: float = np.random.uniform(50.0, 200.0)
                num_packets: int = np.random.randint(1, 10)
                qos_requirement: int = np.random.choice([0, 1, 2])

            self.current_requirement = DummyRequirement()
            # 生成随机路由路径（示例）
            self.current_requirement.routing_path = self._generate_dummy_routing_path()

        # print(f"新业务需求: 源节点 {self.current_requirement.source_id} -> 目标节点 {self.current_requirement.dest_id}, 业务路径{self.current_requirement.routing_path}")
        # 设置初始节点为路由路径的起点
        if self.current_requirement and self.current_requirement.routing_path is not None:
            for node in self.current_requirement.routing_path:
                    self.current_node = int(node)
                    break
        else:
            self.current_node = None

        # print(f
        #     f"新业务需求: 源节点 {self.current_requirement.source_id} -> 目标节点 {self.current_requirement.dest_id}, 业务路径{self.current_requirement.routing_path}")
        obs = self._get_observation()
        # print(f"环境已重置，初始节点: {self.current_node}")  # 调试信息
        return obs, {}

    def _check_schedule_quality(self):
        """检查当前调度方案的质量"""
        if not self.current_requirement:
            return True

        try:
            # 1. 检查是否所有节点都已分配
            if len(self.assigned_nodes) < self.num_nodes:
                return False

            # 2. 检查路由顺序
            # if self.current_requirement.routing_path:
            #     path = self.current_requirement.routing_path
            #     for i in range(len(path) - 1):
            #         slot1 = next((slot for slot, nodes in self.current_schedule.items()
            #                       if path[i] in nodes), None)
            #         slot2 = next((slot for slot, nodes in self.current_schedule.items()
            #                       if path[i + 1] in nodes), None)
            #
            #         if slot1 is None or slot2 is None or slot1 > slot2:
            #             return False

            # 3. 检查时延要求
            current_delay = self._estimate_delay()
            if current_delay > self.current_requirement.delay_requirement:
                return False

            # 4. 检查干扰约束
            for slot, nodes in self.current_schedule.items():
                for i, node1 in enumerate(nodes):
                    for node2 in nodes[i + 1:]:
                        if self.topology_matrix[node1][node2] == 1:
                            return False

            return True

        except Exception as e:
            print(f"Error in _check_schedule_quality: {str(e)}")
            return False

    # only 复用
    def _get_next_node(self):
        """使用图着色方法获取下一个要分配时隙的节点"""
        try:
            # 1. 获取路由路径上的节点
            if self.current_requirement and self.current_requirement.routing_path:
                route_nodes = self.current_requirement.routing_path
            else:
                return None  # 如果没有路由路径，返回None

            # 2. 创建一个用于记录哪些节点已经分配时隙的集合
            assigned_nodes = set()  # 跟踪已分配时隙的节点

            # 遍历当前时隙分配情况，记录已分配的节点
            for slot, nodes in self.current_schedule.items():
                assigned_nodes.update(nodes)  # 添加已分配的节点到集合中
            # print(f"已分配的时隙：{self.current_schedule}")
            # print(f"已分配的节点：{assigned_nodes}")

            # 3. 确保按照路径顺序为每个节点分配时隙
            for node in route_nodes:
                if node in assigned_nodes:
                    # 如果节点已经分配时隙，跳过
                    # print(f"节点 {node} 已经分配过时隙，跳过")
                    continue

                # 4. 寻找可以复用的时隙
                slot_assigned = False
                for slot in range(self.current_num_slots):
                    if node not in self.current_schedule.get(slot, []):
                        # 检查该节点与当前时隙中的其他节点是否存在干扰
                        conflict = False
                        for neighbor in self.current_schedule.get(slot, []):
                            if self._is_conflict(node, neighbor):  # 判断是否有干扰
                                conflict = True
                                break

                        if not conflict:
                            # 如果没有冲突，则将节点分配到该时隙
                            if slot not in self.current_schedule:
                                self.current_schedule[slot] = []  # 初始化时隙
                            self.current_schedule[slot].append(node)
                            assigned_nodes.add(node)  # 将节点标记为已分配
                            # print(f"节点 {node} 复用时隙 {slot}")
                            slot_assigned = True
                            break  # 跳出时隙循环，继续为下一个节点分配

                if not slot_assigned:
                    # 如果找不到合适的复用时隙，为节点分配一个新的时隙
                    new_slot = len(self.current_schedule)  # 新的时隙索引
                    self.current_schedule[new_slot] = [node]
                    assigned_nodes.add(node)  # 将节点标记为已分配
                    # print(f"节点 {node} 被分配到新时隙 {new_slot}")

            # 如果没有满足条件的节点可分配时隙，则返回None
            return None

        except Exception as e:
            print(f"Error in _get_next_node: {str(e)}")
            traceback.print_exc()
            return None

    # 会重用时隙
    # def _get_next_node(self):
    #     """使用图着色方法获取下一个要分配时隙的节点"""
    #     try:
    #         # 1. 获取路由路径上的节点
    #         if self.current_requirement and self.current_requirement.routing_path:
    #             route_nodes = self.current_requirement.routing_path
    #         else:
    #             return None  # 如果没有路由路径，返回None
    #
    #         # 2. 创建一个用于记录哪些节点已经分配时隙的集合
    #         assigned_nodes = set()  # 跟踪已分配时隙的节点
    #
    #         # 遍历当前时隙分配情况，记录已分配的节点
    #         for slot, nodes in self.current_schedule.items():
    #             assigned_nodes.update(nodes)  # 添加已分配的节点到集合中
    #
    #         # 3. 确保按照路径顺序为每个节点分配时隙
    #         for node in route_nodes:
    #             if node in assigned_nodes:
    #                 # 如果节点已经分配时隙，跳过
    #                 continue
    #
    #             # 4. 寻找可以复用的时隙
    #             slot_assigned = False
    #             for slot in range(self.current_num_slots):
    #                 if node not in self.current_schedule.get(slot, []):
    #                     # 检查该节点与当前时隙中的其他节点是否存在干扰
    #                     conflict = False
    #                     for neighbor in self.current_schedule.get(slot, []):
    #                         if self._is_conflict(node, neighbor):  # 判断是否有干扰
    #                             conflict = True
    #                             break
    #
    #                     if not conflict:
    #                         # 如果没有冲突，则将节点分配到该时隙
    #                         if slot not in self.current_schedule:
    #                             self.current_schedule[slot] = []  # 初始化时隙
    #                         self.current_schedule[slot].append(node)
    #                         assigned_nodes.add(node)  # 将节点标记为已分配
    #                         # print(f"节点 {node} 复用时隙 {slot}")
    #                         slot_assigned = True
    #                         break
    #
    #             if not slot_assigned:
    #                 # 如果找不到合适的复用时隙，为节点分配一个新的时隙
    #                 new_slot = len(self.current_schedule)  # 新的时隙索引
    #                 self.current_schedule[new_slot] = [node]
    #                 assigned_nodes.add(node)  # 将节点标记为已分配
    #                 # print(f"节点 {node} 被分配到新时隙 {new_slot}")
    #
    #             return node  # 返回已分配时隙的节点
    #         # # 2. 为其他未分配节点分配时隙
    #         # for node in range(self.num_nodes):
    #         #     if node not in self.assigned_nodes:
    #         #         return node
    #         # 如果没有满足条件的节点可分配时隙，则返回None
    #         return None
    #
    #     except Exception as e:
    #         print(f"Error in _get_next_node: {str(e)}")
    #         traceback.print_exc()
    #         return None

    # def _get_next_node(self):
    #     """获取下一个要分配时隙的节点，并检查复用机会"""
    #     try:
    #         if not self.current_requirement or not self.current_requirement.routing_path:
    #             return None
    #
    #         route_nodes = [n for n in self.current_requirement.routing_path if n != -1]
    #         if not route_nodes:
    #             return None
    #
    #         for node in route_nodes:
    #             if node not in self.assigned_nodes:
    #                 # 检查是否有可以复用的时隙
    #                 two_hop_neighbors = set()
    #                 # 获取一跳邻居
    #                 for neighbor in range(self.num_nodes):
    #                     if self.topology_matrix[node][neighbor] == 1:
    #                         two_hop_neighbors.add(neighbor)
    #                         # 获取二跳邻居
    #                         for second_hop in range(self.num_nodes):
    #                             if self.topology_matrix[neighbor][second_hop] == 1 and second_hop != node:
    #                                 two_hop_neighbors.add(second_hop)
    #
    #                 print(f"节点 {node} 的两跳邻居: {two_hop_neighbors}")
    #                 return node
    #
    #         return None
    #
    #     except Exception as e:
    #         print(f"Error in _get_next_node: {str(e)}")
    #         return None

    def _is_valid_assignment(self, node, action):
        """检查时隙分配是否有效，强制两跳外节点复用"""
        if isinstance(action, np.ndarray):
            action = action.item()

        # print(f"检查时隙分配: 节点 {node}, 时隙 {action}")

        # 基本检查
        if action >= self.current_num_slots or node is None or node >= self.num_nodes:
            return False

        # 获取两跳邻居集合
        two_hop_neighbors = set()
        # 获取一跳邻居
        for neighbor in range(self.num_nodes):
            if self.topology_matrix[node][neighbor] == 1:
                two_hop_neighbors.add(neighbor)
                # 获取二跳邻居
                for second_hop in range(self.num_nodes):
                    if self.topology_matrix[neighbor][second_hop] == 1 and second_hop != node:
                        two_hop_neighbors.add(second_hop)

        # 检查当前时隙中的节点
        if action in self.current_schedule:
            for existing_node in self.current_schedule[action]:
                # 如果现有节点是两跳内的邻居，不允许复用
                if existing_node in two_hop_neighbors:
                    # print(f"节点 {node} 不能与两跳内的节点 {existing_node} 复用时隙 {action}")
                    return False
                # print(f"节点 {node} 可以与节点 {existing_node} 复用时隙 {action}")
            return True  # 可以复用
        else:
            # 检查是否有可以复用的现有时隙
            for slot in range(action):  # 只检查当前action之前的时隙
                if slot in self.current_schedule:
                    can_reuse = True
                    for existing_node in self.current_schedule[slot]:
                        if existing_node in two_hop_neighbors:
                            can_reuse = False
                            break
                    if can_reuse:
                        # print(f"节点 {node} 应该复用时隙 {slot} 而不是新建时隙 {action}")
                        return False
            return True  # 如果没有找到可复用的时隙，允许创建新时隙

        return True
    def is_valid_assignment(self, node, action):
        """检查时隙分配是否有效，强制两跳外节点复用"""
        if isinstance(action, np.ndarray):
            action = action.item()

        # print(f"检查时隙分配: 节点 {node}, 时隙 {action}")

        # 基本检查
        if action >= self.current_num_slots or node is None or node >= self.num_nodes:
            return False

        # 获取两跳邻居集合
        two_hop_neighbors = set()
        # 获取一跳邻居
        for neighbor in range(self.num_nodes):
            if self.topology_matrix[node][neighbor] == 1:
                two_hop_neighbors.add(neighbor)
                # 获取二跳邻居
                for second_hop in range(self.num_nodes):
                    if self.topology_matrix[neighbor][second_hop] == 1 and second_hop != node:
                        two_hop_neighbors.add(second_hop)

        # 检查当前时隙中的节点
        if action in self.current_schedule:
            for existing_node in self.current_schedule[action]:
                # 如果现有节点是两跳内的邻居，不允许复用
                if existing_node in two_hop_neighbors:
                    print(f"节点 {node} 不能与两跳内的节点 {existing_node} 复用时隙 {action}")
                    return False
                # print(f"节点 {node} 可以与节点 {existing_node} 复用时隙 {action}")
            return True  # 可以复用
        else:
            # 检查是否有可以复用的现有时隙
            for slot in range(action):  # 只检查当前action之前的时隙
                if slot in self.current_schedule:
                    can_reuse = True
                    for existing_node in self.current_schedule[slot]:
                        if existing_node in two_hop_neighbors:
                            can_reuse = False
                            break
                    if can_reuse:
                        # print(f"节点 {node} 应该复用时隙 {slot} 而不是新建时隙 {action}")
                        return False
            return True  # 如果没有找到可复用的时隙，允许创建新时隙

        return True
    def _is_conflict(self, node1, node2):
        """检查两个节点是否存在干扰"""
        # 使用网络拓扑和节点位置判断干扰
        # 这里是一个示例，您可以根据实际需求修改判断条件
        distance = euclidean_distance(self.initial_positions[node1], self.initial_positions[node2])
        return distance < self.max_comm_range  # 如果两节点距离小于通信范围则视为干扰

    def _calculate_link_lifetime(self):
        """计算链路生命期"""
        lifetime = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if self.topology_matrix[i][j] == 1:
                    if self.simulator:
                        # 使用实际的链路生命期计算
                        try:
                            from phy.large_scale_fading import link_lifetime_predictor
                            lifetime[i][j] = lifetime[j][i] = link_lifetime_predictor(
                                self.simulator.drones[i],
                                self.simulator.drones[j],
                                self.max_comm_range
                            )
                        except:
                            # 如果计算失败，使用简化版本
                            distance = euclidean_distance(
                                self.initial_positions[i],
                                self.initial_positions[j]
                            )
                            lifetime[i][j] = lifetime[j][i] = 1.0 - (distance / self.max_comm_range)
                    else:
                        # 训练时使用简化版本
                        lifetime[i][j] = lifetime[j][i] = 0.5  # 默认中等生命期
        return lifetime

    def _get_observation(self):
        """获取观察"""
        # 1. 获取拓扑特征矩阵 (N x N)
        topology_features = np.asarray(self.topology_matrix, dtype=np.float32)

        # 2. 获取节点位置 (N x 3)
        node_positions = np.asarray(self.initial_positions, dtype=np.float32)

        # 3. 获取路由路径
        routing_path = self._get_routing_path()  # 已经是numpy数组

        # 4. 获取链路生命期矩阵 (N x N)
        link_lifetime = self._calculate_link_lifetime()  # 已经是numpy数组

        # 5. 获取业务信息
        traffic_info = self._get_traffic_info()  # 已经是numpy数组

        # 6. 计算节点度数
        node_degrees = np.sum(topology_features, axis=1, dtype=np.float32)
        # 7. 计算当前时隙分配状态
        # 修改拓扑特征矩阵：标记当前已分配到同一时隙的节点关系
        for slot, nodes in self.current_schedule.items():
            for node1 in nodes:
                for node2 in nodes:
                    if node1 != node2:
                        # 标记为2表示这两个节点在同一时隙
                        topology_features[node1][node2] = 2
                        topology_features[node2][node1] = 2
        # 打包成字典
        obs_dict = {
            'topology_features': topology_features,
            'node_positions': node_positions,
            'routing_path': routing_path,
            'link_lifetime': link_lifetime,
            'traffic_info': traffic_info,
            'node_degrees': node_degrees
        }

        # 展平观察
        return self._flatten_observation(obs_dict)

    def _flatten_observation(self, obs_dict):
        """将字典格式的观察展平为一维数组"""
        flattened = []

        # 确保所有数据都是numpy数组并展平
        flattened.extend(obs_dict['topology_features'].flatten())
        flattened.extend(obs_dict['node_positions'].flatten())
        flattened.extend(obs_dict['routing_path'].flatten())
        flattened.extend(obs_dict['link_lifetime'].flatten())
        flattened.extend(obs_dict['traffic_info'].flatten())
        flattened.extend(obs_dict['node_degrees'].flatten())

        return np.array(flattened, dtype=np.float32)

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


    def _get_dict_observation(self):
        """获取字典格式的观察（用于内部使用）"""
        # 1. 获取拓扑特征
        topology_features = self.topology_matrix.copy()

        # 2. 获取节点位置
        node_positions = self.initial_positions.copy()

        # 3. 计算路由路径
        routing_path = self._get_routing_path()

        # 4. 计算链路生命期
        link_lifetime = self._calculate_link_lifetime()

        # 5. 获取业务信息
        traffic_info = self._get_traffic_info()

        # 6. 计算节点度数
        node_degrees = np.sum(topology_features, axis=1)

        return {
            'topology_features': topology_features,
            'node_positions': node_positions,
            'routing_path': routing_path,
            'link_lifetime': link_lifetime,
            'traffic_info': traffic_info,
            'node_degrees': node_degrees
        }
    def _get_topology_features(self):
        """基于自定义拓扑结构获取拓扑特征"""
        topology_features = np.zeros((self.num_nodes, self.num_nodes))

        # 设置主链连接
        for i in range(len(self.topology_structure['main_chain']) - 1):
            node1 = self.topology_structure['main_chain'][i]
            node2 = self.topology_structure['main_chain'][i + 1]
            topology_features[node1][node2] = 1
            topology_features[node2][node1] = 1

        # 设置分支1连接
        for i in range(len(self.topology_structure['branch1']) - 1):
            node1 = self.topology_structure['branch1'][i]
            node2 = self.topology_structure['branch1'][i + 1]
            topology_features[node1][node2] = 1
            topology_features[node2][node1] = 1

        # 设置分支2连接
        for i in range(len(self.topology_structure['branch2']) - 1):
            node1 = self.topology_structure['branch2'][i]
            node2 = self.topology_structure['branch2'][i + 1]
            topology_features[node1][node2] = 1
            topology_features[node2][node1] = 1

        # 设置上层节点连接（与主链中间节点相连）
        for upper_node in self.topology_structure['upper_layer']:
            topology_features[1][upper_node] = 1
            topology_features[upper_node][1] = 1
            topology_features[2][upper_node] = 1
            topology_features[upper_node][2] = 1

        # 检查距离约束
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if topology_features[i][j] == 1:
                    distance = euclidean_distance(
                        self.initial_positions[i],
                        self.initial_positions[j]
                    )
                    if distance > self.max_comm_range:
                        print(
                            f"警告: 节点 {i} 和节点 {j} 之间的距离 ({distance:.2f}m) 超过最大通信范围 ({self.max_comm_range:.2f}m)")
                        topology_features[i][j] = 0
                        topology_features[j][i] = 0

        return topology_features

    # def _calculate_reward(self, action):
    #     """改进的奖励计算：鼓励时隙复用和节点分配"""
    #     reward = 0
    #
    #     # 1. 只对路由路径上的节点进行分配
    #     if self.current_requirement and self.current_requirement.routing_path:
    #         path = self.current_requirement.routing_path
    #         if self.current_node not in path:
    #             return reward  # 非路由节点无需分配，奖励为 0
    #     else:
    #         return reward  # 如果没有路由需求，奖励为 0
    #
    #     # 基本奖励：成功分配时隙
    #     if self._is_valid_assignment(self.current_node, action):
    #         reward += 5  # 成功分配奖励
    #     else:
    #         reward -= 3  # 无效分配惩罚
    #         return reward
    #
    #     # 时隙复用奖励
    #     if action in self.current_schedule:
    #         # 计算复用节点数
    #         reuse_count = len(self.current_schedule[action])
    #         if reuse_count > 0:
    #             reuse_reward = 10 * reuse_count
    #             # 检查复用是否高效（节点间距离足够远）
    #             for other_node in self.current_schedule[action]:
    #                 distance = euclidean_distance(
    #                     self.initial_positions[self.current_node],
    #                     self.initial_positions[other_node]
    #                 )
    #                 if distance > 1.2 * self.max_comm_range:  # 较远的节点复用效果更好
    #                     reuse_reward += 10
    #             reward += reuse_reward
    #
    #     # QoS奖励：根据延迟要求来奖励
    #     current_delay = self._estimate_delay()
    #     if self.current_requirement and current_delay <= self.current_requirement.delay_requirement:
    #         reward += 5  # 满足延迟要求奖励
    #     else:
    #         reward -= 3  # 不满足延迟要求惩罚
    #
    #     # 检查所有路由节点是否分配完毕
    #     if all(node in self.assigned_nodes for node in path):
    #         reward += 10  # 路由路径全部分配完成奖励
    #         self._end_allocation()  # 结束分配过程
    #
    #     # 邻居节点干扰惩罚：邻居不应在同一时隙
    #     neighbors = set()
    #     for node, value in enumerate(self.topology_matrix[self.current_node]):
    #         if value == 1:
    #             neighbors.add(node)
    #
    #     if action in self.current_schedule:
    #         for node in self.current_schedule[action]:
    #             if node in neighbors:
    #                 reward -= 2  # 邻居节点不应在同一时隙
    #             else:
    #                 reward += 1  # 非邻居节点轻微奖励
    #
    #     # 全局效率奖励：时隙复用的整体效率
    #     if len(self.current_schedule) > 0:
    #         total_assignments = sum(len(nodes) for nodes in self.current_schedule.values())
    #         avg_reuse = total_assignments / len(self.current_schedule)
    #         if avg_reuse > 1:  # 每个时隙分配多个节点
    #             reward += 5 * (avg_reuse - 1)  # 平均复用率越高，奖励越多
    #
    #     return reward
    def _calculate_reward(self, action):
        """改进的奖励计算"""
        reward = 0

        # 检查是否是路由路径节点
        if self.current_requirement and self.current_requirement.routing_path:
            path = self.current_requirement.routing_path
            if self.current_node not in path:
                return -1  # 非路由节点轻微惩罚

        # 严厉惩罚无效动作
        if not self._is_valid_assignment(self.current_node, action):
            # 检查是否尝试分配到已知有干扰的时隙
            if action in self.current_schedule:
                for node in self.current_schedule[action]:
                    if self.topology_matrix[self.current_node][node] == 1:
                        return -20  # 严厉惩罚尝试分配到有干扰的时隙
            return -10  # 一般无效动作惩罚

        # 基本奖励
        reward += 5

        # 鼓励选择新的可用时隙
        if action not in self.current_schedule:
            reward += 3
        else:
            # 复用奖励
            reuse_count = len(self.current_schedule[action])
            if reuse_count > 0:
                # 检查是否是高效的复用
                can_reuse = True
                for node in self.current_schedule[action]:
                    if self.topology_matrix[self.current_node][node] == 1:
                        can_reuse = False
                        break
                if can_reuse:
                    reward += 15  # 成功复用奖励

        # 完成整个路由路径的奖励
        if self.current_requirement and self.current_requirement.routing_path:
            if all(node in self.assigned_nodes for node in self.current_requirement.routing_path):
                reward += 20
                # 额外的复用效率奖励
                total_slots = len(self.current_schedule)
                total_assignments = sum(len(nodes) for nodes in self.current_schedule.values())
                if total_slots > 0:
                    efficiency = total_assignments / total_slots
                    if efficiency > 1:
                        reward += 10 * (efficiency - 1)

        # 更新累积奖励
        self.episode_reward += reward
        self.total_reward += reward
        self.last_reward = reward

        return reward
    def _end_allocation(self):
        """结束分配过程"""
        self.done = True  # 标志所有分配已完成

    def _are_nodes_connected(self, node1, node2):
        """检查两个节点是否在拓扑中直接相连"""
        topology_features = self._get_topology_features()
        return topology_features[node1][node2] == 1

    def _check_interference(self, node1, node2):
        """检查两个节点之间是否存在干扰"""
        distance = euclidean_distance(
            self.simulator.drones[node1].coords,
            self.simulator.drones[node2].coords
        )
        return distance < self.max_comm_range

    def _get_node_slot(self, node):
        """获取节点的时隙分配"""
        for slot, nodes in self.current_schedule.items():
            if node in nodes:
                return slot
        return None

    def _estimate_delay(self):
        """估计端到端时延"""
        if not self.current_requirement or not self.current_requirement.routing_path:
            return 0

        # 计算总帧长度
        frame_length = self.current_num_slots * config.SLOT_DURATION

        # 计算传输时延
        transmission_delay = (config.DATA_PACKET_LENGTH / config.BIT_RATE) * 1e6

        # 计算路由跳数
        hop_count = len(self.current_requirement.routing_path) - 1

        # 估计总时延
        total_delay = frame_length + transmission_delay * hop_count

        return total_delay
    # #单路径
    # def _get_routing_path(self):
    #     """使用OPAR计算路由路径"""
    #     if not self.current_requirement:
    #         return np.zeros(self.num_nodes, dtype=np.float32)
    #
    #     try:
    #         # 获取源节点和目标节点
    #         src_id = self.current_requirement.source_id
    #         dst_id = self.current_requirement.dest_id
    #
    #         # 创建OPAR实例并计算路由
    #         src_drone = self.simulator.drones[src_id]
    #         opar = Opar(self.simulator, src_drone)
    #         opar = NewOpar(self.simulator, src_drone)
    #         # opar = LastOpar(self.simulator, src_drone)
    #         # 计算代价矩阵
    #         cost_matrix = opar.calculate_cost_matrix()
    #
    #         # 使用Dijkstra算法计算路径
    #         path = opar.dijkstra(cost_matrix, src_id, dst_id, 0)
    #
    #         # 移除源节点
    #         if path:
    #             path.pop(0)
    #
    #         # 转换为numpy数组
    #         routing_path = np.zeros(self.num_nodes, dtype=np.float32)
    #         routing_path[:] = -1
    #         if path:
    #             for idx, node in enumerate(path[:self.num_nodes]):
    #                 routing_path[idx] = float(node)
    #
    #         # 更新当前业务需求中的路由路径
    #         self.current_requirement.routing_path = path
    #
    #         return routing_path
    #
    #     except Exception as e:
    #         print(f"Error in _get_routing_path: {str(e)}")
    #         return np.zeros(self.num_nodes, dtype=np.float32)

    def _get_routing_path(self):
        """计算并合并多条路由路径"""
        if not self.current_requirement:
            return np.zeros(self.num_nodes, dtype=np.float32)

        try:
            src_id = self.current_requirement.source_id
            dst_id = self.current_requirement.dest_id

            # 创建OPAR实例
            src_drone = self.simulator.drones[src_id]
            opar = LastOpar(self.simulator, src_drone)
            # opar = AMLB_OPAR(self.simulator, src_drone)
            # 计算代价矩阵
            cost_matrix = opar.calculate_cost_matrix()

            # 找到多条不同的路径
            k_paths = []
            temp_cost = cost_matrix.copy()

            # 寻找k条路径
            for k in range(3):  # 找3条不同路径
                path = opar.dijkstra(temp_cost, src_id, dst_id, 0)
                if not path:
                    break
                if len(path) != 0:
                    path.pop(0)

                k_paths.append(path)

                # 增加已使用路径的成本以找到其他路径
                if k < 2:
                    for i in range(len(path) - 1):
                        temp_cost[path[i], path[i + 1]] *= 2
                        temp_cost[path[i + 1], path[i]] *= 2

            if k_paths:
                # 合并所有路径的节点
                merged_path = []
                hop_sets = []  # 每一跳的节点集合

                # 获取最大路径长度
                max_length = max(len(path) for path in k_paths)

                # 对每一跳位置进行处理
                for hop in range(max_length):
                    hop_nodes = set()
                    for path in k_paths:
                        if hop < len(path):
                            # 只有在不是最后一跳时才添加节点
                            if hop < len(path) - 1 or path[hop] == dst_id:
                                hop_nodes.add(path[hop])
                    hop_sets.append(hop_nodes)

                # 构建合并路径
                for nodes in hop_sets:
                    merged_path.extend(list(nodes))

                # 移除重复节点，但保持顺序
                seen = set()
                final_path = []
                for node in merged_path:
                    if node not in seen:
                        seen.add(node)
                        final_path.append(node)

                # 确保起点和终点正确，且终点只出现一次
                if final_path[0] != src_id:
                    final_path.insert(0, src_id)
                # 确保终点是dst_id，并移除之前可能出现的dst_id
                # final_path = [node for node in final_path if node != dst_id]
                # final_path.append(dst_id)

                # # 移除源节点
                # if final_path:
                #     final_path.pop(0)

                # 转换为numpy数组
                routing_path = np.zeros(self.num_nodes, dtype=np.float32)
                routing_path[:] = -1
                for idx, node in enumerate(final_path[:self.num_nodes]):
                    routing_path[idx] = float(node)

                # 存储路径信息
                self.current_requirement.routing_paths = k_paths  # 保存所有原始路径
                self.current_requirement.routing_path = final_path  # 保存合并后的路径
                # print(f"Original paths: {k_paths}")
                # print(f"Merged path: {final_path}")
                logging.info(f"Original paths: {k_paths}")
                logging.info(f"Merged path: {final_path}")

                return routing_path

            return np.zeros(self.num_nodes, dtype=np.float32)

        except Exception as e:
            logging.error(f"Error in _get_routing_path: {str(e)}")
            return np.zeros(self.num_nodes, dtype=np.float32)
    def update_requirement(self, requirement):
        """从字典或对象中更新业务需求"""
        if isinstance(requirement, dict):
            self.source_id = requirement.get('source_id', self.source_id)
            self.dest_id = requirement.get('dest_id', self.dest_id)
            self.num_packets = requirement.get('num_packets', self.num_packets)
            self.delay_requirement = requirement.get('delay_req', self.delay_requirement)
            self.qos_requirement = requirement.get('qos_req', self.qos_requirement)
            self.routing_path = requirement.get('routing_path', self.routing_path)
            self.deadline = requirement.get('deadline', self.deadline)
            self.next_hop_id = requirement.get('next_hop_id', self.next_hop_id)
