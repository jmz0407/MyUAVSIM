import logging
import math
import numpy as np
from collections import defaultdict
import copy
import random
from entities.packet import DataPacket
from utils import config
from utils.util_function import euclidean_distance
from phy.large_scale_fading import maximum_communication_range
from routing.mp_olsr.olsr import Olsr
from routing.mp_olsr.direct_olsr import DirectOlsr
from simulator.improved_traffic_generator import TrafficRequirement
# config logging
logging.basicConfig(filename='running_log.log',
                    filemode='w',  # there are two modes: 'a' and 'w'
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=config.LOGGING_LEVEL
                    )


class MP_OLSR(DirectOlsr):
    """
    多路径OLSR实现

    基于标准OLSR协议，提供多路径支持
    1. 维护多条到达目的地的路径
    2. 提供负载均衡和路径选择策略
    3. 支持链路质量评估和自适应路径选择
    """

    def __init__(self, simulator, my_drone):
        super().__init__(simulator, my_drone)

        # 多路径相关数据结构
        self.path_cache = {}  # 路径缓存，存储到每个目的地的多条路径
        self.max_paths = config.MAX_PATHS if hasattr(config, 'MAX_PATHS') else 3  # 每个目的地的最大路径数
        self.path_selection_strategy = (config.PATH_SELECTION_STRATEGY
                                        if hasattr(config, 'PATH_SELECTION_STRATEGY')
                                        else 'adaptive')  # 路径选择策略
        self.current_path_index = {}  # 当前使用的路径索引

        # 路径质量评估
        self.path_quality = {}  # 存储路径质量指标

        # 负载均衡相关
        self.path_use_count = defaultdict(lambda: defaultdict(int))  # 路径使用次数统计
        self.path_success_rate = defaultdict(lambda: defaultdict(list))  # 路径成功率

        logging.info(f"MP-OLSR初始化: UAV-{my_drone.identifier}, "
                     f"最大路径数={self.max_paths}, 路径选择策略={self.path_selection_strategy}")
        self.simulator.env.process(self.global_table_sync())
    # 在DirectOlsr和MP_OLSR类中添加方法
    def use_global_neighbor_table(self):
        """使用全局邻居表更新本地路由信息"""
        my_id = self.my_drone.identifier

        # 更新一跳邻居表
        current_time = self.simulator.env.now
        self.neighbor_table = {}
        for neighbor_id in self.simulator.global_neighbor_table.get(my_id, set()):
            position = self.simulator.drones[neighbor_id].coords
            self.neighbor_table[neighbor_id] = (position, current_time)

        # 更新二跳邻居表
        self.two_hop_neighbors = {}
        for neighbor_id, two_hop_set in self.simulator.global_two_hop_neighbors.get(my_id, {}).items():
            self.two_hop_neighbors[neighbor_id] = set(two_hop_set)

        # 重新计算MPR
        self.compute_mprs()

        # 更新路由表
        self.calculate_routing_table()

        logging.info(f"UAV {my_id} 使用全局邻居表更新了路由信息")
    def calculate_routing_table(self):
        """计算路由表和路径缓存"""
        # 先调用基类方法计算基本路由表
        super().calculate_routing_table()

        # 计算多条路径
        self.compute_multiple_paths()

    def compute_multiple_paths(self):
        """使用全局邻居表计算多条路径"""
        # 重置路径缓存
        self.path_cache = {}

        for dest_id in range(self.simulator.n_drones):
            if dest_id == self.my_drone.identifier:
                continue  # 跳过自身

            # 计算到目的地的多条路径
            paths = []

            # 首先使用全局邻居表构建网络图
            graph = self._create_network_graph_from_global()

            # 第一条路径使用Dijkstra最短路径
            path = self._find_path_in_global_graph(graph, dest_id)
            if path:
                paths.append(path)

            # 使用链路不相交策略查找更多路径
            for _ in range(self.max_paths - 1):
                if not paths:
                    break

                # 基于已有路径创建临时图（排除已使用的链路）
                temp_graph = self._create_temp_graph_from_global(paths)

                # 在临时图上计算新路径
                new_path = self._find_path_in_global_graph(temp_graph, dest_id)

                if new_path and len(new_path) > 0:
                    paths.append(new_path)
                else:
                    break  # 无法找到更多路径

            if paths:
                self.path_cache[dest_id] = paths

                # 初始化路径索引
                if dest_id not in self.current_path_index:
                    self.current_path_index[dest_id] = 0

                # 更新路径质量
                self.update_path_quality(dest_id, paths)

        logging.info('UAV: %s 使用全局邻居表更新了路径缓存，目的地数量: %d',
                     self.my_drone.identifier, len(self.path_cache))
    def find_multiple_paths(self, dest_id):
        """查找到目的地的多条路径"""
        paths = []

        # 第一条路径使用Dijkstra最短路径
        shortest_path = self.construct_path(dest_id)
        if shortest_path:
            paths.append(shortest_path)

        # 使用链路不相交策略查找更多路径
        for _ in range(self.max_paths - 1):
            if not paths:
                break

            # 基于已有路径创建临时图（排除已使用的链路）
            temp_graph = self.create_temp_graph(paths)

            # 在临时图上计算新路径
            new_path = self.find_path_in_graph(temp_graph, dest_id)

            if new_path and len(new_path) > 0:
                paths.append(new_path)
            else:
                break  # 无法找到更多路径

        return paths

    def create_temp_graph(self, existing_paths):
        """创建临时图，排除已使用的链路"""
        temp_graph = defaultdict(set)

        # 添加邻居节点
        for neighbor_id in self.neighbor_table:
            # 添加邻居的邻居
            if neighbor_id in self.two_hop_neighbors:
                for two_hop in self.two_hop_neighbors[neighbor_id]:
                    temp_graph[neighbor_id].add(two_hop)
                    temp_graph[two_hop].add(neighbor_id)

        # 添加拓扑表中的链路
        for dest, entries in self.topology_table.items():
            for last_hop, _ in entries:
                temp_graph[dest].add(last_hop)
                temp_graph[last_hop].add(dest)

        # 添加自身的邻居连接
        for neighbor_id in self.neighbor_table:
            temp_graph[self.my_drone.identifier].add(neighbor_id)
            temp_graph[neighbor_id].add(self.my_drone.identifier)

        # 排除已使用的链路
        for path in existing_paths:
            for i in range(len(path) - 1):
                if i + 1 < len(path):  # 确保索引有效
                    node1 = path[i]
                    node2 = path[i + 1]

                    # 移除双向链路
                    if node2 in temp_graph[node1]:
                        temp_graph[node1].remove(node2)
                    if node1 in temp_graph[node2]:
                        temp_graph[node2].remove(node1)

        return temp_graph

    def find_path_in_graph(self, graph, dest_id):
        """在给定图中查找路径"""
        # 使用BFS查找路径
        visited = {self.my_drone.identifier}
        queue = [(self.my_drone.identifier, [])]

        while queue:
            node, path = queue.pop(0)
            current_path = path + [node]

            if node == dest_id:
                return current_path

            for neighbor in graph.get(node, set()):
                if neighbor not in visited and neighbor not in path:
                    visited.add(neighbor)
                    queue.append((neighbor, current_path))

        return []

    def update_path_quality(self, dest_id, paths):
        """更新路径质量指标"""
        if dest_id not in self.path_quality:
            self.path_quality[dest_id] = []

        # 清空之前的质量评估
        self.path_quality[dest_id] = []

        for path in paths:
            # 计算路径质量（跳数、节点负载、链路稳定性等）
            hop_count = len(path) - 1 if path else 0

            # 计算路径上节点的平均负载
            load = 0
            node_count = 0
            for node_id in path:
                if node_id != self.my_drone.identifier and node_id != dest_id:
                    node = self.simulator.drones[node_id]
                    load += node.transmitting_queue.qsize() / node.max_queue_size
                    node_count += 1

            avg_load = load / max(node_count, 1)

            # 计算路径稳定性（基于链路稳定性）
            stability = 1.0  # 默认最高稳定性
            for j in range(len(path) - 1):
                if j + 1 < len(path):  # 确保索引有效
                    node1 = self.simulator.drones[path[j]]
                    node2 = self.simulator.drones[path[j + 1]]

                    # 计算链路稳定性（基于节点移动性）
                    link_stability = self.calculate_link_stability(node1, node2)
                    stability = min(stability, link_stability)

            # 使用历史成功率调整质量评分
            success_rate = 0.5  # 默认值
            if dest_id in self.path_success_rate:
                path_tuple = tuple(path)
                if path_tuple in self.path_success_rate[dest_id] and self.path_success_rate[dest_id][path_tuple]:
                    # 计算成功率
                    successes = self.path_success_rate[dest_id][path_tuple].count(True)
                    total = len(self.path_success_rate[dest_id][path_tuple])
                    if total > 0:
                        success_rate = successes / total

            # 综合评分，较低越好
            quality_score = (
                    0.3 * hop_count +
                    0.2 * avg_load +
                    0.2 * (1 - stability) +
                    0.3 * (1 - success_rate)
            )

            # 更新质量指标
            self.path_quality[dest_id].append(quality_score)

            # logging.info(f"UAV-{self.my_drone.identifier} 路径质量更新: 目的地={dest_id}, "
            #              f"跳数={hop_count}, 负载={avg_load:.2f}, 稳定性={stability:.2f}, "
            #              f"成功率={success_rate:.2f}, 总评分={quality_score:.2f}")

    def calculate_link_stability(self, node1, node2):
        """计算链路稳定性"""
        # 基于节点距离和相对速度计算链路稳定性
        distance = euclidean_distance(node1.coords, node2.coords)

        # 计算相对速度
        relative_velocity = [
            node1.velocity[0] - node2.velocity[0],
            node1.velocity[1] - node2.velocity[1],
            node1.velocity[2] - node2.velocity[2]
        ]
        rel_speed = math.sqrt(sum(v ** 2 for v in relative_velocity))

        # 距离越近、相对速度越小，稳定性越高
        if rel_speed < 0.001:  # 避免除零错误
            return 1.0  # 相对静止，最高稳定性

        # 计算估计链路持续时间
        remaining_distance = self.max_comm_range - distance
        if remaining_distance <= 0:
            return 0.0  # 已超出通信范围

        estimated_lifetime = remaining_distance / rel_speed

        # 归一化为0-1之间的值
        max_lifetime = self.max_comm_range / 5  # 假设5m/s为基准速度
        stability = min(1.0, estimated_lifetime / max_lifetime)

        return stability

    def select_path(self, dest_id):
        """选择路径"""
        if dest_id not in self.path_cache or not self.path_cache[dest_id]:
            return []

        paths = self.path_cache[dest_id]
        path = []

        if self.path_selection_strategy == 'round_robin':
            # 轮询策略
            index = self.current_path_index.get(dest_id, 0) % len(paths)
            path = paths[index]

            # 更新索引
            self.current_path_index[dest_id] = (index + 1) % len(paths)

            logging.info(f"UAV-{self.my_drone.identifier} 使用轮询策略选择路径: 索引={index}")

        elif self.path_selection_strategy == 'weighted':
            # 加权选择策略（基于路径质量）
            if dest_id in self.path_quality and self.path_quality[dest_id]:
                # 反转质量分数（较低的分数应有较高的概率）
                weights = [1.0 / (q + 0.1) for q in self.path_quality[dest_id]]
                total = sum(weights)
                probs = [w / total for w in weights]

                # 加权随机选择
                r = random.random()
                cdf = 0
                for i, p in enumerate(probs):
                    cdf += p
                    if r <= cdf:
                        path = paths[i]
                        logging.info(f"UAV-{self.my_drone.identifier} 使用加权策略选择路径: "
                                     f"索引={i}, 权重={probs[i]:.2f}")
                        break
                else:
                    path = paths[0]  # 默认使用第一条路径
            else:
                path = paths[0]

        elif self.path_selection_strategy == 'adaptive':
            # 自适应策略
            # 根据网络状况动态选择最佳路径
            if dest_id in self.path_quality and self.path_quality[dest_id]:
                best_index = min(range(len(self.path_quality[dest_id])),
                                 key=lambda i: self.path_quality[dest_id][i])
                path = paths[best_index]
                logging.info(f"UAV-{self.my_drone.identifier} 使用自适应策略选择路径: "
                             f"索引={best_index}, 评分={self.path_quality[dest_id][best_index]:.2f}")
            else:
                path = paths[0]

        else:
            # 默认使用第一条路径（最短路径）
            path = paths[0]
            logging.info(f"UAV-{self.my_drone.identifier} 使用默认策略选择路径")

        # 记录路径使用次数
        if path:
            path_tuple = tuple(path)
            self.path_use_count[dest_id][path_tuple] += 1
            logging.info(f"UAV-{self.my_drone.identifier} 到目的地{dest_id}的路径使用次数: "
                         f"{self.path_use_count[dest_id][path_tuple]}")

        return path

    def next_hop_selection(self, packet):
        """选择下一跳（多路径版本），基于全局邻居表"""
        enquire = False
        has_route = True

        if isinstance(packet, DataPacket) or isinstance(packet, TrafficRequirement):
            logging.info('UAV: %s (MP_OLSR) 为数据包(id: %s)选择下一跳',
                         self.my_drone.identifier, packet.packet_id)
            dst_id = packet.dst_drone.identifier

            # 如果是目的地，就不需要路由
            if dst_id == self.my_drone.identifier:
                has_route = False
                return has_route, packet, enquire

            # 先同步最新的全局邻居状态
            self.use_global_neighbor_table()

            # 选择路径
            path = self.select_path(dst_id)

            if path and len(path) > 1:
                # 设置下一跳
                next_hop_id = path[1]  # 第0个是自己，第1个是下一跳
                packet.next_hop_id = next_hop_id

                # 设置完整路径
                if hasattr(packet, 'routing_path'):
                    packet.routing_path = path[1:]  # 排除自己

                logging.info('UAV: %s (MP_OLSR) 使用全局邻居表为数据包(id: %s)选择下一跳: %s, 目的地: %s, 完整路径: %s',
                             self.my_drone.identifier, packet.packet_id, next_hop_id, dst_id, path)
                packet.routing_path = path

            elif dst_id in self.routing_table:
                # 回退到基本OLSR路由表
                next_hop_id = self.routing_table[dst_id]
                packet.next_hop_id = next_hop_id

                if hasattr(packet, 'routing_path'):
                    packet.routing_path = [next_hop_id, dst_id]

                logging.info('UAV: %s (MP_OLSR回退) 为数据包(id: %s)选择下一跳: %s, 目的地: %s',
                             self.my_drone.identifier, packet.packet_id, next_hop_id, dst_id)
            else:
                # 无路由
                has_route = False
                logging.info('UAV: %s (MP_OLSR) 没有到目的地: %s 的路由',
                             self.my_drone.identifier, dst_id)
        else:
            # 非数据包，直接发送
            has_route = True

        return has_route, packet, enquire

    def process_data_packet(self, packet, src_drone_id, current_time):
        """处理数据包（多路径版本）"""
        # 先处理数据包的基本接收逻辑
        packet_copy = copy.copy(packet)

        # 检查是否是目的地
        if packet_copy.dst_drone.identifier == self.my_drone.identifier:
            # 目的地，处理数据包
            latency = current_time - packet_copy.creation_time
            self.simulator.metrics.deliver_time_dict[packet_copy.packet_id] = latency
            self.simulator.metrics.throughput_dict[packet_copy.packet_id] = config.DATA_PACKET_LENGTH / (latency / 1e6)
            self.simulator.metrics.hop_cnt_dict[packet_copy.packet_id] = packet_copy.get_current_ttl()
            self.simulator.metrics.datapacket_arrived.add(packet_copy.packet_id)

            logging.info('数据包: %s 送达至无人机: %s, 时延: %s us',
                         packet_copy.packet_id, self.my_drone.identifier, latency)

            # 记录路径成功
            if hasattr(packet_copy, 'routing_path') and packet_copy.routing_path:
                dst_id = packet_copy.dst_drone.identifier
                src_id = packet_copy.src_drone.identifier

                # 构建完整路径
                if src_id != self.my_drone.identifier:
                    # 提取成功路径并更新成功统计
                    self.record_path_success(packet_copy, True)

        else:
            # 转发节点，更新路径质量信息并将包放入队列
            if hasattr(packet_copy, 'routing_path') and packet_copy.routing_path:
                self.update_path_statistics(packet_copy)

            # 将包放入队列中转发
            if self.my_drone.transmitting_queue.qsize() < self.my_drone.max_queue_size:
                self.my_drone.transmitting_queue.put(packet_copy)
                logging.info('无人机: %s 转发数据包: %s', self.my_drone.identifier, packet_copy.packet_id)
            else:
                logging.info('无人机: %s 队列已满，丢弃数据包: %s', self.my_drone.identifier, packet_copy.packet_id)

                # 记录路径失败
                self.record_path_success(packet_copy, False)
    # def next_hop_selection(self, packet):
    #     return self.next_hop_selection_multi(packet)
    def update_path_statistics(self, packet):
        """更新路径统计信息"""
        dst_id = packet.dst_drone.identifier

        # 如果有路径缓存，更新使用次数
        if dst_id in self.path_cache:
            for path in self.path_cache[dst_id]:
                if len(path) > 1 and hasattr(packet, 'routing_path'):
                    # 检查当前路径是否匹配某个缓存路径
                    remaining_path = [self.my_drone.identifier] + packet.routing_path
                    for i in range(len(path) - len(remaining_path) + 1):
                        if path[i:i + len(remaining_path)] == remaining_path:
                            # 找到匹配，更新统计
                            path_tuple = tuple(path)
                            self.path_use_count[dst_id][path_tuple] += 1
                            logging.info(f"无人机{self.my_drone.identifier}更新路径统计: 目的地={dst_id}, "
                                         f"路径使用计数={self.path_use_count[dst_id][path_tuple]}")
                            break

    def record_path_success(self, packet, success):
        """记录路径成功/失败"""
        if hasattr(packet, 'routing_path'):
            dst_id = packet.dst_drone.identifier
            src_id = packet.src_drone.identifier

            # 尝试推断完整路径
            if dst_id in self.path_cache:
                for path in self.path_cache[dst_id]:
                    if len(path) > 1:
                        # 检查这个路径是否与当前包匹配
                        path_tuple = tuple(path)

                        # 限制历史记录长度
                        if len(self.path_success_rate[dst_id][path_tuple]) >= 10:
                            self.path_success_rate[dst_id][path_tuple].pop(0)

                        self.path_success_rate[dst_id][path_tuple].append(success)

                        logging.info(f"无人机{self.my_drone.identifier}记录路径{'成功' if success else '失败'}: "
                                     f"目的地={dst_id}, 路径={path}")

                        # 更新路径质量
                        self.update_path_quality(dst_id, self.path_cache[dst_id])
                        break

    def _create_network_graph(self):
        """创建整个网络的拓扑图"""
        graph = defaultdict(set)

        # 添加自己的邻居
        for neighbor_id in self.neighbor_table:
            graph[self.my_drone.identifier].add(neighbor_id)
            graph[neighbor_id].add(self.my_drone.identifier)

        # 添加从拓扑表获取的连接
        for dest_id, entries in self.topology_table.items():
            for last_hop, _ in entries:
                graph[dest_id].add(last_hop)
                graph[last_hop].add(dest_id)

        return graph
    def create_fused_path(self, source_id, dest_id, max_paths=3):
        """
        创建融合路径：从多条路径中提取最优路段，合并形成一条优化路径

        :param source_id: 源节点ID
        :param dest_id: 目的节点ID
        :param max_paths: 考虑的最大路径数
        :return: 融合后的优化路径
        """
        logging.info(f"【MP-OLSR】为节点{source_id}到{dest_id}创建融合路径")

        # 1. 首先计算多条常规路径
        paths = []

        # 如果源节点是当前节点，使用路径缓存
        if source_id == self.my_drone.identifier:
            # 检查缓存或计算新路径
            if dest_id not in self.path_cache or not self.path_cache[dest_id]:
                self.compute_multiple_paths()

            if dest_id in self.path_cache:
                paths = self.path_cache[dest_id]
        else:
            # 为其他源节点计算路径，使用全局邻居表
            temp_graph = self._create_network_graph_from_global()

            # 尝试计算多条路径
            for _ in range(max_paths):
                path = self._find_path_in_global_graph(temp_graph, source_id, dest_id)
                if path:
                    paths.append(path)
                    # 临时删除路径中的一些链路以获取不同路径
                    for i in range(len(path) - 1):
                        if path[i] in temp_graph and path[i + 1] in temp_graph[path[i]]:
                            temp_graph[path[i]].remove(path[i + 1])
                        if path[i + 1] in temp_graph and path[i] in temp_graph[path[i + 1]]:
                            temp_graph[path[i + 1]].remove(path[i])
                else:
                    break

        if not paths:
            logging.warning(f"【MP-OLSR】没有找到从{source_id}到{dest_id}的路径，无法创建融合路径")
            return []

        logging.info(f"【MP-OLSR】找到{len(paths)}条候选路径：{paths}")

        # 2. 创建节点评分表 - 评估每个节点的质量
        node_scores = self._calculate_node_scores(paths)

        # 3. 使用路径融合算法创建优化路径
        fused_path = self._fuse_paths(source_id, dest_id, paths, node_scores)

        logging.info(f"【MP-OLSR】融合路径创建完成：{fused_path}")
        return fused_path

    def _calculate_node_scores(self, paths):
        """
        计算路径中各节点的质量评分
        评分考虑因素：出现频率、队列负载、能量水平、链路稳定性

        :param paths: 多条路径的列表
        :return: 节点评分字典 {node_id: score}
        """
        node_scores = {}
        node_count = defaultdict(int)  # 节点在不同路径中出现的次数

        # 1. 计算每个节点出现的频率
        all_nodes = set()
        for path in paths:
            for node in path:
                node_count[node] += 1
                all_nodes.add(node)

        # 2. 为每个节点评分
        for node in all_nodes:
            # 基础分 - 出现频率 (0-1分)
            frequency_score = node_count[node] / len(paths)

            # 如果是当前模拟器中的节点，获取更多信息
            if node < self.simulator.n_drones:
                drone = self.simulator.drones[node]

                # 队列负载评分 (0-1分，负载越低越好)
                queue_score = 1.0 - (drone.transmitting_queue.qsize() / max(drone.max_queue_size, 1))

                # 能量水平评分 (0-1分，能量越高越好)
                energy_score = drone.residual_energy / config.INITIAL_ENERGY

                # 链路稳定性评分 (只考虑与当前节点的链路，假设当前节点是self.my_drone.identifier)
                stability_score = 0.5  # 默认中等稳定性
                if node in self.neighbor_table:
                    # 计算与当前节点的链路稳定性
                    my_drone = self.simulator.drones[self.my_drone.identifier]
                    node_drone = self.simulator.drones[node]
                    stability_score = self._calculate_link_stability(my_drone, node_drone)

                # 综合评分 (加权平均)
                final_score = (
                        0.3 * frequency_score +
                        0.3 * queue_score +
                        0.2 * energy_score +
                        0.2 * stability_score
                )
            else:
                # 如果不是当前模拟器中的节点，只使用频率评分
                final_score = frequency_score

            node_scores[node] = final_score

        return node_scores

    def _calculate_link_stability(self, node1, node2):
        """
        计算两个节点之间链路的稳定性
        """
        # 计算距离
        distance = euclidean_distance(node1.coords, node2.coords)

        # 计算相对速度
        relative_velocity = [
            node1.velocity[0] - node2.velocity[0],
            node1.velocity[1] - node2.velocity[1],
            node1.velocity[2] - node2.velocity[2]
        ]
        rel_speed = math.sqrt(sum(v ** 2 for v in relative_velocity))

        # 距离越近、相对速度越小，稳定性越高
        max_range = self.max_comm_range

        # 距离评分
        distance_score = 1.0 - (distance / max_range)

        # 相对速度评分
        speed_score = 1.0
        if rel_speed > 0:
            speed_score = max(0, 1.0 - (rel_speed / 50.0))  # 假设50是最大相对速度

        # 综合评分
        stability = 0.6 * distance_score + 0.4 * speed_score

        return max(0, min(1, stability))  # 确保在0-1范围内

    def _fuse_paths(self, source_id, dest_id, paths, node_scores):
        """
        融合多条路径，创建一条优化路径
        使用动态规划方法，基于节点评分选择最优路径

        :param source_id: 源节点ID
        :param dest_id: 目的节点ID
        :param paths: 多条候选路径
        :param node_scores: 节点评分字典
        :return: 融合后的路径
        """
        # 1. 创建一个有向图表示所有路径
        graph = defaultdict(list)

        # 对于每条路径，添加所有边到图中
        for path in paths:
            for i in range(len(path) - 1):
                from_node = path[i]
                to_node = path[i + 1]
                # 使用节点评分作为边的权重
                edge_weight = (node_scores.get(to_node, 0.5) + node_scores.get(from_node, 0.5)) / 2
                graph[from_node].append((to_node, edge_weight))

        # 2. 使用Dijkstra算法在融合图上找到最优路径
        # 初始化
        dist = {node: float('inf') for node in node_scores.keys()}
        dist[source_id] = 0
        prev = {node: None for node in node_scores.keys()}
        visited = set()

        # Dijkstra算法
        while len(visited) < len(dist):
            # 找到未访问节点中距离最小的
            current = None
            min_dist = float('inf')
            for node in dist:
                if node not in visited and dist[node] < min_dist:
                    current = node
                    min_dist = dist[node]

            if current is None or current == dest_id:
                break

            visited.add(current)

            # 更新邻居
            for neighbor, weight in graph[current]:
                distance = dist[current] - weight  # 使用负权重，因为我们想要最大化评分
                if distance < dist[neighbor]:
                    dist[neighbor] = distance
                    prev[neighbor] = current

        # 3. 构建融合路径
        fused_path = []
        current = dest_id

        while current is not None:
            fused_path.insert(0, current)
            current = prev[current]

        # 检查路径是否包含源节点和目的节点
        if not fused_path or fused_path[0] != source_id or fused_path[-1] != dest_id:
            logging.warning(f"【MP-OLSR】融合路径不完整，回退到最佳单一路径")
            # 计算最佳单一路径
            best_path_idx = 0
            best_path_score = -float('inf')

            for i, path in enumerate(paths):
                path_score = sum(node_scores.get(node, 0) for node in path) / len(path)
                if path_score > best_path_score:
                    best_path_score = path_score
                    best_path_idx = i

            return paths[best_path_idx]

        return fused_path

    def compute_path(self, source_id, dest_id, options=0):
        """
        计算从source_id到dest_id的路径，兼容OPAR接口

        :param source_id: 源节点ID
        :param dest_id: 目的节点ID
        :param options: 选项，0:常规路径，1:融合路径
        :return: 路径列表
        """
        logging.info(f"【MP-OLSR】计算从节点{source_id}到节点{dest_id}的路径，选项={options}")

        # 检查是否使用路径融合
        if options == 1:
            return self.create_fused_path(source_id, dest_id)
        # 检查是否使用网格路径
        elif options == 2:
            return self.create_mesh_path(source_id, dest_id)

        # 默认使用常规单一路径
        # 如果源节点是自己，使用路径缓存
        if source_id == self.my_drone.identifier:
            # 检查缓存中是否有路径
            if dest_id in self.path_cache and self.path_cache[dest_id]:
                paths = self.path_cache[dest_id]

                # 根据策略选择合适的路径
                if self.path_selection_strategy == 'adaptive' and dest_id in self.path_quality:
                    # 选择质量最好的路径
                    best_index = min(range(len(self.path_quality[dest_id])),
                                     key=lambda i: self.path_quality[dest_id][i])
                    selected_path = paths[best_index]
                    logging.info(f"【MP-OLSR】使用自适应策略选择路径{selected_path}")
                else:
                    # 默认使用第一条路径(通常是最短路径)
                    selected_path = paths[0]
                    logging.info(f"【MP-OLSR】使用默认路径{selected_path}")

                # 返回完整路径列表（包含源节点）
                return [source_id] + selected_path

            # 如果没有缓存的路径，重新计算
            logging.info(f"【MP-OLSR】路径缓存中没有到节点{dest_id}的路径，重新计算")
            self.compute_multiple_paths()

            # 再次检查是否有路径
            if dest_id in self.path_cache and self.path_cache[dest_id]:
                selected_path = self.path_cache[dest_id][0]
                return [source_id] + selected_path

            # 仍然找不到路径，返回空
            logging.warning(f"【MP-OLSR】无法找到从节点{source_id}到节点{dest_id}的路径")
            return []

        # 如果源节点不是自己，创建临时拓扑并计算路径
        temp_graph = self._create_network_graph()
        path = self._find_path_in_graph(temp_graph, source_id, dest_id)

        logging.info(f"【MP-OLSR】计算得到路径: {path}")
        return path


    def create_routing_mesh(self, source_id, dest_id, max_paths=3):
        """
        创建路由网格：将多条路径合并成一个包含所有节点但不重复的连通子网络

        :param source_id: 源节点ID
        :param dest_id: 目的节点ID
        :param max_paths: 用于构建网格的最大路径数
        :return: 一个连通的路由网格，以邻接表形式返回 {node_id: [neighbor_ids]}
        """
        logging.info(f"【MP-OLSR】为节点{source_id}到{dest_id}创建路由网格")

        # 1. 计算多条路径
        paths = self._get_multiple_paths(source_id, dest_id, max_paths)

        if not paths:
            logging.warning(f"【MP-OLSR】无法找到从{source_id}到{dest_id}的路径，无法创建路由网格")
            return {}

        logging.info(f"【MP-OLSR】找到{len(paths)}条用于构建网格的路径")
        for i, path in enumerate(paths):
            logging.info(f"路径{i + 1}: {path}")

        # 2. 构建路由网格（邻接表）
        mesh = defaultdict(set)
        for path in paths:
            for i in range(len(path) - 1):
                node1 = path[i]
                node2 = path[i + 1]
                mesh[node1].add(node2)
                mesh[node2].add(node1)  # 双向连接

        # 3. 打印网格信息
        logging.info(f"【MP-OLSR】路由网格创建完成，包含{len(mesh)}个节点")
        for node, neighbors in mesh.items():
            logging.info(f"节点{node}的邻居: {neighbors}")

        return mesh


    def _get_multiple_paths(self, source_id, dest_id, max_paths):
        """
        获取多条从源到目的地的路径

        :param source_id: 源节点ID
        :param dest_id: 目的节点ID
        :param max_paths: 最大路径数
        :return: 多条路径的列表
        """
        paths = []

        # 如果源节点是当前节点，使用路径缓存
        if source_id == self.my_drone.identifier:
            # 检查缓存或计算新路径
            if dest_id not in self.path_cache or not self.path_cache[dest_id]:
                self.compute_multiple_paths()

            if dest_id in self.path_cache:
                cached_paths = self.path_cache[dest_id]
                for path in cached_paths:
                    complete_path = [source_id] + path
                    paths.append(complete_path)
        else:
            # 为其他源节点计算路径
            temp_graph = self._create_network_graph()
            # 尝试计算多条路径（变种的Yen算法）
            first_path = self._find_path_in_graph(temp_graph, source_id, dest_id)
            if first_path:
                paths.append(first_path)

                # 计算更多路径
                for k in range(1, max_paths):
                    # 复制原图
                    temp_graph_k = self._copy_graph(temp_graph)

                    # 尝试在前k-1条路径上的每个节点处分叉
                    for i in range(len(paths[-1]) - 1):
                        spur_node = paths[-1][i]
                        root_path = paths[-1][:i + 1]

                        # 移除图中会导致重复计算的边
                        for p in paths:
                            if len(p) > i and p[:i + 1] == root_path:
                                if i + 1 < len(p) and spur_node in temp_graph_k:
                                    if p[i + 1] in temp_graph_k[spur_node]:
                                        temp_graph_k[spur_node].remove(p[i + 1])
                                    if spur_node in temp_graph_k[p[i + 1]]:
                                        temp_graph_k[p[i + 1]].remove(spur_node)

                        # 计算从spur_node到目的地的新路径
                        spur_path = self._find_path_in_graph(temp_graph_k, spur_node, dest_id)

                        if spur_path and len(spur_path) > 1:
                            # 完整路径 = 根路径 + 分叉路径（去除根路径的最后一个节点）
                            candidate_path = root_path + spur_path[1:]
                            if candidate_path not in paths:
                                paths.append(candidate_path)
                                break

                    # 如果没有找到新路径，结束循环
                    if len(paths) <= k:
                        break

        # 去除可能的重复路径
        unique_paths = []
        for path in paths:
            if path not in unique_paths:
                unique_paths.append(path)

        return unique_paths[:max_paths]


    def _copy_graph(self, graph):
        """
        创建图的深拷贝
        """
        new_graph = defaultdict(set)
        for node, neighbors in graph.items():
            new_graph[node] = set(neighbors)
        return new_graph


    def create_mesh_path(self, source_id, dest_id, max_paths=3):
        """
        创建路由网格并转换为单一路径格式

        :param source_id: 源节点ID
        :param dest_id: 目的节点ID
        :param max_paths: 用于构建网格的最大路径数
        :return: 包含所有节点的路径（不重复）
        """
        # 1. 获取路由网格
        mesh = self.create_routing_mesh(source_id, dest_id, max_paths)

        if not mesh:
            logging.warning(f"【MP-OLSR】无法创建路由网格，返回空路径")
            return []

        # 2. 将网格转换为路径 - 使用深度优先搜索捕获所有节点
        visited = set()
        path = []

        def dfs(node):
            if node in visited:
                return

            visited.add(node)
            path.append(node)

            # 特殊处理：如果当前节点是目的地，不继续探索
            if node == dest_id:
                return

            # 优先考虑更接近目的地的邻居
            neighbors = list(mesh[node])
            if neighbors:
                # 计算每个邻居到目的地的距离（使用Manhattan距离作为启发式）
                if all(n < self.simulator.n_drones for n in neighbors + [dest_id]):
                    try:
                        neighbor_distances = []
                        for neighbor in neighbors:
                            if neighbor in visited:
                                # 已访问的邻居距离设为无穷大
                                neighbor_distances.append((neighbor, float('inf')))
                            else:
                                # 计算到目的地的距离
                                n_coords = self.simulator.drones[neighbor].coords
                                d_coords = self.simulator.drones[dest_id].coords
                                distance = sum(abs(a - b) for a, b in zip(n_coords, d_coords))
                                neighbor_distances.append((neighbor, distance))

                        # 按距离排序，优先访问更接近目的地的邻居
                        neighbor_distances.sort(key=lambda x: x[1])
                        for neighbor, _ in neighbor_distances:
                            if neighbor not in visited:
                                dfs(neighbor)
                    except Exception as e:
                        logging.error(f"计算邻居距离时出错: {e}")
                        # 出错时采用普通DFS
                        for neighbor in neighbors:
                            if neighbor not in visited:
                                dfs(neighbor)
                else:
                    # 如果节点ID不在有效范围内，回退到普通DFS
                    for neighbor in neighbors:
                        if neighbor not in visited:
                            dfs(neighbor)

        # 从源节点开始DFS
        dfs(source_id)

        # 确保目的节点在路径末尾
        if dest_id in path and path[-1] != dest_id:
            path.remove(dest_id)
            path.append(dest_id)

        logging.info(f"【MP-OLSR】生成的网格路径: {path}")

        # 3. 优化路径 - 尝试消除不必要的绕路
        optimized_path = self._optimize_path(path, mesh, source_id, dest_id)

        logging.info(f"【MP-OLSR】优化后的网格路径: {optimized_path}")
        return optimized_path


    def _optimize_path(self, path, mesh, source_id, dest_id):
        """
        优化路径，消除不必要的绕路

        :param path: 原始路径
        :param mesh: 路由网格
        :param source_id: 源节点ID
        :param dest_id: 目的节点ID
        :return: 优化后的路径
        """
        if len(path) <= 2:
            return path

        optimized = [source_id]
        i = 0

        while i < len(path) - 1:
            current = path[i]

            # 查找当前节点可以直接到达的最远节点
            furthest_reachable = i
            for j in range(len(path) - 1, i, -1):
                if path[j] in mesh[current]:
                    furthest_reachable = j
                    break

            # 如果找到可以跳过的节点
            if furthest_reachable > i + 1:
                i = furthest_reachable
            else:
                i += 1

            if i < len(path) and path[i] not in optimized:
                optimized.append(path[i])

        # 确保目的节点在路径末尾
        if optimized[-1] != dest_id and dest_id in optimized:
            optimized.remove(dest_id)
            optimized.append(dest_id)
        elif optimized[-1] != dest_id:
            optimized.append(dest_id)

        return optimized


    def _create_network_graph_from_global(self):
        """从全局邻居表创建网络图"""
        graph = defaultdict(set)
        my_id = self.my_drone.identifier

        # 添加自己的邻居
        if my_id in self.simulator.global_neighbor_table:
            for neighbor_id in self.simulator.global_neighbor_table[my_id]:
                graph[my_id].add(neighbor_id)
                graph[neighbor_id].add(my_id)

        # 添加其他节点的连接
        for node_id in range(self.simulator.n_drones):
            if node_id != my_id and node_id in self.simulator.global_neighbor_table:
                for neighbor_id in self.simulator.global_neighbor_table[node_id]:
                    graph[node_id].add(neighbor_id)
                    graph[neighbor_id].add(node_id)

        return graph


    def _create_temp_graph_from_global(self, existing_paths):
        """从全局邻居表创建临时图，排除已使用的链路"""
        graph = self._create_network_graph_from_global()

        # 排除已使用的链路
        for path in existing_paths:
            for i in range(len(path) - 1):
                if i + 1 < len(path):  # 确保索引有效
                    node1 = path[i]
                    node2 = path[i + 1]

                    # 移除双向链路
                    if node2 in graph[node1]:
                        graph[node1].remove(node2)
                    if node1 in graph[node2]:
                        graph[node2].remove(node1)

        return graph


    def _find_path_in_global_graph(self, graph, dest_id):
        """在基于全局邻居表的图中查找路径"""
        # 使用BFS查找路径
        visited = {self.my_drone.identifier}
        queue = [(self.my_drone.identifier, [])]

        while queue:
            node, path = queue.pop(0)
            current_path = path + [node]

            if node == dest_id:
                return current_path

            for neighbor in graph.get(node, set()):
                if neighbor not in visited and neighbor not in path:
                    visited.add(neighbor)
                    queue.append((neighbor, current_path))

        return []