import copy
import logging
import math
import numpy as np
from entities.packet import DataPacket
from topology.virtual_force.vf_packet import VfPacket
from utils import config
from utils.util_function import euclidean_distance
from phy.large_scale_fading import maximum_communication_range
from simulator.TrafficGenerator import TrafficRequirement
import random
# config logging
logging.basicConfig(filename='running_log.log',
                    filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=config.LOGGING_LEVEL)


class LastOpar:
    """优化版OPAR路由算法

    主要改进:
    1. 多维度路由成本计算
    2. 自适应权重调整
    3. 路由缓存机制
    4. 增强型链路预测
    """

    def __init__(self, simulator, my_drone):
        self.simulator = simulator
        self.my_drone = my_drone
        self.cost = None
        self.best_obj = float('inf')
        self.best_path = None

        # 自适应权重
        self.w_distance = 0.3  # 距离权重
        self.w_queue = 0.2  # 队列权重
        self.w_lifetime = 0.3  # 生命周期权重
        self.w_energy = 0.2  # 能量权重

        # 路由缓存
        self.route_cache = {}
        self.cache_lifetime = 1000000  # 缓存有效期(微秒)
        self.cache_timestamp = {}

        self.max_comm_range = maximum_communication_range()
        # 添加多路径支持
        self.k_paths = 2  # 每对节点维护的路径数
        self.path_cache = {}  # 存储多条路径
        self.path_usage = {}  # 记录路径使用次数
        self.current_path_index = {}  # 当前使用的路径索引
        # 启动定期进程
        self.simulator.env.process(self.check_waiting_list())
        self.simulator.env.process(self.update_route_cache())

    def find_k_shortest_paths(self, cost, src_id, dst_id, k=3):
        """找到k条最短路径"""
        paths = []
        cost_matrix = cost.copy()

        for _ in range(k):
            # 使用Dijkstra找到当前的最短路径
            path = self.dijkstra(cost_matrix, src_id, dst_id, 0)
            if not path:
                break

            paths.append(path)

            # 通过增加已使用边的成本来寻找替代路径
            if len(paths) < k:
                for i in range(len(path) - 1):
                    # 增加已使用边的成本，促使寻找替代路径
                    cost_matrix[path[i], path[i + 1]] *= 2
                    cost_matrix[path[i + 1], path[i]] *= 2

        return paths
    def calculate_cost_matrix(self):
        """计算多维度路由成本矩阵"""
        n_drones = self.simulator.n_drones
        cost = np.full((n_drones, n_drones), np.inf)

        for i in range(n_drones):
            for j in range(i + 1, n_drones):
                drone1 = self.simulator.drones[i]
                drone2 = self.simulator.drones[j]

                dist = euclidean_distance(drone1.coords, drone2.coords)

                if dist < self.max_comm_range:
                    # 计算多维度成本
                    distance_cost = dist / self.max_comm_range
                    queue_cost = drone2.transmitting_queue.qsize() / drone2.max_queue_size
                    lifetime = link_lifetime_predictor(drone1, drone2, self.max_comm_range)
                    lifetime_cost = 1.0 / (1.0 + lifetime)
                    energy_cost = 1.0 - (drone2.residual_energy / config.INITIAL_ENERGY)

                    # 综合成本
                    total_cost = (
                            self.w_distance * distance_cost +
                            self.w_queue * queue_cost +
                            self.w_lifetime * lifetime_cost +
                            self.w_energy * energy_cost
                    )

                    cost[i, j] = cost[j, i] = total_cost

        return cost

    def dijkstra(self, cost, src_id, dst_id, minimum_link_lifetime):
        """改进的Dijkstra算法"""
        n_drones = self.simulator.n_drones
        distance = np.full(n_drones, np.inf)
        distance[src_id] = 0
        prev = np.full(n_drones, -1)
        visited = np.zeros(n_drones, dtype=bool)

        while True:
            # 找到最近的未访问节点
            min_dist = np.inf
            current = -1
            for i in range(n_drones):
                if not visited[i] and distance[i] < min_dist:
                    min_dist = distance[i]
                    current = i

            if current == -1 or current == dst_id:
                break

            visited[current] = True

            # 更新邻居距离
            for next_node in range(n_drones):
                if not visited[next_node] and cost[current, next_node] != np.inf:
                    if minimum_link_lifetime > 0:
                        lifetime = link_lifetime_predictor(
                            self.simulator.drones[current],
                            self.simulator.drones[next_node],
                            self.max_comm_range
                        )
                        if lifetime <= minimum_link_lifetime:
                            continue

                    new_dist = distance[current] + cost[current, next_node]
                    if new_dist < distance[next_node]:
                        distance[next_node] = new_dist
                        prev[next_node] = current

        # 构建路径
        if distance[dst_id] == np.inf:
            return []

        path = []
        current = dst_id
        while current != -1:
            path.insert(0, current)
            current = prev[current]

        return path

    def next_hop_selection(self, packet):
        """
        基于路径轮询的下一跳选择策略
        """
        enquire = False
        has_route = True

        if not isinstance(packet, DataPacket):
            self.simulator.metrics.control_packet_num += 1

        if packet.src_drone is self.my_drone:  # 源节点路由选择
            self.cost = self.calculate_cost_matrix()
            src_id = self.my_drone.identifier
            dst_id = packet.dst_drone.identifier
            path_key = (src_id, dst_id)

            # 检查路由缓存
            if path_key not in self.path_cache:
                # 找到k条路径并缓存
                paths = self._find_k_shortest_paths(src_id, dst_id)
                if paths:
                    self.path_cache[path_key] = paths
                    self.path_usage[path_key] = [0] * len(paths)

            if path_key in self.path_cache and self.path_cache[path_key]:
                paths = self.path_cache[path_key]

                # 选择负载最小的路径
                best_path_idx = self._select_least_loaded_path(paths)
                self.path_usage[path_key][best_path_idx] += 1
                selected_path = paths[best_path_idx]

                if selected_path and len(selected_path) > 1:
                    packet.routing_path = selected_path[1:]  # 移除源节点
                    best_next_hop_id = selected_path[1]
                    logging.info('UAV %s using path %d: %s',
                                 self.my_drone.identifier,
                                 best_path_idx,
                                 packet.routing_path)
                else:
                    best_next_hop_id = self.my_drone.identifier
                    has_route = False
            else:
                best_next_hop_id = self.my_drone.identifier
                has_route = False

        else:  # 中继节点处理
            routing_path = packet.routing_path
            if len(routing_path) > 1:
                routing_path.pop(0)
                packet.routing_path = routing_path
                best_next_hop_id = routing_path[0]
            else:
                best_next_hop_id = self.my_drone.identifier
                has_route = False

        if best_next_hop_id is self.my_drone.identifier:
            has_route = False
        else:
            packet.next_hop_id = best_next_hop_id

        return has_route, packet, enquire

    def _find_k_shortest_paths(self, src_id, dst_id):
        """找到k条不同的最短路径"""
        paths = []
        temp_cost = self.cost.copy()

        for _ in range(2):  # 找3条不同路径
            path = self.dijkstra(temp_cost, src_id, dst_id, 0)
            if not path:
                break

            paths.append(path)

            # 增加已使用边的成本以找到其他路径
            if len(paths) < 2:
                for i in range(len(path) - 1):
                    temp_cost[path[i], path[i + 1]] *= 2
                    temp_cost[path[i + 1], path[i]] *= 2
        logging.info('Paths found: %s', paths)
        return paths

    def _select_least_loaded_path(self, paths):
        """选择负载最小的路径"""
        path_loads = []

        for path in paths:
            # 计算路径负载
            path_load = sum(
                self.simulator.drones[node].transmitting_queue.qsize()
                for node in path[1:-1]  # 不包括源节点和目标节点
            )
            # 路径长度惩罚
            length_penalty = len(path) * 0.1

            total_load = path_load + length_penalty
            path_loads.append(total_load)
            # 找到最小负载值
        min_load = min(path_loads)

        # 如果负载相同，则随机选择一条路径
        min_paths = [i for i, load in enumerate(path_loads) if load == min_load]

        # 随机选择一条路径
        selected_path_index = random.choice(min_paths)

        logging.info('Path loads: %s', path_loads)
        logging.info('Selected path index: %d', selected_path_index)
        return selected_path_index

    def _evaluate_path(self, path, packet):
        """评估路径质量"""
        if not path or len(path) < 2:
            return float('inf')

        path_length = len(path) - 1
        total_queue_load = 0
        min_lifetime = float('inf')
        min_energy = float('inf')

        for i in range(len(path) - 1):
            drone1 = self.simulator.drones[path[i]]
            drone2 = self.simulator.drones[path[i + 1]]

            # 队列负载
            queue_load = drone2.transmitting_queue.qsize() / drone2.max_queue_size
            total_queue_load += queue_load

            # 链路生命周期
            lifetime = link_lifetime_predictor(drone1, drone2, self.max_comm_range)
            min_lifetime = min(min_lifetime, lifetime)

            # 节点能量
            min_energy = min(min_energy, drone2.residual_energy)

        # 计算综合评分
        score = (
                self.w_distance * path_length +
                self.w_queue * (total_queue_load / path_length) +
                self.w_lifetime * (1 / min_lifetime) +
                self.w_energy * (1 - min_energy / config.INITIAL_ENERGY)
        )

        return score

    def _select_best_path(self, paths, path_key):
        """选择最佳路径"""
        scores = []

        for i, path in enumerate(paths):
            # 计算路径负载
            path_load = sum(
                self.simulator.drones[node].transmitting_queue.qsize()
                for node in path[1:-1]  # 不包括源节点和目标节点
            )
            # 使用次数惩罚
            usage_penalty = self.path_usage[path_key][i]

            # 路径长度惩罚
            length_penalty = len(path)

            # 综合评分 (分数越低越好)
            score = (0.4 * path_load +
                     0.3 * usage_penalty +
                     0.3 * length_penalty)
            scores.append(score)

        return scores.index(min(scores))
    def packet_reception(self, packet, src_drone_id):
        """数据包接收处理"""
        current_time = self.simulator.env.now

        if isinstance(packet, DataPacket):
            packet_copy = copy.copy(packet)
            logging.info('DataPacket: %s received by UAV: %s at: %s',
                         packet_copy.packet_id, self.my_drone.identifier, current_time)

            if packet_copy.dst_drone.identifier == self.my_drone.identifier:
                # 目标节点处理
                latency = current_time - packet_copy.creation_time
                hop_count = packet_copy.get_current_ttl()
                self.simulator.metrics.deliver_time_dict[packet_copy.packet_id] = latency
                self.simulator.metrics.throughput_dict[packet_copy.packet_id] = (
                        config.DATA_PACKET_LENGTH / (latency / 1e6)
                )
                self.simulator.metrics.hop_cnt_dict[packet_copy.packet_id] = (
                    packet_copy.get_current_ttl()
                )
                self.simulator.metrics.record_packet_reception(
                    packet_copy.packet_id, latency, hop_count)

                logging.info('Packet %s delivered, latency: %s us, throughput: %s',
                             packet_copy.packet_id, latency,
                             self.simulator.metrics.throughput_dict[packet_copy.packet_id])
                self.simulator.metrics.datapacket_arrived.add(packet_copy.packet_id)

                logging.info('Packet %s delivered, latency: %s us, throughput: %s',
                             packet_copy.packet_id, latency,
                             self.simulator.metrics.throughput_dict[packet_copy.packet_id])
            else:
                # 中继转发
                if self.my_drone.transmitting_queue.qsize() < self.my_drone.max_queue_size:
                    self.my_drone.transmitting_queue.put(packet_copy)
                else:
                    logging.warning('Queue full, packet %s dropped', packet_copy.packet_id)

        elif isinstance(packet, VfPacket):
            # VF包处理
            logging.info('VF packet %s from UAV %s received by UAV %s at %s',
                         packet.packet_id, src_drone_id, self.my_drone.identifier,
                         current_time)

            self.my_drone.motion_controller.neighbor_table.add_neighbor(
                packet, current_time)

            if packet.msg_type == 'hello':
                config.GL_ID_VF_PACKET += 1
                ack_packet = VfPacket(
                    src_drone=self.my_drone,
                    creation_time=current_time,
                    id_hello_packet=config.GL_ID_VF_PACKET,
                    hello_packet_length=config.HELLO_PACKET_LENGTH,
                    simulator=self.simulator
                )
                ack_packet.msg_type = 'ack'
                self.my_drone.transmitting_queue.put(ack_packet)

    def update_route_cache(self):
        """定期更新路由缓存"""
        while True:
            yield self.simulator.env.timeout(self.cache_lifetime)

            # 清理缓存
            self.path_cache.clear()
            self.path_usage.clear()
            self.current_path_index.clear()

            logging.info('Route cache and path statistics cleared')

    def check_waiting_list(self):
        """检查等待队列中的数据包"""
        while True:
            if not self.my_drone.sleep:
                yield self.simulator.env.timeout(0.6 * 1e6)
                for waiting_pkd in self.my_drone.waiting_list:
                    if self.simulator.env.now < waiting_pkd.creation_time + waiting_pkd.deadline:
                        self.my_drone.waiting_list.remove(waiting_pkd)
                    else:
                        best_next_hop_id = self.next_hop_selection(waiting_pkd)
                        if best_next_hop_id != self.my_drone.identifier:
                            self.my_drone.transmitting_queue.put(waiting_pkd)
                            self.my_drone.waiting_list.remove(waiting_pkd)
                        else:
                            pass
            else:
                break


def link_lifetime_predictor(drone1, drone2, max_comm_range):
    """优化的链路生命周期预测"""
    coords1 = np.array(drone1.coords)
    coords2 = np.array(drone2.coords)
    velocity1 = np.array(drone1.velocity)
    velocity2 = np.array(drone2.velocity)

    rel_velocity = velocity1 - velocity2
    rel_position = coords1 - coords2

    a = np.sum(rel_velocity ** 2)
    b = 2 * np.sum(rel_velocity * rel_position)
    c = np.sum(rel_position ** 2) - max_comm_range ** 2

    if abs(a) < 1e-6:  # 相对速度接近0
        return float('inf')

    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:  # 无实根
        return 0

    t1 = (-b + np.sqrt(discriminant)) / (2 * a)
    t2 = (-b - np.sqrt(discriminant)) / (2 * a)

    return max(t1, t2) if max(t1, t2) > 0 else 0