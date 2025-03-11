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
from routing.opar.opar import Opar
from collections import defaultdict, deque
from routing.mp_olsr.olsr import OlsrHelloPacket, OlsrTcPacket


class AMLBR(Opar):
    """
    自适应多路径路由协议
    - 结合链路质量、拥塞情况和能量状态动态选择最佳路径
    - 支持主动和被动路径发现
    - 提供路径冗余以提高可靠性
    - 自适应负载均衡机制
    """

    def __init__(self, simulator, my_drone):
        super().__init__(simulator, my_drone)

        # 多路径相关属性
        self.max_paths = config.MAX_PATHS if hasattr(config, 'MAX_PATHS') else 3
        self.path_cache = {}  # {dst_id: [path_list]}
        self.path_stats = {}  # {dst_id: {path_id: {metrics}}}
        self.current_path_index = {}  # {dst_id: index}
        self.path_selection_strategy = config.PATH_SELECTION_STRATEGY if hasattr(config,
                                                                                 'PATH_SELECTION_STRATEGY') else 'round_robin'

        # 链路质量评估
        self.link_quality = {}  # {(src, dst): quality}
        self.link_reliability = {}  # {(src, dst): reliability}
        self.expected_transmission_count = {}  # {(src, dst): etx}

        # 路径监控和管理
        self.last_path_update = {}  # {dst_id: timestamp}
        self.path_update_interval = 5 * 1e6  # 5秒
        self.failed_transmissions = {}  # {dst_id: count}
        self.energy_aware = True  # 是否考虑能量因素

        # 拥塞感知参数
        self.congestion_threshold = 0.7  # 拥塞阈值
        self.queue_history = deque(maxlen=10)  # 队列长度历史

        # 初始化邻居表
        self.neighbor_table = {}  # {node_id: {last_update: time, quality: value}}
        self.two_hop_neighbors = {}  # {node_id: set(two_hop_neighbors)}

        # 启动路径监控进程
        self.simulator.env.process(self.monitor_paths())
        self.simulator.env.process(self.periodic_link_quality_update())

    def packet_reception(self, packet, src_drone_id):
        """数据包接收处理，增强处理逻辑"""
        current_time = self.simulator.env.now

        # 处理OLSR控制包
        if isinstance(packet, OlsrHelloPacket) or isinstance(packet, OlsrTcPacket):
            # 更新邻居表
            self.update_neighbor_info(src_drone_id, packet)
            # 转发给其他节点
            if packet.get_current_ttl() < config.MAX_TTL - 1:
                self.forward_control_packet(packet)
            return

        # 处理数据包
        if isinstance(packet, DataPacket):
            packet_copy = copy.copy(packet)
            logging.info('DataPacket: %s received by UAV: %s at: %s',
                         packet_copy.packet_id, self.my_drone.identifier, current_time)

            # 更新链路质量统计
            self.update_link_quality(src_drone_id)

            if packet_copy.dst_drone.identifier == self.my_drone.identifier:
                # 目标节点处理
                self.handle_packet_at_destination(packet_copy, current_time)
            else:
                # 中继转发
                self.forward_data_packet(packet_copy, src_drone_id)

        # 处理VF包
        elif isinstance(packet, VfPacket):
            # VF包处理
            logging.info('VF packet %s from UAV %s received by UAV %s at %s',
                         packet.packet_id, src_drone_id, self.my_drone.identifier,
                         current_time)

            self.my_drone.motion_controller.neighbor_table.add_neighbor(
                packet, current_time)

            if packet.msg_type == 'hello':
                self.send_vf_ack(packet, current_time)

    def update_neighbor_info(self, neighbor_id, packet):
        """更新邻居信息"""
        current_time = self.simulator.env.now

        # 初始化邻居条目
        if neighbor_id not in self.neighbor_table:
            self.neighbor_table[neighbor_id] = {
                'last_update': current_time,
                'quality': 1.0,
                'reliability': 1.0,
                'etx': 1.0,
                'energy': self.simulator.drones[neighbor_id].residual_energy
            }
        else:
            # 更新现有邻居信息
            self.neighbor_table[neighbor_id]['last_update'] = current_time

            # 更新邻居能量状态
            self.neighbor_table[neighbor_id]['energy'] = self.simulator.drones[neighbor_id].residual_energy

        # 处理OLSR Hello包中的二跳邻居信息
        if isinstance(packet, OlsrHelloPacket) and hasattr(packet, 'neighbors'):
            self.two_hop_neighbors[neighbor_id] = set(packet.neighbors)

    def send_vf_ack(self, packet, current_time):
        """发送VF确认包"""
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

    def handle_packet_at_destination(self, packet, current_time):
        """目标节点处理数据包"""
        # 计算延迟和跳数
        latency = current_time - packet.creation_time
        hop_count = packet.get_current_ttl()

        # 记录指标
        self.simulator.metrics.deliver_time_dict[packet.packet_id] = latency
        self.simulator.metrics.throughput_dict[packet.packet_id] = (
                config.DATA_PACKET_LENGTH / (latency / 1e6)
        )
        self.simulator.metrics.hop_cnt_dict[packet.packet_id] = hop_count

        # 记录全局统计
        self.simulator.metrics.record_packet_reception(
            packet.packet_id, latency, hop_count)
        self.simulator.metrics.datapacket_arrived.add(packet.packet_id)

        # 对于多路径传输，记录成功的路径
        if hasattr(packet, 'current_path_index'):
            self.update_path_stats(packet.dst_drone.identifier,
                                   packet.current_path_index,
                                   {'success': True, 'latency': latency})

        logging.info('Packet %s delivered, latency: %s us, throughput: %s',
                     packet.packet_id, latency,
                     self.simulator.metrics.throughput_dict[packet.packet_id])

    def forward_data_packet(self, packet, src_drone_id):
        """中继转发数据包，带路径选择"""
        # 检查队列容量
        if self.my_drone.transmitting_queue.qsize() < self.my_drone.max_queue_size:
            # 检查路由信息
            if hasattr(packet, 'routing_path') and len(packet.routing_path) > 0:
                # 使用预定路径
                if self.my_drone.identifier in packet.routing_path:
                    # 找到我在路径中的位置
                    idx = packet.routing_path.index(self.my_drone.identifier)
                    if idx < len(packet.routing_path) - 1:
                        # 更新下一跳
                        packet.next_hop_id = packet.routing_path[idx + 1]
                        self.my_drone.transmitting_queue.put(packet)
                        return

            # 如果没有预定路径或我不在路径中，使用路由表
            dst_id = packet.dst_drone.identifier
            next_hop = self.find_best_next_hop(dst_id, packet)

            if next_hop is not None:
                packet.next_hop_id = next_hop
                self.my_drone.transmitting_queue.put(packet)
            else:
                # 无法路由，尝试发现路由
                logging.warning('No route to destination %s, initiating route discovery',
                                dst_id)
                self.discover_route(dst_id)
        else:
            # 队列满，丢弃数据包
            logging.warning('Queue full at UAV %s, dropping packet %s',
                            self.my_drone.identifier, packet.packet_id)

            # 更新拥塞状态
            self.queue_history.append(1.0)  # 队列满

    def find_best_next_hop(self, dst_id, packet=None):
        """找到最佳下一跳，考虑多个因素"""
        # 检查缓存的路径
        if dst_id in self.path_cache and self.path_cache[dst_id]:
            # 选择一条路径
            path = self.select_best_path(dst_id, packet)
            if path and len(path) > 1:
                # 返回下一跳
                return path[1]

        # 如果没有缓存路径，使用直接邻居中最佳的
        best_neighbor = None
        best_score = float('-inf')

        for neighbor_id, info in self.neighbor_table.items():
            # 检查链路是否有效
            if self.simulator.env.now - info['last_update'] > config.HELLO_PACKET_LIFETIME:
                continue

            # 计算邻居评分
            score = self.calculate_neighbor_score(neighbor_id, dst_id)

            if score > best_score:
                best_score = score
                best_neighbor = neighbor_id

        return best_neighbor

    def calculate_neighbor_score(self, neighbor_id, dst_id):
        """计算邻居节点的综合评分"""
        # 获取邻居信息
        info = self.neighbor_table.get(neighbor_id, {})

        # 基础评分
        score = 0

        # 链路质量因素 (0-1)
        link_quality = info.get('quality', 0)
        score += link_quality * 5  # 权重: 5

        # 能量因素 (0-1)
        if self.energy_aware and 'energy' in info:
            energy_ratio = info['energy'] / config.INITIAL_ENERGY
            score += energy_ratio * 3  # 权重: 3

        # 邻居与目标的距离因素
        if neighbor_id != dst_id:
            # 计算距离
            neighbor_pos = self.simulator.drones[neighbor_id].coords
            dst_pos = self.simulator.drones[dst_id].coords
            distance = euclidean_distance(neighbor_pos, dst_pos)
            max_range = maximum_communication_range()

            # 归一化距离 (0-1)，越近越好
            distance_factor = max(0, 1 - distance / (1.5 * max_range))
            score += distance_factor * 4  # 权重: 4

            # 二跳邻居加成
            if neighbor_id in self.two_hop_neighbors and dst_id in self.two_hop_neighbors[neighbor_id]:
                score += 3  # 额外加分
        else:
            # 目标就是邻居，给最高分
            score += 10

        # 拥塞因素 (0-1)
        queue_size = self.simulator.drones[neighbor_id].transmitting_queue.qsize()
        max_queue = self.simulator.drones[neighbor_id].max_queue_size
        congestion_factor = 1 - (queue_size / max_queue)
        score += congestion_factor * 2  # 权重: 2

        return score

    def select_best_path(self, dst_id, packet=None):
        """根据策略选择最佳路径"""
        if dst_id not in self.path_cache or not self.path_cache[dst_id]:
            return None

        paths = self.path_cache[dst_id]
        strategy = self.path_selection_strategy

        if strategy == 'round_robin':
            # 轮询策略
            if dst_id not in self.current_path_index:
                self.current_path_index[dst_id] = 0

            index = self.current_path_index[dst_id]
            self.current_path_index[dst_id] = (index + 1) % len(paths)
            return paths[index]

        elif strategy == 'weighted':
            # 加权选择策略
            if dst_id not in self.path_stats:
                return paths[0]

            # 计算每条路径的权重
            weights = []
            for i, path in enumerate(paths):
                path_id = f"path_{self.my_drone.identifier}_{dst_id}_{i}"
                stats = self.path_stats[dst_id].get(path_id, {})

                # 默认权重为1
                weight = 1.0

                # 根据历史表现调整权重
                success_rate = stats.get('success_rate', 0.5)
                avg_latency = stats.get('avg_latency', float('inf'))

                if avg_latency != float('inf'):
                    # 归一化延迟 (0-1)，越低越好
                    latency_factor = 1.0 / (1.0 + avg_latency / 1e6)
                    weight *= (success_rate * 0.7 + latency_factor * 0.3)
                else:
                    weight *= success_rate

                weights.append(weight)

            # 归一化权重
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]

                # 根据权重随机选择
                r = np.random.random()
                cumsum = 0
                for i, w in enumerate(weights):
                    cumsum += w
                    if r <= cumsum:
                        return paths[i]

            # 默认返回第一条路径
            return paths[0]

        elif strategy == 'best_quality':
            # 选择质量最好的路径
            best_path = None
            best_quality = float('-inf')

            for i, path in enumerate(paths):
                # 计算路径质量
                path_quality = self.calculate_path_quality(path, dst_id)

                if path_quality > best_quality:
                    best_quality = path_quality
                    best_path = path

            return best_path

        elif strategy == 'adaptive':
            # 自适应策略：根据当前网络状况选择

            # 检查拥塞状态
            is_congested = self.is_network_congested()

            if is_congested:
                # 网络拥塞时，选择链路质量最好的路径
                return self.select_best_path(dst_id, packet)
            else:
                # 正常情况下轮询
                if dst_id not in self.current_path_index:
                    self.current_path_index[dst_id] = 0

                index = self.current_path_index[dst_id]
                self.current_path_index[dst_id] = (index + 1) % len(paths)
                return paths[index]

        # 默认返回第一条路径
        return paths[0]

    def calculate_path_quality(self, path, dst_id):
        """计算路径质量"""
        quality = 0

        # 计算路径上所有链路的平均质量
        for i in range(len(path) - 1):
            node1 = path[i]
            node2 = path[i + 1]

            # 获取链路质量
            link_key = (node1, node2)
            link_quality = self.link_quality.get(link_key, 0.5)

            # 累加链路质量
            quality += link_quality

        # 计算平均值
        if len(path) > 1:
            quality /= (len(path) - 1)

        # 考虑路径长度因素（较短路径有优势）
        length_factor = 1.0 / (1.0 + 0.1 * (len(path) - 2))

        # 综合评分
        final_quality = quality * 0.7 + length_factor * 0.3

        return final_quality

    def is_network_congested(self):
        """检测网络是否拥塞"""
        if not self.queue_history:
            return False

        # 计算队列占用率的平均值
        avg_queue_occupancy = sum(self.queue_history) / len(self.queue_history)

        # 如果平均占用率超过阈值，认为网络拥塞
        return avg_queue_occupancy > self.congestion_threshold

    def update_queue_history(self):
        """更新队列历史"""
        current_queue_size = self.my_drone.transmitting_queue.qsize()
        occupancy_ratio = current_queue_size / self.my_drone.max_queue_size
        self.queue_history.append(occupancy_ratio)

    def update_link_quality(self, neighbor_id):
        """更新链路质量信息"""
        if neighbor_id not in self.neighbor_table:
            self.neighbor_table[neighbor_id] = {
                'last_update': self.simulator.env.now,
                'quality': 1.0,
                'reliability': 1.0,
                'etx': 1.0,
                'energy': self.simulator.drones[neighbor_id].residual_energy
            }

        # 更新链路质量（简化模型）
        dist = euclidean_distance(
            self.my_drone.coords,
            self.simulator.drones[neighbor_id].coords
        )
        max_range = maximum_communication_range()

        # 根据距离估计链路质量 (0-1)
        quality = max(0.1, 1.0 - (dist / max_range) ** 0.7)

        # 平滑更新
        old_quality = self.neighbor_table[neighbor_id]['quality']
        new_quality = old_quality * 0.7 + quality * 0.3

        # 更新邻居表
        self.neighbor_table[neighbor_id]['quality'] = new_quality
        self.neighbor_table[neighbor_id]['last_update'] = self.simulator.env.now

        # 更新链路质量表
        link_key = (self.my_drone.identifier, neighbor_id)
        self.link_quality[link_key] = new_quality

        # 计算ETX（Expected Transmission Count）
        reliability = min(1.0, new_quality + 0.2)  # 简化计算
        etx = 1.0 / reliability if reliability > 0 else float('inf')

        self.neighbor_table[neighbor_id]['reliability'] = reliability
        self.neighbor_table[neighbor_id]['etx'] = etx
        self.expected_transmission_count[link_key] = etx

    def calculate_cost_matrix(self, jitter=0):
        """计算成本矩阵，考虑ETX、距离和能量因素"""
        n_drones = self.simulator.n_drones
        cost = np.full((n_drones, n_drones), np.inf)

        # 设置自环成本为0
        for i in range(n_drones):
            cost[i, i] = 0

        # 设置邻居链路成本
        for i in range(n_drones):
            for j in range(n_drones):
                if i != j:
                    drone1 = self.simulator.drones[i]
                    drone2 = self.simulator.drones[j]

                    # 计算距离
                    dist = euclidean_distance(drone1.coords, drone2.coords)
                    max_range = maximum_communication_range()

                    # 如果在通信范围内
                    if dist <= max_range:
                        # 计算基础成本（基于距离）
                        base_cost = dist / max_range

                        # 考虑ETX因素
                        link_key = (i, j)
                        etx = self.expected_transmission_count.get(link_key, 1.0)

                        # 考虑能量因素
                        energy_ratio = drone2.residual_energy / config.INITIAL_ENERGY
                        energy_cost = max(0.1, 1.0 - energy_ratio)

                        # 综合成本
                        total_cost = base_cost * 0.4 + etx * 0.4 + energy_cost * 0.2

                        # 应用抖动（用于生成多路径）
                        if jitter > 0:
                            noise = np.random.uniform(-jitter, jitter)
                            total_cost *= (1 + noise)

                        cost[i, j] = max(0.1, total_cost)  # 确保成本为正

        return cost

    def discover_multiple_paths(self, src_id, dst_id):
        """发现多条路径"""
        paths = []

        # 使用带不同抖动的Dijkstra算法找到多条路径
        for i in range(self.max_paths):
            # 增加抖动以找到不同路径
            jitter = 0.05 * (i + 1)
            cost_matrix = self.calculate_cost_matrix(jitter=jitter)

            path = self.dijkstra(cost_matrix, src_id, dst_id, 0)

            # 如果找到有效路径且与已有路径差异足够大
            if path and self._is_path_diverse(path, paths):
                path.pop(0)
                paths.append(path)

                # 找到足够多的路径就停止
                if len(paths) >= self.max_paths:
                    break

        return paths

    def _is_path_diverse(self, new_path, existing_paths, threshold=0.3):
        """检查新路径是否与现有路径有足够差异"""
        if not existing_paths or not new_path:
            return True

        for path in existing_paths:
            # 计算路径重叠度
            common_nodes = set(new_path) & set(path)
            total_nodes = set(new_path) | set(path)

            if len(total_nodes) > 0:
                overlap_ratio = len(common_nodes) / len(total_nodes)
                if overlap_ratio > (1 - threshold):
                    return False

        return True

    def next_hop_selection(self, packet):
        """改进的下一跳选择，支持多路径"""
        enquire = False
        has_route = True

        if not isinstance(packet, DataPacket):
            # 对于控制包，直接返回
            return super().next_hop_selection(packet)

        dst_id = packet.dst_drone.identifier

        # 如果是源节点，计算或使用缓存的多条路径
        if packet.src_drone.identifier == self.my_drone.identifier:
            # 检查缓存的路径
            if dst_id not in self.path_cache or not self.path_cache[dst_id] or self._should_update_paths(dst_id):
                # 发现新路径
                paths = self.discover_multiple_paths(self.my_drone.identifier, dst_id)

                if paths:
                    self.path_cache[dst_id] = paths
                    self.last_path_update[dst_id] = self.simulator.env.now

                    # 初始化路径统计
                    self.path_stats[dst_id] = {}

                    for i, path in enumerate(paths):
                        path_id = f"path_{self.my_drone.identifier}_{dst_id}_{i}"
                        self.path_stats[dst_id][path_id] = self._initialize_path_stats()

            # 选择最佳路径
            selected_path = self.select_best_path(dst_id, packet)

            if selected_path:
                # 设置路由信息
                packet.routing_path = selected_path

                if len(selected_path) > 1:
                    packet.next_hop_id = selected_path[1]  # 第二个节点是下一跳

                    # 记录路径选择
                    if dst_id in self.path_cache:
                        path_index = self.path_cache[dst_id].index(selected_path)
                        packet.current_path_index = path_index

                    logging.info('选择路径: %s，下一跳: %s',
                                 selected_path, packet.next_hop_id)
                    return True, packet, False

            # 如果没有找到路径，需要路由发现
            has_route = False
            enquire = True

        else:
            # 对于中继节点，使用预设路径
            if hasattr(packet, 'routing_path') and packet.routing_path:
                routing_path = packet.routing_path

                # 找到我在路径中的位置
                if self.my_drone.identifier in routing_path:
                    idx = routing_path.index(self.my_drone.identifier)

                    # 如果不是最后一个节点，设置下一跳
                    if idx < len(routing_path) - 1:
                        packet.next_hop_id = routing_path[idx + 1]
                        return True, packet, False

            # 如果没有有效路径，尝试使用链路状态
            next_hop = self.find_best_next_hop(dst_id, packet)

            if next_hop is not None:
                packet.next_hop_id = next_hop
                return True, packet, False

            # 无法路由
            has_route = False

        return has_route, packet, enquire

    def _should_update_paths(self, dst_id):
        """检查是否应该更新路径"""
        # 如果没有更新记录，应该更新
        if dst_id not in self.last_path_update:
            return True

        # 检查上次更新时间
        time_since_update = self.simulator.env.now - self.last_path_update[dst_id]

        # 如果超过更新间隔，应该更新
        if time_since_update > self.path_update_interval:
            return True

        # 检查失败记录
        if dst_id in self.failed_transmissions and self.failed_transmissions[dst_id] > 2:
            return True

        return False

    def _initialize_path_stats(self):
        """初始化路径统计"""
        return {
            'success_count': 0,
            'attempt_count': 0,
            'success_rate': 0.5,  # 初始化为中等值
            'avg_latency': float('inf'),
            'last_used': self.simulator.env.now,
            'congestion_level': 0.0
        }

    def update_path_stats(self, dst_id, path_index, metrics):
        """更新路径统计信息"""
        if dst_id not in self.path_stats or dst_id not in self.path_cache:
            return

        path_id = f"path_{self.my_drone.identifier}_{dst_id}_{path_index}"

        if path_id not in self.path_stats[dst_id]:
            self.path_stats[dst_id][path_id] = self._initialize_path_stats()

        # 更新统计信息
        stats = self.path_stats[dst_id][path_id]
        stats['last_used'] = self.simulator.env.now

        # 更新成功/失败计数
        if 'success' in metrics:
            stats['attempt_count'] += 1
            if metrics['success']:
                stats['success_count'] += 1

        # 计算成功率
        if stats['attempt_count'] > 0:
            stats['success_rate'] = stats['success_count'] / stats['attempt_count']

        # 更新延迟统计
        if 'latency' in metrics and metrics['latency'] > 0:
            if stats['avg_latency'] == float('inf'):
                stats['avg_latency'] = metrics['latency']
            else:
                # 指数移动平均
                stats['avg_latency'] = stats['avg_latency'] * 0.7 + metrics['latency'] * 0.3

        # 更新拥塞级别
        if 'congestion' in metrics:
            stats['congestion_level'] = metrics['congestion']

    def discover_route(self, dst_id):
        """主动发现路由"""
        # 实现主动路由发现（类似于DSR或AODV）
        # 这里使用简化版本，直接计算路径
        paths = self.discover_multiple_paths(self.my_drone.identifier, dst_id)
        if paths:
            self.path_cache[dst_id] = paths
            self.last_path_update[dst_id] = self.simulator.env.now

            # 初始化路径统计
            self.path_stats[dst_id] = {}
            for i, path in enumerate(paths):
                path_id = f"path_{self.my_drone.identifier}_{dst_id}_{i}"
                self.path_stats[dst_id][path_id] = self._initialize_path_stats()

            logging.info('路由发现完成，找到 %d 条到 %d 的路径', len(paths), dst_id)
            return True

        logging.warning('路由发现失败，未找到到 %d 的路径', dst_id)
        return False

    def forward_control_packet(self, packet):
        """转发控制包"""
        # 复制控制包
        packet_copy = copy.copy(packet)
        packet_copy.increase_ttl()

        # 标记为广播
        packet_copy.transmission_mode = 1  # 广播模式

        # 放入发送队列
        self.my_drone.transmitting_queue.put(packet_copy)

    def monitor_paths(self):
        """定期监控路径状态"""
        while True:
            yield self.simulator.env.timeout(1 * 1e6)  # 每1秒检查一次

            # 更新队列历史
            self.update_queue_history()

            # 清理过期的邻居
            self.clean_neighbor_table()

            # 检查并更新路径状态
            self.check_path_validity()

    def periodic_link_quality_update(self):
        """定期更新链路质量"""
        while True:
            yield self.simulator.env.timeout(0.5 * 1e6)  # 每0.5秒更新一次

            # 更新与所有邻居的链路质量
            for neighbor_id in list(self.neighbor_table.keys()):
                # 检查邻居是否仍然在线
                if self.simulator.env.now - self.neighbor_table[neighbor_id][
                    'last_update'] > config.HELLO_PACKET_LIFETIME:
                    continue

                # 计算链路质量
                dist = euclidean_distance(
                    self.my_drone.coords,
                    self.simulator.drones[neighbor_id].coords
                )
                max_range = maximum_communication_range()

                # 如果超出通信范围，更新为0
                if dist > max_range:
                    quality = 0.0
                else:
                    # 根据距离计算链路质量
                    quality = max(0.1, 1.0 - (dist / max_range) ** 0.7)

                # 平滑更新
                old_quality = self.neighbor_table[neighbor_id]['quality']
                new_quality = old_quality * 0.8 + quality * 0.2

                # 更新链路质量
                self.neighbor_table[neighbor_id]['quality'] = new_quality

                # 更新链路质量表
                link_key = (self.my_drone.identifier, neighbor_id)
                self.link_quality[link_key] = new_quality

                # 更新ETX
                reliability = min(1.0, new_quality + 0.2)
                etx = 1.0 / reliability if reliability > 0 else float('inf')

                self.neighbor_table[neighbor_id]['reliability'] = reliability
                self.neighbor_table[neighbor_id]['etx'] = etx
                self.expected_transmission_count[link_key] = etx

    def clean_neighbor_table(self):
        """清理过期的邻居信息"""
        current_time = self.simulator.env.now
        expired_neighbors = []

        for neighbor_id, info in self.neighbor_table.items():
            # 检查上次更新时间
            if current_time - info['last_update'] > config.HELLO_PACKET_LIFETIME:
                expired_neighbors.append(neighbor_id)

        # 删除过期邻居
        for neighbor_id in expired_neighbors:
            del self.neighbor_table[neighbor_id]

            # 同时删除相关的链路质量信息
            link_key = (self.my_drone.identifier, neighbor_id)
            if link_key in self.link_quality:
                del self.link_quality[link_key]
            if link_key in self.expected_transmission_count:
                del self.expected_transmission_count[link_key]

            # 删除二跳邻居信息
            if neighbor_id in self.two_hop_neighbors:
                del self.two_hop_neighbors[neighbor_id]

    def check_path_validity(self):
        """检查并更新路径有效性"""
        invalid_paths = {}

        for dst_id in list(self.path_cache.keys()):
            invalid_indices = []

            for i, path in enumerate(self.path_cache[dst_id]):
                # 检查路径是否还有效
                if not self._is_path_valid(path):
                    invalid_indices.append(i)

            # 记录无效路径
            if invalid_indices:
                invalid_paths[dst_id] = invalid_indices

        # 处理无效路径
        for dst_id, indices in invalid_paths.items():
            # 删除无效路径
            for index in sorted(indices, reverse=True):
                path = self.path_cache[dst_id][index]
                self.path_cache[dst_id].pop(index)
                logging.info('移除到 %d 的无效路径: %s', dst_id, path)

            # 如果没有路径了，重新发现
            if not self.path_cache[dst_id]:
                self.discover_route(dst_id)

    def _is_path_valid(self, path):
        """检查路径是否有效"""
        if not path:
            return False

        # 检查路径中的每个链路
        for i in range(len(path) - 1):
            node1 = path[i]
            node2 = path[i + 1]

            # 检查节点是否在线
            if self.simulator.drones[node1].sleep or self.simulator.drones[node2].sleep:
                return False

            # 检查是否在通信范围内
            dist = euclidean_distance(
                self.simulator.drones[node1].coords,
                self.simulator.drones[node2].coords
            )

            if dist > maximum_communication_range():
                return False

            # 检查链路质量
            link_key = (node1, node2)
            quality = self.link_quality.get(link_key, 0)

            if quality < 0.2:  # 链路质量太差
                return False

        return True