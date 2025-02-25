import copy
import logging
import numpy as np
from entities.packet import DataPacket
from topology.virtual_force.vf_packet import VfPacket
from utils import config
from routing.opar.link_predictor import LinkPredictor
from routing.opar.path_manager import PathManager
from phy.large_scale_fading import maximum_communication_range, euclidean_distance
class OparMetrics:
    """OPAR性能指标管理"""

    def __init__(self):
        self.path_breaks = {}
        self.route_changes = {}
        self.delay_samples = []
        self.success_rate = {}
        self.link_quality = {}
        self.window_size = 100

    def update_delay(self, delay):
        self.delay_samples.append(delay)
        if len(self.delay_samples) > self.window_size:
            self.delay_samples.pop(0)

    def update_success_rate(self, dst_id, success):
        if dst_id not in self.success_rate:
            self.success_rate[dst_id] = {"success": 0, "total": 0}
        self.success_rate[dst_id]["total"] += 1
        if success:
            self.success_rate[dst_id]["success"] += 1

    def get_average_delay(self):
        return sum(self.delay_samples) / len(self.delay_samples) if self.delay_samples else 0

    def get_success_rate(self, dst_id=None):
        if dst_id:
            if dst_id in self.success_rate:
                stats = self.success_rate[dst_id]
                return stats["success"] / stats["total"] if stats["total"] > 0 else 0
            return 0

        total_success = sum(s["success"] for s in self.success_rate.values())
        total_attempts = sum(s["total"] for s in self.success_rate.values())
        return total_success / total_attempts if total_attempts > 0 else 0


class NewOpar:
    """
    Enhanced OPAR (Optimized Predictive and Adaptive Routing) Implementation
    """

    def __init__(self, simulator, my_drone):
        self.simulator = simulator
        self.my_drone = my_drone
        self.max_comm_range = maximum_communication_range()

        # 核心组件
        self.link_predictor = LinkPredictor()
        self.path_manager = PathManager(simulator, my_drone)
        self.metrics = OparMetrics()

        # 路由权重配置
        self.weights = {
            'distance': 0.3,
            'lifetime': 0.3,
            'quality': 0.2,
            'load': 0.2
        }

        # 初始化定时任务
        self.simulator.env.process(self.check_waiting_list())
        self.simulator.env.process(self.update_metrics())
        self.simulator.env.process(self.adapt_weights())

    def calculate_cost_matrix(self):
        """计算网络代价矩阵"""
        n_drones = self.simulator.n_drones
        cost_matrix = np.full((n_drones, n_drones), np.inf)

        for i in range(n_drones):
            for j in range(i + 1, n_drones):
                drone1 = self.simulator.drones[i]
                drone2 = self.simulator.drones[j]

                # 计算基础代价（距离）
                distance = euclidean_distance(drone1.coords, drone2.coords)
                if distance <= self.max_comm_range:
                    # 计算链路生命期
                    lifetime = self.link_predictor.predict_lifetime(
                        drone1, drone2, self.max_comm_range
                    )

                    # 计算链路质量
                    quality = self.link_predictor.calculate_link_quality(
                        drone1, drone2, self.max_comm_range
                    )

                    # 计算负载因子
                    load_factor = (
                                          drone1.transmitting_queue.qsize() / drone1.max_queue_size +
                                          drone2.transmitting_queue.qsize() / drone2.max_queue_size
                                  ) / 2

                    # 综合代价计算
                    cost = (
                            self.weights['distance'] * (distance / self.max_comm_range) +
                            self.weights['lifetime'] * (1.0 / (lifetime + 1e-6)) +
                            self.weights['quality'] * (1.0 - quality) +
                            self.weights['load'] * load_factor
                    )

                    cost_matrix[i, j] = cost
                    cost_matrix[j, i] = cost

        return cost_matrix

    def next_hop_selection(self, packet):
        """选择下一跳节点"""
        has_route = True
        enquire = False

        if isinstance(packet, DataPacket):
            dst_id = packet.dst_drone.identifier

            if packet.src_drone is self.my_drone:
                # 源节点路由选择
                cost_matrix = self.calculate_cost_matrix()
                path = self.path_manager.find_path(
                    cost_matrix,
                    self.my_drone.identifier,
                    dst_id
                )

                if path and len(path) > 1:
                    packet.routing_path = path[1:]  # 移除源节点
                    packet.next_hop_id = path[1]
                    logging.info(f'Found path for packet {packet.packet_id}: {path}')
                else:
                    has_route = False

            else:
                # 中继节点路由选择
                if hasattr(packet, 'routing_path') and packet.routing_path:
                    routing_path = packet.routing_path
                    if len(routing_path) > 1:
                        routing_path.pop(0)
                        packet.routing_path = routing_path
                        packet.next_hop_id = routing_path[0]
                    else:
                        has_route = False
                else:
                    has_route = False

            if not has_route:
                logging.info(f'No route found for packet {packet.packet_id} to {dst_id}')
                if packet not in self.my_drone.waiting_list:
                    self.my_drone.waiting_list.append(packet)

        return has_route, packet, enquire

    def packet_reception(self, packet, src_drone_id):
        """处理接收到的数据包"""
        current_time = self.simulator.env.now

        if isinstance(packet, DataPacket):
            packet_copy = copy.copy(packet)
            logging.info(f'Received packet {packet_copy.packet_id} from {src_drone_id}')

            if packet_copy.dst_drone.identifier == self.my_drone.identifier:
                # 目标节点处理
                latency = current_time - packet_copy.creation_time
                self.metrics.update_delay(latency)

                # 更新性能指标
                self.simulator.metrics.deliver_time_dict[packet_copy.packet_id] = latency
                self.simulator.metrics.throughput_dict[packet_copy.packet_id] = (
                        config.DATA_PACKET_LENGTH / (latency / 1e6)
                )
                self.simulator.metrics.hop_cnt_dict[packet_copy.packet_id] = (
                    packet_copy.get_current_ttl()
                )
                self.simulator.metrics.datapacket_arrived.add(packet_copy.packet_id)
                self.metrics.update_success_rate(src_drone_id, True)

                logging.info(f'Packet {packet_copy.packet_id} delivered, '
                             f'latency: {latency / 1e6:.3f}s')

            else:
                # 转发处理
                if self.my_drone.transmitting_queue.qsize() < self.my_drone.max_queue_size:
                    self.my_drone.transmitting_queue.put(packet_copy)
                    logging.info(f'Queued packet {packet_copy.packet_id} for forwarding')
                else:
                    logging.warning(f'Queue full, dropped packet {packet_copy.packet_id}')
                    self.metrics.update_success_rate(packet_copy.dst_drone.identifier, False)

        elif isinstance(packet, VfPacket):
            # 虚拟力包处理
            self.my_drone.motion_controller.neighbor_table.add_neighbor(packet, current_time)

            if packet.msg_type == 'hello':
                self._send_vf_ack(packet)

    def update_metrics(self):
        """定期更新性能指标"""
        while True:
            yield self.simulator.env.timeout(1e6)  # 每秒更新

            stats = {
                'avg_delay': self.metrics.get_average_delay(),
                'success_rate': self.metrics.get_success_rate(),
                'queue_load': self.my_drone.transmitting_queue.qsize() / self.my_drone.max_queue_size
            }

            logging.info(f'Performance metrics for UAV {self.my_drone.identifier}:')
            logging.info(f'- Average delay: {stats["avg_delay"] / 1e3:.2f}ms')
            logging.info(f'- Success rate: {stats["success_rate"] * 100:.1f}%')
            logging.info(f'- Queue load: {stats["queue_load"] * 100:.1f}%')

    def adapt_weights(self):
        """自适应调整路由权重"""
        while True:
            yield self.simulator.env.timeout(5e6)  # 每5秒调整

            avg_delay = self.metrics.get_average_delay()
            success_rate = self.metrics.get_success_rate()
            queue_load = self.my_drone.transmitting_queue.qsize() / self.my_drone.max_queue_size

            # 根据性能指标调整权重
            if avg_delay > 100e3:  # 延迟大于100ms
                self.weights['distance'] = min(0.5, self.weights['distance'] + 0.05)
                self.weights['load'] = max(0.1, self.weights['load'] - 0.05)

            if success_rate < 0.9:  # 成功率低于90%
                self.weights['quality'] = min(0.4, self.weights['quality'] + 0.05)
                self.weights['lifetime'] = min(0.4, self.weights['lifetime'] + 0.05)

            if queue_load > 0.8:  # 队列负载大于80%
                self.weights['load'] = min(0.4, self.weights['load'] + 0.05)
                self.weights['distance'] = max(0.2, self.weights['distance'] - 0.05)

            # 归一化权重
            total = sum(self.weights.values())
            self.weights = {k: v / total for k, v in self.weights.items()}

            logging.info(f'Updated routing weights: {self.weights}')

    def check_waiting_list(self):
        """检查等待列表中的数据包"""
        while True:
            if not self.my_drone.sleep:
                yield self.simulator.env.timeout(0.5 * 1e6)  # 每0.5秒检查

                current_waiting_list = self.my_drone.waiting_list.copy()
                for packet in current_waiting_list:
                    # 检查是否过期
                    if self.simulator.env.now >= packet.creation_time + packet.deadline:
                        self.my_drone.waiting_list.remove(packet)
                        self.metrics.update_success_rate(packet.dst_drone.identifier, False)
                        continue

                    # 尝试寻找新路由
                    has_route, packet, _ = self.next_hop_selection(packet)
                    if has_route:
                        self.my_drone.waiting_list.remove(packet)
                        if self.my_drone.transmitting_queue.qsize() < self.my_drone.max_queue_size:
                            self.my_drone.transmitting_queue.put(packet)
                            logging.info(f'Moved packet {packet.packet_id} from waiting list to queue')
            else:
                break

    def dijkstra(self, cost, src_id, dst_id, minimum_link_lifetime):
        """
        Dijkstra's algorithm to find the shortest path
        :param cost: cost matrix
        :param src_id: source node id
        :param dst_id: destination node id
        :param minimum_link_lifetime: used to determine which edges cannot be considered in this iteration
        :return: routing path that has the minimum total cost
        """

        distance_list = [np.inf for _ in range(self.simulator.n_drones)]
        distance_list[src_id] = 0

        prev_list = [-1 for _ in range(self.simulator.n_drones)]
        prev_list[src_id] = -2

        visited_list = [False for _ in range(self.simulator.n_drones)]

        for i in range(self.simulator.n_drones):
            unvisited_list = [(index, value) for index, value in enumerate(distance_list) if
                              not visited_list[index]]
            min_distance_node, _ = min(unvisited_list, key=lambda x: x[1])

            visited_list[min_distance_node] = True

            for j in range(self.simulator.n_drones):
                drone1 = self.simulator.drones[min_distance_node]
                drone2 = self.simulator.drones[j]

                if (visited_list[j] is False) and (cost[min_distance_node, j] != np.inf):
                    delta_temp = self.link_predictor.predict_lifetime(drone1, drone2, self.max_comm_range)

                    if delta_temp <= minimum_link_lifetime:
                        cost[min_distance_node, j] = np.inf
                        cost[j, min_distance_node] = np.inf

                    alt = distance_list[min_distance_node] + cost[min_distance_node, j]
                    if alt < distance_list[j]:
                        distance_list[j] = alt
                        prev_list[j] = min_distance_node

        # path construction
        current_node = dst_id
        path = [dst_id]

        while current_node != -2:
            current_node = prev_list[current_node]

            if current_node != -1:
                path.insert(0, current_node)
            else:
                path = []
                break

        return path

    def _send_vf_ack(self, hello_packet):
        """发送虚拟力ACK包"""
        config.GL_ID_VF_PACKET += 1
        ack_packet = VfPacket(
            src_drone=self.my_drone,
            creation_time=self.simulator.env.now,
            id_hello_packet=config.GL_ID_VF_PACKET,
            hello_packet_length=config.HELLO_PACKET_LENGTH,
            simulator=self.simulator
        )
        ack_packet.msg_type = 'ack'

        if self.my_drone.transmitting_queue.qsize() < self.my_drone.max_queue_size:
            self.my_drone.transmitting_queue.put(ack_packet)
            logging.info(f'Queued VF ACK for packet {hello_packet.packet_id}')