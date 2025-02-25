import copy
import logging
import math
import numpy as np
import heapq
from entities.packet import DataPacket
from topology.virtual_force.vf_packet import VfPacket
from utils import config
from utils.util_function import euclidean_distance
from phy.large_scale_fading import maximum_communication_range
from simulator.TrafficGenerator import TrafficRequirement

# 配置日志
logging.basicConfig(filename='running_log.log',
                    filemode='w',  # 'a' 或 'w'
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=config.LOGGING_LEVEL
                    )


class MutiOpar:
    """
    OPAR 路由协议（v3.0）
    支持多路径路由：计算多条候选路径，然后将它们拼接成一条连续的路径返回
    （重复节点无所谓）。

    Attributes:
        simulator: 仿真平台，包含所有仿真信息
        my_drone: 当前执行路由协议的无人机
        cost: 代价矩阵
        best_obj: 最优目标函数值
        best_path: 对应于最优目标函数的路由路径
        w1, w2: 目标函数中两项的权重
        max_comm_range: 最大通信范围
    """

    def __init__(self, simulator, my_drone):
        self.simulator = simulator
        self.my_drone = my_drone
        self.cost = None
        self.best_obj = float('inf')
        self.best_path = None

        self.w1 = 0.5
        self.w2 = 0.5

        self.max_comm_range = maximum_communication_range()
        self.simulator.env.process(self.check_waiting_list())
        self.slot_schedule = {}  # 初始化时隙调度

    def calculate_cost_matrix(self):
        n = self.simulator.n_drones
        cost = np.full((n, n), np.inf)
        for i in range(n):
            for j in range(i + 1, n):
                drone1 = self.simulator.drones[i]
                drone2 = self.simulator.drones[j]
                if euclidean_distance(drone1.coords, drone2.coords) < self.max_comm_range:
                    # 简单设定代价为1，可根据需求修改为更复杂的公式
                    cost_value = 1
                    cost[i, j] = cost_value
                    cost[j, i] = cost_value
        return cost

    def build_graph(self, cost_matrix):
        """
        将代价矩阵转换为图的字典表示，格式为 {node: [(neighbor, cost), ...], ...}
        """
        n = cost_matrix.shape[0]
        graph = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(n):
                if i != j and cost_matrix[i, j] != np.inf:
                    graph[i].append((j, cost_matrix[i, j]))
        return graph

    def dijkstra_graph(self, graph, source, target):
        """
        基于图结构的 Dijkstra 算法，返回从 source 到 target 的最短路径（节点序列）
        """
        n = len(graph)
        distances = {node: float('inf') for node in graph}
        previous = {node: None for node in graph}
        distances[source] = 0
        queue = [(0, source)]

        while queue:
            dist_u, u = heapq.heappop(queue)
            if u == target:
                break
            if dist_u > distances[u]:
                continue
            for v, cost in graph[u]:
                alt = distances[u] + cost
                if alt < distances[v]:
                    distances[v] = alt
                    previous[v] = u
                    heapq.heappush(queue, (alt, v))

        if distances[target] == float('inf'):
            return None

        path = []
        u = target
        while u is not None:
            path.insert(0, u)
            u = previous[u]
        return path

    def yen_k_shortest_paths(self, cost_matrix, source, target, K=3):
        """
        计算从 source 到 target 的 K 条最短路径（候选路径），使用 Yen 算法
        返回路径列表，每条路径为节点序列
        """
        graph = self.build_graph(cost_matrix)
        A = []
        first_path = self.dijkstra_graph(graph, source, target)
        if first_path is None:
            return A
        A.append(first_path)
        B = []  # 候选路径

        for k in range(1, K):
            for i in range(len(A[-1]) - 1):
                spur_node = A[-1][i]
                root_path = A[-1][:i + 1]
                removed_edges = []
                for path in A:
                    if len(path) > i and path[:i + 1] == root_path:
                        u = path[i]
                        v = path[i + 1]
                        for idx, (nbr, cost_val) in enumerate(graph[u]):
                            if nbr == v:
                                removed_edges.append((u, (nbr, cost_val)))
                                del graph[u][idx]
                                break
                spur_path = self.dijkstra_graph(graph, spur_node, target)
                if spur_path is not None:
                    total_path = root_path[:-1] + spur_path
                    total_cost = 0
                    for j in range(len(total_path) - 1):
                        total_cost += cost_matrix[total_path[j], total_path[j + 1]]
                    B.append((total_cost, total_path))
                for u, edge in removed_edges:
                    graph[u].append(edge)
            if not B:
                break
            B.sort(key=lambda x: x[0])
            cost_val, path = B.pop(0)
            A.append(path)
        return A

    def get_concatenated_paths(self, cost_matrix, source, target, K=3):
        """
        计算 K 条候选路径，并将它们简单拼接成一个连续的路径返回
        重复节点也无所谓
        """
        candidate_paths = self.yen_k_shortest_paths(cost_matrix, source, target, K)
        concatenated_path = []
        for path in candidate_paths:
            concatenated_path.extend(path)
        return concatenated_path

    def next_hop_selection(self, packet):
        """
        多路径路由下的下一跳选择：
        - 如果是源节点，计算 K 条候选路径，将它们拼接成一个连续路径，然后移除源节点，
          下一跳取拼接路径中的第一个节点；
        - 如果是中继节点，则直接使用传递的路由路径。
        """
        enquire = False
        has_route = True

        # 确保数据包有 routing_path 属性
        if not hasattr(packet, 'routing_path'):
            packet.routing_path = []

        if packet.src_drone is self.my_drone:
            self.cost = self.calculate_cost_matrix()
            src_id = self.my_drone.identifier
            dst_id = packet.dst_drone.identifier

            # 计算多条候选路径并拼接
            concatenated_path = self.get_concatenated_paths(self.cost, src_id, dst_id, K=2)
            if concatenated_path:
                # 移除源节点
                if concatenated_path[0] == src_id:
                    concatenated_path.pop(0)
                logging.info('concatenated_path: %s', concatenated_path)
                if isinstance(packet, TrafficRequirement):
                    packet.routing_path = concatenated_path
                # self.current_requirement.routing_path = concatenated_path
                best_next_hop_id = concatenated_path[0] if concatenated_path else self.my_drone.identifier
                logging.info('my_drone: %s routing Path: %s', self.my_drone.identifier, concatenated_path)
            else:
                best_next_hop_id = self.my_drone.identifier
        else:
            routing_path = packet.routing_path
            if len(routing_path) > 1:
                routing_path.pop(0)
                packet.routing_path = routing_path
                best_next_hop_id = routing_path[0]
            else:
                best_next_hop_id = self.my_drone.identifier

        if best_next_hop_id == self.my_drone.identifier:
            has_route = False
        else:
            packet.next_hop_id = best_next_hop_id

        return has_route, packet, enquire

    def _get_routing_path(self):
        """
        使用 OPAR 计算路由路径，返回动态长度的 numpy 数组（只包含实际的路径节点）
        """
        if not self.current_requirement:
            return np.zeros(self.simulator.n_drones, dtype=np.float32)
        try:
            src_id = self.current_requirement.source_id
            dst_id = self.current_requirement.dest_id
            cost_matrix = self.calculate_cost_matrix()
            path = self.dijkstra(cost_matrix, src_id, dst_id, 0)
            if path:
                path.pop(0)  # 移除源节点
            self.current_requirement.routing_path = path
            routing_path = np.array(path, dtype=np.float32)
            return routing_path
        except Exception as e:
            print(f"Error in _get_routing_path: {str(e)}")
            return np.zeros(self.simulator.n_drones, dtype=np.float32)

    def packet_reception(self, packet, src_drone_id):
        current_time = self.simulator.env.now
        if isinstance(packet, DataPacket):
            packet_copy = copy.copy(packet)
            logging.info('~~~ DataPacket: %s is received by UAV: %s at time: %s',
                         packet_copy.packet_id, self.my_drone.identifier, current_time)
            if packet_copy.dst_drone.identifier == self.my_drone.identifier:
                latency = current_time - packet_copy.creation_time
                self.simulator.metrics.deliver_time_dict[packet_copy.packet_id] = latency
                self.simulator.metrics.throughput_dict[packet_copy.packet_id] = config.DATA_PACKET_LENGTH / (
                            latency / 1e6)
                self.simulator.metrics.hop_cnt_dict[packet_copy.packet_id] = packet_copy.get_current_ttl()
                self.simulator.metrics.datapacket_arrived.add(packet_copy.packet_id)
                logging.info('Latency: %s us, Throughput: %s', latency,
                             self.simulator.metrics.throughput_dict[packet_copy.packet_id])
                logging.info('DataPacket: %s delivered to UAV: %s', packet_copy.packet_id, self.my_drone.identifier)
            else:
                if self.my_drone.transmitting_queue.qsize() < self.my_drone.max_queue_size:
                    self.my_drone.transmitting_queue.put(packet_copy)
                else:
                    logging.info('Queue full, DataPacket: %s cannot be added to the queue.', packet_copy.packet_id)
        elif isinstance(packet, VfPacket):
            logging.info('At time %s, UAV: %s receives a VF packet from UAV: %s, packet id: %s',
                         current_time, self.my_drone.identifier, src_drone_id, packet.packet_id)
            self.my_drone.motion_controller.neighbor_table.add_neighbor(packet, current_time)
            if packet.msg_type == 'hello':
                config.GL_ID_VF_PACKET += 1
                ack_packet = VfPacket(src_drone=self.my_drone,
                                      creation_time=current_time,
                                      id_hello_packet=config.GL_ID_VF_PACKET,
                                      hello_packet_length=config.HELLO_PACKET_LENGTH,
                                      simulator=self.simulator)
                ack_packet.msg_type = 'ack'
                self.my_drone.transmitting_queue.put(ack_packet)

    def check_waiting_list(self):
        while True:
            if not self.my_drone.sleep:
                yield self.simulator.env.timeout(0.6 * 1e6)
                for waiting_pkd in self.my_drone.waiting_list.copy():
                    if self.simulator.env.now < waiting_pkd.creation_time + waiting_pkd.deadline:
                        self.my_drone.waiting_list.remove(waiting_pkd)
                    else:
                        has_route, pkt, _ = self.next_hop_selection(waiting_pkd)
                        if has_route:
                            self.my_drone.transmitting_queue.put(pkt)
                            self.my_drone.waiting_list.remove(waiting_pkd)
            else:
                break


def link_lifetime_predictor(drone1, drone2, max_comm_range):
    coords1 = drone1.coords
    coords2 = drone2.coords
    velocity1 = drone1.velocity
    velocity2 = drone2.velocity

    x1 = (velocity1[0] - velocity2[0]) ** 2
    x2 = (velocity1[1] - velocity2[1]) ** 2
    x3 = (velocity1[2] - velocity2[2]) ** 2

    y1 = 2 * (velocity1[0] - velocity2[0]) * (coords1[0] - coords2[0])
    y2 = 2 * (velocity1[1] - velocity2[1]) * (coords1[1] - coords2[1])
    y3 = 2 * (velocity1[2] - velocity2[2]) * (coords1[2] - coords2[2])

    z1 = (coords1[0] - coords2[0]) ** 2
    z2 = (coords1[1] - coords2[1]) ** 2
    z3 = (coords1[2] - coords2[2]) ** 2

    A = x1 + x2 + x3
    B = y1 + y2 + y3
    C = (z1 + z2 + z3) - max_comm_range ** 2

    discriminant = B ** 2 - 4 * A * C
    if discriminant < 0 or A == 0:
        return 0
    delta_t_1 = (-B + math.sqrt(discriminant)) / (2 * A)
    delta_t_2 = (-B - math.sqrt(discriminant)) / (2 * A)
    delta_t = max(delta_t_1, delta_t_2)
    return delta_t
