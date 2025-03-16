import copy
import logging
import numpy as np
from entities.packet import DataPacket
from topology.virtual_force.vf_packet import VfPacket
from utils import config
from utils.util_function import euclidean_distance
from phy.large_scale_fading import maximum_communication_range
from simulator.TrafficGenerator import TrafficRequirement

# config logging
logging.basicConfig(filename='running_log.log',
                    filemode='w',  # there are two modes: 'a' and 'w'
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=config.LOGGING_LEVEL
                    )


class SimpleDijkstraRouting:
    """
    简化版迪杰斯特拉路由协议

    仅使用迪杰斯特拉算法计算最短路径，不包含链路寿命预测和多次迭代优化。
    适用于需要简单、稳定路由决策的无人机网络。

    Attributes:
        simulator: 仿真平台实例
        my_drone: 安装此路由协议的无人机
        cost: 成本矩阵，用于记录所有链路的成本
        max_comm_range: 基于SNR阈值的最大通信范围
    """

    def __init__(self, simulator, my_drone):
        self.simulator = simulator
        self.my_drone = my_drone
        self.cost = None
        self.max_comm_range = maximum_communication_range()

        # 启动待处理列表检查进程
        self.simulator.env.process(self.check_waiting_list())

    def calculate_cost_matrix(self):
        """
        计算成本矩阵

        综合考虑距离、队列负载和能量等因素来计算链路成本

        Returns:
            numpy.ndarray: 成本矩阵
        """
        n_drones = self.simulator.n_drones
        cost = np.full((n_drones, n_drones), np.inf)

        # 设置自环成本为0
        for i in range(n_drones):
            cost[i, i] = 0

        # 计算无人机之间的链路成本
        for i in range(n_drones):
            for j in range(n_drones):
                if i == j:
                    continue

                drone1 = self.simulator.drones[i]
                drone2 = self.simulator.drones[j]

                # 计算距离
                distance = euclidean_distance(drone1.coords, drone2.coords)

                # 检查是否在通信范围内
                if distance < self.max_comm_range:
                    # 基础距离成本 (0-1)
                    distance_cost = distance / self.max_comm_range

                    # 队列负载成本 (0-1)
                    queue_cost = drone2.transmitting_queue.qsize() / drone2.max_queue_size if drone2.max_queue_size > 0 else 0

                    # 能量成本 (0-1)
                    energy_cost = 1.0 - (drone2.residual_energy / config.INITIAL_ENERGY)

                    # 综合评估 (权重可调整)
                    total_cost = (
                            0.5 * distance_cost +
                            0.3 * queue_cost +
                            0.2 * energy_cost
                    )

                    # 确保成本是正数且合理范围内
                    cost[i, j] = max(0.1, total_cost)

        return cost

    def dijkstra(self, cost, src_id, dst_id):
        """
        迪杰斯特拉算法找最短路径

        Args:
            cost: 成本矩阵
            src_id: 源节点ID
            dst_id: 目标节点ID

        Returns:
            list: 路由路径 (节点ID列表)
        """
        # 初始化距离列表、前驱列表和访问列表
        distance_list = [np.inf for _ in range(self.simulator.n_drones)]
        distance_list[src_id] = 0  # 源节点到自身距离为0

        prev_list = [-1 for _ in range(self.simulator.n_drones)]  # 前驱节点列表
        prev_list[src_id] = -2  # 标记源节点

        visited_list = [False for _ in range(self.simulator.n_drones)]  # 访问标记列表

        # 迪杰斯特拉主循环
        for _ in range(self.simulator.n_drones):
            # 找出未访问节点中距离最小的节点
            unvisited_list = [(index, value) for index, value in enumerate(distance_list) if not visited_list[index]]
            if not unvisited_list:
                break

            min_distance_node, _ = min(unvisited_list, key=lambda x: x[1])
            visited_list[min_distance_node] = True

            # 更新邻居节点的距离
            for j in range(self.simulator.n_drones):
                if (not visited_list[j]) and (cost[min_distance_node, j] != np.inf):
                    alt = distance_list[min_distance_node] + cost[min_distance_node, j]
                    if alt < distance_list[j]:
                        distance_list[j] = alt
                        prev_list[j] = min_distance_node

        # 构建路径
        current_node = dst_id
        path = [dst_id]

        # 从目标节点向源节点回溯
        while current_node != -2:
            current_node = prev_list[current_node]

            if current_node == -1:  # 没有路径到达目标
                path = []
                break

            if current_node != -2:  # 不是源节点
                path.insert(0, current_node)

        return path

    def compute_path(self, src_id, dst_id, option=0):
        """
        计算从源节点到目标节点的完整路径

        Args:
            src_id: 源节点ID
            dst_id: 目标节点ID
            option: 选项参数 (保留接口兼容性)

        Returns:
            list: 完整路由路径
        """
        # 计算成本矩阵
        self.cost = self.calculate_cost_matrix()

        # 使用迪杰斯特拉算法计算路径
        path = self.dijkstra(self.cost, src_id, dst_id)

        return path

    def next_hop_selection(self, packet):
        """
        选择下一跳节点

        Args:
            packet: 数据包对象

        Returns:
            tuple: (has_route, packet, enquire)
            - has_route: 是否有可用路由
            - packet: 更新后的数据包
            - enquire: 是否需要查询路由
        """
        enquire = False
        has_route = True

        # 记录控制包
        if not isinstance(packet, DataPacket):
            self.simulator.metrics.control_packet_num += 1

        # 源节点处理
        if packet.src_drone is self.my_drone:
            # 计算成本矩阵
            self.cost = self.calculate_cost_matrix()
            src_id = self.my_drone.identifier
            dst_id = packet.dst_drone.identifier

            # 计算路径
            path = self.dijkstra(self.cost, src_id, dst_id)

            if path and len(path) > 1:
                # 移除源节点自身
                if path[0] == src_id:
                    path.pop(0)

                # 设置路由路径和下一跳
                packet.routing_path = path
                best_next_hop_id = path[0]
                logging.info('源节点 %s 计算路由路径: %s', self.my_drone.identifier, packet.routing_path)
            else:
                # 没有找到路径
                best_next_hop_id = self.my_drone.identifier
                packet.routing_path = []
                logging.warning('未找到从 %s 到 %s 的路由路径', self.my_drone.identifier, packet.dst_drone.identifier)
        else:
            # 中继节点处理
            routing_path = packet.routing_path

            if routing_path and len(routing_path) > 0:
                # 移除当前节点
                if routing_path[0] == self.my_drone.identifier:
                    routing_path.pop(0)

                # 更新路由路径
                packet.routing_path = routing_path

                if routing_path and len(routing_path) > 0:
                    best_next_hop_id = routing_path[0]
                else:
                    best_next_hop_id = self.my_drone.identifier
            else:
                # 路径为空，可能需要重新计算
                best_next_hop_id = self.my_drone.identifier

        # 设置结果
        if best_next_hop_id is self.my_drone.identifier:
            has_route = False  # 没有可用的下一跳
        else:
            packet.next_hop_id = best_next_hop_id  # 有可用的下一跳

        return has_route, packet, enquire

    def packet_reception(self, packet, src_drone_id):
        """
        网络层数据包接收处理

        Args:
            packet: 接收到的数据包
            src_drone_id: 上一跳ID
        """
        current_time = self.simulator.env.now

        if isinstance(packet, DataPacket):
            packet_copy = copy.copy(packet)
            logging.info('数据包 %s 被无人机 %s 接收，时间: %s',
                         packet_copy.packet_id, self.my_drone.identifier, current_time)

            # 目标是当前无人机
            if packet_copy.dst_drone.identifier == self.my_drone.identifier:
                # 计算延迟和吞吐量
                latency = self.simulator.env.now - packet_copy.creation_time  # 微秒
                self.simulator.metrics.deliver_time_dict[packet_copy.packet_id] = latency
                self.simulator.metrics.throughput_dict[packet_copy.packet_id] = config.DATA_PACKET_LENGTH / (
                            latency / 1e6)
                self.simulator.metrics.hop_cnt_dict[packet_copy.packet_id] = packet_copy.get_current_ttl()
                self.simulator.metrics.datapacket_arrived.add(packet_copy.packet_id)

                logging.info('数据包 %s 已送达目标无人机 %s，延迟: %s 微秒, 吞吐量: %s',
                             packet_copy.packet_id, self.my_drone.identifier, latency,
                             self.simulator.metrics.throughput_dict[packet_copy.packet_id])
            else:
                # 目标不是当前无人机，需要转发
                if self.my_drone.transmitting_queue.qsize() < self.my_drone.max_queue_size:
                    self.my_drone.transmitting_queue.put(packet_copy)
                else:
                    logging.warning('队列已满，数据包 %s 无法加入队列', packet_copy.packet_id)

        elif isinstance(packet, VfPacket):
            logging.info('无人机 %s 接收到来自无人机 %s 的VF包 %s，时间: %s',
                         self.my_drone.identifier, src_drone_id, packet.packet_id, current_time)

            # 更新邻居表
            self.my_drone.motion_controller.neighbor_table.add_neighbor(packet, current_time)

            # 如果是hello消息，发送ACK回复
            if packet.msg_type == 'hello':
                config.GL_ID_VF_PACKET += 1
                ack_packet = VfPacket(src_drone=self.my_drone,
                                      creation_time=current_time,
                                      id_hello_packet=config.GL_ID_VF_PACKET,
                                      hello_packet_length=config.HELLO_PACKET_LENGTH,
                                      simulator=self.simulator)
                ack_packet.msg_type = 'ack'
                self.my_drone.transmitting_queue.put(ack_packet)

        elif isinstance(packet, TrafficRequirement):
            logging.info('无人机 %s 接收到业务需求 %s，源: %s, 目标: %s',
                         self.my_drone.identifier, packet.packet_id,
                         packet.source_id, packet.dest_id)

            # 如果是源节点，计算路由路径
            if self.my_drone.identifier == packet.source_id:
                path = self.compute_path(packet.source_id, packet.dest_id)
                packet.routing_path = path
                logging.info('为业务需求计算路径: %s', path)

    def check_waiting_list(self):
        """
        定期检查待处理列表中的数据包
        """
        while True:
            if not self.my_drone.sleep:
                yield self.simulator.env.timeout(0.6 * 1e6)  # 每0.6秒检查一次

                # 复制列表以防迭代时修改
                waiting_list_copy = list(self.my_drone.waiting_list)

                for waiting_pkd in waiting_list_copy:
                    # 检查数据包是否过期
                    if self.simulator.env.now >= waiting_pkd.creation_time + waiting_pkd.deadline:
                        self.my_drone.waiting_list.remove(waiting_pkd)
                        continue

                    # 尝试为等待的数据包找路由
                    has_route, updated_packet, _ = self.next_hop_selection(waiting_pkd)

                    if has_route:
                        # 找到路由，添加到发送队列并从等待列表移除
                        self.my_drone.transmitting_queue.put(updated_packet)
                        self.my_drone.waiting_list.remove(waiting_pkd)
            else:
                # 无人机休眠，退出进程
                break