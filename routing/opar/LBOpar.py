import copy
import logging
import math
import numpy as np
from entities.packet import DataPacket
from topology.virtual_force.vf_packet import VfPacket
from utils import config
from utils.util_function import euclidean_distance

# 配置日志
logging.basicConfig(filename='running_log.log',
                    filemode='w',  # 两种模式: 'a' 和 'w'
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=config.LOGGING_LEVEL
                    )


class LbOpar:
    """
    LB-OPAR: 负载均衡的优化预测自适应路由协议

    基于论文"LB-OPAR: Load balanced optimized predictive and adaptive routing
    for cooperative UAV networks"实现

    属性:
        simulator: 包含所有模拟环境的仿真器
        my_drone: 安装该路由协议的无人机
        cost: 成本矩阵，记录所有链路的成本
        edge_lifetimes: 链路生命期矩阵，记录所有链路的预测生命期
        node_loads: 节点负载向量，记录所有节点的负载情况
        best_obj: 最小目标函数值
        best_path: 对应"best_obj"的最优路由路径
        w1: 目标函数中路径长度的权重
        w2: 目标函数中路径生命期的权重
        w3: 目标函数中路径负载的权重
        max_comm_range: 最大通信范围
    """

    def __init__(self, simulator, my_drone):
        self.simulator = simulator
        self.my_drone = my_drone
        self.cost = None  # 成本矩阵
        self.edge_lifetimes = None  # 链路生命期矩阵
        self.node_loads = None  # 节点负载向量
        self.best_obj = float('inf')  # 最优目标函数值
        self.best_path = None  # 最优路径

        # 目标函数权重
        self.w1 = 0.5  # 路径长度权重，默认值
        self.w2 = 0.3  # 路径生命期权重，默认值
        self.w3 = 0.2  # 路径负载权重，默认值

        # 计算最大通信范围
        try:
            from phy.large_scale_fading import maximum_communication_range
            self.max_comm_range = maximum_communication_range()
        except ImportError:
            # 如果无法导入，使用默认值
            self.max_comm_range = 500.0

        self.simulator.env.process(self.check_waiting_list())

    def calculate_cost_matrix(self):
        """
        计算链路成本矩阵，仅考虑距离因素
        """
        n_drones = self.simulator.n_drones
        cost = np.full((n_drones, n_drones), np.inf)

        for i in range(n_drones):
            for j in range(i + 1, n_drones):
                drone1 = self.simulator.drones[i]
                drone2 = self.simulator.drones[j]

                # 计算欧几里得距离
                distance = euclidean_distance(drone1.coords, drone2.coords)

                # 如果在通信范围内，计算成本
                if distance < self.max_comm_range:
                    # 基于距离的成本
                    cost[i, j] = cost[j, i] = distance / self.max_comm_range

        return cost

    def calculate_node_loads(self):
        """
        计算节点负载
        每个节点的负载定义为其邻居范围内正在发送或转发数据的节点数量
        """
        n_drones = self.simulator.n_drones
        loads = np.zeros(n_drones)

        # 获取当前正在传输的所有流
        active_flows = []
        for drone in self.simulator.drones:
            if not drone.transmitting_queue.empty():
                active_flows.append(drone.identifier)

        # 计算每个节点的负载
        for i in range(n_drones):
            drone_i = self.simulator.drones[i]
            # 邻居节点中，活跃的发送节点数量
            active_neighbors = 0

            for j in range(n_drones):
                if i == j:
                    continue

                drone_j = self.simulator.drones[j]
                distance = euclidean_distance(drone_i.coords, drone_j.coords)

                # 如果是邻居且在活跃流列表中
                if distance < self.max_comm_range and drone_j.identifier in active_flows:
                    active_neighbors += 1

            loads[i] = active_neighbors

        return loads

    def predict_link_lifetime(self, drone1, drone2):
        """
        预测两个无人机之间的链路生命期
        基于论文中的数学模型实现

        参数:
            drone1: 第一个无人机
            drone2: 第二个无人机

        返回:
            预测的链路生命期（微秒）
        """
        # 提取当前坐标和速度
        coords1 = np.array(drone1.coords)
        coords2 = np.array(drone2.coords)
        velocity1 = np.array(drone1.velocity)
        velocity2 = np.array(drone2.velocity)

        # 计算相对速度
        rel_velocity = velocity1 - velocity2

        # 计算相对距离
        rel_distance = coords1 - coords2

        # 计算二次方程系数: A(t^2) + Bt + C = 0
        A = np.sum(rel_velocity ** 2)

        # 如果相对速度为0，生命期无限长
        if A == 0:
            return 1e11

        B = 2 * np.sum(rel_velocity * rel_distance)
        C = np.sum(rel_distance ** 2) - self.max_comm_range ** 2

        # 计算判别式
        discriminant = B ** 2 - 4 * A * C

        # 如果判别式小于0，无解，链路无法建立
        if discriminant < 0:
            return 0

        # 计算二次方程的两个解
        t1 = (-B + math.sqrt(discriminant)) / (2 * A)
        t2 = (-B - math.sqrt(discriminant)) / (2 * A)

        # 返回正的最大解作为链路生命期
        delta_t = max(t1, t2) if t1 > 0 or t2 > 0 else 0

        # 转换为微秒
        return delta_t * 1e6

    def calculate_edge_lifetimes(self):
        """
        计算所有边的生命期矩阵
        """
        n_drones = self.simulator.n_drones
        edge_lifetimes = np.full((n_drones, n_drones), 0.0)

        for i in range(n_drones):
            for j in range(i + 1, n_drones):
                drone1 = self.simulator.drones[i]
                drone2 = self.simulator.drones[j]

                # 计算距离
                distance = euclidean_distance(drone1.coords, drone2.coords)

                # 如果在通信范围内，预测链路生命期
                if distance < self.max_comm_range:
                    lifetime = self.predict_link_lifetime(drone1, drone2)
                    edge_lifetimes[i, j] = edge_lifetimes[j, i] = lifetime

        return edge_lifetimes

    def find_optimized_path(self, src_id, dst_id):
        """
        找到源节点和目标节点之间的优化路径
        基于论文中的算法1实现

        参数:
            src_id: 源节点ID
            dst_id: 目标节点ID

        返回:
            优化路径列表
        """
        # 计算成本矩阵、链路生命期矩阵和节点负载
        self.cost = self.calculate_cost_matrix()
        self.edge_lifetimes = self.calculate_edge_lifetimes()
        self.node_loads = self.calculate_node_loads()

        # 计算链路的生命期-负载成本矩阵
        n_drones = self.simulator.n_drones
        lifetime_load_costs = np.full((n_drones, n_drones), np.inf)

        for i in range(n_drones):
            for j in range(n_drones):
                if self.edge_lifetimes[i, j] > 0:  # 如果链路存在
                    # 计算链路的生命期-负载成本: w2/lifetime + w3*load
                    lifetime_cost = self.w2 / self.edge_lifetimes[i, j] if self.edge_lifetimes[i, j] > 0 else np.inf
                    load_cost = self.w3 * self.node_loads[j]
                    lifetime_load_costs[i, j] = lifetime_cost + load_cost

        # 复制当前图以进行修改
        temp_cost = self.cost.copy()

        # 重置最优解
        self.best_obj = float('inf')
        self.best_path = None

        # 使用BFS查找初始最短路径
        path = self.breadth_first_search(temp_cost, src_id, dst_id)

        # 如果无法找到路径，返回空列表
        if not path:
            return []

        # 迭代优化过程
        while path:
            # 计算当前路径的总成本和目标函数值
            total_cost = 0
            max_lifetime_load_cost = 0

            # 计算路径的总成本和找出最大的生命期-负载成本
            for i in range(len(path) - 1):
                node1 = path[i]
                node2 = path[i + 1]
                # 累加路径长度成本
                total_cost += self.cost[node1, node2]
                # 更新最大生命期-负载成本
                if lifetime_load_costs[node1, node2] > max_lifetime_load_cost:
                    max_lifetime_load_cost = lifetime_load_costs[node1, node2]

            # 计算目标函数值: w1*路径长度 + 生命期-负载成本
            obj_value = self.w1 * total_cost + max_lifetime_load_cost

            # 更新最优解
            if obj_value < self.best_obj:
                self.best_obj = obj_value
                self.best_path = path.copy()

            # 移除所有生命期-负载成本大于等于当前最大值的链路
            for i in range(n_drones):
                for j in range(n_drones):
                    if lifetime_load_costs[i, j] >= max_lifetime_load_cost:
                        temp_cost[i, j] = np.inf

            # 寻找新的路径
            path = self.breadth_first_search(temp_cost, src_id, dst_id)

        return self.best_path

    def breadth_first_search(self, cost, src_id, dst_id):
        """
        使用广度优先搜索查找从源节点到目标节点的最短路径

        参数:
            cost: 成本矩阵
            src_id: 源节点ID
            dst_id: 目标节点ID

        返回:
            最短路径列表
        """
        n_drones = self.simulator.n_drones

        # 初始化访问状态、距离和前驱节点
        visited = [False] * n_drones
        distance = [float('inf')] * n_drones
        predecessor = [-1] * n_drones

        # 设置源节点
        distance[src_id] = 0
        queue = [src_id]
        visited[src_id] = True

        # BFS遍历
        while queue:
            current = queue.pop(0)

            # 如果到达目标节点，结束搜索
            if current == dst_id:
                break

            # 遍历所有邻居
            for neighbor in range(n_drones):
                if not visited[neighbor] and cost[current, neighbor] != np.inf:
                    visited[neighbor] = True
                    distance[neighbor] = distance[current] + cost[current, neighbor]
                    predecessor[neighbor] = current
                    queue.append(neighbor)

        # 如果目标节点不可达，返回空路径
        if predecessor[dst_id] == -1:
            return []

        # 重建路径
        path = [dst_id]
        while path[0] != src_id:
            path.insert(0, predecessor[path[0]])

        return path

    def next_hop_selection(self, packet):
        """
        选择下一跳节点

        参数:
            packet: 数据包

        返回:
            has_route: 是否有路由
            packet: 更新后的数据包
            enquire: 是否需要询问
        """
        enquire = False
        has_route = True

        # 记录路由控制包
        if not isinstance(packet, DataPacket):
            self.simulator.metrics.control_packet_num += 1

        if packet.src_drone is self.my_drone:  # 如果是源节点，需要执行优化
            src_drone = self.my_drone
            dst_drone = packet.dst_drone

            # 使用优化算法找到最佳路径
            path = self.find_optimized_path(src_drone.identifier, dst_drone.identifier)

            if path:
                # 移除源节点自身
                path.pop(0)
                packet.routing_path = path

                if path:
                    best_next_hop_id = path[0]
                    logging.info('my_drone: %s routing Path: %s', self.my_drone.identifier, packet.routing_path)
                else:
                    # 如果路径只有源节点，说明目标就是下一跳
                    best_next_hop_id = dst_drone.identifier
            else:
                # 无法找到路径
                has_route = False
                best_next_hop_id = self.my_drone.identifier
        else:  # 中继节点不需要额外计算
            routing_path = packet.routing_path
            if routing_path and len(routing_path) > 1:
                routing_path.pop(0)  # 移除当前节点
                packet.routing_path = routing_path
                best_next_hop_id = routing_path[0]
            else:
                # 如果路径为空或只有一个节点，目标就是下一跳
                best_next_hop_id = packet.dst_drone.identifier

        # 如果下一跳是自己，说明没有可用路由
        if best_next_hop_id == self.my_drone.identifier:
            has_route = False
        else:
            packet.next_hop_id = best_next_hop_id

        return has_route, packet, enquire

    def packet_reception(self, packet, src_drone_id):
        """
        网络层的数据包接收处理

        参数:
            packet: 接收到的数据包
            src_drone_id: 前一跳节点ID
        """
        current_time = self.simulator.env.now

        if isinstance(packet, DataPacket):
            packet_copy = copy.copy(packet)
            logging.info('~~~ DataPacket: %s is received by UAV: %s at time: %s',
                         packet_copy.packet_id, self.my_drone.identifier, current_time)

            # 如果数据包的目标是当前无人机，则进行处理
            if packet_copy.dst_drone.identifier == self.my_drone.identifier:
                # 计算延迟和吞吐量
                latency = self.simulator.env.now - packet_copy.creation_time  # 微秒
                self.simulator.metrics.deliver_time_dict[packet_copy.packet_id] = latency
                self.simulator.metrics.throughput_dict[packet_copy.packet_id] = config.DATA_PACKET_LENGTH / (
                            latency / 1e6)
                self.simulator.metrics.hop_cnt_dict[packet_copy.packet_id] = packet_copy.get_current_ttl()
                self.simulator.metrics.datapacket_arrived.add(packet_copy.packet_id)

                logging.info('Latency: %s us, Throughput: %s', latency,
                             self.simulator.metrics.throughput_dict[packet_copy.packet_id])
                logging.info('DataPacket: %s delivered to UAV: %s', packet_copy.packet_id, self.my_drone.identifier)
            else:
                # 如果目标不是当前无人机，将包放入队列中等待
                if self.my_drone.transmitting_queue.qsize() < self.my_drone.max_queue_size:
                    self.my_drone.transmitting_queue.put(packet_copy)
                else:
                    logging.info('Queue full, DataPacket: %s cannot be added to the queue.', packet_copy.packet_id)

        elif isinstance(packet, VfPacket):
            logging.info('At time %s, UAV: %s receives a VF packet from UAV: %s, packet id: %s',
                         current_time, self.my_drone.identifier, src_drone_id, packet.packet_id)

            # 更新邻居表
            self.my_drone.motion_controller.neighbor_table.add_neighbor(packet, current_time)

            # 如果是hello消息，发送一个ack
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
        """
        定期检查等待列表中的数据包
        """
        while True:
            if not self.my_drone.sleep:
                yield self.simulator.env.timeout(0.6 * 1e6)  # 等待600,000微秒

                for waiting_pkd in list(self.my_drone.waiting_list):  # 使用list创建副本以避免修改迭代器
                    if self.simulator.env.now >= waiting_pkd.creation_time + waiting_pkd.deadline:
                        # 数据包超时，尝试重新路由
                        has_route, _, _ = self.next_hop_selection(waiting_pkd)

                        if has_route:
                            self.my_drone.transmitting_queue.put(waiting_pkd)
                            self.my_drone.waiting_list.remove(waiting_pkd)
            else:
                break  # 如果无人机休眠，退出循环

    def adjust_weights(self, num_flows=5, network_density=50):
        """
        根据网络负载和密度调整权重

        参数:
            num_flows: 当前并发流数量
            network_density: 网络密度（无人机数量）
        """
        # 基于论文表4和表5的权重分析结果
        # 随着流量增加，w3权重增加；随着密度增加，w1和w2需要平衡

        # 根据流量数调整w3
        if num_flows <= 3:
            self.w3 = 0.0  # 低负载网络不需要考虑负载平衡
        elif num_flows <= 5:
            self.w3 = 0.1
        elif num_flows <= 7:
            self.w3 = 0.3
        elif num_flows <= 9:
            self.w3 = 0.6
        else:
            self.w3 = 0.7

        # 确保w1 + w2 + w3 = 1
        remaining = 1.0 - self.w3

        # 根据网络密度调整w1和w2的比例
        if network_density <= 60:
            # 中低密度网络，路径生命期更重要
            self.w1 = 0.3 * remaining
            self.w2 = 0.7 * remaining
        elif network_density <= 80:
            # 中等密度网络，路径长度和生命期同等重要
            self.w1 = 0.5 * remaining
            self.w2 = 0.5 * remaining
        else:
            # 高密度网络，路径长度相对更重要
            self.w1 = 0.6 * remaining
            self.w2 = 0.4 * remaining

        logging.info('Weights adjusted: w1=%f, w2=%f, w3=%f', self.w1, self.w2, self.w3)