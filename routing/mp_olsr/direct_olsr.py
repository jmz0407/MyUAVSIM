import logging
import math
import numpy as np
from collections import defaultdict
import copy
from entities.packet import Packet, DataPacket
from utils import config
from utils.util_function import euclidean_distance
from phy.large_scale_fading import maximum_communication_range

# config logging
logging.basicConfig(filename='running_log.log',
                    filemode='w',  # there are two modes: 'a' and 'w'
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=config.LOGGING_LEVEL
                    )


class OlsrHelloPacket(Packet):
    """OLSR Hello消息"""

    def __init__(self,
                 src_drone,
                 creation_time,
                 packet_id,
                 packet_length,
                 simulator):
        super().__init__(packet_id, packet_length, creation_time, simulator)

        self.src_drone = src_drone
        self.neighbor_list = []  # 邻居列表
        self.mpr_selector_list = []  # MPR选择器列表
        self.transmission_mode = 1  # 设置为广播模式


class OlsrTcPacket(Packet):
    """OLSR Topology Control消息"""

    def __init__(self,
                 src_drone,
                 creation_time,
                 packet_id,
                 packet_length,
                 simulator):
        super().__init__(packet_id, packet_length, creation_time, simulator)

        self.src_drone = src_drone
        self.mpr_selector_list = []  # MPR选择器列表
        self.transmission_mode = 1  # 设置为广播模式


class DirectOlsr:
    """
    直接发送OLSR - 绕过MAC层直接通过仿真器传递Hello包

    特点:
    1. Hello消息直接通过仿真器发送，无需经过MAC层处理
    2. 其他功能与标准OLSR相同
    """

    def __init__(self, simulator, my_drone):
        self.simulator = simulator
        self.my_drone = my_drone

        # OLSR特有数据结构
        self.neighbor_table = {}  # 一跳邻居表 - {drone_id: (position, last_update_time)}
        self.two_hop_neighbors = {}  # 二跳邻居表 - {neighbor_id: set(two_hop_neighbors)}
        self.mpr_set = set()  # MPR集合 - 节点选择的MPR
        self.mpr_selector_set = set()  # MPR选择器集 - 选择本节点作为MPR的邻居
        self.topology_table = {}  # 拓扑表 - {destination_id: [(last_hop, timestamp)]}
        self.routing_table = {}  # 路由表 - {destination_id: next_hop_id}

        # 时间间隔和序列号
        self.hello_interval = 2e5  # Hello消息间隔 (0.2s)，提高频率加快收敛
        self.tc_interval = 5e5  # TC消息间隔 (0.5s)
        self.neighbor_hold_time = 1e6  # 邻居表项保持时间 (1s)
        self.topology_hold_time = 3e6  # 拓扑表项保持时间 (3s)
        self.sequence_number = 0  # 消息序列号

        # 最大通信范围
        self.max_comm_range = maximum_communication_range()

        # 启动OLSR进程
        # self.simulator.env.process(self.send_hello_messages_direct())
        # self.simulator.env.process(self.send_tc_messages_direct())
        self.simulator.env.process(self.global_table_sync())
        self.simulator.env.process(self.check_tables())
        self.simulator.env.process(self.check_waiting_list())
        self.simulator.env.process(self.periodic_table_print())
        logging.info(f"DirectOLSR初始化: UAV-{my_drone.identifier}, 最大通信范围={self.max_comm_range}米")

    def periodic_table_print(self):
        """定期打印OLSR表格信息"""
        while True:
            if not self.my_drone.sleep:
                yield self.simulator.env.timeout(5 * 1e6)  # 每5秒打印一次
                self.print_neighbor_tables()
            else:
                break  # 如果无人机休眠则退出

    def global_table_sync(self):
        """定期与全局邻居表同步"""
        while True:
            if not self.my_drone.sleep:
                # 从0.2秒增加到0.1秒，提高同步频率
                yield self.simulator.env.timeout(1 * 1e5)
                self.use_global_neighbor_table()
                # 每次同步后重新计算路由表
                self.calculate_routing_table()
            else:
                break
    def compute_mprs(self):
        """计算MPR集合 - 使用贪心算法选择能覆盖所有二跳邻居的最小邻居集合"""
        # 重置MPR集合
        self.mpr_set = set()

        # 获取一跳邻居集合
        one_hop_neighbors = set(self.neighbor_table.keys())

        # 获取二跳邻居集合(不包括一跳邻居和自身)
        all_two_hop_neighbors = set()
        for neighbors in self.two_hop_neighbors.values():
            all_two_hop_neighbors.update(neighbors)
        all_two_hop_neighbors -= one_hop_neighbors
        all_two_hop_neighbors.discard(self.my_drone.identifier)

        # 如果没有二跳邻居，则不需要MPR
        if not all_two_hop_neighbors:
            return

        # 选择MPR的贪心算法
        remaining_two_hop = all_two_hop_neighbors.copy()

        # 计算每个一跳邻居能够覆盖的二跳邻居数量
        while remaining_two_hop:
            max_coverage = -1
            best_neighbor = None

            for neighbor_id in one_hop_neighbors:
                if neighbor_id in self.mpr_set:
                    continue  # 已经是MPR的节点跳过

                # 计算该邻居能覆盖的剩余二跳邻居数量
                coverage = len(remaining_two_hop.intersection(self.two_hop_neighbors.get(neighbor_id, set())))

                if coverage > max_coverage:
                    max_coverage = coverage
                    best_neighbor = neighbor_id

            if best_neighbor is None or max_coverage == 0:
                break  # 没有邻居能覆盖更多二跳邻居

            # 添加最佳邻居到MPR集合
            self.mpr_set.add(best_neighbor)

            # 更新剩余未覆盖的二跳邻居
            remaining_two_hop -= self.two_hop_neighbors.get(best_neighbor, set())

        logging.info('【DirectOLSR】UAV: %s has selected MPRs: %s',
                     self.my_drone.identifier, self.mpr_set)

    def send_hello_messages_direct(self):
        """直接发送Hello消息，绕过MAC层"""
        while True:
            if not self.my_drone.sleep:
                # 创建Hello消息
                hello_msg = OlsrHelloPacket(
                    src_drone=self.my_drone,
                    creation_time=self.simulator.env.now,
                    packet_id=self.get_next_sequence_number(),
                    packet_length=config.HELLO_PACKET_LENGTH,
                    simulator=self.simulator
                )

                # 添加邻居信息
                hello_msg.neighbor_list = list(self.neighbor_table.keys())
                hello_msg.mpr_selector_list = list(self.mpr_selector_set)

                # 设置为广播模式
                hello_msg.transmission_mode = 1

                logging.info('【DirectOLSR】UAV: %s 直接发送HELLO消息，邻居: %s, 时间: %s',
                             self.my_drone.identifier, hello_msg.neighbor_list, self.simulator.env.now)

                # 直接发送Hello消息到所有邻居
                self.direct_broadcast_hello(hello_msg)

                # 等待下一个Hello间隔
                yield self.simulator.env.timeout(self.hello_interval)
            else:
                break

    def direct_broadcast_hello(self, hello_msg):
        """直接广播Hello消息到所有通信范围内的节点"""
        current_time = self.simulator.env.now
        sender_id = self.my_drone.identifier

        # 遍历所有节点，检查是否在通信范围内
        for drone in self.simulator.drones:
            if drone.identifier != sender_id:  # 不发给自己
                distance = euclidean_distance(self.my_drone.coords, drone.coords)

                if distance <= self.max_comm_range:
                    # 在通信范围内，直接传递消息
                    # 创建消息副本
                    msg_copy = copy.copy(hello_msg)

                    # 使用仿真器的channel直接传递消息
                    # 创建inbox消息格式: [packet, insertion_time, transmitter_id, processed]
                    message = [msg_copy, current_time, sender_id, 0]

                    # 添加到目标inbox
                    drone.inbox.append(message)

                    logging.info('【DirectOLSR】UAV: %s 直接发送Hello消息到UAV: %s，距离=%.2f米',
                                 sender_id, drone.identifier, distance)

                    # 立即处理接收到的Hello消息 - 确保不受MAC层影响
                    self.simulator.env.process(self.immediate_hello_processing(drone, msg_copy, sender_id))

    def immediate_hello_processing(self, receiving_drone, packet, sender_id):
        """立即处理接收到的Hello消息"""
        # 创建一个非常短的延迟，确保消息在inbox中
        yield self.simulator.env.timeout(10)

        # 直接调用接收节点的路由协议处理Hello消息
        receiving_drone.routing_protocol.process_hello_message(packet, sender_id, self.simulator.env.now)

        logging.info('【DirectOLSR】UAV: %s 已立即处理来自UAV: %s的Hello消息',
                     receiving_drone.identifier, sender_id)

        # 重新计算路由表
        self.calculate_routing_table()

        # 打印更新后的表格
        self.print_neighbor_tables()

    # 在DirectOlsr类中修改send_tc_messages_direct方法
    def send_tc_messages_direct(self):
        """直接发送TC消息，包含所有邻居信息，不仅是MPR选择器"""
        while True:
            if not self.my_drone.sleep:
                # 不再检查是否有MPR选择器，总是发送TC消息
                # 创建TC消息
                tc_msg = OlsrTcPacket(
                    src_drone=self.my_drone,
                    creation_time=self.simulator.env.now,
                    packet_id=self.get_next_sequence_number(),
                    packet_length=config.HELLO_PACKET_LENGTH,
                    simulator=self.simulator
                )

                # 添加所有邻居信息而不只是MPR选择器
                tc_msg.mpr_selector_list = list(self.neighbor_table.keys())

                # 设置为广播模式
                tc_msg.transmission_mode = 1

                logging.info('【DirectOLSR】UAV: %s 直接发送增强型TC消息，包含所有邻居: %s, 时间: %s',
                             self.my_drone.identifier, tc_msg.mpr_selector_list, self.simulator.env.now)

                # 直接发送TC消息到所有邻居，不只是MPR
                self.direct_broadcast_tc_to_all(tc_msg)

                # 等待下一个TC间隔
                yield self.simulator.env.timeout(self.tc_interval)
            else:
                break

    def direct_broadcast_tc_to_all(self, tc_msg):
        """直接广播TC消息到所有邻居，不只是MPR"""
        current_time = self.simulator.env.now
        sender_id = self.my_drone.identifier

        # 遍历所有节点，检查是否在通信范围内
        for drone in self.simulator.drones:
            if drone.identifier != sender_id:  # 不发给自己
                distance = euclidean_distance(self.my_drone.coords, drone.coords)

                if distance <= self.max_comm_range:  # 在通信范围内就发送，不再检查是否是MPR
                    # 创建消息副本
                    msg_copy = copy.copy(tc_msg)

                    # 使用仿真器的channel直接传递消息
                    message = [msg_copy, current_time, sender_id, 0]

                    # 添加到目标inbox
                    drone.inbox.append(message)

                    logging.info('【DirectOLSR】UAV: %s 直接发送TC消息到UAV: %s，距离=%.2f米',
                                 sender_id, drone.identifier, distance)

                    # 立即处理接收到的TC消息
                    self.simulator.env.process(self.immediate_tc_processing(drone, msg_copy, sender_id))
    # def send_tc_messages_direct(self):
    #     """直接发送TC消息"""
    #     while True:
    #         if not self.my_drone.sleep:
    #             # 只有当节点有MPR选择器时才发送TC消息
    #             if self.mpr_selector_set:
    #                 # 创建TC消息
    #                 tc_msg = OlsrTcPacket(
    #                     src_drone=self.my_drone,
    #                     creation_time=self.simulator.env.now,
    #                     packet_id=self.get_next_sequence_number(),
    #                     packet_length=config.HELLO_PACKET_LENGTH,
    #                     simulator=self.simulator
    #                 )
    #
    #                 # 添加MPR选择器信息
    #                 tc_msg.mpr_selector_list = list(self.mpr_selector_set)
    #
    #                 # 设置为广播模式
    #                 tc_msg.transmission_mode = 1
    #
    #                 logging.info('【DirectOLSR】UAV: %s 直接发送TC消息，MPR选择器: %s, 时间: %s',
    #                              self.my_drone.identifier, tc_msg.mpr_selector_list, self.simulator.env.now)
    #
    #                 # 直接发送TC消息到MPR节点
    #                 self.direct_broadcast_tc(tc_msg)
    #
    #             # 等待下一个TC间隔
    #             yield self.simulator.env.timeout(self.tc_interval)
    #         else:
    #             break
    #
    # def direct_broadcast_tc(self, tc_msg):
    #     """直接广播TC消息到MPR节点"""
    #     current_time = self.simulator.env.now
    #     sender_id = self.my_drone.identifier
    #
    #     # 遍历所有节点，检查是否是MPR且在通信范围内
    #     for drone in self.simulator.drones:
    #         if drone.identifier != sender_id:  # 不发给自己
    #             distance = euclidean_distance(self.my_drone.coords, drone.coords)
    #
    #             if distance <= self.max_comm_range and drone.identifier in self.mpr_set:
    #                 # 在通信范围内且是MPR，直接传递消息
    #                 # 创建消息副本
    #                 msg_copy = copy.copy(tc_msg)
    #
    #                 # 使用仿真器的channel直接传递消息
    #                 # 创建inbox消息格式: [packet, insertion_time, transmitter_id, processed]
    #                 message = [msg_copy, current_time, sender_id, 0]
    #
    #                 # 添加到目标inbox
    #                 drone.inbox.append(message)
    #
    #                 logging.info('【DirectOLSR】UAV: %s 直接发送TC消息到MPR: %s，距离=%.2f米',
    #                              sender_id, drone.identifier, distance)
    #
    #                 # 立即处理接收到的TC消息
    #                 self.simulator.env.process(self.immediate_tc_processing(drone, msg_copy, sender_id))

    def immediate_tc_processing(self, receiving_drone, packet, sender_id):
        """立即处理接收到的TC消息"""
        # 创建一个非常短的延迟，确保消息在inbox中
        yield self.simulator.env.timeout(10)

        # 直接调用接收节点的路由协议处理TC消息
        receiving_drone.routing_protocol.process_tc_message(packet, sender_id, self.simulator.env.now)

        logging.info('【DirectOLSR】UAV: %s 已立即处理来自UAV: %s的TC消息',
                     receiving_drone.identifier, sender_id)

    def check_tables(self):
        """定期检查并清理过期表项"""
        while True:
            if not self.my_drone.sleep:
                current_time = self.simulator.env.now

                # 清理邻居表
                expired_neighbors = []
                for neighbor_id, (_, last_time) in self.neighbor_table.items():
                    if current_time - last_time > self.neighbor_hold_time:
                        expired_neighbors.append(neighbor_id)

                for neighbor_id in expired_neighbors:
                    self.neighbor_table.pop(neighbor_id, None)
                    self.two_hop_neighbors.pop(neighbor_id, None)

                # 清理拓扑表
                expired_entries = []
                for dest_id, entries in self.topology_table.items():
                    expired_entries_for_dest = []
                    for i, (last_hop, last_time) in enumerate(entries):
                        if current_time - last_time > self.topology_hold_time:
                            expired_entries_for_dest.append(i)

                    # 从后向前删除，避免索引混乱
                    for i in sorted(expired_entries_for_dest, reverse=True):
                        del entries[i]

                    if not entries:
                        expired_entries.append(dest_id)

                for dest_id in expired_entries:
                    self.topology_table.pop(dest_id, None)

                # 重新计算MPR
                self.compute_mprs()

                # 重新计算路由表
                self.calculate_routing_table()

                # 每隔一定时间检查一次
                yield self.simulator.env.timeout(1e5)  # 0.1秒
            else:
                break

    def get_next_sequence_number(self):
        """获取下一个序列号"""
        self.sequence_number += 1
        return self.sequence_number

    def calculate_routing_table(self):
        """使用全局邻居表计算路由表 - 基于Dijkstra算法"""
        # 记录旧路由表以便比较
        old_routing_table = self.routing_table.copy()

        # 重置路由表
        self.routing_table = {}

        # 使用Dijkstra算法计算最短路径
        n_drones = self.simulator.n_drones
        dist = {i: float('inf') for i in range(n_drones)}
        prev = {i: None for i in range(n_drones)}

        # 自身距离为0
        dist[self.my_drone.identifier] = 0

        # 待处理节点集合
        unprocessed = set(range(n_drones))

        # Dijkstra算法
        while unprocessed:
            # 找到距离最小的未处理节点
            current = min(unprocessed, key=lambda x: dist[x])

            if dist[current] == float('inf'):
                break  # 无法到达更多节点

            unprocessed.remove(current)

            # 直接使用全局邻居表更新距离
            if current in self.simulator.global_neighbor_table:
                for neighbor in self.simulator.global_neighbor_table[current]:
                    if neighbor in unprocessed and dist[neighbor] > dist[current] + 1:
                        dist[neighbor] = dist[current] + 1
                        prev[neighbor] = current

        # 构建路由表
        for dest in range(n_drones):
            if dest != self.my_drone.identifier and dist[dest] < float('inf'):
                # 回溯找到第一跳
                path = []
                current = dest
                while current != self.my_drone.identifier:
                    if current is None or prev[current] is None:
                        break  # 避免循环引用
                    path.append(current)
                    current = prev[current]

                if path:  # 确保找到有效路径
                    # 第一跳是下一跳
                    next_hop = path[-1]
                    self.routing_table[dest] = next_hop

        # 检查路由表是否有变化
        if self.routing_table != old_routing_table:
            logging.info('【DirectOLSR】UAV: %s 路由表更新为: %s',
                         self.my_drone.identifier, self.routing_table)

    def next_hop_selection(self, packet):
        """选择下一跳"""
        enquire = False
        has_route = True

        if isinstance(packet, DataPacket):
            dst_id = packet.dst_drone.identifier

            # 如果是目的地，就不需要路由
            if dst_id == self.my_drone.identifier:
                has_route = False
                return has_route, packet, enquire

            # 查找路由表
            if dst_id in self.routing_table:
                next_hop_id = self.routing_table[dst_id]
                packet.next_hop_id = next_hop_id

                # 如果设置了路由路径属性，更新路径
                if hasattr(packet, 'routing_path'):
                    if not packet.routing_path:
                        # 路径为空，构建新路径
                        path = self.construct_path(dst_id)
                        packet.routing_path = path
                    else:
                        # 路径不为空，移除第一个节点（自己）
                        if packet.routing_path and len(packet.routing_path) > 0:
                            packet.routing_path.pop(0)

                logging.info('【DirectOLSR】UAV: %s 为数据包(id: %s)选择下一跳: %s, 目的地: %s',
                             self.my_drone.identifier, packet.packet_id, next_hop_id, dst_id)
            else:
                # 无路由
                has_route = False
                logging.info('【DirectOLSR】UAV: %s 没有到目的地: %s 的路由',
                             self.my_drone.identifier, dst_id)
        else:
            # 非数据包，直接发送
            has_route = True

        return has_route, packet, enquire

    def construct_path(self, dst_id):
        """构建到目的地的完整路径"""
        path = []

        # 如果没有路由到目的地，返回空路径
        if dst_id not in self.routing_table:
            return path

        # 使用路由表回溯构建路径
        current = dst_id
        while current != self.my_drone.identifier:
            path.insert(0, current)

            # 查找前一跳
            prev_found = False
            for node_id, next_hop in self.routing_table.items():
                if next_hop == current:
                    current = node_id
                    prev_found = True
                    break

            if not prev_found or current in path:  # 避免循环
                break

        return path

    def process_hello_message(self, packet, src_drone_id, current_time):
        """处理Hello消息"""
        logging.info('【DirectOLSR】UAV: %s收到来自UAV: %s的HELLO消息，时间: %s，邻居列表: %s',
                     self.my_drone.identifier, src_drone_id, current_time, packet.neighbor_list)

        # 更新邻居表
        position = self.simulator.drones[src_drone_id].coords
        self.neighbor_table[src_drone_id] = (position, current_time)

        # 更新二跳邻居表 - 从邻居的邻居列表中排除自己的ID
        filtered_two_hop = set(n for n in packet.neighbor_list if n != self.my_drone.identifier)
        self.two_hop_neighbors[src_drone_id] = filtered_two_hop

        # 如果我在它的MPR选择器列表中，更新我的MPR选择器集合
        if self.my_drone.identifier in packet.mpr_selector_list:
            self.mpr_selector_set.add(src_drone_id)
        else:
            self.mpr_selector_set.discard(src_drone_id)

        # 重新计算MPR
        self.compute_mprs()

        # 重新计算路由表
        self.calculate_routing_table()

    # 在DirectOlsr和MP_OLSR类中添加方法
    def use_global_neighbor_table(self):
        """使用全局邻居表更新本地路由信息"""
        my_id = self.my_drone.identifier
        current_time = self.simulator.env.now

        # 清空当前邻居表，完全依赖全局表
        self.neighbor_table = {}
        self.two_hop_neighbors = {}
        self.topology_table = {}  # 也清空拓扑表，将通过全局表重建

        # 更新一跳邻居表
        for neighbor_id in self.simulator.global_neighbor_table.get(my_id, set()):
            position = self.simulator.drones[neighbor_id].coords
            self.neighbor_table[neighbor_id] = (position, current_time)

        # 更新二跳邻居表
        for neighbor_id, two_hop_set in self.simulator.global_two_hop_neighbors.get(my_id, {}).items():
            self.two_hop_neighbors[neighbor_id] = set(two_hop_set)

        # 构建拓扑表 - 从全局邻居表中提取
        for node_id in range(self.simulator.n_drones):
            if node_id != my_id:  # 排除自己
                if node_id in self.simulator.global_neighbor_table:
                    for neighbor_id in self.simulator.global_neighbor_table[node_id]:
                        if neighbor_id != my_id:  # 排除自己作为邻居
                            if neighbor_id not in self.topology_table:
                                self.topology_table[neighbor_id] = []
                            self.topology_table[neighbor_id].append((node_id, current_time))

        # 重新计算MPR
        self.compute_mprs()

        # 更新路由表
        self.calculate_routing_table()

        logging.info(f"UAV {my_id} 使用全局邻居表更新了路由信息，"
                     f"邻居数: {len(self.neighbor_table)}, "
                     f"二跳邻居节点数: {len(self.two_hop_neighbors)}")
    def process_tc_message(self, packet, src_drone_id, current_time):
        """处理TC消息"""
        logging.info('【DirectOLSR】UAV: %s收到来自UAV: %s的TC消息，时间: %s',
                     self.my_drone.identifier, src_drone_id, current_time)

        # 只处理来自MPR的TC消息或自己转发的TC消息
        if src_drone_id in self.mpr_set or src_drone_id == self.my_drone.identifier:
            # 更新拓扑表
            for mpr_selector in packet.mpr_selector_list:
                if mpr_selector not in self.topology_table:
                    self.topology_table[mpr_selector] = []

                # 检查是否已存在，如果存在则更新时间
                entry_exists = False
                for i, (hop, _) in enumerate(self.topology_table[mpr_selector]):
                    if hop == packet.src_drone.identifier:
                        self.topology_table[mpr_selector][i] = (hop, current_time)
                        entry_exists = True
                        break

                if not entry_exists:
                    self.topology_table[mpr_selector].append((packet.src_drone.identifier, current_time))

            # 如果我是MPR，需要转发此TC消息
            if self.mpr_set and packet.get_current_ttl() < config.MAX_TTL:
                # 避免重复转发，检查TTL和上次转发的消息ID
                tc_copy = copy.copy(packet)
                tc_copy.increase_ttl()

                # 直接转发TC消息给邻居
                self.direct_broadcast_tc_to_all(tc_copy)

            # 重新计算路由表
            self.calculate_routing_table()

    def packet_reception(self, packet, src_drone_id):
        """处理接收到的数据包"""
        current_time = self.simulator.env.now

        if isinstance(packet, OlsrHelloPacket):
            # 处理Hello消息
            self.process_hello_message(packet, src_drone_id, current_time)
        elif isinstance(packet, OlsrTcPacket):
            # 处理TC消息
            self.process_tc_message(packet, src_drone_id, current_time)
        elif isinstance(packet, DataPacket):
            # 处理数据包
            self.process_data_packet(packet, src_drone_id, current_time)

    def process_data_packet(self, packet, src_drone_id, current_time):
        """处理数据包"""
        packet_copy = copy.copy(packet)

        # 检查是否是目的地
        if packet_copy.dst_drone.identifier == self.my_drone.identifier:
            # 目的地，处理数据包
            latency = current_time - packet_copy.creation_time
            self.simulator.metrics.deliver_time_dict[packet_copy.packet_id] = latency
            self.simulator.metrics.throughput_dict[packet_copy.packet_id] = config.DATA_PACKET_LENGTH / (latency / 1e6)
            self.simulator.metrics.hop_cnt_dict[packet_copy.packet_id] = packet_copy.get_current_ttl()
            self.simulator.metrics.datapacket_arrived.add(packet_copy.packet_id)

            logging.info('【DirectOLSR】数据包: %s 送达至无人机: %s, 延迟: %s us',
                         packet_copy.packet_id, self.my_drone.identifier, latency)
        else:
            # 转发节点，将包放入队列中
            if self.my_drone.transmitting_queue.qsize() < self.my_drone.max_queue_size:
                self.my_drone.transmitting_queue.put(packet_copy)
                logging.info('【DirectOLSR】UAV: %s 转发数据包: %s',
                             self.my_drone.identifier, packet_copy.packet_id)
            else:
                logging.info('【DirectOLSR】UAV: %s 队列已满，丢弃数据包: %s',
                             self.my_drone.identifier, packet_copy.packet_id)

    def print_neighbor_tables(self):
        """打印OLSR协议的邻居表信息"""
        logging.info("=" * 50)
        logging.info(f"OLSR无人机 {self.my_drone.identifier} 的邻居表信息:")
        logging.info("=" * 50)

        # 打印一跳邻居
        logging.info("一跳邻居表:")
        if self.neighbor_table:
            for neighbor_id, (position, last_update_time) in sorted(self.neighbor_table.items()):
                time_diff = (self.simulator.env.now - last_update_time) / 1e6  # 转换为秒
                distance = euclidean_distance(self.my_drone.coords, position)
                logging.info(
                    f"  UAV {neighbor_id}: 位置={position}, 距离={distance:.2f}米, 上次更新={time_diff:.2f}秒前")
        else:
            logging.info("  没有一跳邻居")

        # 打印二跳邻居
        logging.info("\n二跳邻居表:")
        if self.two_hop_neighbors:
            for neighbor_id, two_hop_set in sorted(self.two_hop_neighbors.items()):
                logging.info(f"  通过UAV {neighbor_id}可达的二跳邻居: {sorted(two_hop_set)}")
        else:
            logging.info("  没有二跳邻居信息")

        # 打印MPR信息
        logging.info("\nMPR集合:")
        if self.mpr_set:
            logging.info(f"  选择的MPR节点: {sorted(self.mpr_set)}")
        else:
            logging.info("  没有选择MPR节点")

        logging.info("\nMPR选择器集合:")
        if self.mpr_selector_set:
            logging.info(f"  选择我作为MPR的节点: {sorted(self.mpr_selector_set)}")
        else:
            logging.info("  没有节点选择我作为MPR")

        # 打印路由表
        logging.info("\n路由表:")
        if self.routing_table:
            for dest_id, next_hop in sorted(self.routing_table.items()):
                logging.info(f"  目的地UAV {dest_id} -> 下一跳UAV {next_hop}")
        else:
            logging.info("  路由表为空")

        logging.info("=" * 50)
    def check_waiting_list(self):
        """检查等待列表中的数据包"""
        while True:
            if not self.my_drone.sleep:
                yield self.simulator.env.timeout(0.6 * 1e6)  # 每0.6秒检查一次

                for waiting_pkd in self.my_drone.waiting_list[:]:  # 使用副本遍历，避免修改原列表时的问题
                    if self.simulator.env.now < waiting_pkd.creation_time + waiting_pkd.deadline:
                        # 数据包未过期，尝试重新路由
                        has_route, packet, _ = self.next_hop_selection(waiting_pkd)

                        if has_route:
                            # 找到路由，放入发送队列
                            self.my_drone.transmitting_queue.put(packet)
                            self.my_drone.waiting_list.remove(waiting_pkd)
                            logging.info('【DirectOLSR】UAV: %s 为等待数据包找到路由: %s',
                                         self.my_drone.identifier, waiting_pkd.packet_id)
                    else:
                        # 数据包过期，移除
                        self.my_drone.waiting_list.remove(waiting_pkd)
                        logging.info('【DirectOLSR】UAV: %s 移除过期的等待数据包: %s',
                                     self.my_drone.identifier, waiting_pkd.packet_id)
            else:
                break

# DirectMP_OLSR也可以类似实现，提供多路径支持