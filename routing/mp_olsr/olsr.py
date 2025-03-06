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
        self.transmission_mode = 1

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


class Olsr:
    """
    OLSR (Optimized Link State Routing) 协议实现

    特点:
    1. MPR (Multipoint Relay) 机制减少控制消息洪泛
    2. 通过Hello和TC消息维护链路状态信息
    3. 使用Dijkstra算法计算最短路径
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
        self.hello_interval = 1e6  # Hello消息间隔 (1s)
        self.tc_interval = 3e6  # TC消息间隔 (3s)
        self.neighbor_hold_time = 3e6  # 邻居表项保持时间 (3s)
        self.topology_hold_time = 9e6  # 拓扑表项保持时间 (9s)
        self.sequence_number = 0  # 消息序列号

        # 最大通信范围
        self.max_comm_range = maximum_communication_range()

        # 启动OLSR进程
        self.simulator.env.process(self.send_hello_messages())
        self.simulator.env.process(self.send_tc_messages())
        self.simulator.env.process(self.check_tables())
        self.simulator.env.process(self.check_waiting_list())

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

        logging.info('UAV: %s has selected MPRs: %s', self.my_drone.identifier, self.mpr_set)

    def send_hello_messages(self):
        """定期发送Hello消息"""
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

                logging.info('UAV: %s is sending HELLO message at time: %s',
                             self.my_drone.identifier, self.simulator.env.now)

                # 将Hello消息放入发送队列
                self.my_drone.transmitting_queue.put(hello_msg)

                # 等待下一个Hello间隔
                yield self.simulator.env.timeout(self.hello_interval)
            else:
                break

    def send_tc_messages(self):
        """定期发送TC消息 - 只有当节点有MPR选择器时才发送"""
        while True:
            if not self.my_drone.sleep:
                # 只有当节点有MPR选择器时才发送TC消息
                if self.mpr_selector_set:
                    # 创建TC消息
                    tc_msg = OlsrTcPacket(
                        src_drone=self.my_drone,
                        creation_time=self.simulator.env.now,
                        packet_id=self.get_next_sequence_number(),
                        packet_length=config.HELLO_PACKET_LENGTH,
                        simulator=self.simulator
                    )

                    # 添加MPR选择器信息
                    tc_msg.mpr_selector_list = list(self.mpr_selector_set)

                    # 设置为广播模式
                    tc_msg.transmission_mode = 1

                    logging.info('UAV: %s is sending TC message at time: %s',
                                 self.my_drone.identifier, self.simulator.env.now)

                    # 将TC消息放入发送队列
                    self.my_drone.transmitting_queue.put(tc_msg)

                # 等待下一个TC间隔
                yield self.simulator.env.timeout(self.tc_interval)
            else:
                break

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
                yield self.simulator.env.timeout(1e6)  # 1s
            else:
                break

    def get_next_sequence_number(self):
        """获取下一个序列号"""
        self.sequence_number += 1
        return self.sequence_number

    def calculate_routing_table(self):
        """计算路由表 - 使用Dijkstra算法"""
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

            # 处理直接邻居
            if current == self.my_drone.identifier:
                for neighbor_id in self.neighbor_table:
                    if dist[neighbor_id] > 1:
                        dist[neighbor_id] = 1
                        prev[neighbor_id] = current

            # 处理拓扑表中的连接
            if current in self.topology_table:
                for last_hop, _ in self.topology_table[current]:
                    if last_hop in unprocessed and dist[last_hop] > dist[current] + 1:
                        dist[last_hop] = dist[current] + 1
                        prev[last_hop] = current

            # 处理邻居节点的邻居
            if current in self.two_hop_neighbors:
                for two_hop in self.two_hop_neighbors[current]:
                    if two_hop in unprocessed and dist[two_hop] > dist[current] + 1:
                        dist[two_hop] = dist[current] + 1
                        prev[two_hop] = current

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

        logging.info('UAV: %s updated routing table: %s', self.my_drone.identifier, self.routing_table)

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

                logging.info('UAV: %s selects next hop: %s for data packet (id: %s) to destination: %s',
                             self.my_drone.identifier, next_hop_id, packet.packet_id, dst_id)
            else:
                # 无路由
                has_route = False
                logging.info('UAV: %s has no route to destination: %s', self.my_drone.identifier, dst_id)
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

    def packet_reception(self, packet, src_drone_id):
        """处理接收到的数据包"""
        current_time = self.simulator.env.now
        logging.info('UAV: %s received olsr packet: %s from UAV: %s at time: %s', self.my_drone.identifier,self.my_drone.identifier, packet,
                     src_drone_id, current_time)
        if isinstance(packet, OlsrHelloPacket):
            # 处理Hello消息
            logging.info('UAV: %s received olsr hello packet: %s from UAV: %s at time: %s', self.my_drone.identifier,
                         self.my_drone.identifier, packet,
                         src_drone_id, current_time)
            self.process_hello_message(packet, src_drone_id, current_time)
        elif isinstance(packet, OlsrTcPacket):
            # 处理TC消息
            logging.info('UAV: %s received TC packet: %s from UAV: %s at time: %s', self.my_drone.identifier,
                         self.my_drone.identifier, packet,
                         src_drone_id, current_time)
            self.process_tc_message(packet, src_drone_id, current_time)
        elif isinstance(packet, DataPacket):
            # 处理数据包
            logging.info('UAV: %s received DATa packet: %s from UAV: %s at time: %s', self.my_drone.identifier,
                         self.my_drone.identifier, packet,
                         src_drone_id, current_time)
            self.process_data_packet(packet, src_drone_id, current_time)

    def process_hello_message(self, packet, src_drone_id, current_time):
        """处理Hello消息"""
        logging.info('UAV: %s received HELLO message from UAV: %s at time: %s',
                     self.my_drone.identifier, src_drone_id, current_time)

        # 更新邻居表
        position = self.simulator.drones[src_drone_id].coords
        self.neighbor_table[src_drone_id] = (position, current_time)

        # 更新二跳邻居表
        self.two_hop_neighbors[src_drone_id] = set(packet.neighbor_list)

        # 如果我在它的MPR选择器列表中，更新我的MPR选择器集合
        if self.my_drone.identifier in packet.mpr_selector_list:
            self.mpr_selector_set.add(src_drone_id)
        else:
            self.mpr_selector_set.discard(src_drone_id)

        # 重新计算MPR
        self.compute_mprs()

        # 重新计算路由表
        self.calculate_routing_table()

    def process_tc_message(self, packet, src_drone_id, current_time):
        """处理TC消息"""
        logging.info('UAV: %s received TC message from UAV: %s at time: %s',
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

                # 放入发送队列
                self.my_drone.transmitting_queue.put(tc_copy)

            # 重新计算路由表
            self.calculate_routing_table()

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

            logging.info('DataPacket: %s delivered to UAV: %s, Latency: %s us',
                         packet_copy.packet_id, self.my_drone.identifier, latency)
        else:
            # 转发节点，将包放入队列中
            if self.my_drone.transmitting_queue.qsize() < self.my_drone.max_queue_size:
                self.my_drone.transmitting_queue.put(packet_copy)
                logging.info('UAV: %s forwarding DataPacket: %s', self.my_drone.identifier, packet_copy.packet_id)
            else:
                logging.info('UAV: %s queue full, DataPacket: %s dropped', self.my_drone.identifier,
                             packet_copy.packet_id)

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
                            logging.info('UAV: %s found route for waiting packet: %s',
                                         self.my_drone.identifier, waiting_pkd.packet_id)
                    else:
                        # 数据包过期，移除
                        self.my_drone.waiting_list.remove(waiting_pkd)
                        logging.info('UAV: %s removed expired waiting packet: %s',
                                     self.my_drone.identifier, waiting_pkd.packet_id)
            else:
                break