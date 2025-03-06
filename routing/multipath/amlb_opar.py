# 创建一个继承自OPAR的新类
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
from collections import defaultdict
class AMLB_OPAR(Opar):
    def __init__(self, simulator, my_drone):
        super().__init__(simulator, my_drone)

        # 多路径相关属性
        self.max_paths = config.MAX_PATHS if hasattr(config, 'MAX_PATHS') else 3
        self.path_cache = {}  # {dst_id: [path_list]}
        self.path_stats = {}  # {dst_id: {path_id: {metrics}}}
        self.current_path_index = {}  # {dst_id: index}
        self.routing_table = {}  # 添加一个空的路由表以兼容
        # 启动路径监控进程
        self.simulator.env.process(self.monitor_paths())
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
    def calculate_cost_matrix(self, jitter=0):
        """扩展OPAR的成本矩阵计算，添加抖动以生成不同路径"""
        # 获取原始成本矩阵
        cost = super().calculate_cost_matrix()

        if jitter > 0:
            # 添加随机抖动以生成不同的路径
            noise = np.random.uniform(-jitter, jitter, cost.shape)
            # 仅对有限值添加抖动
            mask = cost != np.inf
            cost[mask] = cost[mask] * (1 + noise[mask])

        return cost

    def discover_multiple_paths(self, src_id, dst_id):
        """发现多条路径"""
        paths = []

        # 使用不同的jitter参数多次调用dijkstra算法
        for i in range(self.max_paths):
            # 增加jitter以促使算法发现不同路径
            jitter = 0.05 * (i + 1)
            temp_cost = self.calculate_cost_matrix(jitter=jitter)

            path = self.dijkstra(temp_cost, src_id, dst_id, 0)
            if len(path) != 0:
                path.pop(0)
            # 如果找到有效路径且与已有路径有足够差异
            if path and len(path) > 1 and self._is_path_diverse(path, paths):
                paths.append(path)

        # 如果需要，可以添加更复杂的路径多样性计算

        return paths

    def _is_path_diverse(self, new_path, existing_paths, threshold=0.3):
        """检查新路径是否与现有路径有足够差异"""
        if not existing_paths:
            return True

        for path in existing_paths:
            # 计算路径重叠度
            overlap = len(set(new_path) & set(path)) / len(set(new_path) | set(path))
            if overlap > (1 - threshold):
                return False

        return True

    def next_hop_selection(self, packet):
        """扩展OPAR的下一跳选择，支持多路径"""
        enquire = False
        has_route = True

        if not isinstance(packet, DataPacket):
            # 对于控制包，使用原始方法
            return super().next_hop_selection(packet)

        dst_id = packet.dst_drone.identifier

        # 如果是源节点，计算多条路径
        if packet.src_drone is self.my_drone:
            # 检查缓存中是否有路径
            if dst_id not in self.path_cache or not self.path_cache[dst_id]:
                # 没有缓存路径，发现新路径
                paths = self.discover_multiple_paths(self.my_drone.identifier, dst_id)

                if paths:
                    self.path_cache[dst_id] = paths

                    # 初始化路径统计
                    if dst_id not in self.path_stats:
                        self.path_stats[dst_id] = {}

                    for i, path in enumerate(paths):
                        path_id = f"path_{self.my_drone.identifier}_{dst_id}_{i}"
                        self.path_stats[dst_id][path_id] = self._initialize_path_stats(path)

            # 选择最佳路径
            selected_path = self._select_path(dst_id, packet)

            if selected_path:
                # 设置路由信息
                packet.routing_path = selected_path[1:]  # 排除源节点
                if len(selected_path) > 1:
                    packet.next_hop_id = selected_path[1]
                    logging.info('AMLB-OPAR: my_drone: %s 选择路径: %s',
                                 self.my_drone.identifier, packet.routing_path)
                    return True, packet, False

            # 没有找到路径
            has_route = False
        else:
            # 对于中继节点，按OPAR的方式处理
            routing_path = packet.routing_path
            if len(routing_path) > 1:
                routing_path.pop(0)
                packet.routing_path = routing_path
                packet.next_hop_id = routing_path[0]
                return True, packet, False
            else:
                has_route = False

        return has_route, packet, enquire

    def _select_path(self, dst_id, packet=None):
        """从多条路径中选择最佳路径"""
        if dst_id not in self.path_cache or not self.path_cache[dst_id]:
            return None

        paths = self.path_cache[dst_id]

        # 使用轮询方式选择路径
        if dst_id not in self.current_path_index:
            self.current_path_index[dst_id] = 0

        index = self.current_path_index[dst_id]
        self.current_path_index[dst_id] = (index + 1) % len(paths)

        return paths[index]

    def _initialize_path_stats(self, path):
        """初始化路径统计"""
        return {
            'delay': 0,
            'loss_rate': 0,
            'throughput': 0,
            'stability': 1.0,
            'usage_count': 0,
            'last_used': self.simulator.env.now
        }

    def monitor_paths(self):
        """定期监控路径状态"""
        while True:
            yield self.simulator.env.timeout(1 * 1e6)  # 每1秒检查一次

            # 更新路径状态
            for dst_id in list(self.path_cache.keys()):
                for path_index, path in enumerate(self.path_cache[dst_id]):
                    # 检查路径有效性
                    valid = self._check_path_validity(path)

                    if not valid:
                        # 移除无效路径
                        logging.info('AMLB-OPAR: 移除到 %s 的无效路径: %s',
                                     dst_id, path)
                        self.path_cache[dst_id].pop(path_index)

                        # 如果没有路径了，发现新路径
                        if not self.path_cache[dst_id]:
                            new_paths = self.discover_multiple_paths(
                                self.my_drone.identifier, dst_id)
                            if new_paths:
                                self.path_cache[dst_id] = new_paths

    def _check_path_validity(self, path):
        """检查路径是否有效"""
        # 检查路径中的每个链路
        for i in range(len(path) - 1):
            drone1 = self.simulator.drones[path[i]]
            drone2 = self.simulator.drones[path[i + 1]]

            # 检查节点是否在线
            if drone1.sleep or drone2.sleep:
                return False

            # 检查是否在通信范围内
            distance = euclidean_distance(drone1.coords, drone2.coords)
            if distance > self.max_comm_range:
                return False

            # 可以添加更多检查...

        return True

