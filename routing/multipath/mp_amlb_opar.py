import copy
import logging
import math
import numpy as np
import random
from entities.packet import DataPacket
from topology.virtual_force.vf_packet import VfPacket
from utils import config
from utils.util_function import euclidean_distance
from phy.large_scale_fading import maximum_communication_range
from simulator.TrafficGenerator import TrafficRequirement
from routing.opar.opar import Opar
from collections import defaultdict




class MP_AMLB_OPAR(Opar):
    def __init__(self, simulator, my_drone):
        super().__init__(simulator, my_drone)

        # 多路径相关属性
        self.max_paths = config.MAX_PATHS if hasattr(config, 'MAX_PATHS') else 3
        self.path_cache = {}  # {dst_id: [path_list]}
        self.path_stats = {}  # {dst_id: {path_id: {metrics}}}
        self.current_path_index = {}  # {dst_id: index}
        self.routing_table = {}  # 添加一个空的路由表以兼容

        # 流量相关
        self.flow_to_path = {}  # {flow_id: path_index} 记录每个流量使用的路径
        self.path_load = {}  # {dst_id: {path_index: load}} 记录每条路径的负载

        # 多路径传输模式：'round_robin', 'parallel', 'adaptive'
        self.multipath_mode = getattr(config, 'PATH_SELECTION_STRATEGY', 'parallel')

        # 启动路径监控进程
        self.simulator.env.process(self.monitor_paths())

        # 启动负载监控
        self.simulator.env.process(self.monitor_path_load())

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

                # 更新路径统计
                self._update_path_stats(packet_copy)

                # 将数据包ID添加到已接收集合
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

    def _update_path_stats(self, packet):
        """更新路径统计数据"""
        if not hasattr(packet, 'flow_id') or not hasattr(packet, 'path_index'):
            return

        dst_id = packet.dst_drone.identifier
        flow_id = packet.flow_id
        path_index = packet.path_index

        if dst_id in self.path_stats and path_index in self.path_stats[dst_id]:
            path_id = f"path_{self.my_drone.identifier}_{dst_id}_{path_index}"
            if path_id in self.path_stats[dst_id]:
                stats = self.path_stats[dst_id][path_id]
                # 更新统计数据
                latency = self.simulator.env.now - packet.creation_time
                stats['delay'] = (stats['delay'] * stats['usage_count'] + latency) / (stats['usage_count'] + 1)
                stats['usage_count'] += 1
                stats['last_used'] = self.simulator.env.now
                # 计算吞吐量(bps)
                stats['throughput'] = (config.DATA_PACKET_LENGTH * stats['usage_count']) / (
                            self.simulator.env.now / 1e6)

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

        # 先尝试计算主路径
        main_cost = self.calculate_cost_matrix()
        main_path = self.dijkstra(main_cost, src_id, dst_id, 0)

        if main_path and len(main_path) > 1:
            main_path.pop(0)
            paths.append(main_path)

            # 使用不同的jitter参数多次调用dijkstra算法寻找备用路径
            for i in range(1, self.max_paths):
                # 增加jitter以促使算法发现不同路径
                jitter = 0.05 * (i + 1)
                temp_cost = self.calculate_cost_matrix(jitter=jitter)

                path = self.dijkstra(temp_cost, src_id, dst_id, 0)
                path.pop(0)
                # 如果找到有效路径且与已有路径有足够差异
                if path and len(path) > 1 and self._is_path_diverse(path, paths):
                    paths.append(path)
                    logging.info(f"AMLB-OPAR: 找到从 {src_id} 到 {dst_id} 的第 {len(paths)} 条路径: {path}")

        logging.info(f"AMLB-OPAR: 共找到 {len(paths)} 条从 {src_id} 到 {dst_id} 的路径")
        logging.info(f"AMLB-OPAR: 所有路径: {paths}")
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

        # 生成唯一流ID
        if not hasattr(packet, 'flow_id') or not packet.flow_id:
            packet.flow_id = f"flow_{packet.src_drone.identifier}_{dst_id}_{packet.priority}"

        # 是否已经有路径索引
        has_path_index = hasattr(packet, 'path_index') and packet.path_index is not None

        # 如果是源节点，计算多条路径
        if packet.src_drone is self.my_drone:
            # 检查缓存中是否有路径
            if dst_id not in self.path_cache or not self.path_cache[dst_id]:
                # 没有缓存路径，发现新路径
                paths = self.discover_multiple_paths(self.my_drone.identifier, dst_id)

                if paths:
                    self.path_cache[dst_id] = paths

                    # 初始化负载信息
                    if dst_id not in self.path_load:
                        self.path_load[dst_id] = {}

                    for i in range(len(paths)):
                        self.path_load[dst_id][i] = 0

                    # 初始化路径统计
                    if dst_id not in self.path_stats:
                        self.path_stats[dst_id] = {}

                    for i, path in enumerate(paths):
                        path_id = f"path_{self.my_drone.identifier}_{dst_id}_{i}"
                        self.path_stats[dst_id][path_id] = self._initialize_path_stats(path)

            # 选择合适的路径
            if self.multipath_mode == 'parallel' and not has_path_index:
                # 首次处理此数据包，分配路径索引
                path_index = self._assign_path_for_flow(packet.flow_id, dst_id)
                packet.path_index = path_index
                logging.info(f"AMLB-OPAR: 流 {packet.flow_id} 使用 {path_index} 号路径")
            elif not has_path_index:
                # 默认使用轮询方式
                path_index = self._select_next_path_index(dst_id)
                packet.path_index = path_index

            # 根据路径索引获取路径
            selected_path = self._get_path_by_index(dst_id, packet.path_index)

            if selected_path:
                # 更新路径负载
                if dst_id in self.path_load and packet.path_index in self.path_load[dst_id]:
                    self.path_load[dst_id][packet.path_index] += 1

                # 设置路由信息
                # if selected_path:
                #     selected_path.pop(0)  # 确保路径不包含源节点自身

                # packet.routing_path = selected_path  # 设置完整路径
                packet.routing_path = selected_path[1:]  # 排除源节点
                if len(selected_path) > 1:
                    packet.next_hop_id = selected_path[1]
                    logging.info('MP-AMLB-OPAR: my_drone: %s 选择路径: %s',
                                 self.my_drone.identifier, packet.routing_path)
                    return True, packet, False

            # 没有找到路径
            has_route = False
        else:
            # 对于中继节点，按OPAR的方式处理
            routing_path = packet.routing_path
            if routing_path and len(routing_path) > 0:
                next_hop = routing_path[0]
                routing_path = routing_path[1:]  # 移除已使用的下一跳
                packet.routing_path = routing_path
                packet.next_hop_id = next_hop
                return True, packet, False
            else:
                has_route = False

        return has_route, packet, enquire

    def _assign_path_for_flow(self, flow_id, dst_id):
        """为流量分配固定路径"""
        if flow_id in self.flow_to_path:
            return self.flow_to_path[flow_id]

        # 检查是否有可用路径
        if dst_id not in self.path_cache or not self.path_cache[dst_id]:
            return 0

        # 选择负载最小的路径
        path_index = 0
        min_load = float('inf')

        if dst_id in self.path_load:
            for idx, load in self.path_load[dst_id].items():
                if load < min_load:
                    min_load = load
                    path_index = idx

        # 记录分配给该流的路径
        self.flow_to_path[flow_id] = path_index
        return path_index

    def _select_next_path_index(self, dst_id):
        """轮询选择下一个路径索引"""
        if dst_id not in self.path_cache or not self.path_cache[dst_id]:
            return 0

        if dst_id not in self.current_path_index:
            self.current_path_index[dst_id] = 0

        paths = self.path_cache[dst_id]
        index = self.current_path_index[dst_id]
        self.current_path_index[dst_id] = (index + 1) % len(paths)

        return index

    def _get_path_by_index(self, dst_id, path_index):
        """根据索引获取路径"""
        if dst_id not in self.path_cache or not self.path_cache[dst_id]:
            return None

        paths = self.path_cache[dst_id]
        if path_index >= len(paths):
            path_index = 0

        return paths[path_index]

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

                        # 更新使用此路径的流分配
                        for flow_id, idx in list(self.flow_to_path.items()):
                            if idx == path_index:
                                # 为此流重新分配路径
                                new_idx = self._select_next_path_index(dst_id)
                                self.flow_to_path[flow_id] = new_idx
                                logging.info(f"AMLB-OPAR: 流 {flow_id} 的路径已失效，重新分配路径 {new_idx}")

                        # 如果没有路径了，发现新路径
                        if not self.path_cache[dst_id]:
                            new_paths = self.discover_multiple_paths(
                                self.my_drone.identifier, dst_id)
                            if new_paths:
                                self.path_cache[dst_id] = new_paths

    def monitor_path_load(self):
        """监控路径负载并重新平衡"""
        while True:
            yield self.simulator.env.timeout(2 * 1e6)  # 每2秒检查一次

            # 检查每个目的地的路径负载情况
            for dst_id in self.path_load:
                if dst_id not in self.path_cache or not self.path_cache[dst_id]:
                    continue

                # 计算负载总和和平均值
                total_load = sum(self.path_load[dst_id].values())
                if total_load == 0:
                    continue

                avg_load = total_load / len(self.path_load[dst_id])

                # 检查是否有路径负载显著高于平均值
                imbalanced = False
                for idx, load in self.path_load[dst_id].items():
                    if load > avg_load * 1.5:  # 负载超过平均值的1.5倍
                        imbalanced = True
                        break

                if imbalanced and self.multipath_mode == 'adaptive':
                    # 需要重新平衡
                    self._rebalance_flows(dst_id)

                # 定期重置负载计数，以适应动态变化
                for idx in self.path_load[dst_id]:
                    # 衰减而不是完全重置
                    self.path_load[dst_id][idx] = max(0, self.path_load[dst_id][idx] * 0.8)

    def _rebalance_flows(self, dst_id):
        """重新平衡到特定目的地的流量"""
        # 获取所有使用此目的地路径的流
        target_flows = {}
        for flow_id, path_idx in self.flow_to_path.items():
            if flow_id.split('_')[2] == str(dst_id):  # flow_src_dst_priority
                target_flows[flow_id] = path_idx

        if not target_flows:
            return

        # 找出负载最高和最低的路径
        max_load = -1
        min_load = float('inf')
        max_path = None
        min_path = None

        for idx, load in self.path_load[dst_id].items():
            if load > max_load:
                max_load = load
                max_path = idx
            if load < min_load:
                min_load = load
                min_path = idx

        if max_path is None or min_path is None or max_path == min_path:
            return

        # 找出使用负载最高路径的流
        flows_on_max_path = [f for f, p in target_flows.items() if p == max_path]

        if flows_on_max_path:
            # 随机选择一个流转移到负载最低的路径
            flow_to_move = random.choice(flows_on_max_path)
            self.flow_to_path[flow_to_move] = min_path

            # 更新负载计数
            self.path_load[dst_id][max_path] -= 1
            self.path_load[dst_id][min_path] += 1

            logging.info(f"AMLB-OPAR: 负载平衡 - 将流 {flow_to_move} 从路径 {max_path} 转移到路径 {min_path}")

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

        return True

    def compute_path(self, source_id, dest_id, options=0):
        """
        兼容MP_OLSR的接口，计算从source_id到dest_id的路径

        :param source_id: 源节点ID
        :param dest_id: 目的节点ID
        :param options: 选项，0:常规路径，1:其他方式
        :return: 路径列表
        """
        logging.info(f"【AMLB-OPAR】计算从节点{source_id}到节点{dest_id}的路径，选项={options}")

        # 基于不同选项使用不同的路径计算方法
        if options == 1:
            # 类似于融合路径的功能，可以调用discover_multiple_paths并合并
            paths = self.discover_multiple_paths(source_id, dest_id)
            if paths:
                # 返回第一条路径或合并路径
                return self._merge_paths(paths, source_id, dest_id)
            return []

        # 默认使用标准OPAR路径
        cost_matrix = self.calculate_cost_matrix()
        path = self.dijkstra(cost_matrix, source_id, dest_id, 0)

        if not path:
            return []

        # 确保路径中包含源节点
        if path and path[0] != source_id:
            path.insert(0, source_id)

        return path

    def _merge_paths(self, paths, source_id, dest_id):
        """
        合并多条路径，模拟MP_OLSR的融合路径功能
        简单实现，实际应用中可以根据需要优化
        """
        if not paths:
            return []

        # 如果只有一条路径，直接返回
        if len(paths) == 1:
            return paths[0]

        # 创建节点集合用于合并
        nodes_set = set()
        for path in paths:
            nodes_set.update(path)

        # 确保源节点和目标节点在路径中
        nodes_set.add(source_id)
        nodes_set.add(dest_id)

        # 构建一个临时图
        graph = defaultdict(set)
        for path in paths:
            for i in range(len(path) - 1):
                if i + 1 < len(path):
                    node1 = path[i]
                    node2 = path[i + 1]
                    graph[node1].add(node2)
                    graph[node2].add(node1)  # 双向连接

        # 使用BFS从源节点开始构建一条到目标节点的路径
        visited = {source_id}
        queue = [(source_id, [source_id])]

        while queue:
            node, path = queue.pop(0)

            if node == dest_id:
                return path

            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        # 如果无法构建完整路径，则返回第一条路径
        return paths[0] if paths else []