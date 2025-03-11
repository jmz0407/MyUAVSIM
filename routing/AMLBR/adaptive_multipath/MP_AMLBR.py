import copy
import logging
import math
import numpy as np
import random
from collections import defaultdict, deque

from entities.packet import DataPacket
from topology.virtual_force.vf_packet import VfPacket
from utils import config
from utils.util_function import euclidean_distance
from phy.large_scale_fading import maximum_communication_range
from routing.AMLBR.adaptive_multipath.AMLBR import AMLBR


class MP_AMLBR(AMLBR):
    """
    真正的多路径路由协议 (MP-AMLBR)
    - 支持同时使用多条路径传输数据
    - 实现数据包分发、重组和重排序
    - 提供流量和负载均衡
    - 动态路径监控和切换
    """

    def __init__(self, simulator, my_drone):
        super().__init__(simulator, my_drone)

        # 多路径传输配置
        self.use_multipath = True  # 是否启用真正的多路径传输
        self.max_active_paths = 3  # 最大活跃路径数
        self.reorder_buffer_size = 50  # 重排序缓冲区大小
        self.duplicate_detection = True  # 是否检测重复数据包

        # 路径分配和管理
        self.active_paths = {}  # {dst_id: [active_path_indices]}
        self.path_load = {}  # {dst_id: {path_id: load}}
        self.path_delay = {}  # {dst_id: {path_id: delay}}
        self.path_failure_count = {}  # {dst_id: {path_id: failures}}

        # 数据包分发和重组
        self.packet_sequence = {}  # {flow_id: next_sequence_number}
        self.packet_distribution = {}  # {flow_id: {path_id: count}}
        self.reorder_buffer = {}  # {dst_id: {sequence: packet}}
        self.expected_sequence = {}  # {dst_id: next_expected_sequence}
        self.reorder_timeout = 5 * 1e6  # 重排序超时时间 (5s)

        # 路径质量监控
        self.path_rtt = {}  # {dst_id: {path_id: [rtt_samples]}}
        self.path_loss_rate = {}  # {dst_id: {path_id: loss_rate}}
        self.path_bandwidth = {}  # {dst_id: {path_id: bandwidth}}
        self.monitor_window = 10  # 监控窗口大小

        # 负载均衡策略
        self.load_balancing_mode = 'adaptive'  # 'round_robin', 'weighted', 'adaptive'
        self.weight_delay = 0.5  # 延迟权重
        self.weight_loss = 0.3  # 丢包率权重
        self.weight_bandwidth = 0.2  # 带宽权重

        # 数据包分片和汇聚 (可选)
        self.enable_fragmentation = False  # 是否启用数据包分片
        self.max_fragment_size = 512  # 最大分片大小(字节)
        self.fragment_buffers = {}  # {packet_id: {fragment_id: data}}

        # 启动多路径管理进程
        self.simulator.env.process(self.manage_multipath_transmission())
        self.simulator.env.process(self.process_reorder_buffer())

    def packet_reception(self, packet, src_drone_id):
        """增强的数据包接收处理，支持多路径重组"""
        current_time = self.simulator.env.now

        # 处理数据包
        if isinstance(packet, DataPacket):
            packet_copy = copy.copy(packet)
            logging.info('DataPacket: %s received by UAV: %s at: %s',
                         packet_copy.packet_id, self.my_drone.identifier, current_time)

            # 更新链路质量统计
            self.update_link_quality(src_drone_id)

            # 目标节点处理
            if packet_copy.dst_drone.identifier == self.my_drone.identifier:
                # 多路径处理：检查序列号，放入重排序缓冲区
                if hasattr(packet_copy, 'mp_sequence'):
                    self._handle_multipath_packet(packet_copy, current_time)
                else:
                    # 兼容非多路径数据包
                    self.handle_packet_at_destination(packet_copy, current_time)
            else:
                # 中继节点处理
                self._forward_multipath_packet(packet_copy, src_drone_id)
        else:
            # 将其他类型的包传递给父类处理
            super().packet_reception(packet, src_drone_id)

    def _handle_multipath_packet(self, packet, current_time):
        """处理多路径数据包：重排序和汇聚"""
        flow_id = f"flow_{packet.src_drone.identifier}_{packet.dst_drone.identifier}"
        mp_sequence = getattr(packet, 'mp_sequence', 0)

        # 初始化重排序缓冲区
        if flow_id not in self.reorder_buffer:
            self.reorder_buffer[flow_id] = {}
            self.expected_sequence[flow_id] = 0

        # 检查是否重复包
        if mp_sequence in self.reorder_buffer[flow_id]:
            logging.info('检测到重复数据包: %s，序列号: %s',
                         packet.packet_id, mp_sequence)
            return

        # 存入重排序缓冲区
        packet.arrival_time = current_time
        self.reorder_buffer[flow_id][mp_sequence] = packet

        # 记录路径统计
        if hasattr(packet, 'current_path_index') and hasattr(packet, 'routing_path'):
            dst_id = packet.dst_drone.identifier
            path_index = packet.current_path_index
            path_id = f"path_{packet.src_drone.identifier}_{dst_id}_{path_index}"

            # 计算端到端延迟
            delay = current_time - packet.creation_time

            # 更新路径RTT统计
            if dst_id not in self.path_rtt:
                self.path_rtt[dst_id] = {}
            if path_id not in self.path_rtt[dst_id]:
                self.path_rtt[dst_id][path_id] = deque(maxlen=self.monitor_window)

            self.path_rtt[dst_id][path_id].append(delay)

            # 更新路径统计
            self.update_path_stats(dst_id, path_index, {
                'success': True,
                'latency': delay
            })

            logging.info('多路径数据包 %s 通过路径 %s 到达，延迟: %s us',
                         packet.packet_id, path_id, delay)

        # 检查是否可以按序递交数据包
        self._deliver_in_order_packets(flow_id)

    def _deliver_in_order_packets(self, flow_id):
        """按序递交数据包"""
        if flow_id not in self.reorder_buffer or flow_id not in self.expected_sequence:
            return

        buffer = self.reorder_buffer[flow_id]
        expected = self.expected_sequence[flow_id]
        delivered_count = 0

        # 递交所有连续的数据包
        while expected in buffer:
            packet = buffer[expected]

            # 递交数据包
            latency = self.simulator.env.now - packet.creation_time
            hop_count = packet.get_current_ttl()

            # 记录性能指标
            self.simulator.metrics.deliver_time_dict[packet.packet_id] = latency
            self.simulator.metrics.throughput_dict[packet.packet_id] = (
                    config.DATA_PACKET_LENGTH / (latency / 1e6)
            )
            self.simulator.metrics.hop_cnt_dict[packet.packet_id] = hop_count

            # 记录全局统计
            self.simulator.metrics.record_packet_reception(
                packet.packet_id, latency, hop_count)
            self.simulator.metrics.datapacket_arrived.add(packet.packet_id)

            # 从缓冲区中移除
            del buffer[expected]
            expected += 1
            delivered_count += 1

            logging.info('递交顺序数据包: %s，序列号: %s，延迟: %s us',
                         packet.packet_id, getattr(packet, 'mp_sequence', 'NA'), latency)

        # 更新下一个期望的序列号
        self.expected_sequence[flow_id] = expected

        if delivered_count > 0:
            logging.info('流 %s: 连续递交了 %d 个数据包，下一个期望序列号: %d',
                         flow_id, delivered_count, expected)

    def _forward_multipath_packet(self, packet, src_drone_id):
        """转发多路径数据包"""
        # 检查队列容量
        if self.my_drone.transmitting_queue.qsize() >= self.my_drone.max_queue_size:
            logging.warning('队列已满，无法转发数据包 %s', packet.packet_id)
            return

        # 检查是否有预定路径
        if hasattr(packet, 'routing_path') and packet.routing_path:
            routing_path = packet.routing_path

            # 找到我在路径中的位置
            if self.my_drone.identifier in routing_path:
                idx = routing_path.index(self.my_drone.identifier)

                # 如果不是最后一个节点，设置下一跳
                if idx < len(routing_path) - 1:
                    packet.next_hop_id = routing_path[idx + 1]
                    self.my_drone.transmitting_queue.put(packet)
                    logging.info('转发多路径数据包 %s 到下一跳 %s',
                                 packet.packet_id, packet.next_hop_id)
                    return

        # 如果没有预定路径，使用标准路由
        dst_id = packet.dst_drone.identifier
        next_hop = self.find_best_next_hop(dst_id, packet)

        if next_hop is not None:
            packet.next_hop_id = next_hop
            self.my_drone.transmitting_queue.put(packet)
            logging.info('使用标准路由转发数据包 %s 到 %s',
                         packet.packet_id, next_hop)
        else:
            logging.warning('无法找到到 %s 的路由，丢弃数据包 %s',
                            dst_id, packet.packet_id)

    def next_hop_selection(self, packet):
        """多路径的下一跳选择"""
        enquire = False
        has_route = True

        if not isinstance(packet, DataPacket):
            # 对于控制包，使用原始方法
            return super().next_hop_selection(packet)

        dst_id = packet.dst_drone.identifier

        # 如果是源节点，应用多路径传输策略
        if packet.src_drone.identifier == self.my_drone.identifier:
            # 发现和维护多路径
            if self._should_discover_paths(dst_id):
                paths = self.discover_multiple_paths(self.my_drone.identifier, dst_id)
                if paths:
                    self.path_cache[dst_id] = paths
                    self.last_path_update[dst_id] = self.simulator.env.now

                    # 初始化路径统计
                    if dst_id not in self.path_stats:
                        self.path_stats[dst_id] = {}

                    for i, path in enumerate(paths):
                        path_id = f"path_{self.my_drone.identifier}_{dst_id}_{i}"
                        if path_id not in self.path_stats[dst_id]:
                            self.path_stats[dst_id][path_id] = self._initialize_path_stats()

                    # 更新活跃路径
                    self._update_active_paths(dst_id)

            # 使用多路径传输
            if self.use_multipath and dst_id in self.path_cache and self.path_cache[dst_id]:
                # 为当前包选择路径
                path_index = self._select_path_for_packet(dst_id, packet)

                if path_index is not None and path_index < len(self.path_cache[dst_id]):
                    selected_path = self.path_cache[dst_id][path_index]

                    # 设置多路径序列号
                    flow_id = f"flow_{self.my_drone.identifier}_{dst_id}"
                    if flow_id not in self.packet_sequence:
                        self.packet_sequence[flow_id] = 0

                    packet.mp_sequence = self.packet_sequence[flow_id]
                    self.packet_sequence[flow_id] += 1

                    # 设置路由信息
                    packet.routing_path = selected_path
                    packet.current_path_index = path_index

                    if len(selected_path) > 0:
                        packet.next_hop_id = selected_path[1] if selected_path else None

                        # 更新路径使用统计
                        path_id = f"path_{self.my_drone.identifier}_{dst_id}_{path_index}"
                        if dst_id not in self.packet_distribution:
                            self.packet_distribution[dst_id] = {}
                        if path_id not in self.packet_distribution[dst_id]:
                            self.packet_distribution[dst_id][path_id] = 0
                        self.packet_distribution[dst_id][path_id] += 1

                        logging.info('多路径传输: 数据包 %s (序列号: %s) 通过路径 %s 发送',
                                     packet.packet_id, packet.mp_sequence, path_id)

                        return True, packet, False

            # 回退到单路径选择
            selected_path = self.select_best_path(dst_id, packet)

            if selected_path:
                # 设置路由信息
                packet.routing_path = selected_path

                if len(selected_path) > 0:
                    packet.next_hop_id = selected_path[1] if selected_path else None

                    # 记录路径选择
                    if dst_id in self.path_cache:
                        try:
                            path_index = self.path_cache[dst_id].index(selected_path)
                            packet.current_path_index = path_index
                        except ValueError:
                            pass

                    return True, packet, False

            # 没有找到路径
            has_route = False
            enquire = True
        else:
            # 中继节点处理
            if hasattr(packet, 'routing_path') and packet.routing_path:
                routing_path = packet.routing_path

                # 找到我在路径中的位置
                if self.my_drone.identifier in routing_path:
                    idx = routing_path.index(self.my_drone.identifier)

                    # 如果不是最后一个节点，设置下一跳
                    if idx < len(routing_path) - 1:
                        packet.next_hop_id = routing_path[idx + 1]
                        return True, packet, False

            # 如果没有路由信息，使用标准方法
            next_hop = self.find_best_next_hop(dst_id, packet)

            if next_hop is not None:
                packet.next_hop_id = next_hop
                return True, packet, False

            # 无法路由
            has_route = False

        return has_route, packet, enquire

    def _select_path_for_packet(self, dst_id, packet):
        """为当前数据包选择最佳路径"""
        if dst_id not in self.path_cache or not self.path_cache[dst_id]:
            return None

        # 获取活跃路径
        active_paths = self._get_active_paths(dst_id)
        if not active_paths:
            return None

        # 根据不同策略选择路径
        if self.load_balancing_mode == 'round_robin':
            # 轮询策略
            if dst_id not in self.current_path_index:
                self.current_path_index[dst_id] = 0

            # 在活跃路径中轮询
            active_idx = self.current_path_index[dst_id] % len(active_paths)
            path_index = active_paths[active_idx]

            # 更新索引
            self.current_path_index[dst_id] = (self.current_path_index[dst_id] + 1) % len(active_paths)

            return path_index

        elif self.load_balancing_mode == 'weighted':
            # 加权选择
            weights = []

            for idx in active_paths:
                path_id = f"path_{self.my_drone.identifier}_{dst_id}_{idx}"

                # 计算权重
                weight = 1.0  # 默认权重

                # 考虑路径指标
                if dst_id in self.path_rtt and path_id in self.path_rtt[dst_id] and self.path_rtt[dst_id][path_id]:
                    avg_rtt = sum(self.path_rtt[dst_id][path_id]) / len(self.path_rtt[dst_id][path_id])
                    # RTT越小权重越大
                    rtt_weight = 1.0 / (1.0 + avg_rtt / 1e6)
                    weight *= rtt_weight

                # 考虑路径失败次数
                failure_count = self.path_failure_count.get(dst_id, {}).get(path_id, 0)
                reliability = 1.0 / (1.0 + failure_count)
                weight *= reliability

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
                        return active_paths[i]

            # 如果权重计算失败，随机选择
            return np.random.choice(active_paths)

        elif self.load_balancing_mode == 'adaptive':
            # 自适应选择

            # 对于高优先级数据包，选择最低延迟路径
            if hasattr(packet, 'priority') and packet.priority > 0:
                best_path = None
                lowest_delay = float('inf')

                for idx in active_paths:
                    path_id = f"path_{self.my_drone.identifier}_{dst_id}_{idx}"

                    # 计算平均RTT
                    if dst_id in self.path_rtt and path_id in self.path_rtt[dst_id] and self.path_rtt[dst_id][path_id]:
                        avg_rtt = sum(self.path_rtt[dst_id][path_id]) / len(self.path_rtt[dst_id][path_id])

                        if avg_rtt < lowest_delay:
                            lowest_delay = avg_rtt
                            best_path = idx

                if best_path is not None:
                    return best_path

            # 检查是否有严重拥塞的路径
            congested_paths = []
            for idx in active_paths:
                path_id = f"path_{self.my_drone.identifier}_{dst_id}_{idx}"

                # 检查路径负载
                path_load = self.path_load.get(dst_id, {}).get(path_id, 0)

                if path_load > 10:  # 负载阈值
                    congested_paths.append(idx)

            # 从非拥塞路径中选择
            available_paths = [idx for idx in active_paths if idx not in congested_paths]

            if available_paths:
                # 仍然有可用路径，使用轮询
                if dst_id not in self.current_path_index:
                    self.current_path_index[dst_id] = 0

                active_idx = self.current_path_index[dst_id] % len(available_paths)
                path_index = available_paths[active_idx]

                self.current_path_index[dst_id] = (self.current_path_index[dst_id] + 1) % len(available_paths)

                return path_index

        # 默认随机选择一条活跃路径
        return np.random.choice(active_paths)

    def _get_active_paths(self, dst_id):
        """获取当前活跃的路径索引列表"""
        if dst_id not in self.active_paths:
            self._update_active_paths(dst_id)

        return self.active_paths.get(dst_id, [])

    def _update_active_paths(self, dst_id):
        """更新活跃路径列表"""
        if dst_id not in self.path_cache or not self.path_cache[dst_id]:
            # 尝试发现路径
            logging.info('目的地 %s 没有缓存路径，尝试发现路径', dst_id)
            paths = self.discover_multiple_paths(self.my_drone.identifier, dst_id)
            if paths:
                self.path_cache[dst_id] = paths
                self.last_path_update[dst_id] = self.simulator.env.now
                logging.info('为目的地 %s 发现了 %d 条路径', dst_id, len(paths))
            else:
                self.active_paths[dst_id] = []
                logging.warning('无法为目的地 %s 发现路径', dst_id)
                return []

        paths = self.path_cache[dst_id]
        if not paths:
            self.active_paths[dst_id] = []
            logging.warning('目的地 %s 的路径缓存为空', dst_id)
            return []

        # 评估所有路径
        path_scores = []

        for i, path in enumerate(paths):
            path_id = f"path_{self.my_drone.identifier}_{dst_id}_{i}"

            # 检查路径有效性 - 添加调试信息
            if not self._is_path_valid(path):
                score = -1  # 无效路径得分为-1
                logging.info('路径 %s 被判定为无效', path_id)
            else:
                # 计算路径得分
                score = self._calculate_path_score(dst_id, i)
                logging.info('路径 %s 得分: %.2f', path_id, score)

            path_scores.append((i, score))

        # 过滤掉无效路径并按得分排序
        valid_paths = [(i, score) for i, score in path_scores if score >= 0]

        # 如果没有有效路径，则至少选择一条路径(选得分最高的，即使是负分)
        if not valid_paths and path_scores:
            # 按得分排序，选择最高分(即使是负分)
            path_scores.sort(key=lambda x: x[1], reverse=True)
            best_path = path_scores[0]
            valid_paths = [best_path]
            logging.warning('没有完全有效的路径，选择得分最高的路径: %s (得分: %.2f)',
                            f"path_{self.my_drone.identifier}_{dst_id}_{best_path[0]}", best_path[1])

        valid_paths.sort(key=lambda x: x[1], reverse=True)

        # 选择前N条最好的路径作为活跃路径
        active_indices = [i for i, _ in valid_paths[:self.max_active_paths]]

        # 确保至少有一条路径(选第一条)
        if not active_indices and paths:
            active_indices = [0]
            logging.warning('没有活跃路径，默认选择第一条路径')

        # 更新活跃路径
        self.active_paths[dst_id] = active_indices

        logging.info('更新到目的地 %s 的活跃路径: %s', dst_id, active_indices)

        return active_indices

    def _calculate_path_score(self, dst_id, path_index):
        """计算路径得分"""
        path_id = f"path_{self.my_drone.identifier}_{dst_id}_{path_index}"

        # 基础得分 - 确保有一个正的最小值
        score = 1.0

        # 获取路径
        path = self.path_cache[dst_id][path_index]

        # 考虑路径长度 - 较短路径有优势，但不要完全排除长路径
        if len(path) > 0:
            length_factor = 1.0 / (1.0 + 0.05 * len(path))  # 减小长度惩罚
            score *= length_factor

        # 考虑RTT (如果有历史数据)
        if dst_id in self.path_rtt and path_id in self.path_rtt[dst_id] and self.path_rtt[dst_id][path_id]:
            avg_rtt = sum(self.path_rtt[dst_id][path_id]) / len(self.path_rtt[dst_id][path_id])
            rtt_score = 1.0 / (1.0 + avg_rtt / 1e6)
            score *= (rtt_score ** self.weight_delay)

        # 考虑丢包率 (使用默认值0.1，确保新路径有机会)
        loss_rate = self.path_loss_rate.get(dst_id, {}).get(path_id, 0.1)
        loss_score = 1.0 - loss_rate
        score *= (loss_score ** self.weight_loss)

        # 考虑带宽 (使用默认值，确保新路径有机会)
        bandwidth = self.path_bandwidth.get(dst_id, {}).get(path_id, config.BIT_RATE / 2)
        bandwidth_score = min(1.0, bandwidth / config.BIT_RATE)
        score *= (bandwidth_score ** self.weight_bandwidth)

        # 考虑链路质量
        link_quality_score = 1.0
        for i in range(len(path) - 1):
            if i >= len(path) - 1:
                continue

            node1 = path[i]
            node2 = path[i + 1]
            link_key = (node1, node2)

            # 获取链路质量，默认0.5确保新路径有机会
            quality = self.link_quality.get(link_key, 0.5)
            link_quality_score *= quality

        # 链路质量影响得分，但权重较小
        if len(path) > 1:
            link_quality_score = link_quality_score ** (1.0 / (len(path) - 1))  # 几何平均
            score *= link_quality_score

        # 确保得分至少为0.1，给所有路径一个机会
        score = max(0.1, score)

        logging.debug('路径 %s 得分计算: 长度=%d, 得分=%.2f', path_id, len(path), score)

        return score

    def _should_discover_paths(self, dst_id):
        """判断是否需要重新发现路径"""
        # 如果没有路径，需要发现
        if dst_id not in self.path_cache or not self.path_cache[dst_id]:
            return True

        # 检查上次更新时间
        if dst_id not in self.last_path_update:
            return True

        time_since_update = self.simulator.env.now - self.last_path_update[dst_id]
        if time_since_update > self.path_update_interval:
            return True

        # 检查活跃路径数量
        active_paths = self._get_active_paths(dst_id)
        if len(active_paths) < min(2, self.max_active_paths):
            return True

        return False

    def manage_multipath_transmission(self):
        """多路径传输管理进程"""
        while True:
            yield self.simulator.env.timeout(1 * 1e6)  # 每1秒执行一次

            try:
                # 更新所有目的地的活跃路径
                for dst_id in list(self.path_cache.keys()):
                    self._update_active_paths(dst_id)

                # 更新路径负载统计
                self._update_path_load_stats()

                # 更新路径带宽估计
                self._estimate_path_bandwidth()

                # 检测并处理路径失败
                self._detect_path_failures()

                # 平衡路径负载
                self._balance_path_load()

                # 记录当前多路径状态
                self._log_multipath_status()

            except Exception as e:
                logging.error('多路径管理进程异常: %s', str(e))

    def _update_path_load_stats(self):
        """更新路径负载统计"""
        for dst_id in self.packet_distribution:
            if dst_id not in self.path_load:
                self.path_load[dst_id] = {}

            # 重置所有路径的负载计数
            for path_id in self.packet_distribution[dst_id]:
                count = self.packet_distribution[dst_id][path_id]

                # 更新移动平均
                if path_id not in self.path_load[dst_id]:
                    self.path_load[dst_id][path_id] = count
                else:
                    self.path_load[dst_id][path_id] = 0.7 * self.path_load[dst_id][path_id] + 0.3 * count

                # 重置计数
                self.packet_distribution[dst_id][path_id] = 0

    def _estimate_path_bandwidth(self):
        """估计路径带宽"""
        current_time = self.simulator.env.now

        for dst_id in self.path_load:
            if dst_id not in self.path_bandwidth:
                self.path_bandwidth[dst_id] = {}

            for path_id, load in self.path_load[dst_id].items():
                # 计算数据率 (bps)
                if load > 0:
                    data_sent = load * config.DATA_PACKET_LENGTH  # bits
                    bandwidth = data_sent / 1.0  # 1秒内的数据量

                    # 更新移动平均
                    if path_id not in self.path_bandwidth[dst_id]:
                        self.path_bandwidth[dst_id][path_id] = bandwidth
                    else:
                        self.path_bandwidth[dst_id][path_id] = (
                                0.7 * self.path_bandwidth[dst_id][path_id] + 0.3 * bandwidth
                        )

    def _detect_path_failures(self):
        """检测路径失败"""
        for dst_id in self.path_rtt:
            if dst_id not in self.path_failure_count:
                self.path_failure_count[dst_id] = {}

            for path_id, rtt_samples in self.path_rtt[dst_id].items():
                if not rtt_samples:
                    continue

                # 计算RTT统计
                avg_rtt = sum(rtt_samples) / len(rtt_samples)

                # 检查RTT异常
                if avg_rtt > 5 * 1e6:  # 5秒RTT被视为异常
                    # 记录路径失败
                    if path_id not in self.path_failure_count[dst_id]:
                        self.path_failure_count[dst_id][path_id] = 0

                    self.path_failure_count[dst_id][path_id] += 1

                    # 如果连续失败次数过多，从活跃路径中移除
                    if self.path_failure_count[dst_id][path_id] > 3:
                        self._deactivate_path(dst_id, path_id)
                        logging.warning('路径 %s 失败次数过多，已从活跃路径中移除', path_id)
                else:
                    # 重置失败计数
                    if path_id in self.path_failure_count[dst_id]:
                        self.path_failure_count[dst_id][path_id] = max(0, self.path_failure_count[dst_id][path_id] - 1)

    def _deactivate_path(self, dst_id, path_id):
        """将路径从活跃路径中移除"""
        if dst_id not in self.active_paths:
            return

        # 找到路径索引
        path_index = None
        for i in range(len(self.path_cache[dst_id])):
            curr_path_id = f"path_{self.my_drone.identifier}_{dst_id}_{i}"
            if curr_path_id == path_id:
                path_index = i
                break

        if path_index is not None and path_index in self.active_paths[dst_id]:
            self.active_paths[dst_id].remove(path_index)

            # 如果没有活跃路径了，尝试重新发现路径
            if not self.active_paths[dst_id]:
                self._update_active_paths(dst_id)

    def _balance_path_load(self):
        """平衡各路径负载"""
        for dst_id in self.active_paths:
            active_indices = self.active_paths[dst_id]
            if len(active_indices) <= 1:
                continue  # 只有一条路径无需平衡

            # 获取各路径负载
            path_loads = []
            for idx in active_indices:
                path_id = f"path_{self.my_drone.identifier}_{dst_id}_{idx}"
                load = self.path_load.get(dst_id, {}).get(path_id, 0)
                path_loads.append((idx, load))

            # 计算平均负载
            avg_load = sum(load for _, load in path_loads) / len(path_loads)

            # 检查负载不平衡程度
            max_load = max(load for _, load in path_loads)
            min_load = min(load for _, load in path_loads)

            if max_load > 0 and max_load / avg_load > 2:
                # 负载不平衡，调整路径权重
                for idx, load in path_loads:
                    path_id = f"path_{self.my_drone.identifier}_{dst_id}_{idx}"

                    # 计算权重调整因子
                    adjustment = avg_load / (load + 0.1)  # 避免除以零

                    # 更新路径统计中的权重
                    if dst_id in self.path_stats and path_id in self.path_stats[dst_id]:
                        self.path_stats[dst_id][path_id]['weight'] = adjustment

                logging.info('路径负载不平衡，已调整路径权重: %s', [(idx, load) for idx, load in path_loads])

    def _log_multipath_status(self):
        """记录当前多路径状态"""
        for dst_id in self.active_paths:
            active_indices = self.active_paths[dst_id]
            if not active_indices:
                continue

            status_log = f"目的地 {dst_id} 的多路径状态:\n"
            status_log += f"  活跃路径: {active_indices}\n"

            # 添加每条路径的详细信息
            for idx in active_indices:
                path_id = f"path_{self.my_drone.identifier}_{dst_id}_{idx}"

                # 获取各项指标
                load = self.path_load.get(dst_id, {}).get(path_id, 0)

                avg_rtt = 0
                if dst_id in self.path_rtt and path_id in self.path_rtt[dst_id] and self.path_rtt[dst_id][path_id]:
                    avg_rtt = sum(self.path_rtt[dst_id][path_id]) / len(self.path_rtt[dst_id][path_id])

                bandwidth = self.path_bandwidth.get(dst_id, {}).get(path_id, 0)
                failures = self.path_failure_count.get(dst_id, {}).get(path_id, 0)

                status_log += f"  路径 {idx}: 负载={load:.2f}, RTT={avg_rtt / 1e3:.2f}ms, "
                status_log += f"带宽={bandwidth / 1e6:.2f}Mbps, 失败计数={failures}\n"

            logging.info(status_log)

    def process_reorder_buffer(self):
        """处理重排序缓冲区，递交超时数据包"""
        while True:
            yield self.simulator.env.timeout(0.5 * 1e6)  # 每0.5秒检查一次

            current_time = self.simulator.env.now

            for flow_id in list(self.reorder_buffer.keys()):
                if not self.reorder_buffer[flow_id]:
                    continue

                # 检查是否有超时数据包
                expired_sequences = []
                expected = self.expected_sequence.get(flow_id, 0)

                for seq, packet in self.reorder_buffer[flow_id].items():
                    # 检查是否已过期
                    if seq < expected or current_time - packet.arrival_time > self.reorder_timeout:
                        expired_sequences.append(seq)

                # 排序过期序列号
                expired_sequences.sort()

                if expired_sequences:
                    # 找到最大的连续序列
                    continuous_end = expired_sequences[0]
                    for i in range(1, len(expired_sequences)):
                        if expired_sequences[i] == continuous_end + 1:
                            continuous_end = expired_sequences[i]
                        else:
                            break

                    # 递交连续的过期数据包
                    for seq in range(expired_sequences[0], continuous_end + 1):
                        if seq in self.reorder_buffer[flow_id]:
                            packet = self.reorder_buffer[flow_id][seq]

                            # 递交数据包
                            latency = current_time - packet.creation_time
                            hop_count = packet.get_current_ttl()

                            # 记录性能指标
                            self.simulator.metrics.deliver_time_dict[packet.packet_id] = latency
                            self.simulator.metrics.throughput_dict[packet.packet_id] = (
                                    config.DATA_PACKET_LENGTH / (latency / 1e6)
                            )
                            self.simulator.metrics.hop_cnt_dict[packet.packet_id] = hop_count

                            # 记录全局统计
                            self.simulator.metrics.record_packet_reception(
                                packet.packet_id, latency, hop_count)
                            self.simulator.metrics.datapacket_arrived.add(packet.packet_id)

                            # 从缓冲区中移除
                            del self.reorder_buffer[flow_id][seq]

                            logging.info('递交超时数据包: %s，序列号: %s，延迟: %s us',
                                         packet.packet_id, seq, latency)

                    # 更新下一个期望的序列号
                    self.expected_sequence[flow_id] = continuous_end + 1

                    logging.info('流 %s: 递交了 %d 个超时数据包，更新期望序列号为 %d',
                                 flow_id, continuous_end - expired_sequences[0] + 1, continuous_end + 1)

                # 清理过大的缓冲区
                if len(self.reorder_buffer[flow_id]) > self.reorder_buffer_size:
                    # 保留最近的数据包
                    sequences = sorted(self.reorder_buffer[flow_id].keys())
                    to_remove = sequences[:-self.reorder_buffer_size]

                    # 移除过旧的数据包
                    for seq in to_remove:
                        del self.reorder_buffer[flow_id][seq]

                    logging.warning('流 %s: 缓冲区过大，已清理 %d 个旧数据包',
                                    flow_id, len(to_remove))

    # 数据包分片相关方法（可选实现）
    def _fragment_packet(self, packet):
        """将大数据包分片"""
        if not self.enable_fragmentation:
            return [packet]

        # 检查是否需要分片
        if packet.packet_length <= self.max_fragment_size * 8:
            return [packet]

        # 计算需要的分片数量
        fragment_count = math.ceil(packet.packet_length / (self.max_fragment_size * 8))
        fragments = []

        # 创建分片
        for i in range(fragment_count):
            fragment = copy.copy(packet)
            fragment.fragment_id = i
            fragment.fragment_count = fragment_count
            fragment.is_fragment = True
            fragments.append(fragment)

        logging.info('数据包 %s 被分为 %d 个分片', packet.packet_id, fragment_count)
        return fragments

    def _reassemble_fragments(self, packet):
        """重组分片数据包"""
        if not self.enable_fragmentation:
            return packet

        # 检查是否为分片
        if not hasattr(packet, 'is_fragment') or not packet.is_fragment:
            return packet

        packet_id = packet.packet_id
        fragment_id = packet.fragment_id
        fragment_count = packet.fragment_count

        # 初始化分片缓冲区
        if packet_id not in self.fragment_buffers:
            self.fragment_buffers[packet_id] = {}

        # 存储分片
        self.fragment_buffers[packet_id][fragment_id] = packet

        # 检查是否已收集到所有分片
        if len(self.fragment_buffers[packet_id]) == fragment_count:
            # 重组数据包
            reassembled = self.fragment_buffers[packet_id][0]  # 使用第一个分片作为基础
            reassembled.is_fragment = False
            del self.fragment_buffers[packet_id]

            logging.info('数据包 %s 的所有分片已重组', packet_id)
            return reassembled

        return None