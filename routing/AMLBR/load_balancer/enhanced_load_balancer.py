import logging
import numpy as np
from collections import defaultdict, deque
from utils import config
from entities.packet import DataPacket


class EnhancedLoadBalancer:
    """
    增强型负载均衡器，用于自适应多路径路由
    - 实时监控流量负载和网络状态
    - 智能分配流量到多条路径
    - 支持QoS感知的负载均衡
    - 提供拥塞控制和故障恢复机制
    """

    def __init__(self, router, window_size=10):
        self.router = router
        self.simulator = router.simulator
        self.my_drone = router.my_drone

        # 流量统计
        self.flow_stats = {}  # {flow_id: stats}
        self.path_stats = {}  # {dst_id: {path_id: stats}}
        self.node_stats = {}  # {node_id: stats}

        # 监控窗口
        self.window_size = window_size
        self.delay_windows = {}  # {flow_id: deque}
        self.throughput_windows = {}  # {flow_id: deque}
        self.loss_windows = {}  # {flow_id: deque}

        # 负载均衡参数
        self.congestion_threshold = 0.7  # 拥塞阈值
        self.max_imbalance = 0.3  # 最大不平衡度
        self.rebalance_interval = 2 * 1e6  # 重平衡间隔 (2秒)
        self.last_rebalance_time = 0  # 上次重平衡时间

        # QoS需求映射
        self.qos_requirements = {
            0: {'delay': 5000, 'loss': 0.1, 'priority': 1},  # 低优先级
            1: {'delay': 1000, 'loss': 0.05, 'priority': 2},  # 中优先级
            2: {'delay': 500, 'loss': 0.01, 'priority': 3}  # 高优先级
        }

        # 路径分配策略
        self.path_allocation = {}  # {flow_id: {path_id: weight}}

        # 启动定期监控
        self.simulator.env.process(self.periodic_monitor())

    def register_flow(self, packet):
        """注册新的数据流"""
        if not isinstance(packet, DataPacket):
            return

        flow_id = self._get_flow_id(packet)

        # 如果是新流量，初始化统计
        if flow_id not in self.flow_stats:
            self.flow_stats[flow_id] = {
                'packets_sent': 0,
                'packets_received': 0,
                'bytes_sent': 0,
                'start_time': self.simulator.env.now,
                'last_packet_time': self.simulator.env.now,
                'avg_delay': 0,
                'loss_rate': 0,
                'throughput': 0,
                'qos_class': packet.priority if hasattr(packet, 'priority') else 0
            }

            # 初始化监控窗口
            self.delay_windows[flow_id] = deque(maxlen=self.window_size)
            self.throughput_windows[flow_id] = deque(maxlen=self.window_size)
            self.loss_windows[flow_id] = deque(maxlen=self.window_size)

            # 为流量分配路径
            self._allocate_paths_for_flow(flow_id, packet)

            logging.info('注册新流量 %s，QoS类别: %d',
                         flow_id, self.flow_stats[flow_id]['qos_class'])

    def update_flow_stats(self, packet, event_type):
        """更新流量统计信息"""
        if not isinstance(packet, DataPacket):
            return

        flow_id = self._get_flow_id(packet)

        # 确保流量已注册
        if flow_id not in self.flow_stats:
            self.register_flow(packet)

        stats = self.flow_stats[flow_id]

        if event_type == 'sent':
            # 发送事件统计
            stats['packets_sent'] += 1
            stats['bytes_sent'] += packet.packet_length
            stats['last_packet_time'] = self.simulator.env.now

            # 记录路径使用情况
            if hasattr(packet, 'current_path_index'):
                path_id = f"path_{packet.src_drone.identifier}_{packet.dst_drone.identifier}_{packet.current_path_index}"
                if path_id in self.path_stats:
                    self.path_stats[path_id]['packets_sent'] += 1

        elif event_type == 'received':
            # 接收事件统计
            stats['packets_received'] += 1

            # 计算延迟
            if hasattr(packet, 'creation_time'):
                delay = self.simulator.env.now - packet.creation_time
                self.delay_windows[flow_id].append(delay)
                stats['avg_delay'] = sum(self.delay_windows[flow_id]) / len(self.delay_windows[flow_id])

            # 更新丢包率
            if stats['packets_sent'] > 0:
                loss_rate = 1.0 - (stats['packets_received'] / stats['packets_sent'])
                self.loss_windows[flow_id].append(loss_rate)
                stats['loss_rate'] = sum(self.loss_windows[flow_id]) / len(self.loss_windows[flow_id])

            # 更新吞吐量 (bps)
            elapsed_time = (self.simulator.env.now - stats['start_time']) / 1e6  # 秒
            if elapsed_time > 0:
                throughput = (stats['bytes_sent'] * 8) / elapsed_time
                self.throughput_windows[flow_id].append(throughput)
                stats['throughput'] = sum(self.throughput_windows[flow_id]) / len(self.throughput_windows[flow_id])

            # 记录路径接收情况
            if hasattr(packet, 'current_path_index'):
                path_id = f"path_{packet.src_drone.identifier}_{packet.dst_drone.identifier}_{packet.current_path_index}"
                if path_id in self.path_stats:
                    self.path_stats[path_id]['packets_received'] += 1

    def get_best_path(self, packet):
        """为数据包选择最佳路径"""
        if not isinstance(packet, DataPacket):
            return None

        flow_id = self._get_flow_id(packet)
        dst_id = packet.dst_drone.identifier

        # 检查是否有缓存的路径
        if dst_id not in self.router.path_cache or not self.router.path_cache[dst_id]:
            return None

        # 获取可用路径
        available_paths = self.router.path_cache[dst_id]

        # 如果没有分配策略，使用轮询
        if flow_id not in self.path_allocation:
            return self._round_robin_selection(dst_id)

        # 获取路径分配权重
        path_weights = self.path_allocation[flow_id]

        # 根据QoS需求选择合适的策略
        qos_class = self.flow_stats[flow_id]['qos_class']

        if qos_class >= 2:  # 高优先级：选择最低延迟路径
            return self._lowest_delay_selection(dst_id)
        elif qos_class == 1:  # 中优先级：考虑延迟和可靠性的加权选择
            return self._weighted_selection(dst_id, path_weights)
        else:  # 低优先级：负载平衡的轮询
            return self._round_robin_selection(dst_id)

    def _allocate_paths_for_flow(self, flow_id, packet):
        """为流量分配路径权重"""
        if not isinstance(packet, DataPacket):
            return

        dst_id = packet.dst_drone.identifier

        # 检查是否有可用路径
        if dst_id not in self.router.path_cache or not self.router.path_cache[dst_id]:
            return

        paths = self.router.path_cache[dst_id]

        # 为每条路径计算初始权重
        weights = {}

        for i, path in enumerate(paths):
            path_id = f"path_{packet.src_drone.identifier}_{dst_id}_{i}"

            # 初始化路径统计
            if path_id not in self.path_stats:
                self.path_stats[path_id] = {
                    'packets_sent': 0,
                    'packets_received': 0,
                    'avg_delay': None,
                    'loss_rate': 0,
                    'congestion': 0
                }

            # 计算路径质量
            path_quality = self._calculate_path_quality(path, dst_id)

            # 设置初始权重
            weights[path_id] = max(0.1, path_quality)

        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            for path_id in weights:
                weights[path_id] /= total_weight

        # 保存路径分配
        self.path_allocation[flow_id] = weights

    def _calculate_path_quality(self, path, dst_id):
        """计算路径质量评分"""
        if not path:
            return 0.0

        # 基础评分
        score = 1.0

        # 考虑路径长度
        path_length = len(path)
        length_factor = 1.0 / path_length if path_length > 0 else 0
        score *= (0.3 + 0.7 * length_factor)  # 路径越短越好

        # 考虑链路质量
        avg_link_quality = 0.0
        link_count = 0

        for i in range(len(path) - 1):
            node1 = path[i]
            node2 = path[i + 1]
            link_key = (node1, node2)

            if link_key in self.router.link_quality:
                avg_link_quality += self.router.link_quality[link_key]
                link_count += 1

        if link_count > 0:
            avg_link_quality /= link_count
            score *= (0.2 + 0.8 * avg_link_quality)

        # 考虑节点拥塞
        congestion_factor = 1.0
        for node_id in path[1:-1]:  # 跳过源节点和目标节点
            if node_id in self.node_stats:
                node_congestion = self.node_stats[node_id].get('congestion', 0)
                congestion_factor *= (1.0 - node_congestion)

        score *= congestion_factor

        return score

    def _lowest_delay_selection(self, dst_id):
        """选择延迟最低的路径"""
        if dst_id not in self.router.path_cache or not self.router.path_cache[dst_id]:
            return None

        paths = self.router.path_cache[dst_id]
        best_path = None
        lowest_delay = float('inf')

        for i, path in enumerate(paths):
            path_id = f"path_{self.my_drone.identifier}_{dst_id}_{i}"

            # 获取路径延迟
            if path_id in self.path_stats and self.path_stats[path_id]['avg_delay'] is not None:
                delay = self.path_stats[path_id]['avg_delay']
            else:
                # 估计延迟
                delay = self._estimate_path_delay(path)

            if delay < lowest_delay:
                lowest_delay = delay
                best_path = path

        return best_path or paths[0]

    def _weighted_selection(self, dst_id, weights):
        """根据权重选择路径"""
        if dst_id not in self.router.path_cache or not self.router.path_cache[dst_id]:
            return None

        paths = self.router.path_cache[dst_id]

        # 准备选择
        path_weights = []
        for i, path in enumerate(paths):
            path_id = f"path_{self.my_drone.identifier}_{dst_id}_{i}"
            weight = weights.get(path_id, 1.0 / len(paths))
            path_weights.append(weight)

        # 归一化权重
        total_weight = sum(path_weights)
        if total_weight > 0:
            path_weights = [w / total_weight for w in path_weights]
        else:
            path_weights = [1.0 / len(paths)] * len(paths)

        # 根据权重随机选择
        r = np.random.random()
        cumsum = 0
        for i, weight in enumerate(path_weights):
            cumsum += weight
            if r <= cumsum:
                return paths[i]

        return paths[0]

    def _round_robin_selection(self, dst_id):
        """轮询选择路径"""
        if dst_id not in self.router.path_cache or not self.router.path_cache[dst_id]:
            return None

        paths = self.router.path_cache[dst_id]

        if dst_id not in self.router.current_path_index:
            self.router.current_path_index[dst_id] = 0

        index = self.router.current_path_index[dst_id]
        self.router.current_path_index[dst_id] = (index + 1) % len(paths)

        return paths[index]

    def _estimate_path_delay(self, path):
        """估计路径延迟"""
        if not path:
            return float('inf')

        # 基础传输延迟 (假设每跳1ms)
        hop_delay = (len(path) - 1) * 1000  # 微秒

        # 考虑队列延迟
        queue_delay = 0
        for node_id in path[1:-1]:  # 跳过源节点和目标节点
            if node_id in self.node_stats:
                # 估计队列延迟
                queue_size = self.node_stats[node_id].get('queue_size', 0)
                avg_service_time = 500  # 假设平均服务时间500微秒
                queue_delay += queue_size * avg_service_time

        # 考虑链路延迟
        link_delay = 0
        for i in range(len(path) - 1):
            node1 = path[i]
            node2 = path[i + 1]

            # 基于ETX估计链路延迟
            link_key = (node1, node2)
            etx = self.router.expected_transmission_count.get(link_key, 1.0)
            link_delay += 200 * etx  # 假设基础链路延迟200微秒

        total_delay = hop_delay + queue_delay + link_delay
        return total_delay

    def update_node_stats(self, node_id, queue_size=None, throughput=None, delay=None):
        """更新节点统计信息"""
        if node_id not in self.node_stats:
            self.node_stats[node_id] = {
                'queue_size': 0,
                'throughput': 0,
                'avg_delay': 0,
                'congestion': 0,
                'last_update': self.simulator.env.now
            }

        stats = self.node_stats[node_id]
        stats['last_update'] = self.simulator.env.now

        # 更新队列大小
        if queue_size is not None:
            stats['queue_size'] = queue_size

            # 计算拥塞度
            max_queue = self.my_drone.max_queue_size
            stats['congestion'] = min(1.0, queue_size / max_queue)

        # 更新吞吐量
        if throughput is not None:
            stats['throughput'] = throughput

        # 更新延迟
        if delay is not None:
            stats['avg_delay'] = delay

    def _get_flow_id(self, packet):
        """获取数据流ID"""
        if hasattr(packet, 'flow_id'):
            return packet.flow_id

        # 默认使用源目标对作为流ID
        return f"flow_{packet.src_drone.identifier}_{packet.dst_drone.identifier}"

    def periodic_monitor(self):
        """定期监控并调整负载平衡"""
        while True:
            yield self.simulator.env.timeout(1 * 1e6)  # 每1秒监控一次

            current_time = self.simulator.env.now

            # 清理过期流量
            self._clean_expired_flows()

            # 定期重新平衡
            if current_time - self.last_rebalance_time >= self.rebalance_interval:
                self._rebalance_flows()
                self.last_rebalance_time = current_time

    def _clean_expired_flows(self):
        """清理过期的流量记录"""
        current_time = self.simulator.env.now
        expired_flows = []

        for flow_id, stats in self.flow_stats.items():
            # 如果超过10秒没有新的数据包，认为流量已结束
            if current_time - stats['last_packet_time'] > 10 * 1e6:
                expired_flows.append(flow_id)

        # 删除过期流量
        for flow_id in expired_flows:
            if flow_id in self.flow_stats:
                del self.flow_stats[flow_id]
            if flow_id in self.delay_windows:
                del self.delay_windows[flow_id]
            if flow_id in self.throughput_windows:
                del self.throughput_windows[flow_id]
            if flow_id in self.loss_windows:
                del self.loss_windows[flow_id]
            if flow_id in self.path_allocation:
                del self.path_allocation[flow_id]

    def _rebalance_flows(self):
        """重新平衡所有流量的路径分配"""
        # 检查是否需要重平衡
        if not self._check_imbalance():
            return

        logging.info('开始重新平衡流量分配')

        # 更新所有流量的路径分配
        for flow_id in list(self.flow_stats.keys()):
            if flow_id in self.path_allocation:
                self._update_flow_allocation(flow_id)

        # 输出新的分配结果
        self._log_allocation()

    def _check_imbalance(self):
        """检查路径负载是否不平衡"""
        # 计算每条路径的负载
        path_loads = {}

        for path_id, stats in self.path_stats.items():
            if stats['packets_sent'] > 0:
                path_loads[path_id] = stats['packets_sent']

        if not path_loads:
            return False

        # 计算负载均值和标准差
        load_values = list(path_loads.values())
        avg_load = sum(load_values) / len(load_values)

        if avg_load == 0:
            return False

        # 计算变异系数
        std_load = np.std(load_values)
        cv = std_load / avg_load if avg_load > 0 else 0

        # 如果变异系数超过阈值，需要重平衡
        return cv > self.max_imbalance

    def _update_flow_allocation(self, flow_id):
        """更新流量的路径分配权重"""
        if flow_id not in self.flow_stats:
            return

        # 提取源目标信息
        try:
            parts = flow_id.split('_')
            if len(parts) < 3:
                return

            src_id = int(parts[-2])
            dst_id = int(parts[-1])

            # 检查是否有路径
            if dst_id not in self.router.path_cache or not self.router.path_cache[dst_id]:
                return

            paths = self.router.path_cache[dst_id]

            # 获取流量的QoS级别
            qos_class = self.flow_stats[flow_id]['qos_class']

            # 根据QoS级别选择不同的分配策略
            if qos_class >= 2:  # 高优先级：优先低延迟
                self._update_delay_sensitive_allocation(flow_id, src_id, dst_id, paths)
            elif qos_class == 1:  # 中优先级：平衡延迟和吞吐量
                self._update_balanced_allocation(flow_id, src_id, dst_id, paths)
            else:  # 低优先级：最大化吞吐量
                self._update_throughput_allocation(flow_id, src_id, dst_id, paths)
        except Exception as e:
            logging.error(f"更新流量分配时发生错误: {e}")

    def _update_delay_sensitive_allocation(self, flow_id, src_id, dst_id, paths):
        """为延迟敏感型流量更新路径分配"""
        # 计算每条路径的延迟
        path_delays = {}

        for i, path in enumerate(paths):
            path_id = f"path_{src_id}_{dst_id}_{i}"
            delay = self._estimate_path_delay(path)
            path_delays[path_id] = delay

        # 如果没有延迟信息，使用均匀分配
        if not path_delays:
            weights = {f"path_{src_id}_{dst_id}_{i}": 1.0 / len(paths) for i in range(len(paths))}
            self.path_allocation[flow_id] = weights
            return

        # 计算基于延迟的权重（延迟越低，权重越高）
        max_delay = max(path_delays.values())
        weights = {}

        for path_id, delay in path_delays.items():
            # 反比例关系：延迟越低，权重越高
            weights[path_id] = max_delay / delay if delay > 0 else 1.0

        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            for path_id in weights:
                weights[path_id] /= total_weight

        # 更新权重，倾向于选择少数最佳路径
        for path_id in weights:
            weights[path_id] = weights[path_id] ** 2  # 平方增加权重差异

        # 再次归一化
        total_weight = sum(weights.values())
        if total_weight > 0:
            for path_id in weights:
                weights[path_id] /= total_weight

        self.path_allocation[flow_id] = weights

    def _update_balanced_allocation(self, flow_id, src_id, dst_id, paths):
        """为平衡型流量更新路径分配"""
        # 计算每条路径的综合得分
        path_scores = {}

        for i, path in enumerate(paths):
            path_id = f"path_{src_id}_{dst_id}_{i}"

            # 计算延迟、丢包率和拥塞度
            delay = self._estimate_path_delay(path)
            loss_rate = self._estimate_path_loss(path)
            congestion = self._estimate_path_congestion(path)

            # 归一化各指标（值越小越好）
            delay_score = 1.0 / (1.0 + delay / 1e6)  # 延迟越低分数越高
            loss_score = 1.0 - loss_rate  # 丢包率越低分数越高
            congestion_score = 1.0 - congestion  # 拥塞度越低分数越高

            # 计算综合得分（权重可调）
            score = delay_score * 0.4 + loss_score * 0.3 + congestion_score * 0.3
            path_scores[path_id] = score

        # 转换得分为权重
        weights = path_scores.copy()

        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            for path_id in weights:
                weights[path_id] /= total_weight

        self.path_allocation[flow_id] = weights

    def _update_throughput_allocation(self, flow_id, src_id, dst_id, paths):
        """为吞吐量敏感型流量更新路径分配"""
        # 计算每条路径的带宽容量
        path_capacities = {}

        for i, path in enumerate(paths):
            path_id = f"path_{src_id}_{dst_id}_{i}"

            # 估计路径容量（瓶颈带宽）
            capacity = self._estimate_path_capacity(path)
            path_capacities[path_id] = capacity

        # 根据容量分配权重
        weights = path_capacities.copy()

        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            for path_id in weights:
                weights[path_id] /= total_weight

        self.path_allocation[flow_id] = weights

    def _estimate_path_loss(self, path):
        """估计路径丢包率"""
        if not path:
            return 1.0

        # 默认丢包率
        default_loss = 0.01

        # 计算链路质量导致的丢包
        loss_rate = 0.0
        for i in range(len(path) - 1):
            node1 = path[i]
            node2 = path[i + 1]
            link_key = (node1, node2)

            # 基于链路质量估计丢包率
            quality = self.router.link_quality.get(link_key, 0.5)
            link_loss = max(0, 1.0 - quality)

            # 累加丢包率（考虑级联效应）
            loss_rate = loss_rate + (1 - loss_rate) * link_loss

        return max(default_loss, loss_rate)

    def _estimate_path_congestion(self, path):
        """估计路径拥塞度"""
        if not path:
            return 1.0

        # 计算平均拥塞度
        total_congestion = 0.0
        node_count = 0

        for node_id in path[1:-1]:  # 跳过源节点和目标节点
            if node_id in self.node_stats:
                total_congestion += self.node_stats[node_id].get('congestion', 0)
                node_count += 1

        # 如果没有节点信息，假设中等拥塞度
        if node_count == 0:
            return 0.5

        return total_congestion / node_count

    def _estimate_path_capacity(self, path):
        """估计路径带宽容量"""
        if not path:
            return 0

        # 假设基础链路容量
        base_capacity = config.BIT_RATE  # 使用系统配置的比特率

        # 找出瓶颈链路容量
        min_capacity = base_capacity

        for i in range(len(path) - 1):
            node1 = path[i]
            node2 = path[i + 1]
            link_key = (node1, node2)

            # 考虑链路质量影响
            quality = self.router.link_quality.get(link_key, 0.5)
            link_capacity = base_capacity * quality

            # 考虑节点拥塞影响
            if node2 in self.node_stats:
                congestion = self.node_stats[node2].get('congestion', 0)
                available_capacity = link_capacity * (1 - congestion)
                min_capacity = min(min_capacity, available_capacity)

        return min_capacity

    def _log_allocation(self):
        """记录路径分配情况"""
        log_msg = "当前流量路径分配:\n"

        for flow_id, weights in self.path_allocation.items():
            qos_class = self.flow_stats.get(flow_id, {}).get('qos_class', 0)
            log_msg += f"流量 {flow_id} (QoS: {qos_class}):\n"

            for path_id, weight in weights.items():
                log_msg += f"  - {path_id}: {weight:.2f}\n"

        logging.info(log_msg)

    def get_path_stats(self):
        """获取路径性能统计"""
        return self.path_stats

    def get_flow_stats(self):
        """获取流量性能统计"""
        return self.flow_stats

    def get_node_stats(self):
        """获取节点性能统计"""
        return self.node_stats