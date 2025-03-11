import logging
import copy
import numpy as np
from utils import config
from utils.util_function import euclidean_distance
from entities.packet import DataPacket
from phy.large_scale_fading import maximum_communication_range
from routing.AMLBR.gat.gat_network_module import GATNetworkAnalyzer


class GATEnhancedRoutingManager:
    """
    整合GAT的增强型路由管理器
    - 利用图注意力网络分析网络拓扑
    - 智能感知网络结构和节点重要性
    - 基于学习的路径评估和选择
    """

    def __init__(self, simulator, my_drone):
        self.simulator = simulator
        self.my_drone = my_drone

        # 初始化原始路由协议
        from routing.AMLBR.adaptive_multipath.AMLBR import AMLBR
        self.router = AMLBR(simulator, my_drone)

        # 初始化负载均衡器
        from routing.AMLBR.load_balancer.enhanced_load_balancer import EnhancedLoadBalancer
        self.load_balancer = EnhancedLoadBalancer(self.router)

        # 初始化GAT网络分析器
        self.gat_analyzer = GATNetworkAnalyzer(self.simulator,num_drones=simulator.n_drones)

        # 配置参数
        self.max_paths = config.MAX_PATHS if hasattr(config, 'MAX_PATHS') else 3
        self.path_selection_strategy = config.PATH_SELECTION_STRATEGY if hasattr(config,
                                                                                 'PATH_SELECTION_STRATEGY') else 'gat'

        # 缓存和指标
        self.route_cache = {}  # {dst_id: {routes, timestamp}}
        self.performance_metrics = {}  # {metric_name: value}

        # GAT相关参数
        self.gat_update_interval = 2 * 1e6  # 2秒更新一次GAT模型
        self.gat_enabled = True  # 是否启用GAT
        self.use_attention_weights = True  # 是否使用注意力权重

        # 能量感知参数
        self.energy_aware = True  # 是否启用能量感知
        self.energy_threshold = 0.3  # 触发能量感知的阈值 (30%)

        # 初始化性能监控
        self.simulator.env.process(self.monitor_network_performance())
        self.simulator.env.process(self.gat_analyzer.periodic_gat_monitoring())

        # 初始化GAT模型定期更新
        self.simulator.env.process(self.update_gat_model())

        logging.info("GAT增强型路由管理器已初始化: 策略=%s, 最大路径数=%d",
                     self.path_selection_strategy, self.max_paths)

    def packet_reception(self, packet, src_drone_id):
        """处理收到的数据包"""
        # 更新流量统计
        if isinstance(packet, DataPacket):
            self.load_balancer.update_flow_stats(packet, 'received')

        # 交给路由协议处理
        self.router.packet_reception(packet, src_drone_id)

    def next_hop_selection(self, packet):
        """选择下一跳节点"""
        # 数据包预处理
        if isinstance(packet, DataPacket):
            # 注册流量
            self.load_balancer.register_flow(packet)

            # 更新发送统计
            self.load_balancer.update_flow_stats(packet, 'sent')

            # 如果是源节点，应用路径选择策略
            if packet.src_drone.identifier == self.my_drone.identifier:
                # 使用GAT增强的路径选择
                self._apply_gat_enhanced_path_selection(packet)

                # 记录转发指标
                self._update_forwarding_metrics(packet)

        # 调用路由协议选择下一跳
        return self.router.next_hop_selection(packet)

    def _apply_gat_enhanced_path_selection(self, packet):
        """使用GAT增强的路径选择策略，增加详细日志"""
        if not isinstance(packet, DataPacket):
            return

        dst_id = packet.dst_drone.identifier

        # 根据流量特性选择合适的路径
        if hasattr(packet, 'priority') and packet.priority > 0:
            # 高优先级流量
            selected_path = self.load_balancer.get_best_path(packet)
            path_source = "负载均衡器(高优先级)"
        else:
            # 普通流量，使用GAT增强的路径选择
            selected_path = self._select_path_with_gat(dst_id, packet)
            path_source = "GAT"

        if selected_path:
            # 设置路由路径和下一跳
            packet.routing_path = selected_path
            if len(selected_path) > 1:
                packet.next_hop_id = selected_path[1]

                # 记录使用的路径
                if dst_id in self.router.path_cache:
                    try:
                        path_index = self.router.path_cache[dst_id].index(selected_path)
                        packet.current_path_index = path_index

                        # 记录GAT使用情况
                        if path_source == "GAT":
                            if 'gat_usage' not in self.performance_metrics:
                                self.performance_metrics['gat_usage'] = 0

                            self.performance_metrics['gat_usage'] += 1

                            # 评估路径质量
                            path_quality = self.gat_analyzer.evaluate_path_quality(selected_path)
                            logging.info(
                                f"无人机 {self.my_drone.identifier} 使用GAT选择到 {dst_id} 的路径: {selected_path}, 质量: {path_quality:.4f}")

                    except ValueError:
                        pass

            # 日志记录路径选择
            logging.info(f"无人机 {self.my_drone.identifier} 为数据包 {packet.packet_id} 选择路径来源: {path_source}")

    def _select_path_with_gat(self, dst_id, packet=None):
        """使用GAT模型选择路径"""
        # 检查缓存中是否有路径
        if dst_id not in self.router.path_cache or not self.router.path_cache[dst_id]:
            # 尝试发现路径
            paths = self.router.discover_multiple_paths(self.my_drone.identifier, dst_id)
            if paths:
                self.router.path_cache[dst_id] = paths
            else:
                return None

        paths = self.router.path_cache[dst_id]
        strategy = self.path_selection_strategy

        # 如果GAT未启用或不可用，回退到其他策略
        if not self.gat_enabled or strategy != 'gat':
            return self._select_path_by_strategy(dst_id, packet)

        # 使用GAT推荐最佳路径
        recommended_paths = self.gat_analyzer.recommend_paths(
            self.my_drone.identifier, dst_id, paths, top_k=1)

        if recommended_paths:
            return recommended_paths[0]

        # 如果GAT推荐失败，回退到其他策略
        return self._select_path_by_strategy(dst_id, packet)

    def _select_path_by_strategy(self, dst_id, packet=None):
        """根据策略选择路径"""
        # 检查缓存中是否有路径
        if dst_id not in self.router.path_cache or not self.router.path_cache[dst_id]:
            return None

        paths = self.router.path_cache[dst_id]
        strategy = self.path_selection_strategy

        # 根据策略选择路径
        if strategy == 'round_robin':
            # 轮询策略
            if dst_id not in self.router.current_path_index:
                self.router.current_path_index[dst_id] = 0

            index = self.router.current_path_index[dst_id]
            self.router.current_path_index[dst_id] = (index + 1) % len(paths)
            return paths[index]

        elif strategy == 'best_quality':
            # 基于质量的选择
            best_path = None
            best_quality = float('-inf')

            for path in paths:
                quality = self.router.calculate_path_quality(path, dst_id)
                if quality > best_quality:
                    best_quality = quality
                    best_path = path

            return best_path

        elif strategy == 'adaptive':
            # 自适应策略 - 基于网络状态动态选择

            # 检查网络拥塞状况
            is_congested = self._is_network_congested()

            # 检查能量状况
            is_energy_critical = self._is_energy_critical()

            if is_energy_critical:
                # 如果能量状态临界，选择能耗最小的路径
                return self._select_energy_efficient_path(dst_id)
            elif is_congested:
                # 如果网络拥塞，选择拥塞最小的路径
                return self._select_least_congested_path(dst_id)
            elif packet and hasattr(packet, 'priority') and packet.priority > 0:
                # 如果是高优先级流量，选择延迟最小的路径
                return self._select_lowest_delay_path(dst_id)
            else:
                # 默认使用轮询
                if dst_id not in self.router.current_path_index:
                    self.router.current_path_index[dst_id] = 0

                index = self.router.current_path_index[dst_id]
                self.router.current_path_index[dst_id] = (index + 1) % len(paths)
                return paths[index]

        # 默认使用第一条路径
        return paths[0] if paths else None

    def update_gat_model(self):
        """定期更新GAT模型，确保模型得到训练"""
        # 等待初始化阶段完成
        yield self.simulator.env.timeout(1 * 1e6)  # 等待1秒

        # 第一次强制进行训练，确保模型有基础权重
        if self.gat_enabled:
            logging.info(f"无人机 {self.my_drone.identifier} 初始化GAT模型...")
            self.gat_analyzer.update_network_state(self.simulator)
            self.gat_analyzer.train_model(self.simulator, epochs=5)  # 第一次用更多轮次
            logging.info(f"无人机 {self.my_drone.identifier} GAT模型初始化完成")

        # 定期更新与训练
        while True:
            yield self.simulator.env.timeout(self.gat_update_interval)

            if self.gat_enabled:
                # 更新网络状态
                updated = self.gat_analyzer.update_network_state(self.simulator)
                logging.info(f"无人机 {self.my_drone.identifier} 更新网络状态，结果: {updated}")

                # 每次都进行训练，但控制轮次
                train_now = True

                if train_now:
                    logging.info(f"无人机 {self.my_drone.identifier} 训练GAT模型开始...")
                    success = self.gat_analyzer.train_model(self.simulator, epochs=3)
                    logging.info(f"无人机 {self.my_drone.identifier} 训练GAT模型完成，状态: {success}")

                    # 获取节点重要性
                    node_importance = self.gat_analyzer.get_node_importance()
                    if node_importance is not None:
                        # 记录自身重要性
                        my_importance = node_importance[self.my_drone.identifier]
                        logging.info(f"无人机 {self.my_drone.identifier} 的重要性: {my_importance:.4f}")

                    # 打印注意力矩阵摘要
                    attention_matrix = self.gat_analyzer.get_attention_matrix()
                    if attention_matrix is not None:
                        # 找出注意力值最高的5个链路
                        attention_values = []
                        for i in range(len(attention_matrix)):
                            for j in range(len(attention_matrix[0])):
                                if i != j:  # 排除自环
                                    attention_values.append((i, j, attention_matrix[i][j]))

                        # 排序并打印前5个
                        attention_values.sort(key=lambda x: x[2], reverse=True)
                        top_links = attention_values[:5]

                        logging.info(f"无人机 {self.my_drone.identifier} 检测到的重要链路: " +
                                     ", ".join([f"({src}->{dst}: {val:.2f})" for src, dst, val in top_links]))

    def _is_network_congested(self):
        """检测网络是否拥塞"""
        # 检查队列占用
        current_queue_size = self.my_drone.transmitting_queue.qsize()
        queue_threshold = 0.7 * self.my_drone.max_queue_size

        # 检查平均延迟趋势
        recent_delays = self.performance_metrics.get('recent_delays', [])
        delay_increasing = False

        if len(recent_delays) >= 3:
            # 判断延迟是否持续增加
            delay_increasing = all(
                recent_delays[i] < recent_delays[i + 1] for i in range(len(recent_delays) - 3, len(recent_delays) - 1))

        # 综合判断
        return current_queue_size > queue_threshold or delay_increasing

    def _is_energy_critical(self):
        """检测能量是否处于临界状态"""
        if not self.energy_aware:
            return False

        # 计算能量比例
        current_energy = self.my_drone.residual_energy
        initial_energy = config.INITIAL_ENERGY
        energy_ratio = current_energy / initial_energy

        # 如果能量低于阈值，认为处于临界状态
        return energy_ratio < self.energy_threshold

    def _select_energy_efficient_path(self, dst_id):
        """选择能耗最小的路径"""
        if dst_id not in self.router.path_cache or not self.router.path_cache[dst_id]:
            return None

        paths = self.router.path_cache[dst_id]
        best_path = None
        min_energy_cost = float('inf')

        for path in paths:
            # 计算路径能耗
            energy_cost = self._estimate_path_energy_cost(path)

            if energy_cost < min_energy_cost:
                min_energy_cost = energy_cost
                best_path = path

        return best_path

    def _select_least_congested_path(self, dst_id):
        """选择拥塞最小的路径"""
        if dst_id not in self.router.path_cache or not self.router.path_cache[dst_id]:
            return None

        paths = self.router.path_cache[dst_id]
        best_path = None
        min_congestion = float('inf')

        for path in paths:
            # 估计路径拥塞度
            congestion = self._estimate_path_congestion(path)

            if congestion < min_congestion:
                min_congestion = congestion
                best_path = path

        return best_path

    def _select_lowest_delay_path(self, dst_id):
        """选择延迟最低的路径"""
        if dst_id not in self.router.path_cache or not self.router.path_cache[dst_id]:
            return None

        paths = self.router.path_cache[dst_id]
        best_path = None
        min_delay = float('inf')

        for path in paths:
            # 估计路径延迟
            delay = self._estimate_path_delay(path)

            if delay < min_delay:
                min_delay = delay
                best_path = path

        return best_path

    def _estimate_path_energy_cost(self, path):
        """估计路径能耗"""
        if not path:
            return float('inf')

        # 基础传输能耗 (每跳固定能耗)
        hop_count = len(path) - 1
        base_energy = hop_count * config.TRANSMITTING_POWER * 500  # 假设每跳500微秒传输时间

        # 考虑节点剩余能量
        total_energy_ratio = 0
        for node_id in path[1:-1]:  # 跳过源节点和目标节点
            if 0 <= node_id < len(self.simulator.drones):
                drone = self.simulator.drones[node_id]
                energy_ratio = drone.residual_energy / config.INITIAL_ENERGY
                total_energy_ratio += energy_ratio

        # 节点平均能量比例
        avg_energy_ratio = total_energy_ratio / (hop_count - 1) if hop_count > 1 else 1

        # 考虑距离因素
        total_distance = 0
        for i in range(len(path) - 1):
            node1 = path[i]
            node2 = path[i + 1]
            if 0 <= node1 < len(self.simulator.drones) and 0 <= node2 < len(self.simulator.drones):
                dist = euclidean_distance(
                    self.simulator.drones[node1].coords,
                    self.simulator.drones[node2].coords
                )
                total_distance += dist

        # 距离越远，能耗越高
        distance_factor = 1 + (total_distance / (hop_count * maximum_communication_range()))

        # 计算能耗成本 (考虑能量状态和距离)
        energy_cost = base_energy * distance_factor / avg_energy_ratio

        # 如果启用GAT，考虑节点重要性
        if self.gat_enabled and self.gat_analyzer.node_embeddings is not None:
            node_importance = self.gat_analyzer.get_node_importance()
            if node_importance is not None:
                # 计算路径上中继节点的平均重要性
                path_importance = 0
                for node_id in path[1:-1]:  # 跳过源节点和目标节点
                    if 0 <= node_id < len(node_importance):
                        path_importance += node_importance[node_id]

                avg_importance = path_importance / (hop_count - 1) if hop_count > 1 else 0

                # 重要节点能耗更高，避免过度使用
                energy_cost *= (1 + avg_importance)

        return energy_cost

    def _estimate_path_congestion(self, path):
        """估计路径拥塞度"""
        if not path:
            return float('inf')

        total_congestion = 0
        node_count = 0

        # 检查每个中继节点的队列状态
        for node_id in path[1:-1]:  # 跳过源节点和目标节点
            if 0 <= node_id < len(self.simulator.drones):
                drone = self.simulator.drones[node_id]
                queue_size = drone.transmitting_queue.qsize()
                max_queue = drone.max_queue_size

                # 计算拥塞度 (0-1)
                congestion = queue_size / max_queue
                total_congestion += congestion
                node_count += 1

        # 计算平均拥塞度
        avg_congestion = total_congestion / node_count if node_count > 0 else 0

        # 如果启用GAT，考虑GAT的路径质量
        if self.gat_enabled:
            gat_quality = self.gat_analyzer.evaluate_path_quality(path)
            # 将GAT质量因素融入拥塞评估 (质量越高，拥塞度越低)
            avg_congestion = avg_congestion * (1 - 0.5 * gat_quality)

        return avg_congestion

    def _estimate_path_delay(self, path):
        """估计路径延迟"""
        if not path:
            return float('inf')

        # 基础传输延迟 (考虑跳数)
        hop_count = len(path) - 1
        transmission_delay = hop_count * (config.DATA_PACKET_LENGTH / config.BIT_RATE * 1e6)

        # 考虑队列延迟
        queue_delay = 0
        for node_id in path[1:-1]:  # 跳过源节点和目标节点
            if 0 <= node_id < len(self.simulator.drones):
                drone = self.simulator.drones[node_id]
                queue_size = drone.transmitting_queue.qsize()

                # 估计队列延迟
                queue_delay += queue_size * 500  # 假设每个包处理时间500微秒

        # 考虑链路质量导致的延迟
        link_delay = 0
        for i in range(len(path) - 1):
            node1 = path[i]
            node2 = path[i + 1]
            link_key = (node1, node2)

            # 使用ETX估计链路延迟
            etx = self.router.expected_transmission_count.get(link_key, 1.0)
            link_delay += transmission_delay * (etx - 1)

        # 使用GAT注意力权重调整延迟估计
        if self.gat_enabled and self.use_attention_weights:
            attention_matrix = self.gat_analyzer.get_attention_matrix()
            if attention_matrix is not None:
                attention_factor = 1.0
                for i in range(len(path) - 1):
                    src, dst = path[i], path[i + 1]
                    # 注意：这里需要根据实际的注意力矩阵结构进行调整
                    if 0 <= src < len(attention_matrix) and 0 <= dst < len(attention_matrix[0]):
                        # 链路注意力权重越高，延迟估计越低
                        attention_weight = attention_matrix[src][dst]
                        attention_factor *= (1 - 0.3 * attention_weight)  # 减少30%的影响

                # 应用注意力因子
                link_delay *= attention_factor

        # 总延迟
        total_delay = transmission_delay + queue_delay + link_delay

        return total_delay

    def _update_forwarding_metrics(self, packet):
        """更新数据包转发指标"""
        if not isinstance(packet, DataPacket):
            return

        # 记录延迟历史
        if hasattr(packet, 'waiting_start_time') and packet.waiting_start_time is not None:
            queueing_delay = self.simulator.env.now - packet.waiting_start_time

            if 'recent_delays' not in self.performance_metrics:
                self.performance_metrics['recent_delays'] = []

            self.performance_metrics['recent_delays'].append(queueing_delay)

            # 保持最近的10条记录
            if len(self.performance_metrics['recent_delays']) > 10:
                self.performance_metrics['recent_delays'].pop(0)

        # 更新吞吐量指标
        if 'packets_sent' not in self.performance_metrics:
            self.performance_metrics['packets_sent'] = 0

        self.performance_metrics['packets_sent'] += 1

        # 更新目的地统计
        dst_id = packet.dst_drone.identifier
        if 'destinations' not in self.performance_metrics:
            self.performance_metrics['destinations'] = {}

        if dst_id not in self.performance_metrics['destinations']:
            self.performance_metrics['destinations'][dst_id] = 0

        self.performance_metrics['destinations'][dst_id] += 1

        # 记录GAT的使用情况
        if self.gat_enabled:
            if 'gat_usage' not in self.performance_metrics:
                self.performance_metrics['gat_usage'] = 0

            self.performance_metrics['gat_usage'] += 1

    def monitor_network_performance(self):
        """监控网络性能"""
        while True:
            yield self.simulator.env.timeout(1 * 1e6)  # 每1秒监控一次

            # 收集网络性能指标
            self._collect_performance_metrics()

            # 检查是否需要调整路由策略
            self._check_and_adjust_strategy()

            # 更新节点状态
            self._update_node_stats()

            # 定期清理过期数据
            self._clean_expired_data()

    def _collect_performance_metrics(self):
        """收集网络性能指标"""
        # 收集端到端延迟
        avg_delay = 0
        delay_count = 0

        for packet_id in self.simulator.metrics.deliver_time_dict:
            delay = self.simulator.metrics.deliver_time_dict[packet_id]
            avg_delay += delay
            delay_count += 1

        if delay_count > 0:
            avg_delay /= delay_count
            self.performance_metrics['avg_delay'] = avg_delay

        # 收集吞吐量
        throughput = self.simulator.metrics.calculate_throughput()
        self.performance_metrics['throughput'] = throughput

        # 收集数据包投递率
        pdr = self.simulator.metrics.calculate_pdr(self.simulator)
        self.performance_metrics['pdr'] = pdr

        # 收集平均跳数
        avg_hops = 0
        hop_count = 0

        for packet_id in self.simulator.metrics.hop_cnt_dict:
            hops = self.simulator.metrics.hop_cnt_dict[packet_id]
            avg_hops += hops
            hop_count += 1

        if hop_count > 0:
            avg_hops /= hop_count
            self.performance_metrics['avg_hops'] = avg_hops

        # 收集能量消耗
        energy_consumption = 0
        for drone_id in self.simulator.metrics.energy_consumption:
            energy_consumption += self.simulator.metrics.energy_consumption[drone_id]

        self.performance_metrics['energy_consumption'] = energy_consumption

        # 记录GAT使用情况
        if 'gat_usage' in self.performance_metrics:
            self.performance_metrics['gat_usage_ratio'] = self.performance_metrics['gat_usage'] / max(1,
                                                                                                      self.performance_metrics[
                                                                                                          'packets_sent'])

        # 记录日志
        logging.info("网络性能: 延迟=%.2f ms, 吞吐量=%.2f bps, PDR=%.2f%%, 平均跳数=%.2f, 能耗=%.2f J",
                     avg_delay / 1000 if 'avg_delay' in self.performance_metrics else 0,
                     throughput,
                     pdr * 100,
                     avg_hops if 'avg_hops' in self.performance_metrics else 0,
                     energy_consumption)

    def _check_and_adjust_strategy(self):
        """检查并调整路由策略"""
        # 获取当前性能指标
        pdr = self.performance_metrics.get('pdr', 0)
        avg_delay = self.performance_metrics.get('avg_delay', float('inf'))
        throughput = self.performance_metrics.get('throughput', 0)

        # 临界性能阈值
        pdr_threshold = 0.7  # 70%的PDR
        delay_threshold = 1000 * 1000  # 1秒延迟

        # 检测是否需要调整策略
        strategy_changed = False

        # 情况1: PDR过低，切换到GAT或best_quality
        if pdr < pdr_threshold and self.path_selection_strategy != 'gat' and self.gat_enabled:
            self.path_selection_strategy = 'gat'
            strategy_changed = True
            logging.info("PDR过低(%.2f%%), 调整为GAT策略", pdr * 100)

        # 情况2: 延迟过高，调整为lowest_delay优先
        elif avg_delay > delay_threshold and self.path_selection_strategy != 'adaptive':
            self.path_selection_strategy = 'adaptive'
            strategy_changed = True
            logging.info("延迟过高(%.2f ms), 调整为自适应策略", avg_delay / 1000)

        # 情况3: 能量消耗过大，启用能量感知模式
        elif not self.energy_aware and self.my_drone.residual_energy < config.INITIAL_ENERGY * self.energy_threshold:
            self.energy_aware = True
            strategy_changed = True
            logging.info("能量不足(%.2f%%), 启用能量感知模式",
                         self.my_drone.residual_energy / config.INITIAL_ENERGY * 100)

        # 情况4: 性能稳定时，尝试重新训练GAT模型
        elif self.gat_enabled and pdr > 0.85 and 'gat_usage_ratio' in self.performance_metrics:
            gat_usage = self.performance_metrics['gat_usage_ratio']
            if gat_usage < 0.5:  # GAT使用率低于50%
                self.gat_analyzer.train_model(self.simulator, epochs=5)
                logging.info("性能良好但GAT使用率低(%.2f%%), 重新训练GAT模型", gat_usage * 100)

        # 如果策略变化，重新发现所有路径
        if strategy_changed:
            self._rediscover_all_paths()

    def _rediscover_all_paths(self):
        """重新发现所有目的地的路径"""
        # 清空当前路径缓存
        old_destinations = list(self.router.path_cache.keys())
        self.router.path_cache = {}

        # 为每个目的地重新发现路径
        for dst_id in old_destinations:
            paths = self.router.discover_multiple_paths(self.my_drone.identifier, dst_id)
            if paths:
                self.router.path_cache[dst_id] = paths
                logging.info("重新发现到目的地 %d 的 %d 条路径", dst_id, len(paths))

    def _update_node_stats(self):
        """更新节点状态信息"""
        # 更新自身状态
        self.load_balancer.update_node_stats(
            self.my_drone.identifier,
            queue_size=self.my_drone.transmitting_queue.qsize(),
            throughput=self.performance_metrics.get('throughput', 0),
            delay=self.performance_metrics.get('avg_delay', 0)
        )

        # 更新邻居状态
        for neighbor_id in self.router.neighbor_table:
            if 0 <= neighbor_id < len(self.simulator.drones):
                # 估计邻居状态
                neighbor = self.simulator.drones[neighbor_id]
                queue_size = neighbor.transmitting_queue.qsize()
                residual_energy = neighbor.residual_energy

                # 更新到负载均衡器
                self.load_balancer.update_node_stats(
                    neighbor_id,
                    queue_size=queue_size
                )

    def _clean_expired_data(self):
        """清理过期数据"""
        # 清理过期的路由缓存
        current_time = self.simulator.env.now

        # 默认缓存超时 (10秒)
        cache_timeout = 10 * 1e6

        # 检查路径缓存
        for dst_id in list(self.router.path_cache.keys()):
            if dst_id in self.router.last_path_update:
                if current_time - self.router.last_path_update[dst_id] > cache_timeout:
                    # 缓存超时，从路径缓存中删除
                    del self.router.path_cache[dst_id]
                    del self.router.last_path_update[dst_id]
                    logging.info("清理到目的地 %d 的过期路径缓存", dst_id)

        # 清理性能指标历史
        if 'recent_delays' in self.performance_metrics:
            # 只保留最近10条记录
            if len(self.performance_metrics['recent_delays']) > 10:
                self.performance_metrics['recent_delays'] = self.performance_metrics['recent_delays'][-10:]

    def get_performance_metrics(self):
        """获取性能指标"""
        return self.performance_metrics

    def get_routing_stats(self):
        """获取路由统计信息"""
        stats = {
            'path_cache_size': sum(len(paths) for paths in self.router.path_cache.values()),
            'neighbor_count': len(self.router.neighbor_table),
            'strategy': self.path_selection_strategy,
            'energy_aware': self.energy_aware,
            'gat_enabled': self.gat_enabled
        }

        # 添加GAT相关统计
        if self.gat_enabled:
            # 获取节点重要性
            node_importance = self.gat_analyzer.get_node_importance()
            if node_importance is not None:
                stats['node_importance'] = node_importance[self.my_drone.identifier]

            # 添加GAT使用率
            if 'gat_usage_ratio' in self.performance_metrics:
                stats['gat_usage_ratio'] = self.performance_metrics['gat_usage_ratio']

        return stats

    def get_path_details(self, dst_id=None):
        """获取路径详细信息"""
        if dst_id is not None:
            if dst_id in self.router.path_cache:
                paths = self.router.path_cache[dst_id]

                # 如果启用GAT，添加路径质量评估
                path_qualities = {}
                if self.gat_enabled:
                    for i, path in enumerate(paths):
                        path_qualities[i] = self.gat_analyzer.evaluate_path_quality(path)

                return {
                    'destination': dst_id,
                    'path_count': len(paths),
                    'paths': paths,
                    'path_qualities': path_qualities,
                    'last_update': self.router.last_path_update.get(dst_id, 0)
                }
            return None

        # 返回所有路径信息
        result = {}
        for dst_id in self.router.path_cache:
            paths = self.router.path_cache[dst_id]

            # 计算路径质量
            path_qualities = {}
            if self.gat_enabled:
                for i, path in enumerate(paths):
                    path_qualities[i] = self.gat_analyzer.evaluate_path_quality(path)

            result[dst_id] = {
                'path_count': len(paths),
                'paths': paths,
                'path_qualities': path_qualities,
                'last_update': self.router.last_path_update.get(dst_id, 0)
            }

        return result

    def visualize_network(self, output_file='network_visualization.png'):
        """可视化网络拓扑和GAT结果"""
        if not self.gat_enabled:
            logging.warning("GAT未启用，无法生成可视化")
            return False

        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            import numpy as np

            # 创建图
            G = nx.Graph()

            # 添加节点
            for i in range(self.simulator.n_drones):
                drone = self.simulator.drones[i]
                G.add_node(i, pos=(drone.coords[0], drone.coords[1]))

            # 添加边
            edges = []
            for i in range(self.simulator.n_drones):
                for j in range(i + 1, self.simulator.n_drones):
                    drone1 = self.simulator.drones[i]
                    drone2 = self.simulator.drones[j]

                    dist = euclidean_distance(drone1.coords, drone2.coords)
                    if dist <= maximum_communication_range():
                        G.add_edge(i, j)
                        edges.append((i, j))

            # 获取节点位置
            pos = nx.get_node_attributes(G, 'pos')

            # 获取节点重要性
            node_importance = self.gat_analyzer.get_node_importance()
            if node_importance is None:
                node_importance = np.ones(self.simulator.n_drones)

            # 获取注意力权重
            attention_weights = self.gat_analyzer.get_attention_matrix()

            # 设置节点颜色
            node_colors = node_importance

            # 设置节点大小
            node_sizes = [300 + 500 * imp for imp in node_importance]

            # 创建图
            plt.figure(figsize=(12, 10))

            # 绘制节点
            nx.draw_networkx_nodes(G, pos,
                                   node_color=node_colors,
                                   node_size=node_sizes,
                                   cmap=plt.cm.Reds,
                                   alpha=0.8)

            # 绘制边
            if attention_weights is not None:
                # 设置边宽度基于注意力权重
                edge_widths = []
                for u, v in edges:
                    if u < len(attention_weights) and v < len(attention_weights[0]):
                        weight = attention_weights[u][v]
                        edge_widths.append(1 + 5 * weight)
                    else:
                        edge_widths.append(1.0)

                nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5)
            else:
                nx.draw_networkx_edges(G, pos, alpha=0.5)

            # 添加标签
            nx.draw_networkx_labels(G, pos)

            # 添加标题
            plt.title('无人机网络拓扑与GAT分析')

            # 保存图像
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            logging.info(f"网络可视化已保存至 {output_file}")
            return True

        except Exception as e:
            logging.error(f"生成网络可视化时出错: {str(e)}")
            return False