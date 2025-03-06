import logging
from utils import config
from entities.packet import DataPacket
from collections import defaultdict
import math
import random


class MultipathIntegration:
    """
    将多路径负载均衡集成到现有无人机网络系统中的辅助类
    该类负责：
    1. 修改现有路由协议以支持多路径路由
    2. 在适当的节点上添加多路径路由组件
    3. 协调多路径路由和负载均衡决策
    """

    def __init__(self, simulator):
        self.simulator = simulator
        self.multipath_routers = {}  # drone_id -> MultipathRouter实例

        # 初始化集成配置
        self._initialize_multipath_config()

        # 为每个无人机添加多路径路由功能
        self._setup_multipath_routing()

    def _initialize_multipath_config(self):
        """初始化多路径相关配置"""
        # 确保config中有必要的多路径配置
        if not hasattr(config, 'MULTIPATH_ENABLED'):
            config.MULTIPATH_ENABLED = True

        if not hasattr(config, 'MAX_PATHS'):
            config.MAX_PATHS = 3

        if not hasattr(config, 'PATH_SELECTION_STRATEGY'):
            config.PATH_SELECTION_STRATEGY = 'adaptive'  # 可选: weighted, round_robin, adaptive

        logging.info(f"多路径负载均衡已启用，最大路径数: {config.MAX_PATHS}，"
                     f"路径选择策略: {config.PATH_SELECTION_STRATEGY}")

    def _setup_multipath_routing(self):
        """为每个无人机设置多路径路由功能"""
        # 在每个无人机上创建多路径路由实例
        from routing.multipath.multipath_load_balancer import MultipathRouter, MultiPathLoadBalancer

        for drone in self.simulator.drones:
            try:
                # 创建多路径路由器
                router = MultipathRouter(drone, max_paths=config.MAX_PATHS)

                # 保存路由器实例
                self.multipath_routers[drone.identifier] = router

                # 修改原有路由协议，添加多路径支持
                self._enhance_routing_protocol(drone, router)

                logging.info(f"无人机 {drone.identifier} 已启用多路径路由")

            except Exception as e:
                logging.error(f"为无人机 {drone.identifier} 设置多路径路由时出错: {str(e)}")

    def _enhance_routing_protocol(self, drone, multipath_router):
        """增强现有路由协议以支持多路径路由"""
        # 保存原始next_hop_selection方法
        original_next_hop = drone.routing_protocol.next_hop_selection

        # 创建新的路由决策方法，优先使用多路径路由，当多路径路由失败时回退到原始方法
        def enhanced_next_hop_selection(packet):
            # 仅处理数据包
            if isinstance(packet, DataPacket):
                # 如果启用了多路径模式，且非高负载情况下，尝试多路径路由
                if config.MULTIPATH_ENABLED:
                    # 获取多路径路由器的下一跳决策
                    next_hop = multipath_router.get_next_hop(packet)

                    if next_hop is not None:
                        # 多路径路由找到了有效的下一跳
                        packet.next_hop_id = next_hop

                        # 标记为使用多路径传输
                        packet.is_multipath = True

                        # 设置传输模式为单播
                        packet.transmission_mode = 0

                        logging.debug(f"UAV {drone.identifier}: 使用多路径路由决策，"
                                      f"数据包 {packet.packet_id} 下一跳为 {next_hop}")

                        return True, packet, False

            # 如果多路径路由未启用或未找到有效路径，回退到原始路由方法
            return original_next_hop(packet)

        # 替换路由协议的next_hop_selection方法
        drone.routing_protocol.next_hop_selection = enhanced_next_hop_selection

        # 保存多路径路由器的引用
        drone.multipath_router = multipath_router

        # 修改packet_reception方法以适应多路径传输
        original_packet_reception = drone.routing_protocol.packet_reception

        def enhanced_packet_reception(packet, sender):
            # 处理收到的数据包
            if isinstance(packet, DataPacket) and hasattr(packet, 'is_multipath') and packet.is_multipath:
                # 多路径数据包的特殊处理
                if packet.dst_drone.identifier == drone.identifier:
                    # 目的地是本节点，处理数据包
                    self._handle_multipath_packet_arrival(packet, drone)

                    # 调用原始方法完成处理
                    return original_packet_reception(packet, sender)

                # 作为中继节点处理
                # 更新路径统计
                self._update_path_statistics(packet, sender, drone)

            # 调用原始方法完成处理
            return original_packet_reception(packet, sender)

        # 替换路由协议的packet_reception方法
        drone.routing_protocol.packet_reception = enhanced_packet_reception

    def _handle_multipath_packet_arrival(self, packet, drone):
        """处理多路径数据包到达目的地的情况"""
        flow_id = f"{packet.src_drone.identifier}_{packet.dst_drone.identifier}"

        # 检查是否为并行传输
        if hasattr(packet, 'parallel_paths') and packet.parallel_paths:
            # 记录哪条路径成功传输了数据包
            if hasattr(packet, 'path_id') and packet.path_id:
                multipath_router = drone.multipath_router
                load_balancer = multipath_router.load_balancer

                # 更新成功路径集合
                packet.successful_paths.add(packet.path_id)

                # 更新路径统计
                if hasattr(load_balancer, 'path_stats'):
                    src_id = packet.src_drone.identifier
                    dst_id = drone.identifier

                    if dst_id in load_balancer.path_stats[src_id] and packet.path_id in \
                            load_balancer.path_stats[src_id][dst_id]:
                        # 根据端到端延迟更新路径质量
                        e2e_delay = (drone.env.now - packet.creation_time) / 1e3  # 转换为毫秒

                        load_balancer.update_path_stats(
                            src_id,
                            dst_id,
                            packet.path_id,
                            {
                                'delay': e2e_delay,
                                'loss_rate': 0.0,  # 成功传输，丢包率为0
                                'stability': 0.9,  # 成功传输，稳定性高
                            }
                        )

                        logging.info(
                            f"UAV {drone.identifier}: 数据包 {packet.packet_id} 通过路径 {packet.path_id} 成功到达，"
                            f"端到端延迟 {e2e_delay:.2f}ms")

        # 更新路由协议统计信息 (在原始路由协议中完成)

    def _update_path_statistics(self, packet, sender, drone):
        """更新多路径数据包的路径统计信息"""
        if hasattr(drone, 'multipath_router'):
            multipath_router = drone.multipath_router

            # 计算单跳延迟
            if hasattr(packet, 'time_transmitted_at_last_hop') and packet.time_transmitted_at_last_hop > 0:
                hop_delay = (drone.env.now - packet.time_transmitted_at_last_hop) / 1e3  # 转换为毫秒

                # 更新链路统计
                if sender in packet.routing_path and drone.identifier in packet.routing_path:
                    # 获取当前链路在路径中的位置
                    sender_idx = packet.routing_path.index(sender)
                    current_idx = packet.routing_path.index(drone.identifier)

                    if current_idx == sender_idx + 1:  # 确保是直接相连的链路
                        # 更新链路质量信息
                        for path_id in multipath_router.paths_cache.get(packet.dst_drone.identifier, []):
                            path = multipath_router._get_path(path_id)
                            if path and sender in path and drone.identifier in path:
                                # 这条链路是该路径的一部分
                                src_id = packet.src_drone.identifier
                                dst_id = packet.dst_drone.identifier

                                # 获取链路的当前统计
                                current_stats = multipath_router.load_balancer.path_stats.get(src_id, {}).get(dst_id,
                                                                                                              {}).get(
                                    path_id, {})

                                if current_stats:
                                    # 使用指数加权移动平均更新链路延迟
                                    alpha = 0.3  # 新样本权重
                                    if 'delay' in current_stats:
                                        updated_delay = (1 - alpha) * current_stats['delay'] + alpha * hop_delay
                                    else:
                                        updated_delay = hop_delay

                                    # 更新路径统计
                                    multipath_router.load_balancer.update_path_stats(
                                        src_id,
                                        dst_id,
                                        path_id,
                                        {
                                            'delay': updated_delay,
                                            'stability': 0.8,  # 成功中继，稳定性高
                                        }
                                    )

                logging.debug(f"UAV {drone.identifier}: 从 {sender} 接收到数据包 {packet.packet_id}，"
                              f"单跳延迟 {hop_delay:.2f}ms")


    def monitor_network_performance(self):
        """监控网络整体性能，收集多路径路由的效果数据"""
        # 创建性能指标存储
        self.performance_metrics = {
            'end_to_end_delay': defaultdict(list),  # flow_id -> [delays]
            'path_utilization': defaultdict(lambda: defaultdict(int)),  # src_dst -> path_id -> count
            'load_distribution': defaultdict(list),  # node_id -> [load_values]
            'successful_paths': defaultdict(set),  # flow_id -> {successful_path_ids}
        }

        # 监控循环
        while True:
            yield self.simulator.env.timeout(config.MONITOR_INTERVAL)

            # 收集所有无人机的性能数据
            self._collect_performance_data()

            # 分析和记录性能
            self._analyze_performance()

            # 定期打印摘要
            if self.simulator.env.now % (10 * config.MONITOR_INTERVAL) == 0:
                self._print_performance_summary()


    def _collect_performance_data(self):
        """收集所有无人机的性能数据"""
        # 遍历所有无人机
        for drone in self.simulator.drones:
            if drone.identifier in self.multipath_routers:
                router = self.multipath_routers[drone.identifier]

                # 收集队列长度，更新负载分布
                if hasattr(drone, 'transmitting_queue'):
                    queue_size = drone.transmitting_queue.qsize()
                    max_queue = config.MAX_QUEUE_SIZE
                    load = queue_size / max_queue if max_queue > 0 else 0
                    self.performance_metrics['load_distribution'][drone.identifier].append(load)

                    # 更新路由器的负载统计
                    router.load_balancer.update_node_load(drone.identifier, queue_size)

                # 收集路径使用情况
                for dst_id in router.paths_cache:
                    src_dst = f"{drone.identifier}_{dst_id}"

                    # 获取路径使用计数
                    flow_id = f"{drone.identifier}_{dst_id}"
                    if flow_id in router.load_balancer.active_flows:
                        path_counters = router.load_balancer.active_flows[flow_id].get('path_counters', {})

                        for path_id, count in path_counters.items():
                            self.performance_metrics['path_utilization'][src_dst][path_id] = count


    def _analyze_performance(self):
        """分析收集的性能数据"""
        # 计算平均负载
        node_loads = {}
        for node_id, loads in self.performance_metrics['load_distribution'].items():
            if loads:
                node_loads[node_id] = sum(loads) / len(loads)

        # 检测负载不平衡情况
        if node_loads:
            avg_load = sum(node_loads.values()) / len(node_loads)
            high_load_nodes = []
            low_load_nodes = []

            for node_id, load in node_loads.items():
                if load > avg_load * 1.5:
                    high_load_nodes.append((node_id, load))
                elif load < avg_load * 0.5:
                    low_load_nodes.append((node_id, load))

            # 记录负载不平衡情况
            if high_load_nodes:
                high_load_nodes.sort(key=lambda x: x[1], reverse=True)
                logging.warning(f"检测到高负载节点: {high_load_nodes[:3]}")

            # 分析路径利用率
            for src_dst, path_usage in self.performance_metrics['path_utilization'].items():
                if path_usage:
                    total_packets = sum(path_usage.values())
                    if total_packets > 0:
                        # 计算路径分布
                        distribution = {path_id: count / total_packets for path_id, count in path_usage.items()}

                        # 检查分布是否均衡
                        values = list(distribution.values())
                        if len(values) > 1:
                            # 计算变异系数 (CV) 作为不平衡的度量
                            mean = sum(values) / len(values)
                            std_dev = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))
                            cv = std_dev / mean if mean > 0 else 0

                            if cv > 0.5:  # 如果CV大于0.5，认为分布不均衡
                                logging.info(f"流 {src_dst} 的路径利用不平衡 (CV={cv:.2f}): {distribution}")


    def _print_performance_summary(self):
        """打印多路径性能摘要"""
        logging.info("\n==== 多路径负载均衡性能摘要 ====")

        # 打印节点负载情况
        node_loads = {}
        for node_id, loads in self.performance_metrics['load_distribution'].items():
            if loads:
                node_loads[node_id] = sum(loads[-10:]) / min(10, len(loads))  # 最近10个样本的平均值

        if node_loads:
            sorted_loads = sorted([(node_id, load) for node_id, load in node_loads.items()],
                                  key=lambda x: x[1], reverse=True)

            logging.info("节点负载情况 (前5个):")
            for node_id, load in sorted_loads[:5]:
                logging.info(f"  UAV {node_id}: {load * 100:.1f}%")

        # 打印路径利用情况
        active_flows = []
        for src_dst, path_usage in self.performance_metrics['path_utilization'].items():
            if path_usage:
                total_packets = sum(path_usage.values())
                if total_packets > 10:  # 只考虑有足够数据包的流
                    active_flows.append((src_dst, path_usage, total_packets))

        if active_flows:
            active_flows.sort(key=lambda x: x[2], reverse=True)  # 按数据包总数排序

            logging.info("\n活跃流路径分布 (前3个):")
            for src_dst, path_usage, total in active_flows[:3]:
                src_id, dst_id = src_dst.split('_')
                logging.info(f"  流 {src_id}->{dst_id} (共 {total} 个数据包):")

                # 按使用量排序路径
                sorted_paths = sorted(path_usage.items(), key=lambda x: x[1], reverse=True)
                for path_id, count in sorted_paths:
                    percentage = (count / total) * 100
                    logging.info(f"    路径 {path_id}: {count} 个数据包 ({percentage:.1f}%)")

        logging.info("============================\n")


    def adjust_multipath_distribution(self):
        """根据网络状态动态调整多路径分布策略"""
        while True:
            # 每5秒评估一次
            yield self.simulator.env.timeout(5 * 1e6)  # 5秒

            # 分析当前网络状态
            high_load_nodes = self._get_high_load_nodes()

            if high_load_nodes:
                logging.info(f"检测到高负载节点: {high_load_nodes}")

                # 对每个高负载节点进行负载重分配
                for node_id, load in high_load_nodes:
                    self._redistribute_load(node_id)


    def _get_high_load_nodes(self):
        """识别高负载节点"""
        high_load_nodes = []

        for drone in self.simulator.drones:
            if hasattr(drone, 'transmitting_queue'):
                queue_size = drone.transmitting_queue.qsize()
                max_queue = config.MAX_QUEUE_SIZE

                # 如果队列超过80%容量，认为是高负载
                if max_queue > 0 and queue_size > 0.8 * max_queue:
                    high_load_nodes.append((drone.identifier, queue_size / max_queue))

        # 按负载降序排序
        high_load_nodes.sort(key=lambda x: x[1], reverse=True)

        return high_load_nodes


    def _redistribute_load(self, high_load_node_id):
        """尝试为高负载节点重新分配负载"""
        # 找出经过该节点的所有活跃流
        flows_through_node = []

        for drone_id, router in self.multipath_routers.items():
            for dst_id in router.paths_cache:
                # 检查每个缓存的路径
                for path_id in router.paths_cache[dst_id]:
                    path = router._get_path(path_id)

                    if path and high_load_node_id in path:
                        # 这个流经过高负载节点
                        flow = (drone_id, dst_id, path_id)
                        flows_through_node.append(flow)

        if not flows_through_node:
            logging.info(f"未找到经过节点 {high_load_node_id} 的活跃流")
            return

        # 随机选择一部分流进行重新路由
        # 在实际应用中，可以使用更复杂的策略来选择流
        random.shuffle(flows_through_node)
        flows_to_reroute = flows_through_node[:min(3, len(flows_through_node))]

        for src_id, dst_id, path_id in flows_to_reroute:
            if src_id in self.multipath_routers:
                router = self.multipath_routers[src_id]

                # 尝试发现避开高负载节点的新路径
                logging.info(f"尝试为流 {src_id}->{dst_id} 发现避开节点 {high_load_node_id} 的新路径")

                # 这里可以添加特定的路径发现逻辑
                new_paths = self._discover_alternate_paths(src_id, dst_id, high_load_node_id)

                if new_paths:
                    logging.info(f"为流 {src_id}->{dst_id} 发现 {len(new_paths)} 条新路径")

                    # 更新路径缓存
                    for new_path_id in new_paths:
                        if new_path_id not in router.paths_cache.get(dst_id, []):
                            router.paths_cache[dst_id].append(new_path_id)


    def _discover_alternate_paths(self, src_id, dst_id, avoid_node_id):
        """发现避开特定节点的替代路径"""
        if src_id not in self.multipath_routers:
            return []

        router = self.multipath_routers[src_id]

        # 调用路由器的路径发现方法，但避开指定节点
        try:
            # 假设MultipathRouter类有一个discover_paths_avoiding方法
            # 这需要在MultipathRouter类中实现
            if hasattr(router, '_discover_paths_avoiding'):
                return router._discover_paths_avoiding(dst_id, [avoid_node_id])

            # 如果没有专门的方法，我们可以简单地调用常规的discover_paths方法
            # 然后过滤掉包含避开节点的路径
            new_paths = router.discover_paths(dst_id)
            filtered_paths = []

            for path_id in new_paths:
                path = router._get_path(path_id)
                if path and avoid_node_id not in path:
                    filtered_paths.append(path_id)

            return filtered_paths
        except Exception as e:
            logging.error(f"发现替代路径时出错: {str(e)}")
            return []