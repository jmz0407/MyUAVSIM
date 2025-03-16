import logging
import math
from phy.phy import Phy
from utils import config
from utils.util_function import euclidean_distance
from entities.packet import DataPacket
from mac.LinkQualityManager import LinkQualityManager
from mac.LoadBalancer import LoadBalancer
from simulator.TrafficGenerator import TrafficRequirement
import traceback
import copy
import numpy as np
import heapq


class BasicStdma:
    """
    基础STDMA实现，使用启发式算法进行时隙分配
    不依赖强化学习，适用于可变规模无人机网络
    """

    def __init__(self, drone):
        self.my_drone = drone
        self.simulator = drone.simulator
        self.env = drone.env
        self.time_slot_duration = config.SLOT_DURATION
        self.num_slots = config.NUMBER_OF_DRONES  # 初始时隙数
        self.current_slot = 0
        self.phy = Phy(self)
        self.current_transmission = None

        # 存储时隙分配表
        # 格式: {slot_id: [drone_ids]}
        self.slot_schedule = {}

        # 链路质量管理
        self.link_quality_manager = LinkQualityManager()

        # 负载管理
        self.load_balancer = LoadBalancer()

        # 流量需求
        self.traffic_requirements = {}

        # 流量队列和统计
        self.flow_queue = {}
        self.flow_stats = {}

        # 邻居表缓存 - 缓存结果以提高性能
        self._neighbor_cache = {}
        self._neighbor_cache_time = 0
        self._cache_validity = 1000000  # 缓存有效期 (ns)

        # 推迟初始化时隙表
        self.env.process(self._delayed_initialization())

        # 启动时隙同步进程
        self.env.process(self._slot_synchronization())

        # 启动性能监控
        self.env.process(self._monitor_performance())

    def _delayed_initialization(self):
        """推迟初始化，等待所有无人机完成初始化"""
        # 延迟一小段时间，让模拟器完成所有无人机的初始化
        yield self.env.timeout(100)
        # 创建初始调度表
        self.slot_schedule = self._create_slot_schedule()
        logging.info(f"UAV{self.my_drone.identifier} 完成时隙初始化")

    def _slot_synchronization(self):
        """时隙同步进程"""
        while True:
            self.current_slot = (self.env.now // self.time_slot_duration) % self.num_slots
            yield self.env.timeout(self.time_slot_duration)

    def _monitor_performance(self):
        """监控STDMA性能"""
        while True:
            yield self.env.timeout(5 * 1e6)  # 每5秒监控一次

            # 计算性能指标
            self._calculate_performance_metrics()

    def _calculate_performance_metrics(self):
        """计算并记录STDMA性能指标"""
        try:
            # 计算时隙利用率
            total_slots = len(self.slot_schedule)
            if total_slots == 0:
                return

            total_assignments = sum(len(nodes) for nodes in self.slot_schedule.values())
            slot_utilization = total_assignments / total_slots

            # 计算空间复用率
            spatial_reuse = total_assignments / self.simulator.n_drones if self.simulator.n_drones > 0 else 0

            # 记录指标
            logging.info(f"STDMA性能: 时隙数={total_slots}, 分配数={total_assignments}, " +
                         f"时隙利用率={slot_utilization:.2f}, 空间复用率={spatial_reuse:.2f}")

            # 检查最大延迟
            max_delay = 0
            for flow_id, stats in self.flow_stats.items():
                if 'avg_delay' in stats and stats['avg_delay'] > max_delay:
                    max_delay = stats['avg_delay']

            logging.info(f"最大流延迟: {max_delay:.2f} ms")

        except Exception as e:
            logging.error(f"计算性能指标出错: {e}")

    def _get_neighbor_table(self, refresh=False):
        """获取全局邻居表，带缓存"""
        current_time = self.env.now

        # 检查缓存是否有效
        if (not refresh and
                self._neighbor_cache and
                current_time - self._neighbor_cache_time < self._cache_validity):
            return self._neighbor_cache

        # 重建邻居表
        neighbor_table = {}
        for i in range(self.simulator.n_drones):
            neighbors = set()
            for j in range(self.simulator.n_drones):
                if i == j:
                    continue

                # 计算物理距离
                distance = euclidean_distance(
                    self.simulator.drones[i].coords,
                    self.simulator.drones[j].coords
                )

                # 检查是否在通信范围内
                if distance <= self._get_communication_range():
                    neighbors.add(j)

            neighbor_table[i] = neighbors

        # 更新缓存
        self._neighbor_cache = neighbor_table
        self._neighbor_cache_time = current_time

        return neighbor_table

    def _get_communication_range(self):
        """获取通信范围"""
        from phy.large_scale_fading import maximum_communication_range
        return maximum_communication_range()

    def _get_interference_range(self):
        """获取干扰范围，通常大于通信范围"""
        return self._get_communication_range() * 1.5

    def _get_two_hop_neighbors(self, node_id):
        """获取二跳邻居集合"""
        neighbor_table = self._get_neighbor_table()

        # 一跳邻居
        one_hop = neighbor_table.get(node_id, set())

        # 二跳邻居
        two_hop = set()
        for neighbor in one_hop:
            two_hop.update(neighbor_table.get(neighbor, set()))

        # 移除自身和一跳邻居
        two_hop -= {node_id}
        two_hop -= one_hop

        return one_hop, two_hop

    def _calculate_node_priority(self, node_id):
        """计算节点优先级，用于时隙分配顺序

        优先级考虑因素:
        1. 节点流量负载
        2. 节点度数（邻居数量）
        3. 节点位置中心性
        """
        priority = 0

        # 考虑流量负载
        if hasattr(self.simulator.drones[node_id], 'transmitting_queue'):
            queue_size = self.simulator.drones[node_id].transmitting_queue.qsize()
            priority += queue_size * 10  # 队列长度权重

        # 考虑节点度数（连接的邻居数）
        neighbors, _ = self._get_two_hop_neighbors(node_id)
        priority += len(neighbors) * 5  # 邻居数权重

        # 考虑位置中心性
        center_x = sum(drone.coords[0] for drone in self.simulator.drones) / self.simulator.n_drones
        center_y = sum(drone.coords[1] for drone in self.simulator.drones) / self.simulator.n_drones
        center_z = sum(drone.coords[2] for drone in self.simulator.drones) / self.simulator.n_drones

        distance_to_center = euclidean_distance(
            self.simulator.drones[node_id].coords,
            (center_x, center_y, center_z)
        )

        # 距离中心越近，优先级越高
        center_factor = 1.0 - (distance_to_center / (3 * self._get_communication_range()))
        priority += center_factor * 3  # 中心性权重

        return priority

    def _create_slot_schedule(self):
        """创建时隙分配表，使用图着色启发式算法"""
        logging.info("使用图着色算法创建STDMA时隙分配表")

        # 初始化时隙表
        schedule = {}

        # 获取干扰图
        interference_graph = self._build_interference_graph()

        # 计算每个节点的优先级
        node_priorities = [(self._calculate_node_priority(i), i) for i in range(self.simulator.n_drones)]

        # 根据优先级从高到低排序节点
        node_priorities.sort(reverse=True)

        # 逐个为节点分配时隙
        for _, node_id in node_priorities:
            # 找到该节点可用的最小时隙
            assigned_slot = self._find_available_slot(node_id, schedule, interference_graph)

            # 将节点添加到时隙
            if assigned_slot not in schedule:
                schedule[assigned_slot] = []
            schedule[assigned_slot].append(node_id)

            logging.info(f"为节点 {node_id} 分配时隙 {assigned_slot}")

        # 输出最终的时隙分配情况
        logging.info(f"STDMA时隙分配完成，使用 {len(schedule)} 个时隙")
        for slot, nodes in schedule.items():
            logging.info(f"时隙 {slot}: {nodes}")

        # 优化时隙编号，使其从0开始连续
        optimized_schedule = self._optimize_slot_numbering(schedule)

        return optimized_schedule

    def _build_interference_graph(self):
        """构建干扰图，表示哪些节点会相互干扰"""
        graph = {}

        try:
            # 获取干扰范围
            interference_range = self._get_interference_range()

            # 确保不超出无人机列表范围
            n_drones = len(self.simulator.drones)

            # 构建干扰图
            for i in range(n_drones):
                graph[i] = set()

                for j in range(n_drones):
                    if i == j:
                        continue

                    # 计算距离
                    try:
                        distance = euclidean_distance(
                            self.simulator.drones[i].coords,
                            self.simulator.drones[j].coords
                        )

                        # 检查是否在干扰范围内
                        if distance <= interference_range:
                            graph[i].add(j)
                    except Exception as e:
                        logging.error(f"计算节点{i}和{j}之间干扰时出错: {e}")
        except Exception as e:
            logging.error(f"构建干扰图出错: {e}")

        return graph

    def _find_available_slot(self, node_id, schedule, interference_graph):
        """为节点找到可用的时隙

        使用贪心着色算法 - 找到可用的最小时隙索引
        """
        # 获取该节点的干扰邻居
        interfering_nodes = interference_graph.get(node_id, set())

        # 找出邻居已占用的时隙
        occupied_slots = set()
        for slot, nodes in schedule.items():
            for neighbor in interfering_nodes:
                if neighbor in nodes:
                    occupied_slots.add(slot)
                    break

        # 找到可用的最小时隙
        slot = 0
        while slot in occupied_slots:
            slot += 1

        return slot

    def _optimize_slot_numbering(self, schedule):
        """优化时隙编号，确保从0开始连续编号"""
        if not schedule:
            return {}

        # 获取所有时隙编号并排序
        slot_numbers = sorted(schedule.keys())

        # 创建优化的编号映射
        slot_map = {old: new for new, old in enumerate(slot_numbers)}

        # 创建优化后的调度表
        optimized = {}
        for old_slot, nodes in schedule.items():
            new_slot = slot_map[old_slot]
            optimized[new_slot] = nodes

        return optimized

    def update_slot_schedule(self):
        """根据当前网络状态更新时隙分配"""
        # 检查是否需要更新
        if self._should_update_schedule():
            logging.info("开始更新STDMA时隙分配")

            # 保存旧的时隙分配，用于比较
            old_schedule = copy.deepcopy(self.slot_schedule)

            # 创建新的时隙分配
            new_schedule = self._create_slot_schedule()

            # 更新分配
            self.slot_schedule = new_schedule

            # 计算变化百分比
            change_percentage = self._calculate_schedule_change(old_schedule, new_schedule)
            logging.info(f"时隙分配更新完成，变化率: {change_percentage:.2f}%")

    def _should_update_schedule(self):
        """判断是否需要更新时隙分配

        根据以下条件判断:
        1. 网络拓扑变化程度
        2. 性能下降程度
        3. 时间间隔
        """
        # 实现一个简单的判断逻辑
        return False  # 默认不更新，在实际实现中可以根据条件判断

    def _calculate_schedule_change(self, old_schedule, new_schedule):
        """计算两个时隙分配表之间的变化百分比"""
        total_assignments = 0
        changed_assignments = 0

        # 创建老分配的节点到时隙的映射
        old_mapping = {}
        for slot, nodes in old_schedule.items():
            for node in nodes:
                old_mapping[node] = slot

        # 计算变化的分配数
        for slot, nodes in new_schedule.items():
            for node in nodes:
                total_assignments += 1
                if node not in old_mapping or old_mapping[node] != slot:
                    changed_assignments += 1

        # 计算变化百分比
        if total_assignments == 0:
            return 0

        return (changed_assignments / total_assignments) * 100

    def _handle_assignment_conflicts(self, conflicting_nodes):
        """处理分配冲突"""
        # 这个函数可以根据需要实现，例如根据优先级重新分配时隙
        pass

    def _estimate_slot_quality(self, node_id, slot_id):
        """估计时隙质量，用于优化分配"""
        quality = 1.0  # 基础质量

        # 如果时隙存在
        if slot_id in self.slot_schedule:
            # 检查时隙中的其他节点
            for other_node in self.slot_schedule[slot_id]:
                # 计算距离
                distance = euclidean_distance(
                    self.simulator.drones[node_id].coords,
                    self.simulator.drones[other_node].coords
                )

                # 距离越远，干扰越小，质量越高
                interference_range = self._get_interference_range()
                if distance < interference_range:
                    # 距离小于干扰范围，质量降低
                    quality *= (distance / interference_range)
                else:
                    # 距离大于干扰范围，质量不受影响
                    pass

                # 考虑链路质量
                link_quality = self.link_quality_manager.get_link_quality(node_id, other_node)
                if link_quality > 0:  # 有链路质量记录
                    quality *= link_quality

        return quality

    def _is_valid_slot_assignment(self, node_id, slot_id):
        """检查时隙分配是否有效"""
        # 如果时隙不存在，一定有效
        if slot_id not in self.slot_schedule:
            return True

        # 获取干扰图
        interference_graph = self._build_interference_graph()

        # 获取干扰节点
        interfering_nodes = interference_graph.get(node_id, set())

        # 检查时隙中是否有干扰节点
        for other_node in self.slot_schedule[slot_id]:
            if other_node in interfering_nodes:
                return False

        return True

    def _find_best_available_slot(self, node_id, max_slots_to_check=None):
        """找到最佳可用时隙

        考虑因素:
        1. 时隙冲突
        2. 时隙质量
        3. 复用机会
        """
        # 初始化最佳时隙和质量
        best_slot = None
        best_quality = -1

        # 检查时隙数量
        max_slot = max(self.slot_schedule.keys()) if self.slot_schedule else 0
        slots_to_check = max_slot + 2  # 检查额外的一个空时隙

        # 如果指定了检查上限，使用较小值
        if max_slots_to_check is not None:
            slots_to_check = min(slots_to_check, max_slots_to_check)

        # 检查每个可能的时隙
        for slot in range(slots_to_check):
            # 检查分配有效性
            if self._is_valid_slot_assignment(node_id, slot):
                # 计算时隙质量
                quality = self._estimate_slot_quality(node_id, slot)

                # 如果是当前最佳，更新
                if quality > best_quality:
                    best_slot = slot
                    best_quality = quality

        # 若未找到有效时隙，创建新时隙
        if best_slot is None:
            best_slot = max_slot + 1

        return best_slot

    def mac_send(self, packet):
        """MAC层发送函数"""
        logging.info(f"Time {self.env.now}: UAV{self.my_drone.identifier} MAC layer received {type(packet).__name__}")

        if isinstance(packet, TrafficRequirement):
            # 处理业务需求
            self._handle_traffic_requirement(packet)
            yield self.env.process(self._transmit_packet(packet))

        elif isinstance(packet, DataPacket):
            # 处理数据包发送
            self._handle_data_packet_send(packet)

            # 获取分配的时隙
            assigned_slot = self._get_assigned_slot(self.my_drone.identifier)
            if assigned_slot is None:
                logging.error(f"无人机{self.my_drone.identifier}未分配时隙")
                return

            # 等待时隙开始
            yield self.env.process(self._wait_for_slot(assigned_slot))

            # 发送数据包
            yield self.env.process(self._transmit_packet(packet))

            # 记录MAC时延
            mac_delay = (self.env.now - packet.waiting_start_time) / 1e3 if packet.waiting_start_time else 0
            self.simulator.metrics.mac_delay.append(mac_delay)
            logging.info(f"UAV{self.my_drone.identifier} MAC 时延: {mac_delay} ms")

        else:
            # 处理其他类型的包
            yield self.env.process(self._transmit_packet(packet))

    def _wait_for_slot(self, slot_id):
        """等待直到指定时隙开始"""
        current_time = self.env.now
        current_slot = (current_time // self.time_slot_duration) % self.num_slots

        if current_slot == slot_id:
            # 已经在目标时隙内，无需等待
            return

        # 计算需要等待的时间
        if slot_id > current_slot:
            # 目标时隙在当前轮
            wait_slots = slot_id - current_slot
        else:
            # 目标时隙在下一轮
            wait_slots = self.num_slots - current_slot + slot_id

        # 等待时间
        wait_time = wait_slots * self.time_slot_duration

        # 加上当前时隙内已经经过的时间
        elapsed_in_slot = current_time % self.time_slot_duration
        wait_time -= elapsed_in_slot

        # 等待
        yield self.env.timeout(wait_time)

    def _get_assigned_slot(self, node_id):
        """获取节点分配的时隙"""
        for slot, nodes in self.slot_schedule.items():
            if node_id in nodes:
                return slot
        return None

    def _handle_traffic_requirement(self, requirement):
        """处理业务需求"""
        # 记录业务需求
        req_id = f"{requirement.source_id}_{requirement.dest_id}_{self.env.now}"
        self.traffic_requirements[req_id] = requirement

        logging.info(f"处理业务需求: 从节点{requirement.source_id}到节点{requirement.dest_id}")

        # 检查是否需要更新时隙分配
        if len(self.traffic_requirements) > 5:  # 简单触发条件
            self.update_slot_schedule()

    def _handle_data_packet_send(self, packet):
        """处理数据包发送"""
        # 记录等待开始时间
        packet.waiting_start_time = self.env.now

        # 更新流统计
        flow_id = f"flow_{packet.src_drone.identifier}_{packet.dst_drone.identifier}"
        if flow_id not in self.flow_queue:
            self.flow_queue[flow_id] = []

        self.flow_queue[flow_id].append(packet)

        # 初始化流统计
        if flow_id not in self.flow_stats:
            self.flow_stats[flow_id] = {
                'sent_packets': 0,
                'received_packets': 0,
                'avg_delay': 0,
                'queue_size': 0
            }

        # 更新流统计
        self.flow_stats[flow_id]['sent_packets'] += 1
        self.flow_stats[flow_id]['queue_size'] = len(self.flow_queue[flow_id])

    def _transmit_packet(self, packet):
        """传输数据包"""
        self.current_transmission = packet

        # 单播处理
        if packet.transmission_mode == 0:
            packet.increase_ttl()
            self.phy.unicast(packet, packet.next_hop_id)
            logging.info(
                f"Time {self.env.now}: UAV{self.my_drone.identifier} transmitted {type(packet).__name__} to {packet.next_hop_id}")
            yield self.env.timeout(packet.packet_length / config.BIT_RATE * 1e6)

        # 广播处理
        elif packet.transmission_mode == 1:
            packet.increase_ttl()
            self.phy.broadcast(packet)
            yield self.env.timeout(packet.packet_length / config.BIT_RATE * 1e6)

        # 更新流队列
        if isinstance(packet, DataPacket):
            flow_id = f"flow_{packet.src_drone.identifier}_{packet.dst_drone.identifier}"
            if flow_id in self.flow_queue and packet in self.flow_queue[flow_id]:
                self.flow_queue[flow_id].remove(packet)

        self.current_transmission = None