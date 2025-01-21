import logging
import simpy
import math
from phy.phy import Phy
from utils import config
from utils.util_function import euclidean_distance
from topology.virtual_force.vf_packet import VfPacket
from entities.packet import DataPacket
from mac.LinkQualityManager import LinkQualityManager
from mac.LoadBalancer import LoadBalancer
class Stdma:
    def __init__(self, drone):
        self.my_drone = drone
        self.simulator = drone.simulator
        self.env = drone.env
        self.time_slot_duration = config.SLOT_DURATION
        self.num_slots = config.NUMBER_OF_DRONES
        self.current_slot = 0
        self.phy = Phy(self)
        self.current_transmission = None
        self.slot_schedule = None
        # 链路质量管理
        self.link_quality_manager = LinkQualityManager()

        # 添加负载均衡器
        self.load_balancer = LoadBalancer()

        # 添加负载监控进程
        self.env.process(self._monitor_load())


        # 数据流管理
        self.flow_queue = {}  # 存储数据流
        self.flow_stats = {}  # 流量统计

        # 启动进程
        self.env.process(self._slot_synchronization())
        self.env.process(self._delayed_schedule_creation())
        self.env.process(self._monitor_flows())

    def _slot_synchronization(self):
        while True:
            self.current_slot = (self.env.now // self.time_slot_duration) % self.num_slots
            yield self.env.timeout(self.time_slot_duration)

    def _monitor_load(self):
        """
        监控网络负载情况并触发必要的调整
        """
        while True:
            yield self.env.timeout(1000)  # 每1ms检查一次

            # 更新每个节点的统计信息
            for drone_id in range(config.NUMBER_OF_DRONES):
                queue_length = len(self.flow_queue.get(f"flow_{drone_id}", []))
                throughput = self.flow_stats.get(f"flow_{drone_id}", {}).get('throughput', 0)
                delay = self.flow_stats.get(f"flow_{drone_id}", {}).get('avg_delay', 0)

                self.load_balancer.update_node_stats(drone_id, queue_length, throughput, delay)

            # 检查是否需要重新分配时隙
            high_load_nodes = self.load_balancer.get_high_load_nodes()
            if high_load_nodes:
                self._adjust_slot_allocation_for_load(high_load_nodes)

    def _adjust_slot_allocation_for_load(self, high_load_nodes):
        """
        为高负载节点调整时隙分配
        """
        current_distribution = self.load_balancer.get_load_distribution()

        # 记录调整前的状态
        logging.info("\n负载均衡调整开始:")
        logging.info("-" * 50)
        logging.info("当前负载分布:")
        for node_id, stats in current_distribution.items():
            logging.info(f"节点 {node_id}: 负载分数={stats['load_score']:.2f}, "
                         f"队列长度={stats['queue_length']}")

        # 尝试为高负载节点分配额外时隙
        schedule = self.slot_schedule.copy()
        for node_id in high_load_nodes:
            current_slots = [slot for slot, nodes in schedule.items()
                             if node_id in nodes]

            # 如果节点当前时隙数低于平均值，考虑分配额外时隙
            avg_slots_per_node = sum(len(nodes) for nodes in schedule.values()) / config.NUMBER_OF_DRONES
            if len(current_slots) < avg_slots_per_node:
                self._allocate_extra_slot(node_id, schedule)

        # 更新时隙分配
        self.slot_schedule = schedule

        # 记录调整结果
        logging.info("\n调整后的时隙分配:")
        self._print_schedule_info(schedule)

    def _allocate_extra_slot(self, node_id, schedule):
        """
        为指定节点分配额外的时隙
        """
        from phy.large_scale_fading import maximum_communication_range
        interference_range = maximum_communication_range() * 1.5

        # 寻找可以容纳该节点的时隙
        for slot in range(self.num_slots):
            if slot not in schedule:
                schedule[slot] = [node_id]
                return True

            # 检查是否可以加入现有时隙
            can_add = True
            for existing_node in schedule[slot]:
                dist = euclidean_distance(
                    self.simulator.drones[node_id].coords,
                    self.simulator.drones[existing_node].coords
                )
                if dist < interference_range:
                    can_add = False
                    break

            if can_add and len(schedule[slot]) < 4:  # 保持每个时隙最多3个节点
                schedule[slot].append(node_id)
                return True

        # 如果没有找到合适的时隙，创建新的时隙
        new_slot = self.num_slots
        schedule[new_slot] = [node_id]
        self.num_slots += 1
        return True

    def _create_slot_schedule(self):
        """改进的时隙分配算法，确保合理的时隙数量"""
        schedule = {}
        from phy.large_scale_fading import maximum_communication_range
        interference_range = maximum_communication_range() * 1.5

        # 初始化时隙数为节点总数的2/3（可以根据实际情况调整）
        self.num_slots = math.ceil(config.NUMBER_OF_DRONES * 2 / 3)
        min_slots = math.ceil(config.NUMBER_OF_DRONES / 3)  # 最少需要的时隙数
        max_slots = config.NUMBER_OF_DRONES  # 最大时隙数

        # 计算节点兼容性矩阵
        compatibility_matrix = self._calculate_compatibility_matrix(interference_range)

        best_schedule = None
        best_metric = float('inf')  # 用于评估分配方案的优劣

        # 尝试不同的时隙数，找到最优解
        for num_slots in range(min_slots, max_slots + 1):
            current_schedule = {}
            unassigned = set(range(config.NUMBER_OF_DRONES))

            # 为每个时隙分配节点
            for slot in range(num_slots):
                current_schedule[slot] = []
                candidates = list(unassigned)

                # 根据干扰关系选择合适的节点组合
                candidates.sort(key=lambda x: sum(compatibility_matrix[x][y]
                                                  for y in unassigned), reverse=True)

                for drone_id in candidates[:]:
                    # 检查是否可以加入当前时隙
                    if self._can_add_to_slot(drone_id, current_schedule[slot],
                                             compatibility_matrix):
                        current_schedule[slot].append(drone_id)
                        unassigned.remove(drone_id)

            # 如果所有节点都已分配且评估指标更好，更新最优解
            if not unassigned:
                metric = self._evaluate_schedule(current_schedule, compatibility_matrix)
                if metric < best_metric:
                    best_metric = metric
                    best_schedule = current_schedule
                    self.num_slots = num_slots

        if best_schedule:
            schedule = best_schedule
            logging.info(f"找到最优时隙分配方案，使用 {self.num_slots} 个时隙")
        else:
            logging.warning("未找到有效的时隙分配方案，使用默认分配")
            schedule = self._create_default_schedule()

        self._print_schedule_info(schedule)
        return schedule

    def _calculate_compatibility_matrix(self, interference_range):
        """计算节点兼容性矩阵"""
        n_drones = config.NUMBER_OF_DRONES
        matrix = [[False] * n_drones for _ in range(n_drones)]

        for i in range(n_drones):
            for j in range(n_drones):
                if i != j:
                    drone1 = self.simulator.drones[i]
                    drone2 = self.simulator.drones[j]

                    # 结合距离和链路质量判断兼容性
                    dist = euclidean_distance(drone1.coords, drone2.coords)
                    link_quality = self.link_quality_manager.get_link_quality(i, j)

                    matrix[i][j] = (dist >= interference_range and
                                    (link_quality == -1 or link_quality >= 0.7))

        return matrix
    def _can_add_to_slot(self, drone_id, slot_nodes, compatibility_matrix):
        """检查节点是否可以加入当前时隙"""
        # 检查与时隙中所有已有节点的兼容性
        return all(compatibility_matrix[drone_id][assigned_id]
                   for assigned_id in slot_nodes)

    def _evaluate_schedule(self, schedule, compatibility_matrix):
        """评估时隙分配方案的质量"""
        # 评估指标：时隙数 + 平均每个时隙的节点干扰程度
        interference_score = 0
        total_pairs = 0

        for slot_nodes in schedule.values():
            if len(slot_nodes) > 1:
                for i in range(len(slot_nodes)):
                    for j in range(i + 1, len(slot_nodes)):
                        if not compatibility_matrix[slot_nodes[i]][slot_nodes[j]]:
                            interference_score += 1
                        total_pairs += 1

        avg_interference = interference_score / total_pairs if total_pairs > 0 else 0
        return len(schedule) + avg_interference * 5  # 权重可调

    def _print_schedule_info(self, schedule):
        """打印详细的时隙分配信息"""
        info = "\nSTDMA时隙分配详情:\n" + "-" * 50 + "\n"

        for slot, nodes in schedule.items():
            info += f"时隙 {slot}:\n"
            info += f"  节点: {', '.join(f'UAV-{n}' for n in nodes)}\n"

            # 打印该时隙内节点间的链路质量
            if len(nodes) > 1:
                info += "  节点间链路质量:\n"
                for i, node1 in enumerate(nodes):
                    for node2 in nodes[i + 1:]:
                        quality = self.link_quality_manager.get_link_quality(node1, node2)
                        info += f"    UAV-{node1} <-> UAV-{node2}: {quality:.2f}\n"

        info += "-" * 50
        logging.info(info)


    def _delayed_schedule_creation(self):
        yield self.env.timeout(1)
        self.slot_schedule = self._create_slot_schedule()

    def _monitor_flows(self):
        while True:
            yield self.env.timeout(5000)
            self._update_flow_stats()
            self._adjust_slots()

    def _update_flow_stats(self):
        for flow_id, stats in self.flow_stats.items():
            queue = self.flow_queue.get(flow_id, [])
            if queue:
                stats.update({
                    'queue_size': len(queue),
                    'avg_delay': sum(self.env.now - p.creation_time for p in queue) / len(queue),
                    'throughput': stats['sent_packets'] / (self.env.now / 1e6)
                })

    def _adjust_slots(self):
        # 检查是否需要重新分配
        needs_reallocation = any(
            stats['avg_delay'] > 2000 or stats['throughput'] < 1
            for stats in self.flow_stats.values()
        )
        if needs_reallocation:
            self.slot_schedule = self._create_slot_schedule()

    # def _create_slot_schedule(self):
    #     schedule = {}
    #     from phy.large_scale_fading import maximum_communication_range
    #     interference_range = maximum_communication_range() * 1.2
    #
    #     unassigned = set(range(config.NUMBER_OF_DRONES))
    #     slot = 0
    #
    #     while unassigned:
    #         if slot >= self.num_slots:
    #             self.num_slots += 1
    #
    #         schedule[slot] = []
    #         remaining = list(unassigned)
    #
    #         for drone_id in remaining:
    #             if all(euclidean_distance(
    #                     self.simulator.drones[drone_id].coords,
    #                     self.simulator.drones[assigned_id].coords
    #             ) >= interference_range
    #                    for assigned_id in schedule[slot]):
    #                 schedule[slot].append(drone_id)
    #                 unassigned.remove(drone_id)
    #         slot += 1
    #
    #     logging.info(f"STDMA调度表: {schedule}")
    #     return schedule

    def mac_send(self, packet):
        if not self.slot_schedule:
            logging.error("时隙表未创建")
            return

        # 控制包直接发送
        if isinstance(packet, VfPacket):
            yield self.env.process(self._transmit_packet(packet))
            return

        # 数据包流管理
        if isinstance(packet, DataPacket):
            flow_id = f"flow_{packet.src_drone.identifier}_{packet.dst_drone.identifier}"
            self.flow_queue.setdefault(flow_id, []).append(packet)
            self.flow_stats.setdefault(flow_id, {
                'sent_packets': 0,
                'avg_delay': 0,
                'throughput': 0,
                'queue_size': 0
            })

            mac_start_time = self.env.now
            assigned_slot = next((slot for slot, drones in self.slot_schedule.items()
                                  if self.my_drone.identifier in drones), None)

            if assigned_slot is None:
                logging.error(f"无人机{self.my_drone.identifier}未分配时隙")
                return

            current_time = self.env.now
            slot_start = (
                                     current_time // self.time_slot_duration) * self.time_slot_duration + assigned_slot * self.time_slot_duration
            slot_end = slot_start + self.time_slot_duration

            if current_time < slot_start:
                yield self.env.timeout(slot_start - current_time)
            elif current_time >= slot_end:
                yield self.env.timeout(self.num_slots * self.time_slot_duration - (current_time - slot_start))

            yield self.env.process(self._transmit_packet(packet))
            self.simulator.metrics.mac_delay.append(self.env.now - mac_start_time)

    def _transmit_packet(self, packet):
        self.current_transmission = packet

        if packet.transmission_mode == 0:
            packet.increase_ttl()
            self.phy.unicast(packet, packet.next_hop_id)
            yield self.env.timeout(packet.packet_length / config.BIT_RATE * 1e6)

        elif packet.transmission_mode == 1:
            packet.increase_ttl()
            self.phy.broadcast(packet)
            yield self.env.timeout(packet.packet_length / config.BIT_RATE * 1e6)

        if isinstance(packet, DataPacket):
            flow_id = f"flow_{packet.src_drone.identifier}_{packet.dst_drone.identifier}"
            self.flow_stats[flow_id]['sent_packets'] += 1
            self.flow_queue[flow_id].remove(packet)

        self.current_transmission = None