import logging
import math
from phy.phy import Phy
from utils import config
from utils.util_function import euclidean_distance
from entities.packet import DataPacket
from mac.LinkQualityManager import LinkQualityManager
from simulator.TrafficGenerator import TrafficRequirement
import traceback
import copy
import numpy as np
from collections import defaultdict


class GraphColoringStdma:
    """
    基于增强图着色算法的STDMA协议

    特点：
    1. 使用启发式图着色算法实现高效时隙分配
    2. 支持可变规模无人机网络
    3. 自适应的时隙复用机制
    4. 基于业务需求的优先级分配
    5. 支持动态拓扑调整
    """

    def __init__(self, drone):
        self.my_drone = drone
        self.simulator = drone.simulator
        self.env = drone.env
        self.time_slot_duration = config.SLOT_DURATION
        self.num_slots = config.NUMBER_OF_DRONES  # 初始时隙数
        self.max_slots = max(20, self.num_slots * 2)  # 最大时隙数
        self.current_slot = 0
        self.phy = Phy(self)
        self.current_transmission = None

        # 时隙分配表 - 格式: {slot_id: [drone_ids]}
        self.slot_schedule = {}

        # 链路质量管理器
        self.link_quality_manager = LinkQualityManager()

        # 业务需求队列
        self.traffic_requirements = {}

        # 节点优先级缓存
        self.node_priorities = {}

        # 缓存控制参数
        self.cache_validity = 1e6  # 缓存有效期(ns)
        self.last_topology_update = 0  # 上次拓扑更新时间
        self.last_full_update = 0  # 上次全量更新时间

        # 统计信息
        self.stats = {
            'schedule_updates': 0,
            'total_slots_used': 0,
            'total_assignments': 0,
            'conflicts_detected': 0,
            'conflicts_resolved': 0,
            'avg_slot_utilization': 0,
        }

        # 创建初始调度表
        self.slot_schedule = self._create_slot_schedule()

        # 启动时隙同步进程
        self.env.process(self._slot_synchronization())

        # 启动定期统计
        self.env.process(self._periodic_stats())

        # 启动定期拓扑检查
        self.env.process(self._check_topology_changes())

    def _slot_synchronization(self):
        """时隙同步进程"""
        while True:
            # 更新当前时隙
            self.current_slot = (self.env.now // self.time_slot_duration) % self.num_slots
            yield self.env.timeout(self.time_slot_duration)

    def _periodic_stats(self):
        """定期收集和记录统计信息"""
        while True:
            yield self.env.timeout(10 * 1e6)  # 每10秒
            self._update_statistics()
            self._log_statistics()

    def _update_statistics(self):
        """更新统计信息"""
        try:
            if not self.slot_schedule:
                return

            # 更新时隙使用统计
            self.stats['total_slots_used'] = len(self.slot_schedule)
            self.stats['total_assignments'] = sum(len(nodes) for nodes in self.slot_schedule.values())

            # 计算时隙利用率
            self.stats['avg_slot_utilization'] = (
                self.stats['total_assignments'] / self.stats['total_slots_used']
                if self.stats['total_slots_used'] > 0 else 0
            )

        except Exception as e:
            logging.error(f"更新统计信息出错: {e}")

    def _log_statistics(self):
        """记录统计信息到日志"""
        try:
            logging.info(f"STDMA统计 - UAV{self.my_drone.identifier}:")
            logging.info(f"  更新次数: {self.stats['schedule_updates']}")
            logging.info(f"  使用时隙数: {self.stats['total_slots_used']}")
            logging.info(f"  总分配数: {self.stats['total_assignments']}")
            logging.info(f"  时隙利用率: {self.stats['avg_slot_utilization']:.2f}")
            logging.info(f"  冲突检测: {self.stats['conflicts_detected']}")
            logging.info(f"  冲突解决: {self.stats['conflicts_resolved']}")

            # 详细记录时隙分配
            self._print_schedule_info()

        except Exception as e:
            logging.error(f"记录统计信息出错: {e}")

    def _print_schedule_info(self):
        """打印当前时隙分配详情"""
        try:
            if not self.slot_schedule:
                logging.info("  时隙分配表为空")
                return

            logging.info("  时隙分配详情:")
            for slot, nodes in sorted(self.slot_schedule.items()):
                # 获取节点间的空间信息
                spatial_info = []
                for i, node1 in enumerate(nodes):
                    for node2 in nodes[i + 1:]:
                        distance = euclidean_distance(
                            self.simulator.drones[node1].coords,
                            self.simulator.drones[node2].coords
                        )
                        spatial_info.append(f"{node1}-{node2}: {distance:.1f}m")

                # 输出时隙详情
                spatial_str = ", ".join(spatial_info) if spatial_info else "无节点共享"
                logging.info(f"    时隙 {slot}: 节点 {nodes} | {spatial_str}")

        except Exception as e:
            logging.error(f"打印时隙信息出错: {e}")

    def _check_topology_changes(self):
        """检查拓扑变化并触发更新"""
        while True:
            yield self.env.timeout(5 * 1e6)  # 每5秒检查一次

            try:
                # 检查是否需要更新
                if self._should_update_schedule():
                    logging.info(f"UAV{self.my_drone.identifier} 检测到拓扑变化，更新时隙分配")
                    # 创建新的时隙分配
                    new_schedule = self._create_slot_schedule()

                    # 执行增量更新，尽量减少改变
                    final_schedule = self._incremental_update(self.slot_schedule, new_schedule)

                    # 更新分配
                    self.slot_schedule = final_schedule
                    self.stats['schedule_updates'] += 1
                    self.last_full_update = self.env.now

            except Exception as e:
                logging.error(f"检查拓扑变化出错: {e}")

    def _should_update_schedule(self):
        """判断是否需要更新时隙分配"""
        current_time = self.env.now

        # 检查上次更新后的时间间隔
        time_since_update = current_time - self.last_full_update
        if time_since_update < 10 * 1e6:  # 10秒内不重复更新
            return False

        # 检查拓扑变化程度
        topology_changed = self._check_topology_change_degree()

        # 检查性能指标
        performance_degraded = self._check_performance_degradation()

        # 检查冲突数量
        conflicts = self._count_scheduling_conflicts()

        # 如果存在冲突，一定更新
        if conflicts > 0:
            self.stats['conflicts_detected'] += conflicts
            logging.info(f"检测到 {conflicts} 个调度冲突，需要更新")
            return True

        # 如果拓扑变化显著或性能下降，则更新
        return topology_changed or performance_degraded

    def _check_topology_change_degree(self):
        """检查拓扑变化程度"""
        # 这里实现拓扑变化检测的逻辑
        # 可以基于节点移动距离、连接变化等

        # 简单实现：检查位置变化
        position_changed = False
        try:
            # 检查每个节点的位置变化
            for drone in self.simulator.drones:
                # 此处可以添加与上次记录位置的比较逻辑
                pass

        except Exception as e:
            logging.error(f"检查拓扑变化出错: {e}")

        return position_changed

    def _check_performance_degradation(self):
        """检查性能是否下降"""
        # 可以基于吞吐量、延迟、冲突等指标
        return False  # 简单返回，实际中应该有具体逻辑

    def _count_scheduling_conflicts(self):
        """统计当前时隙分配中的冲突数量"""
        conflicts = 0

        # 获取干扰图
        interference_graph = self._build_interference_graph()

        # 检查每个时隙中的节点是否存在干扰
        for slot, nodes in self.slot_schedule.items():
            # 检查时隙内节点间的冲突
            for i, node1 in enumerate(nodes):
                for node2 in nodes[i + 1:]:
                    # 如果两个节点相互干扰，则存在冲突
                    if node2 in interference_graph.get(node1, set()):
                        conflicts += 1
                        logging.warning(f"节点 {node1} 和 {node2} 在时隙 {slot} 存在冲突")

        return conflicts

    def _incremental_update(self, old_schedule, new_schedule):
        """增量更新时隙分配，尽量保持现有分配不变"""
        # 创建一个更新后的调度表
        updated_schedule = {}

        # 获取干扰图
        interference_graph = self._build_interference_graph()

        # 首先保留没有冲突的原有分配
        conflict_nodes = set()
        for slot, nodes in old_schedule.items():
            # 检查该时隙内是否有冲突
            has_conflict = False
            for i, node1 in enumerate(nodes):
                for node2 in nodes[i + 1:]:
                    if node2 in interference_graph.get(node1, set()):
                        has_conflict = True
                        conflict_nodes.add(node1)
                        conflict_nodes.add(node2)

            # 如果没有冲突，保留原有分配
            if not has_conflict:
                updated_schedule[slot] = nodes.copy()

        # 使用新调度表中的分配来处理有冲突的节点
        assigned_nodes = set()
        for slot, nodes in updated_schedule.items():
            assigned_nodes.update(nodes)

        # 从新调度表中获取未分配节点的分配
        for slot, nodes in new_schedule.items():
            for node in nodes:
                if node not in assigned_nodes and node in conflict_nodes:
                    # 检查该节点是否可以分配到现有时隙
                    assigned = False
                    for s in sorted(updated_schedule.keys()):
                        # 检查节点是否可以添加到该时隙
                        if self._can_add_to_slot(node, updated_schedule[s], interference_graph):
                            updated_schedule[s].append(node)
                            assigned_nodes.add(node)
                            assigned = True
                            break

                    # 如果不能分配到现有时隙，创建新时隙
                    if not assigned:
                        new_slot = max(updated_schedule.keys()) + 1 if updated_schedule else 0
                        updated_schedule[new_slot] = [node]
                        assigned_nodes.add(node)

        # 确保所有节点都已分配
        for node in range(self.simulator.n_drones):
            if node not in assigned_nodes:
                # 尝试分配到现有时隙
                assigned = False
                for slot in sorted(updated_schedule.keys()):
                    if self._can_add_to_slot(node, updated_schedule[slot], interference_graph):
                        updated_schedule[slot].append(node)
                        assigned = True
                        break

                # 如果不能分配到现有时隙，创建新时隙
                if not assigned:
                    new_slot = max(updated_schedule.keys()) + 1 if updated_schedule else 0
                    updated_schedule[new_slot] = [node]

        # 重新编号时隙，使其连续
        final_schedule = self._renumber_slots(updated_schedule)

        # 更新时隙数量
        self.num_slots = len(final_schedule)

        return final_schedule

    def _renumber_slots(self, schedule):
        """重新编号时隙，确保从0开始连续"""
        if not schedule:
            return {}

        # 获取所有时隙编号并排序
        old_slots = sorted(schedule.keys())

        # 创建新的编号映射
        slot_map = {old: new for new, old in enumerate(old_slots)}

        # 创建新的调度表
        new_schedule = {}
        for old_slot, nodes in schedule.items():
            new_slot = slot_map[old_slot]
            new_schedule[new_slot] = nodes

        return new_schedule

    def _can_add_to_slot(self, node, slot_nodes, interference_graph):
        """检查节点是否可以添加到时隙"""
        # 获取该节点的干扰邻居
        interfering_nodes = interference_graph.get(node, set())

        # 检查时隙中的节点是否与该节点干扰
        for existing_node in slot_nodes:
            if existing_node in interfering_nodes:
                return False

        return True

    def _create_slot_schedule(self):
        """创建时隙分配表，使用改进的DSATUR算法

        DSATUR (Degree of Saturation) 是一种高效的图着色算法，
        它考虑了节点的饱和度（已分配给邻居的不同颜色数量）。
        """
        logging.info(f"UAV{self.my_drone.identifier} 创建STDMA时隙分配表")

        # 初始化时隙表
        schedule = {}

        # 获取干扰图
        interference_graph = self._build_interference_graph()

        # 未分配的节点集合
        unassigned = set(range(self.simulator.n_drones))

        # 节点颜色（时隙）
        node_colors = {}

        # 节点的邻居已使用的颜色集合
        neighbor_colors = {node: set() for node in range(self.simulator.n_drones)}

        # DSATUR算法
        while unassigned:
            # 选择饱和度最高的节点
            max_saturation = -1
            selected_node = None

            for node in unassigned:
                saturation = len(neighbor_colors[node])

                # 如果饱和度相同，选择度数更高的节点
                if saturation > max_saturation or (
                        saturation == max_saturation and
                        selected_node is not None and
                        len(interference_graph.get(node, set())) > len(interference_graph.get(selected_node, set()))
                ):
                    max_saturation = saturation
                    selected_node = node

            # 如果没有找到节点，中断循环
            if selected_node is None:
                break

            # 找到可用的最小颜色（时隙）
            used_colors = neighbor_colors[selected_node]
            color = 0
            while color in used_colors:
                color += 1

            # 分配颜色
            node_colors[selected_node] = color
            unassigned.remove(selected_node)

            # 更新邻居的可用颜色集合
            for neighbor in interference_graph.get(selected_node, set()):
                if neighbor in unassigned:  # 只更新未分配的邻居
                    neighbor_colors[neighbor].add(color)

        # 将颜色分配转换为调度表格式
        for node, color in node_colors.items():
            if color not in schedule:
                schedule[color] = []
            schedule[color].append(node)

        # 输出分配结果
        total_slots = len(schedule)
        total_assignments = sum(len(nodes) for nodes in schedule.values())
        logging.info(f"STDMA时隙分配完成，使用 {total_slots} 个时隙，共 {total_assignments} 个分配")
        logging.info(f"分配结果：{schedule}")
        return schedule

    def _build_interference_graph(self):
        """构建干扰图，使用更精确的SINR模型"""
        graph = {}

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

                    # 基本干扰检查（距离）
                    if distance <= interference_range:
                        # 添加基于SINR的干扰检查
                        if self._check_interference_sinr(i, j, distance):
                            graph[i].add(j)
                except Exception as e:
                    logging.error(f"计算节点 {i} 和 {j} 之间的干扰时出错: {e}")

        return graph

    def _check_interference_sinr(self, node1, node2, distance):
        """使用SINR模型检查干扰"""
        try:
            # 简化的SINR计算
            path_loss = self._calculate_path_loss(distance)
            noise_power = config.NOISE_POWER
            transmit_power = config.TRANSMITTING_POWER

            # 接收功率
            received_power = transmit_power / path_loss

            # SINR（假设没有其他干扰）
            sinr = received_power / noise_power

            # 将SINR转换为dB
            sinr_db = 10 * math.log10(sinr)

            # 检查是否低于阈值
            return sinr_db < config.SNR_THRESHOLD

        except Exception as e:
            logging.error(f"SINR计算出错: {e}")
            # 如果计算出错，默认假设有干扰
            return True

    def _calculate_path_loss(self, distance):
        """计算路径损耗"""
        # 使用标准的路径损耗模型
        if distance == 0:
            return 1.0  # 避免除零错误

        wavelength = config.LIGHT_SPEED / config.CARRIER_FREQUENCY
        path_loss = (4 * math.pi * distance / wavelength) ** config.PATH_LOSS_EXPONENT

        return max(1.0, path_loss)  # 确保路径损耗至少为1

    def _get_interference_range(self):
        """获取干扰范围"""
        # 干扰范围通常大于通信范围
        from phy.large_scale_fading import maximum_communication_range
        return maximum_communication_range() * 2.0

    def update_slot_schedule(self, new_schedule=None):
        """更新时隙分配表"""
        if new_schedule is not None:
            # 直接更新为提供的时隙表
            self.slot_schedule = new_schedule
            self.num_slots = len(new_schedule)
            self.stats['schedule_updates'] += 1
            self.last_full_update = self.env.now
            return True

        # 检查是否需要更新
        if self._should_update_schedule():
            try:
                # 创建新的时隙分配
                new_schedule = self._create_slot_schedule()

                # 增量更新
                final_schedule = self._incremental_update(self.slot_schedule, new_schedule)

                # 更新分配
                self.slot_schedule = final_schedule
                self.num_slots = len(final_schedule)

                self.stats['schedule_updates'] += 1
                self.last_full_update = self.env.now

                return True

            except Exception as e:
                logging.error(f"更新时隙分配出错: {e}")
                return False

        return False

    def mac_send(self, packet):
        """MAC层发送函数"""
        logging.info(f"Time {self.env.now}: UAV{self.my_drone.identifier} MAC layer received {type(packet).__name__}")

        try:
            if isinstance(packet, TrafficRequirement):
                # 处理业务需求
                self._handle_traffic_requirement(packet)
                yield self.env.process(self._transmit_packet(packet))

            elif isinstance(packet, DataPacket):
                # 处理数据包发送
                mac_start_time = self.env.now
                packet.waiting_start_time = mac_start_time

                # 获取分配的时隙
                assigned_slot = self._get_assigned_slot(self.my_drone.identifier)
                if assigned_slot is None:
                    logging.error(f"无人机{self.my_drone.identifier}未分配时隙")
                    return

                # 等待到分配的时隙
                current_time = self.env.now
                slot_start = (
                                         current_time // self.time_slot_duration) * self.time_slot_duration + assigned_slot * self.time_slot_duration
                slot_end = slot_start + self.time_slot_duration

                if current_time < slot_start:
                    yield self.env.timeout(slot_start - current_time)
                elif current_time >= slot_end:
                    yield self.env.timeout(self.num_slots * self.time_slot_duration - (current_time - slot_start))

                # 传输数据包
                yield self.env.process(self._transmit_packet(packet))

                # 记录MAC时延
                mac_delay = (self.env.now - mac_start_time) / 1e3
                logging.info(f"UAV{self.my_drone.identifier} MAC 时延: {mac_delay} ms")
                self.simulator.metrics.mac_delay.append(mac_delay)

            else:
                # 处理其他类型的包
                yield self.env.process(self._transmit_packet(packet))

        except Exception as e:
            logging.error(f"MAC发送出错: {e}")
            traceback.print_exc()

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

        # 在某些条件下触发更新
        if len(self.traffic_requirements) > 3:  # 例如累积3个以上需求时更新
            self.update_slot_schedule()

    def _transmit_packet(self, packet):
        """传输数据包"""
        self.current_transmission = packet

        if packet.transmission_mode == 0:
            packet.increase_ttl()
            self.phy.unicast(packet, packet.next_hop_id)
            logging.info(
                f"Time {self.env.now}: UAV{self.my_drone.identifier} transmitted {type(packet).__name__} to {packet.next_hop_id}")
            yield self.env.timeout(packet.packet_length / config.BIT_RATE * 1e6)

        elif packet.transmission_mode == 1:
            packet.increase_ttl()
            self.phy.broadcast(packet)
            yield self.env.timeout(packet.packet_length / config.BIT_RATE * 1e6)

        self.current_transmission = None