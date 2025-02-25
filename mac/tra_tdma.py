import logging
import simpy
from phy.phy import Phy
from utils import config
from entities.packet import DataPacket
from simulator.TrafficGenerator import TrafficRequirement


class Tra_Tdma:
    def __init__(self, drone):
        self.my_drone = drone
        self.simulator = drone.simulator
        self.env = drone.env
        self.time_slot_duration = config.SLOT_DURATION
        self.num_slots = config.NUMBER_OF_DRONES
        self.slot_schedule = self._create_initial_slot_schedule()
        self.current_slot = 0
        self.phy = Phy(self)
        self.current_transmission = None

    def _validate_slot_schedule(self, new_schedule):
        """验证新的时隙分配方案是否有效"""
        try:
            # 检查时隙数量是否正确
            if len(new_schedule) != self.num_slots:
                return False

            # 检查时隙分配是否完整且无重复
            assigned_slots = set()
            for node_id, slot in new_schedule.items():
                if slot in assigned_slots or slot >= self.num_slots:
                    return False
                assigned_slots.add(slot)

            # 确保每个节点都有时隙分配
            if len(assigned_slots) != self.num_slots:
                return False

            return True

        except Exception as e:
            logging.error(f"Slot schedule validation error: {str(e)}")
            return False

    def _apply_new_schedule(self, new_schedule):
        """应用新的时隙分配方案"""
        # 检查当前是否正在传输
        if self.current_transmission:
            # 等待当前传输完成
            current_slot_end = ((self.env.now // self.time_slot_duration) + 1) * self.time_slot_duration
            wait_time = current_slot_end - self.env.now
            if wait_time > 0:
                yield self.env.timeout(wait_time)

        # 更新时隙分配
        self.slot_schedule = new_schedule
        self.optimized_schedule = new_schedule

        # 重新同步时隙
        self.current_slot = (self.env.now // self.time_slot_duration) % self.num_slots

        # 更新本地调度参数
        self._update_scheduling_parameters()

    def _update_scheduling_parameters(self):
        """更新与时隙调度相关的本地参数"""
        # 获取本节点的新时隙
        my_slot = self.slot_schedule[self.my_drone.identifier]

        # 更新传输时间和等待时间的计算参数
        current_frame = self.env.now // (self.time_slot_duration * self.num_slots)
        self.next_transmission_time = (current_frame * self.num_slots + my_slot) * self.time_slot_duration

        if self.next_transmission_time < self.env.now:
            self.next_transmission_time += (self.num_slots * self.time_slot_duration)

    def _send_schedule_ack(self, source_id):
        """发送时隙分配确认消息"""

        class ScheduleAckMessage:
            def __init__(self, source_id, target_id, creation_time, simulator):
                self.source_id = source_id
                self.target_id = target_id
                self.creation_time = creation_time
                self.packet_id = f"slot_ack_{creation_time}"
                self.packet_length = 500  # 较小的确认消息
                self.deadline = config.PACKET_LIFETIME
                self.number_retransmission_attempt = {i: 0 for i in range(config.NUMBER_OF_DRONES)}
                self.transmission_mode = 0  # 单播模式
                self.simulator = simulator
                self.next_hop_id = target_id
                self.waiting_start_time = creation_time
                self.msg_type = 'slot_schedule_ack'

            def increase_ttl(self):
                pass

            def get_current_ttl(self):
                return 0

        # 创建确认消息
        ack_msg = ScheduleAckMessage(
            source_id=self.my_drone.identifier,
            target_id=source_id,
            creation_time=self.env.now,
            simulator=self.simulator
        )

        # 发送确认消息
        if self.my_drone.transmitting_queue.qsize() < self.my_drone.max_queue_size:
            self.my_drone.transmitting_queue.put(ack_msg)
            logging.info(f"Drone {self.my_drone.identifier} sent schedule ACK to Drone {source_id}")
        else:
            logging.warning(f"Queue full, schedule ACK to Drone {source_id} dropped")
        # 新增:存储优化后的时隙分配
        self.optimized_schedule = {}

        self.env.process(self._slot_synchronization())

    def _create_initial_slot_schedule(self):
        """创建初始时隙分配表"""
        schedule = {}
        for i in range(self.num_slots):
            schedule[i] = i
        return schedule

    def handle_control_packet(self, packet, sender):
        """处理控制包，包括业务需求消息和时隙分配消息"""
        if isinstance(packet, TrafficRequirement):
            logging.info(f"Time {self.env.now}: Drone {self.my_drone.identifier} "
                         f"processing traffic requirement for path {packet.routing_path}")

            if packet.routing_path:
                self._optimize_slot_schedule(packet.routing_path)
                # 将优化后的时隙分配广播给其他节点
                # self._broadcast_new_schedule()

        elif hasattr(packet, 'msg_type') and packet.msg_type == 'slot_schedule':
            logging.info(f"Time {self.env.now}: Drone {self.my_drone.identifier} "
                         f"received new slot schedule from Drone {sender}")

            # 验证发送者的合法性和消息的新鲜度
            if packet.source_id != self.my_drone.identifier:  # 不处理自己发送的消息
                # 验证新的时隙分配方案
                is_valid = self._validate_slot_schedule(packet.schedule)
                if is_valid:
                    # 备份当前时隙以便回滚
                    old_schedule = self.slot_schedule.copy()
                    try:
                        # 应用新的时隙分配
                        self._apply_new_schedule(packet.schedule)

                        # 发送确认消息给源节点
                        self._send_schedule_ack(packet.source_id)

                        logging.info(
                            f"Drone {self.my_drone.identifier} successfully updated slot schedule: {self.slot_schedule}")

                        # 继续转发时隙分配消息，确保网络同步
                        if packet.number_retransmission_attempt[self.my_drone.identifier] == 0:
                            packet.number_retransmission_attempt[self.my_drone.identifier] += 1
                            if self.my_drone.transmitting_queue.qsize() < self.my_drone.max_queue_size:
                                self.my_drone.transmitting_queue.put(packet)
                                logging.info(f"Drone {self.my_drone.identifier} forwarding slot schedule message")

                    except Exception as e:
                        # 如果更新过程出错，回滚到旧的时隙分配
                        self.slot_schedule = old_schedule
                        self.optimized_schedule = old_schedule
                        logging.error(f"Drone {self.my_drone.identifier} failed to update slot schedule: {str(e)}")
                else:
                    logging.warning(f"Drone {self.my_drone.identifier} received invalid slot schedule from {sender}")

    def _optimize_slot_schedule(self, path):
        """基于业务路径优化时隙分配"""
        if not path:
            return

        logging.info(f"Optimizing slot schedule for path: {path}")


        # 构建优化的时隙分配
        new_schedule = {}
        assigned_slots = set()
        current_slot = 0

        # 首先为路径上的节点分配连续时隙
        for node_id in path:
            if current_slot < self.num_slots and current_slot not in assigned_slots:
                new_schedule[node_id] = current_slot
                assigned_slots.add(current_slot)
                current_slot += 1

        # 为其他节点分配剩余时隙
        for i in range(self.num_slots):
            if i not in new_schedule:
                while current_slot in assigned_slots:
                    current_slot += 1
                if current_slot < self.num_slots:
                    new_schedule[i] = current_slot
                    assigned_slots.add(current_slot)
                    current_slot += 1

        # 更新时隙分配
        self.optimized_schedule = new_schedule
        self.slot_schedule = new_schedule
        for node in self.simulator.drones:
            node.mac_protocol.slot_schedule = self.slot_schedule
        logging.info(f"New slot schedule: {self.slot_schedule}")

        # 计算理论延迟改善
        self._calculate_delay_improvement(path)

    def _calculate_delay_improvement(self, path):
        """计算优化后的理论延迟改善"""
        original_delay = len(path) * self.time_slot_duration

        # 计算优化后的延迟
        optimized_delay = 0
        for i in range(len(path) - 1):
            current_slot = self.slot_schedule[path[i]]
            next_slot = self.slot_schedule[path[i + 1]]
            slot_diff = (next_slot - current_slot) if next_slot > current_slot else (
                        self.num_slots - current_slot + next_slot)
            optimized_delay += slot_diff * self.time_slot_duration

        improvement = ((original_delay - optimized_delay) / original_delay) * 100
        logging.info(f"Theoretical delay improvement: {improvement:.2f}%")
        return improvement

    def _broadcast_new_schedule(self):
        """广播新的时隙分配给网络中的其他节点"""

        class SlotScheduleMessage:
            def __init__(self, source_id, schedule, creation_time, simulator):
                self.source_id = source_id
                self.schedule = schedule
                self.creation_time = creation_time
                self.packet_id = f"slot_schedule_{creation_time}"
                self.packet_length = 1000  # 固定长度
                self.deadline = config.PACKET_LIFETIME
                self.number_retransmission_attempt = {i: 0 for i in range(config.NUMBER_OF_DRONES)}
                self.transmission_mode = 1  # 广播模式
                self.simulator = simulator
                self.next_hop_id = None
                self.waiting_start_time = creation_time
                self.msg_type = 'slot_schedule'

            def increase_ttl(self):
                pass

            def get_current_ttl(self):
                return 0

        # 创建时隙分配消息
        schedule_msg = SlotScheduleMessage(
            source_id=self.my_drone.identifier,
            schedule=self.slot_schedule,
            creation_time=self.env.now,
            simulator=self.simulator
        )

        logging.info(f"Drone {self.my_drone.identifier} broadcasting new slot schedule at {self.env.now}")

        # 将消息加入发送队列
        if self.my_drone.transmitting_queue.qsize() < self.my_drone.max_queue_size:
            self.my_drone.transmitting_queue.put(schedule_msg)
            logging.info(f"Slot schedule message queued for broadcast from Drone {self.my_drone.identifier}")
        else:
            logging.warning(f"Queue full, slot schedule message dropped from Drone {self.my_drone.identifier}")

        return schedule_msg

    def _slot_synchronization(self):
        """时隙同步进程"""
        while True:
            self.current_slot = (self.env.now // self.time_slot_duration) % self.num_slots
            yield self.env.timeout(self.time_slot_duration)

    def mac_send(self, packet):
        """MAC层发送函数"""
        self.handle_control_packet(packet, self.my_drone.identifier)
        if isinstance(packet, DataPacket):
            self.current_transmission = packet
            mac_start_time = self.env.now

            # 获取当前节点的时隙
            assigned_slot = self.slot_schedule.get(self.my_drone.identifier,
                                                   self.my_drone.identifier % self.num_slots)
            current_time = self.env.now

            # 计算下一个可用时隙的开始时间
            current_frame = current_time // (self.time_slot_duration * self.num_slots)
            slot_start_time = (current_frame * self.num_slots + assigned_slot) * self.time_slot_duration

            if slot_start_time < current_time:
                slot_start_time += (self.num_slots * self.time_slot_duration)

            # 等待直到自己的时隙
            wait_time = slot_start_time - current_time
            if wait_time > 0:
                yield self.env.timeout(wait_time)

            # 发送数据包
            yield self.env.process(self._transmit_packet(packet))

            # 计算MAC延迟并记录
            mac_delay = self.env.now - mac_start_time
            self.simulator.metrics.mac_delay.append(mac_delay / 1e3)
            logging.info(f"MAC delay for packet {packet.packet_id}: {mac_delay} us")

    def _transmit_packet(self, packet):
        """执行实际的数据包传输"""
        transmission_mode = packet.transmission_mode

        if transmission_mode == 0:  # 单播
            next_hop_id = packet.next_hop_id
            packet.increase_ttl()
            logging.info(f"Sending unicast packet from UAV {self.my_drone.identifier} "
                         f"to UAV {next_hop_id}")
            self.phy.unicast(packet, next_hop_id)
            yield self.env.timeout(packet.packet_length / config.BIT_RATE * 1e6)

        elif transmission_mode == 1:  # 广播
            packet.increase_ttl()
            logging.info(f"Broadcasting packet from UAV {self.my_drone.identifier}")
            self.phy.broadcast(packet)
            yield self.env.timeout(packet.packet_length / config.BIT_RATE * 1e6)

        self.current_transmission = None