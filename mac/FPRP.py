import random
import logging
import simpy
from phy.phy import Phy
from utils import config
from utils.util_function import euclidean_distance
from entities.packet import DataPacket


class Fprp:
    """
    Five-Phase Reservation Protocol (FPRP) Implementation
    """

    def __init__(self, drone):
        self.my_drone = drone
        self.simulator = drone.simulator
        self.env = drone.env
        self.phy = Phy(self)

        # FPRP parameters
        self.frame_length = config.SLOT_DURATION * config.NUMBER_OF_DRONES
        self.num_slots = config.NUMBER_OF_DRONES
        self.time_slot_duration = config.SLOT_DURATION
        self.reservation_cycles = 3  # 每帧开始前的预约循环次数

        # Phase durations (微秒)
        self.phase_duration = {
            'request': 100,
            'collision': 100,
            'confirm': 100,
            'ack': 100,
            'final': 100
        }

        # Protocol states
        self.slot_schedule = {}  # 时隙分配表
        self.reserved_slots = set()  # 已预约的时隙
        self.current_slot = 0
        self.current_transmission = None

        # Reservation state
        self.requesting_slot = None  # 正在请求的时隙
        self.request_probability = 0.8  # 发送预约请求的概率
        self.neighbors = set()  # 邻居节点集合

        # Start processes
        self.env.process(self._slot_synchronization())
        self.env.process(self._reservation_process())
        self.env.process(self._update_neighbors())

    def _update_neighbors(self):
        """更新邻居节点信息"""
        while True:
            self.neighbors.clear()
            for drone in self.simulator.drones:
                if drone.identifier != self.my_drone.identifier:
                    distance = euclidean_distance(self.my_drone.coords, drone.coords)
                    if distance <= config.SENSING_RANGE:
                        self.neighbors.add(drone.identifier)
            yield self.env.timeout(1e6)  # 每秒更新一次

    def _slot_synchronization(self):
        """时隙同步进程"""
        while True:
            self.current_slot = (self.env.now // self.time_slot_duration) % self.num_slots
            yield self.env.timeout(self.time_slot_duration)

    def _reservation_process(self):
        """预约过程"""
        while True:
            # 开始新的预约帧
            for cycle in range(self.reservation_cycles):
                # 如果没有预约时隙，尝试预约
                if self.my_drone.identifier not in self.slot_schedule:
                    desired_slot = random.randint(0, self.num_slots - 1)
                    if desired_slot not in self.reserved_slots:
                        yield self.env.process(self._five_phase_reservation(desired_slot, cycle))

            # 等待下一个预约周期
            yield self.env.timeout(self.frame_length)

    def _five_phase_reservation(self, desired_slot, cycle):
        """执行五阶段预约过程"""
        self.requesting_slot = desired_slot

        # Phase 1: Reservation Request
        if random.random() < self.request_probability:
            yield self.env.timeout(self.phase_duration['request'])
            self._send_reservation_request(desired_slot)

        # Phase 2: Collision Report
        yield self.env.timeout(self.phase_duration['collision'])
        collision = self._check_collision(desired_slot)
        if collision:
            self._send_collision_report()
            self.requesting_slot = None
            return

        # Phase 3: Reservation Confirmation
        yield self.env.timeout(self.phase_duration['confirm'])
        self._send_reservation_confirm(desired_slot)

        # Phase 4: Reservation Acknowledgment
        yield self.env.timeout(self.phase_duration['ack'])
        success = self._wait_for_ack()
        if not success:
            self.requesting_slot = None
            return

        # Phase 5: Final Confirmation
        yield self.env.timeout(self.phase_duration['final'])
        self._send_final_confirm(desired_slot)

        # 预约成功
        self.slot_schedule[self.my_drone.identifier] = desired_slot
        self.reserved_slots.add(desired_slot)
        self.requesting_slot = None

        logging.info(f"UAV {self.my_drone.identifier} successfully reserved slot {desired_slot}")

    def _send_reservation_request(self, slot):
        """发送预约请求"""
        message = {
            'type': 'request',
            'drone_id': self.my_drone.identifier,
            'slot': slot
        }
        self.phy.broadcast(self._create_control_packet(message))

    def _check_collision(self, slot):
        """检查预约冲突"""
        for neighbor in self.neighbors:
            if neighbor in self.slot_schedule and self.slot_schedule[neighbor] == slot:
                return True
        return False

    def _send_collision_report(self):
        """发送冲突报告"""
        message = {
            'type': 'collision',
            'drone_id': self.my_drone.identifier
        }
        self.phy.broadcast(self._create_control_packet(message))

    def _send_reservation_confirm(self, slot):
        """发送预约确认"""
        message = {
            'type': 'confirm',
            'drone_id': self.my_drone.identifier,
            'slot': slot
        }
        self.phy.broadcast(self._create_control_packet(message))

    def _wait_for_ack(self):
        """等待预约确认的ACK"""
        # 模拟接收ACK的逻辑
        for neighbor in self.neighbors:
            if neighbor in self.slot_schedule:
                if self.slot_schedule[neighbor] == self.requesting_slot:
                    logging.info(f"UAV {self.my_drone.identifier} detected a conflict on slot {self.requesting_slot}")
                    return False
        return True

    def _send_final_confirm(self, slot):
        """发送最终确认信息"""
        message = {
            'type': 'final_confirm',
            'drone_id': self.my_drone.identifier,
            'slot': slot
        }
        self.phy.broadcast(self._create_control_packet(message))

    def _create_control_packet(self, message):
        """创建控制包"""
        packet = DataPacket(
            src_id=self.my_drone.identifier,
            dst_id=None,  # 广播不指定目标
            payload_size=0,  # 控制包没有有效负载
            payload=message
        )
        return packet

    def receive_packet(self, packet):
        """处理接收到的包"""
        if not isinstance(packet.payload, dict):
            return

        message = packet.payload
        msg_type = message.get('type', None)
        slot = message.get('slot', None)
        sender = message.get('drone_id', None)

        if msg_type == 'request':
            logging.info(f"UAV {self.my_drone.identifier} received a reservation request from UAV {sender} for slot {slot}")
            # 检查是否发生冲突
            if slot in self.reserved_slots:
                self._send_collision_report()
        elif msg_type == 'collision':
            logging.info(f"UAV {self.my_drone.identifier} received a collision report from UAV {sender}")
        elif msg_type == 'confirm':
            logging.info(f"UAV {self.my_drone.identifier} received a reservation confirmation from UAV {sender} for slot {slot}")
            self.reserved_slots.add(slot)
        elif msg_type == 'final_confirm':
            logging.info(f"UAV {self.my_drone.identifier} received a final confirmation from UAV {sender} for slot {slot}")
            self.reserved_slots.add(slot)
            self.slot_schedule[sender] = slot
        else:
            logging.warning(f"UAV {self.my_drone.identifier} received an unknown message type from UAV {sender}")

    def transmit(self, payload):
        """在预约的时隙中发送数据"""
        if self.my_drone.identifier in self.slot_schedule:
            reserved_slot = self.slot_schedule[self.my_drone.identifier]
            current_slot = self.current_slot
            if reserved_slot == current_slot:
                logging.info(f"UAV {self.my_drone.identifier} is transmitting data during its reserved slot {reserved_slot}")
                self.phy.broadcast(payload)
            else:
                logging.debug(f"UAV {self.my_drone.identifier} is waiting for its reserved slot {reserved_slot}")
        else:
            logging.warning(f"UAV {self.my_drone.identifier} has no reserved slot to transmit data")