import logging
import random
from phy.phy import Phy
from utils import config
from utils.util_function import euclidean_distance

class Stdma:
    def __init__(self, drone):
        self.my_drone = drone
        self.simulator = drone.simulator
        self.env = drone.env

        # 基本参数初始化
        self.time_slot_duration = config.SLOT_DURATION if hasattr(config, 'SLOT_DURATION') else 1000  # 1ms
        self.frame_duration = config.FRAME_DURATION if hasattr(config,
                                                               'FRAME_DURATION') else self.time_slot_duration * 100
        self.num_slots = config.NUMBER_OF_DRONES
        self.report_rate = 2  # 每帧发送2个数据包
        self.selection_interval_ratio = 0.2

        # 时隙管理
        self.nominal_slots = []  # 标称时隙列表
        self.reserved_slots = {}  # 预约时隙表 {slot_id: (timeout, owner)}
        self.timeout_rng = random.Random(2024 + self.my_drone.identifier)
        self.current_slot = 0

        # 物理层和传输状态
        self.phy = Phy(self)
        self.current_transmission = None

        # 启动初始化和同步进程
        self.env.process(self._initialization_phase())
        self.env.process(self._slot_synchronization())

        logging.info(f"STDMA initialized for UAV {self.my_drone.identifier} with "
                     f"{self.num_slots} slots, slot duration: {self.time_slot_duration}us")

    def _initialization_phase(self):
        """初始化阶段:监听一帧来了解网络状态"""
        logging.info('------> UAV: %s starting initialization phase at time %s',
                     self.my_drone.identifier, self.env.now)
        yield self.env.timeout(self.frame_duration)
        self._select_nominal_slots()
        logging.info('------> UAV: %s completed initialization phase at time %s, starting normal operation',
                     self.my_drone.identifier, self.env.now)

    def _select_nominal_slots(self):
        """选择标称时隙"""
        ni = self.num_slots // self.report_rate  # 标称增量
        nss = random.randint(0, ni - 1)  # 随机选择第一个标称时隙

        self.nominal_slots = []
        for i in range(self.report_rate):
            nominal_slot = (nss + i * ni) % self.num_slots
            self.nominal_slots.append(nominal_slot)

        logging.info(f"UAV {self.my_drone.identifier} selected nominal slots: {self.nominal_slots}")

    def _slot_synchronization(self):
        """时隙同步进程"""
        while True:
            self.current_slot = (self.env.now // self.time_slot_duration) % self.num_slots
            # 更新预约时隙超时状态
            self._update_timeouts()
            yield self.env.timeout(self.time_slot_duration)

    def _update_timeouts(self):
        """更新预约时隙的超时状态"""
        expired = []
        for slot_id, (timeout, owner) in self.reserved_slots.items():
            if timeout <= 0:
                expired.append(slot_id)
        for slot_id in expired:
            del self.reserved_slots[slot_id]

    def _select_transmission_slot(self, nominal_slot):
        """基于虚拟力邻居表的时隙选择"""
        logging.info('------> UAV: %s starting slot selection at time %s', self.my_drone.identifier, self.env.now)
        si_half = max(1, int(self.num_slots * self.selection_interval_ratio / 2))
        candidates = []

        # 获取虚拟力邻居表
        neighbors = set(self.my_drone.motion_controller.neighbor_table.neighbor_table.keys())

        # 计算选择区间
        start = (nominal_slot - si_half) % self.num_slots
        end = (nominal_slot + si_half + 1) % self.num_slots
        slot_range = []
        if end > start:
            slot_range = list(range(start, end))
        else:
            slot_range = list(range(start, self.num_slots)) + list(range(0, end))

        for slot_id in slot_range:
            if slot_id not in self.reserved_slots:
                candidates.append((slot_id, 1.0))  # 空闲时隙
            else:
                timeout, owner = self.reserved_slots[slot_id]

                if owner not in neighbors:  # 非邻居节点的时隙可复用
                    candidates.append((slot_id, 0.8))
                else:
                    # 根据距离计算复用权重
                    neighbor_pos = self.my_drone.motion_controller.neighbor_table.neighbor_table[owner][0]
                    distance = euclidean_distance(self.my_drone.coords, neighbor_pos)
                    if distance > self.my_drone.motion_controller.neighbor_table.desired_distance:
                        candidates.append((slot_id, 0.3))
                        logging.info(
                            '------> UAV: %s can reuse slot %s from far neighbor UAV %s (distance: %.2f) at time %s',
                            self.my_drone.identifier, slot_id, owner, distance, self.env.now)

        if not candidates:
            candidates = [(slot_id, 0.1) for slot_id in slot_range]

        weights = [score for _, score in candidates]
        selected_slot = random.choices([slot_id for slot_id, _ in candidates], weights=weights, k=1)[0]

        timeout = self.timeout_rng.randint(3, 7)
        logging.info('------> UAV: %s selected slot %s with timeout %s at time %s',
                     self.my_drone.identifier, selected_slot, timeout, self.env.now)
        return selected_slot, timeout

    def mac_send(self, packet):
        """MAC层发送函数"""
        self.current_transmission = packet
        mac_start_time = self.env.now

        # 选择发送时隙
        nominal_slot = self.nominal_slots[0] if self.nominal_slots else 0  # 使用第一个标称时隙
        tx_slot, timeout = self._select_transmission_slot(nominal_slot)

        # 计算等待时间
        current_time = self.env.now
        slot_start_time = (
                                  current_time // self.time_slot_duration
                          ) * self.time_slot_duration + tx_slot * self.time_slot_duration

        wait_time = slot_start_time - current_time
        if wait_time < 0:
            wait_time += self.frame_duration

        # 等待到发送时隙
        yield self.env.timeout(wait_time)

        # 预约时隙
        self.reserved_slots[tx_slot] = (timeout, self.my_drone.identifier)
        logging.info(f"UAV {self.my_drone.identifier} reserved slot {tx_slot} "
                     f"for {timeout} frames")

        # 发送数据包
        yield self.env.process(self._transmit_packet(packet))

        # 计算MAC延迟
        mac_delay = self.env.now - mac_start_time
        self.simulator.metrics.mac_delay.append(mac_delay)
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