import logging
import simpy
from phy.phy import Phy
from utils import config

# 配置日志
logging.basicConfig(filename='running_log.log',
                    filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=config.LOGGING_LEVEL)


class Tdma:
    """
    Time Division Multiple Access (TDMA) MAC Protocol Implementation

    Attributes:
        my_drone: 当前安装该协议的无人机
        simulator: 仿真器实例
        env: SimPy仿真环境
        time_slot_duration: 时隙长度(μs)
        num_slots: 总时隙数
        slot_schedule: 时隙分配表
        current_slot: 当前时隙
        phy: 物理层接口

    Author: Claude
    Created at: 2024/1/12
    """

    def __init__(self, drone):
        self.my_drone = drone
        self.simulator = drone.simulator
        self.env = drone.env
        self.time_slot_duration = config.SLOT_DURATION
        self.num_slots = config.NUMBER_OF_DRONES
        self.slot_schedule = self._create_slot_schedule()
        self.current_slot = 0
        self.phy = Phy(self)

        # 当前传输任务
        self.current_transmission = None

        # 启动时隙同步进程
        self.env.process(self._slot_synchronization())

    def _create_slot_schedule(self):
        """创建时隙分配表"""
        schedule = {}
        for i in range(self.num_slots):
            schedule[i] = i % self.num_slots
        logging.info(f"TDMA schedule created for drone {self.my_drone.identifier}: {schedule}")
        return schedule


    def _slot_synchronization(self):
        """时隙同步进程"""
        while True:
            self.current_slot = (self.env.now // self.time_slot_duration) % self.num_slots
            yield self.env.timeout(self.time_slot_duration)

    def mac_send(self, packet):
        """
        MAC层发送函数

        Args:
            packet: 需要发送的数据包
        """
        self.current_transmission = packet
        mac_start_time = self.env.now  # 记录开始等待的时间
        # 获取当前无人机的时隙
        assigned_slot = self.slot_schedule[self.my_drone.identifier]
        current_time = self.env.now

        # 计算当前时隙的开始和结束时间
        slot_start_time = (
                                      current_time // self.time_slot_duration) * self.time_slot_duration + assigned_slot * self.time_slot_duration
        slot_end_time = slot_start_time + self.time_slot_duration

        logging.info(f"UAV {self.my_drone.identifier} assigned to slot {assigned_slot} "
                     f"from {slot_start_time} to {slot_end_time}")

        if current_time >= slot_start_time and current_time < slot_end_time:
            # 当前是分配的时隙,可以发送
            logging.info(f"UAV {self.my_drone.identifier} sending packet {packet.packet_id} "
                         f"in its slot at {self.env.now}")
            yield self.env.process(self._transmit_packet(packet))
        else:
            # 不是当前时隙,等待下一个时隙
            wait_time = slot_end_time - current_time
            logging.info(f"UAV {self.my_drone.identifier} waiting {wait_time} for next slot")
            yield self.env.timeout(wait_time)
            yield self.env.process(self._transmit_packet(packet))


        # 计算MAC延迟并记录
        mac_delay = self.env.now - mac_start_time
        self.simulator.metrics.mac_delay.append(mac_delay)
        logging.info(f"MAC delay for packet {packet.packet_id}: {mac_delay} us")
    def _transmit_packet(self, packet):
        """
        执行实际的数据包传输

        Args:
            packet: 要传输的数据包
        """
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

        self.current_transmission = None  # 传输完成