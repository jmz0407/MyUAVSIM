import logging
import simpy
from phy.phy import Phy
from utils import config
from entities.packet import DataPacket

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
        # self.slot_schedule = self._create_slot_schedule()
        self.current_slot = 0
        self.phy = Phy(self)

        # 当前传输任务
        self.current_transmission = None
        self.cluster_id = self._calculate_cluster()
        self.slot_schedule = self._create_reuse_schedule()  # 修改为复用调度

        # 启动时隙同步进程
        self.env.process(self._slot_synchronization())



    def _create_slot_schedule(self):
        """创建时隙分配表"""
        schedule = {}
        for i in range(self.num_slots):
            schedule[i] = i % self.num_slots
        logging.info(f"TDMA schedule created for drone {self.my_drone.identifier}: {schedule}")
        return schedule

    def _calculate_cluster(self):
        """根据位置计算分簇ID(0~CLUSTER_SIZE-1)"""
        x = self.my_drone.coords[0]
        y = self.my_drone.coords[1]
        return hash((x // config.REUSE_DISTANCE, y // config.REUSE_DISTANCE)) % config.CLUSTER_SIZE

    def _create_reuse_schedule(self):
        """创建可复用的时隙分配表"""
        schedule = {}
        slots_per_cluster = self.num_slots // config.CLUSTER_SIZE

        for slot in range(self.num_slots):
            cluster = slot // slots_per_cluster
            schedule[slot] = [cluster]

            # 添加可复用的非相邻簇（间隔至少一个簇）
            if (slot % config.CLUSTER_SIZE) == 0:
                reuse_cluster = (cluster + 2) % config.TOTAL_CLUSTERS  # 确保复用簇不同
                schedule[slot].append(reuse_cluster)
        logging.info(schedule)

        return schedule
    def _slot_synchronization(self):
        """时隙同步进程"""
        while True:
            self.current_slot = (self.env.now // self.time_slot_duration) % self.num_slots
            yield self.env.timeout(self.time_slot_duration)

    def mac_send(self, packet):
        if isinstance(packet, DataPacket):
            self.current_transmission = packet
            mac_start_time = self.env.now

            # 获取当前允许发送的时隙列表
            allowed_slots = [
                slot for slot, clusters in self.slot_schedule.items()
                if self.cluster_id in clusters
            ]

            current_time = self.env.now
            slot_cycle = self.num_slots * self.time_slot_duration
            current_cycle = current_time // slot_cycle

            # 计算所有允许时隙的下一可用时间
            candidate_slots = []
            for slot in allowed_slots:
                slot_time_in_cycle = current_cycle * slot_cycle + slot * self.time_slot_duration
                if slot_time_in_cycle < current_time:
                    # 当前周期已过，使用下一周期
                    slot_time_in_cycle += slot_cycle
                candidate_slots.append(slot_time_in_cycle)

            nearest_slot_start = min(candidate_slots)
            wait_time = max(nearest_slot_start - current_time, 0)

            # 等待到目标时隙
            if wait_time > 0:
                logging.info(f"UAV {self.my_drone.identifier} 等待 {wait_time / 1e3}ms 直到时隙 {nearest_slot_start}")
                yield self.env.timeout(wait_time)

            # 冲突检测(基于位置信息)
            neighbors = self.my_drone.get_neighbors()
            interfering_drones = [
                drone for drone in neighbors
                if drone.cluster_id == self.cluster_id
                   and drone.current_transmission is not None
            ]

            if not interfering_drones:
                yield self.env.process(self._transmit_packet(packet))
            else:
                logging.warning(f"冲突风险! 检测到 {len(interfering_drones)} 个相邻节点正在发送")
                yield self.env.process(self._handle_collision(packet))

            # 记录时延
            mac_delay = self.env.now - mac_start_time
            self.simulator.metrics.mac_delay.append(mac_delay / 1e3)

    def _handle_collision(self, packet):
        """冲突退避处理"""
        max_retries = 3
        backoff_slots = [2, 4, 8]  # 指数退避

        for attempt in range(max_retries):
            # 随机退避
            backoff = backoff_slots[attempt] * self.time_slot_duration
            yield self.env.timeout(backoff)

            # 重新检测信道
            if not self._check_interference():
                yield self.env.process(self._transmit_packet(packet))
                return

        logging.error(f"包 {packet.packet_id} 经过 {max_retries} 次重试后丢弃")
        self.simulator.metrics.lost_packets += 1

    def _check_interference(self):
        """仅检测信号强度足够的干扰节点"""
        neighbors = self.phy.get_interfering_drones(self.cluster_id)
        return any(
            drone.current_transmission is not None
            for drone in neighbors
        )
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