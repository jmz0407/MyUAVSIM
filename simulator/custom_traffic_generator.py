import random
import logging
from utils import config
from entities.packet import DataPacket
from simulator.TrafficGenerator import TrafficRequirement


class EnhancedTrafficGenerator:
    """增强型业务流生成器，专注于VBR和Burst业务流"""

    def __init__(self, simulator):
        self.simulator = simulator
        self.env = simulator.env
        self.active_flows = {}  # {flow_id: flow_info}
        self.flow_counter = 0
        self.packet_counter = 0

    def generate_vbr_traffic(self, source_id, dest_id,
                             mean_rate=5.0,  # 平均速率(Mbps)
                             peak_rate=10.0,  # 峰值速率(Mbps)
                             packet_size=1024,  # 包大小(bytes)
                             duration=10.0,  # 持续时间(秒)
                             variability=0.3,  # 速率变化程度(0-1)
                             burst_probability=0.1,  # 突发概率
                             start_time=0):  # 开始时间(ns)
        """
        生成VBR(可变比特率)业务流

        参数：
        - source_id: 源节点ID
        - dest_id: 目标节点ID
        - mean_rate: 平均数据速率(Mbps)
        - peak_rate: 峰值数据速率(Mbps)
        - packet_size: 数据包大小(bytes)
        - duration: 业务流持续时间(秒)
        - variability: 速率变化程度(0-1)，值越大变化越大
        - burst_probability: 突发概率(0-1)
        - start_time: 开始时间(ns)
        """
        # 创建唯一的流ID
        flow_id = f"vbr_{self.flow_counter}"
        self.flow_counter += 1

        # 记录流信息
        self.active_flows[flow_id] = {
            'type': 'VBR',
            'source_id': source_id,
            'dest_id': dest_id,
            'mean_rate': mean_rate,
            'peak_rate': peak_rate,
            'packet_size': packet_size,
            'variability': variability,
            'start_time': start_time,
            'end_time': start_time + duration * 1e6,  # 转换为ns
            'packets_sent': 0,
            'packets_received': 0,
            'burst_probability': burst_probability
        }

        # 启动业务流生成进程
        self.simulator.env.process(self._vbr_generation_process(flow_id))
        logging.info(f"VBR业务流 {flow_id} 已创建: {source_id} -> {dest_id}, 平均速率: {mean_rate}Mbps")

        return flow_id

    def generate_burst_traffic(self, source_id, dest_id,
                               burst_size=10,  # 每次突发的包数
                               num_bursts=5,  # 突发次数
                               burst_interval=2.0,  # 突发间隔(秒)
                               packet_interval=0.01,  # 包间隔(秒)
                               packet_size=1024,  # 包大小(bytes)
                               jitter=0.2,  # 时间抖动(0-1)
                               start_time=0):  # 开始时间(ns)
        """
        生成Burst(突发)业务流

        参数：
        - source_id: 源节点ID
        - dest_id: 目标节点ID
        - burst_size: 每次突发的数据包数量
        - num_bursts: 突发次数
        - burst_interval: 突发间隔(秒)
        - packet_interval: 同一突发内包间隔(秒)
        - packet_size: 数据包大小(bytes)
        - jitter: 时间抖动(0-1)，值越大抖动越大
        - start_time: 开始时间(ns)
        """
        # 创建唯一的流ID
        flow_id = f"burst_{self.flow_counter}"
        self.flow_counter += 1

        # 记录流信息
        self.active_flows[flow_id] = {
            'type': 'BURST',
            'source_id': source_id,
            'dest_id': dest_id,
            'burst_size': burst_size,
            'num_bursts': num_bursts,
            'burst_interval': burst_interval * 1e6,  # 转换为ns
            'packet_interval': packet_interval * 1e6,  # 转换为ns
            'packet_size': packet_size,
            'jitter': jitter,
            'start_time': start_time,
            'packets_sent': 0,
            'packets_received': 0
        }

        # 启动业务流生成进程
        self.simulator.env.process(self._burst_generation_process(flow_id))
        logging.info(f"Burst业务流 {flow_id} 已创建: {source_id} -> {dest_id}, 突发次数: {num_bursts}")

        return flow_id

    def _vbr_generation_process(self, flow_id):
        """VBR业务流生成进程"""
        flow = self.active_flows[flow_id]

        # 等待开始时间
        if flow['start_time'] > 0:
            yield self.env.timeout(flow['start_time'])

        source_drone = self.simulator.drones[flow['source_id']]
        dest_drone = self.simulator.drones[flow['dest_id']]

        # 发送数据包直到结束时间
        while self.env.now < flow['end_time']:
            # 计算当前速率(根据变化程度随机波动)
            variability = flow['variability']
            min_rate = flow['mean_rate'] * (1 - variability)
            max_rate = min(flow['peak_rate'], flow['mean_rate'] * (1 + variability))

            # 判断是否进入突发模式
            if random.random() < flow['burst_probability']:
                # 突发模式：使用峰值速率
                current_rate = flow['peak_rate']
                burst_packets = random.randint(3, 8)  # 突发模式下发送3-8个包

                for _ in range(burst_packets):
                    self._send_packet(source_drone, dest_drone, flow_id, packet_size=flow['packet_size'], priority=2)
                    # 突发模式下的包间隔更短
                    interval = (flow['packet_size'] / (current_rate * 1e6 / 8)) * 1e6 * 0.5
                    yield self.env.timeout(max(1000, interval))  # 至少1微秒

                # 突发后暂停一小段时间
                yield self.env.timeout(random.uniform(0.2, 0.5) * 1e6)
            else:
                # 正常模式：随机速率
                current_rate = random.uniform(min_rate, max_rate)

                # 发送单个数据包
                self._send_packet(source_drone, dest_drone, flow_id, packet_size=flow['packet_size'], priority=2)

                # 根据当前速率计算下一个包的时间间隔
                # 公式：间隔(ns) = 包大小(bits) / 速率(bits/s) * 1e9
                interval = (flow['packet_size'] * 8) / (current_rate * 1e6) * 1e9
                yield self.env.timeout(interval)

        # 完成后记录
        logging.info(f"VBR业务流 {flow_id} 完成发送, 共发送 {flow['packets_sent']} 个数据包")

    def _burst_generation_process(self, flow_id):
        """Burst业务流生成进程"""
        flow = self.active_flows[flow_id]

        # 等待开始时间
        if flow['start_time'] > 0:
            yield self.env.timeout(flow['start_time'])

        source_drone = self.simulator.drones[flow['source_id']]
        dest_drone = self.simulator.drones[flow['dest_id']]

        # 发送多次突发
        for burst in range(flow['num_bursts']):
            # 生成此次突发的大小(带随机变化)
            actual_burst_size = max(1, int(flow['burst_size'] * random.uniform(0.8, 1.2)))

            # 发送一组突发数据包
            for i in range(actual_burst_size):
                self._send_packet(source_drone, dest_drone, flow_id, packet_size=flow['packet_size'], priority=1)

                # 计算包间隔(带抖动)
                jitter_factor = 1.0 + random.uniform(-flow['jitter'], flow['jitter'])
                interval = flow['packet_interval'] * jitter_factor
                yield self.env.timeout(max(100, interval))  # 至少100ns

            # 记录突发完成
            logging.info(
                f"Burst业务流 {flow_id} 完成第 {burst + 1}/{flow['num_bursts']} 次突发，发送 {actual_burst_size} 个数据包")

            # 突发之间的间隔(也带抖动)
            if burst < flow['num_bursts'] - 1:  # 不是最后一次突发
                jitter_factor = 1.0 + random.uniform(-flow['jitter'] / 2, flow['jitter'] / 2)
                burst_interval = flow['burst_interval'] * jitter_factor
                yield self.env.timeout(burst_interval)

        # 完成后记录
        logging.info(f"Burst业务流 {flow_id} 完成所有突发，共发送 {flow['packets_sent']} 个数据包")

    def _send_packet(self, source_drone, dest_drone, flow_id, packet_size, priority):
        """发送单个数据包"""
        # 增加全局包计数
        self.packet_counter += 1

        # 创建数据包
        packet = DataPacket(
            src_drone=source_drone,
            dst_drone=dest_drone,
            creation_time=self.env.now,
            data_packet_id=self.simulator.get_next_packet_id() if hasattr(self.simulator,
                                                                          'get_next_packet_id') else self.packet_counter,
            data_packet_length=packet_size,
            simulator=self.simulator,
            priority=priority
        )

        # 为数据包标记流ID
        packet.flow_id = flow_id

        # 设置等待开始时间
        packet.waiting_start_time = self.env.now

        # 放入发送队列
        if source_drone.transmitting_queue.qsize() < source_drone.max_queue_size:
            source_drone.transmitting_queue.put(packet)

            # 更新流的包计数
            flow = self.active_flows[flow_id]
            flow['packets_sent'] += 1

            # 日志
            logging.debug(
                f"业务流 {flow_id} 生成数据包 ID: {packet.packet_id}, 源: {source_drone.identifier}, 目标: {dest_drone.identifier}")
        else:
            logging.warning(f"源无人机 {source_drone.identifier} 队列已满，数据包丢弃")