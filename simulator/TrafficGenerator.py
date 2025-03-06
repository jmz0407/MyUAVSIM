import random
from utils import config
from entities.packet import DataPacket
import logging
from entities.packet import Packet

# 全局数据包ID
GLOBAL_DATA_PACKET_ID = 0
GL_ID_TRAFFIC_REQUIREMENT = 60000
class TrafficGenerator:
    """业务流生成器"""

    def __init__(self, simulator):
        self.simulator = simulator
        self.env = simulator.env
        self.traffic_patterns = {}
        self.active_requirements = []  # 存储活跃的业务需求

    def setup_cbr_traffic(self, source_id, dest_id, data_rate,
                          packet_size=1024,  # 数据包大小(bytes)
                          batch_size=10,  # 每批次包数量
                          num_batches=1,  # 批次数量
                          batch_interval=1e6,  # 批次间隔(ns)
                          start_time=0):
        """设置CBR业务"""
        self.traffic_patterns[(source_id, dest_id)] = {
            'type': 'CBR',
            'data_rate': data_rate,
            'packet_size': packet_size,
            'batch_size': batch_size,
            'num_batches': num_batches,
            'batch_interval': batch_interval,
            'start_time': start_time
        }

    def setup_vbr_traffic(self, source_id, dest_id, mean_rate, peak_rate,
                          packet_size=1024,
                          min_batch_size=5,  # 最小批次大小
                          max_batch_size=15,  # 最大批次大小
                          num_batches=1,
                          batch_interval=1e6,
                          start_time=0):
        """设置VBR业务"""
        self.traffic_patterns[(source_id, dest_id)] = {
            'type': 'VBR',
            'mean_rate': mean_rate,
            'peak_rate': peak_rate,
            'packet_size': packet_size,
            'min_batch_size': min_batch_size,
            'max_batch_size': max_batch_size,
            'num_batches': num_batches,
            'batch_interval': batch_interval,
            'start_time': start_time
        }

    def setup_burst_traffic(self, source_id, dest_id,
                            packet_size=1024,
                            burst_size=10,  # 每次突发的包数量
                            num_bursts=1,  # 突发次数
                            burst_interval=1e6,  # 突发间隔
                            packet_interval=1000,  # 包间隔
                            start_time=0):
        """设置突发业务"""
        self.traffic_patterns[(source_id, dest_id)] = {
            'type': 'BURST',
            'packet_size': packet_size,
            'burst_size': burst_size,
            'num_bursts': num_bursts,
            'burst_interval': burst_interval,
            'packet_interval': packet_interval,
            'start_time': start_time
        }

    def setup_random_traffic(self, num_flows):
        """随机设置多条业务流"""
        n_drones = self.simulator.n_drones
        for _ in range(num_flows):
            source_id = random.randint(0, n_drones - 1)
            dest_id = random.randint(0, n_drones - 1)
            while dest_id == source_id:
                dest_id = random.randint(0, n_drones - 1)

            traffic_type = random.choice(['CBR', 'VBR', 'BURST'])

            # 随机生成业务流大小参数
            packet_size = random.choice([512, 1024, 2048])  # 常用数据包大小

            if traffic_type == 'CBR':
                self.setup_cbr_traffic(
                    source_id,
                    dest_id,
                    data_rate=random.uniform(1, 10),
                    packet_size=packet_size,
                    batch_size=random.randint(5, 15),
                    num_batches=random.randint(1, 5)
                )
            elif traffic_type == 'VBR':
                mean_rate = random.uniform(1, 8)
                self.setup_vbr_traffic(
                    source_id,
                    dest_id,
                    mean_rate=mean_rate,
                    peak_rate=mean_rate * random.uniform(1.2, 2.0),
                    packet_size=packet_size,
                    min_batch_size=random.randint(5, 8),
                    max_batch_size=random.randint(12, 15),
                    num_batches=random.randint(1, 5)
                )
            else:  # BURST
                self.setup_burst_traffic(
                    source_id,
                    dest_id,
                    packet_size=packet_size,
                    burst_size=random.randint(10, 20),
                    num_bursts=random.randint(1, 3),
                    burst_interval=random.randint(1e6, 2e6)
                )

    def _generate_cbr_traffic(self, source_drone, dest_drone, pattern):
        """生成CBR业务流"""
        yield self.env.timeout(pattern['start_time'])

        for batch in range(pattern['num_batches']):
            # 生成一批数据包
            for i in range(pattern['batch_size']):
                packet = DataPacket(
                    src_drone=source_drone,
                    dst_drone=dest_drone,
                    creation_time=self.env.now,
                    data_packet_id=self.simulator.get_next_packet_id(),
                    data_packet_length=pattern['packet_size'],
                    simulator=self.simulator,
                    priority=2
                )

                if source_drone.transmitting_queue.qsize() < source_drone.max_queue_size:
                    source_drone.transmitting_queue.put(packet)

                # 按照数据率计算下一个包的间隔
                interval = (pattern['packet_size'] / pattern['data_rate']) * 1e6
                yield self.env.timeout(interval)

            # 等待下一批次
            if batch < pattern['num_batches'] - 1:
                yield self.env.timeout(pattern['batch_interval'])

    def _generate_vbr_traffic(self, source_drone, dest_drone, pattern):
        """生成VBR业务流"""
        yield self.env.timeout(pattern['start_time'])

        for batch in range(pattern['num_batches']):
            # 随机决定这一批次的包数量
            batch_size = random.randint(pattern['min_batch_size'],
                                        pattern['max_batch_size'])

            for i in range(batch_size):
                # 当前速率在平均速率和峰值速率之间随机变化
                current_rate = random.uniform(pattern['mean_rate'],
                                              pattern['peak_rate'])

                packet = DataPacket(
                    src_drone=source_drone,
                    dst_drone=dest_drone,
                    creation_time=self.env.now,
                    data_packet_id=self.simulator.get_next_packet_id(),
                    data_packet_length=pattern['packet_size'],
                    simulator=self.simulator,
                    priority=2
                )

                if source_drone.transmitting_queue.qsize() < source_drone.max_queue_size:
                    source_drone.transmitting_queue.put(packet)

                interval = (pattern['packet_size'] / current_rate) * 1e6
                yield self.env.timeout(interval)

            # 等待下一批次
            if batch < pattern['num_batches'] - 1:
                yield self.env.timeout(pattern['batch_interval'])

    def _generate_burst_traffic(self, source_drone, dest_drone, pattern):
        """生成突发业务流"""
        yield self.env.timeout(pattern['start_time'])

        for burst in range(pattern['num_bursts']):
            # 生成一次突发的数据包
            for i in range(pattern['burst_size']):
                packet = DataPacket(
                    src_drone=source_drone,
                    dst_drone=dest_drone,
                    creation_time=self.env.now,
                    data_packet_id=self.simulator.get_next_packet_id(),
                    data_packet_length=pattern['packet_size'],
                    simulator=self.simulator,
                    priority=1
                )

                if source_drone.transmitting_queue.qsize() < source_drone.max_queue_size:
                    source_drone.transmitting_queue.put(packet)
                    yield self.env.timeout(pattern['packet_interval'])

            # 等待下一次突发
            if burst < pattern['num_bursts'] - 1:
                yield self.env.timeout(pattern['burst_interval'])

    def generate_traffic(self, source_id, dest_id, num_packets, packet_interval=2000):
        source_drone = self.simulator.drones[source_id]
        dest_drone = self.simulator.drones[dest_id]

        def generate():
            global GLOBAL_DATA_PACKET_ID

            for i in range(num_packets):
                GLOBAL_DATA_PACKET_ID += 1

                # 创建数据包
                packet = DataPacket(
                    src_drone=source_drone,
                    dst_drone=dest_drone,
                    creation_time=self.env.now,
                    data_packet_id=GLOBAL_DATA_PACKET_ID,
                    data_packet_length=config.DATA_PACKET_LENGTH,
                    simulator=self.simulator,
                    priority=1
                )
                packet.transmission_mode = 0  # unicast模式

                # 增加生成数据包计数
                self.simulator.metrics.datapacket_generated_num += 1

                # 放入发送队列
                if source_drone.transmitting_queue.qsize() < source_drone.max_queue_size:
                    # 设置等待开始时间
                    packet.waiting_start_time = self.env.now

                    source_drone.transmitting_queue.put(packet)
                    logging.info(f'UAV {source_id} generates packet {i + 1}/{num_packets} '
                                 f'(id: {packet.packet_id}, dst: {dest_id}) at: {self.env.now}')
                    logging.info(f'UAV: {source_id}, queue size: {source_drone.transmitting_queue.qsize()}')
                else:
                    logging.warning(f'UAV {source_id} queue full, packet dropped')
                    break

                yield self.env.timeout(packet_interval)


        # 启动生成过程
        self.env.process(generate())

class TrafficRequirement(Packet):
    global GL_ID_TRAFFIC_REQUIREMENT
    GL_ID_TRAFFIC_REQUIREMENT += 1
    def __init__(self, source_id, dest_id, num_packets, delay_req, qos_req, simulator, creation_time = 0, data_packet_length = 0,data_packet_id = GL_ID_TRAFFIC_REQUIREMENT):
        super().__init__(data_packet_id, data_packet_length, creation_time, simulator)
        self.source_id = source_id
        self.dest_id = dest_id
        self.num_packets = num_packets
        self.delay_requirement = delay_req
        self.qos_requirement = qos_req
        self.packet_id = 11111  # 将由simulator设置
        self.creation_time = None  # 将由simulator设置
        self.message_type = 'traffic_requirement'
        self.dst_id = dest_id
        self.src_drone = None
        self.dst_drone = None
        self.data_packet_id = data_packet_id
        # 添加所需属性
        self.deadline = 1e6  # 设置一个较大的默认值，单位为ns
        self.number_retransmission_attempt = {i: 0 for i in range(config.NUMBER_OF_DRONES)}  # 用于重传次数统计
        self.next_hop_id = None  # 用于路由
        self.routing_path = []   # 用于记录路由路径
        self.waiting_start_time = None  # 等待开始时间
        self.transmission_mode = 0  # 0: unicast, 1: multicast



