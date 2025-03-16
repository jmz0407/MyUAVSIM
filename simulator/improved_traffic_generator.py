import random
import logging
import math
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Any, Union
from entities.packet import DataPacket, Packet
from utils import config
# 配置日志
logging.basicConfig(
    filename='traffic_generator.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 全局数据包ID计数器
GLOBAL_DATA_PACKET_ID = 0
GL_ID_TRAFFIC_REQUIREMENT = 60000




# 业务流类型枚举
class TrafficType(Enum):
    CBR = "Constant Bit Rate"  # 恒定比特率
    VBR = "Variable Bit Rate"  # 可变比特率
    BURST = "Burst Traffic"  # 突发业务
    PERIODIC = "Periodic Traffic"  # 周期性业务
    POISSON = "Poisson Traffic"  # 泊松分布业务
    PARETO = "Pareto Traffic"  # 帕累托分布业务
    CUSTOM = "Custom Traffic"  # 自定义业务


# 优先级等级枚举
class PriorityLevel(Enum):
    LOW = 0  # 低优先级
    NORMAL = 1  # 普通优先级
    HIGH = 2  # 高优先级
    CRITICAL = 3  # 关键业务


# 业务统计数据结构
@dataclass
class TrafficStats:
    total_packets: int = 0  # 总数据包数
    sent_packets: int = 0  # 已发送数据包数
    received_packets: int = 0  # 已接收数据包数
    dropped_packets: int = 0  # 丢弃的数据包数
    start_time: int = 0  # 开始时间
    end_time: int = 0  # 结束时间
    avg_delay: float = 0.0  # 平均延迟
    min_delay: float = float('inf')  # 最小延迟
    throughout: float = 0.0  # 吞吐量
    max_delay: float = 0.0  # 最大延迟
    queue_overflow_count: int = 0  # 队列溢出次数
    retransmission_count: int = 0  # 重传次数
    packet_ids: Set[int] = field(default_factory=set)  # 跟踪已生成的包ID


# 业务流请求数据结构
@dataclass
class TrafficRequirement(Packet):
    source_id: int  # 源节点ID
    dest_id: int  # 目标节点ID
    num_packets: int  # 数据包数量
    delay_requirement: float  # 延迟要求(ms)
    qos_requirement: float  # QoS要求(0-1)
    traffic_type: TrafficType = TrafficType.CBR  # 业务类型
    packet_length: int = config.DATA_PACKET_LENGTH  # 数据包长度
    priority: PriorityLevel = PriorityLevel.NORMAL  # 优先级
    start_time: int = 0  # 开始时间
    duration: int = 0  # 持续时间
    packet_size: int = config.DATA_PACKET_LENGTH  # 数据包大小
    data_rate: float = 4  # 数据速率(Mbps)
    burst_size: int = 1  # 突发大小
    simulator: Any = None  # 模拟器引用
    src_drone: Any = None  # 源无人机引用
    dst_drone: Any = None  # 目标无人机引用
    routing_path: List[int] = field(default_factory=list)  # 路由路径
    creation_time: int = 0  # 创建时间
    message_type: str = 'traffic_requirement'  # 消息类型
    data_packet_id: int = None  # 数据包ID
    packet_id: int = None  # 包ID
    deadline: int = 1_000_000  # 截止时间(ns)
    number_retransmission_attempt: Dict[int, int] = field(default_factory=dict)  # 重传尝试
    next_hop_id: int = None  # 下一跳ID
    waiting_start_time: int = None  # 等待开始时间
    transmission_mode: int = 0  # 传输模式(0:单播,1:多播)

    def __post_init__(self):
        """初始化后处理"""
        global GL_ID_TRAFFIC_REQUIREMENT

        if self.packet_id is None:
            GL_ID_TRAFFIC_REQUIREMENT += 1
            self.packet_id = GL_ID_TRAFFIC_REQUIREMENT

        if self.data_packet_id is None:
            self.data_packet_id = self.packet_id

        # 初始化重传尝试字典
        if not self.number_retransmission_attempt and self.simulator:
            self.number_retransmission_attempt = {
                i: 0 for i in range(self.simulator.n_drones)
            }

    def get_current_ttl(self):
        """获取当前TTL值"""
        return 0  # 控制包不应该随TTL递减而过期

    def increase_ttl(self):
        """增加TTL，对于控制包这是无操作"""
        pass


class TrafficGenerator:
    """
    改进的业务流生成器
    支持多种业务流类型和动态生成策略
    """

    def __init__(self, simulator):
        """
        初始化业务流生成器

        Args:
            simulator: 仿真器实例
        """
        self.simulator = simulator
        self.env = simulator.env
        self.traffic_flows = {}  # 所有业务流 {flow_id: flow_config}
        self.active_generators = {}  # 活跃的生成器进程 {flow_id: process}
        self.traffic_stats = {}  # 业务流统计 {flow_id: TrafficStats}
        self.requirements = {}  # 业务需求 {req_id: TrafficRequirement}
        self.ongoing_bursts = {}  # 进行中的突发 {flow_id: remaining_packets}
        self.packet_rate_tracker = {}  # 数据包生成率跟踪器
        self.last_generation_time = {}  # 上次生成时间 {flow_id: timestamp}
        self.backlogged_packets = {}  # 由于队列满而延迟的数据包 {drone_id: [packets]}
        self.max_retries = 3  # 最大重试次数

        # 启动统计监控进程
        self.env.process(self._monitor_traffic_stats())

    def _generate_flow_id(self, source_id, dest_id, traffic_type, priority=PriorityLevel.NORMAL):
        """
        生成唯一的流ID

        Args:
            source_id: 源节点ID
            dest_id: 目标节点ID
            traffic_type: 业务类型
            priority: 优先级

        Returns:
            str: 流ID
        """
        prefix = f"{priority.name.lower()}"
        return f"{prefix}_flow_{traffic_type.name}_{source_id}_{dest_id}_{self.env.now}"

    def setup_traffic_flow(self,
                           source_id,
                           dest_id,
                           traffic_type=TrafficType.CBR,
                           num_packets=100,
                           packet_size=None,
                           data_rate=1.0,
                           start_time=0,
                           duration=None,
                           priority=PriorityLevel.NORMAL,
                           delay_req=5000,
                           qos_req=0.9,
                           **params):
        """
        设置业务流配置

        Args:
            source_id: 源节点ID
            dest_id: 目标节点ID
            traffic_type: 业务类型
            num_packets: 数据包数量
            packet_size: 数据包大小(字节)
            data_rate: 数据速率(Mbps)
            start_time: 开始时间(ns)
            duration: 持续时间(ns)
            priority: 优先级
            delay_req: 延迟要求(ms)
            qos_req: QoS要求(0-1)
            **params: 特定业务类型的其他参数

        Returns:
            str: 流ID
        """
        # 使用默认包大小如果未指定
        if packet_size is None:
            packet_size = config.DATA_PACKET_LENGTH

        # 计算默认持续时间如果未指定(基于数据包数和速率)
        if duration is None and num_packets > 0:
            # 计算传输所有数据包所需的时间
            bits_per_packet = packet_size
            bits_total = bits_per_packet * num_packets
            duration_sec = bits_total / (data_rate * 1e6)  # data_rate单位是Mbps
            duration = int(duration_sec * 1e9)  # 转换为ns

        # 生成流ID
        flow_id = self._generate_flow_id(source_id, dest_id, traffic_type, priority)

        # 创建流配置
        flow_config = {
            'flow_id': flow_id,
            'source_id': source_id,
            'dest_id': dest_id,
            'traffic_type': traffic_type,
            'num_packets': num_packets,
            'packet_size': packet_size,
            'data_rate': data_rate,
            'start_time': start_time,
            'duration': duration,
            'priority': priority,
            'delay_req': delay_req,
            'qos_req': qos_req,
            'creation_time': self.env.now,
            'params': params  # 特定业务类型的额外参数
        }

        # 存储流配置
        self.traffic_flows[flow_id] = flow_config

        # 初始化统计
        self.traffic_stats[flow_id] = TrafficStats(
            start_time=self.env.now + start_time
        )

        # 设置上次生成时间
        self.last_generation_time[flow_id] = self.env.now + start_time

        logging.info(f"设置业务流 {flow_id}: {source_id}->{dest_id}, {traffic_type.name}, 包数: {num_packets}")

        return flow_id

    def start_traffic_flow(self, flow_id, immediate=False):
        """
        启动业务流生成

        Args:
            flow_id: 业务流ID
            immediate: 是否立即开始(忽略start_time)

        Returns:
            bool: 成功启动返回True
        """
        if flow_id not in self.traffic_flows:
            logging.error(f"未找到业务流 {flow_id}")
            return False

        # 检查是否已经有活跃的生成器
        if flow_id in self.active_generators:
            logging.warning(f"业务流 {flow_id} 已经在运行")
            return False

        # 获取流配置
        flow_config = self.traffic_flows[flow_id]

        # 为所有类型的业务流创建业务需求报文
        self._create_traffic_requirement_for_flow(flow_config)

        # 创建生成器进程
        if immediate:
            generator = self.env.process(self._generate_traffic(flow_id))
        else:
            generator = self.env.process(self._delayed_start(flow_id, flow_config['start_time']))

        # 存储生成器进程
        self.active_generators[flow_id] = generator

        logging.info(f"启动业务流 {flow_id}")
        return True

    def _create_traffic_requirement_for_flow(self, flow_config):
        """
        为任意类型的业务流创建业务需求报文

        Args:
            flow_config: 业务流配置

        Returns:
            TrafficRequirement: 创建的业务需求对象
        """
        # 提取流配置参数
        source_id = flow_config['source_id']
        dest_id = flow_config['dest_id']
        traffic_type = flow_config['traffic_type']
        num_packets = flow_config['num_packets']
        delay_req = flow_config.get('delay_req', 5000)  # 默认延迟要求
        qos_req = flow_config.get('qos_req', 0.9)  # 默认QoS要求
        priority = flow_config.get('priority', PriorityLevel.NORMAL)
        start_time = flow_config.get('start_time', 0)
        packet_size = flow_config.get('packet_size', config.DATA_PACKET_LENGTH)
        data_rate = flow_config.get('data_rate', 1.0)

        # 创建基本参数字典
        req_params = {
            'source_id': source_id,
            'dest_id': dest_id,
            'num_packets': num_packets,
            'delay_requirement': delay_req,
            'qos_requirement': qos_req,
            'traffic_type': traffic_type,
            'priority': priority,
            'start_time': start_time,
            'packet_size': packet_size,
            'data_rate': data_rate,
            'simulator': self.simulator
        }

        # 处理特定业务类型的额外参数
        # 注意：我们不直接传递这些参数，而是将它们存储在req_params中处理
        if 'params' in flow_config:
            # 对于有特定参数的业务类型，将它们存储在一个字典中
            # 但不直接传递给构造函数
            extra_params = {}

            if traffic_type == TrafficType.VBR:
                if 'peak_rate' in flow_config['params']:
                    extra_params['peak_rate'] = flow_config['params']['peak_rate']
                    extra_params['min_rate'] = flow_config['params'].get('min_rate', data_rate * 0.5)
            elif traffic_type == TrafficType.BURST:
                extra_params['burst_size'] = flow_config['params'].get('burst_size', 10)
                extra_params['num_bursts'] = flow_config['params'].get('num_bursts', 5)
                extra_params['burst_interval'] = flow_config['params'].get('burst_interval', 100000)
            elif traffic_type == TrafficType.PERIODIC:
                # 对于周期性业务，将period和jitter作为额外信息存储
                extra_params['period_info'] = flow_config['params'].get('period', 1e3)
                extra_params['jitter_info'] = flow_config['params'].get('jitter', 0.1)
            elif traffic_type == TrafficType.POISSON:
                extra_params['lambda_rate'] = flow_config['params'].get('lambda', 500)
            elif traffic_type == TrafficType.PARETO:
                extra_params['alpha'] = flow_config['params'].get('alpha', 1.5)
                extra_params['scale'] = flow_config['params'].get('scale', 100000)
            elif traffic_type == TrafficType.CUSTOM:
                # 自定义参数直接传递
                extra_params.update(flow_config['params'])

            # 将extra_params作为一个单独的参数传递
            req_params['extra_params'] = extra_params

        # 创建业务需求报文
        requirement = self.create_traffic_requirement(**req_params)

        logging.info(f"为{traffic_type.name}类型业务流创建业务需求报文: {source_id}->{dest_id}, 包数: {num_packets}")

        return requirement

    def _delayed_start(self, flow_id, delay):
        """延迟启动业务流"""
        yield self.env.timeout(delay)
        yield self.env.process(self._generate_traffic(flow_id))

    def _generate_traffic(self, flow_id):
        """
        根据业务流类型生成业务

        Args:
            flow_id: 业务流ID
        """
        if flow_id not in self.traffic_flows:
            logging.error(f"未找到业务流 {flow_id}")
            return

        flow_config = self.traffic_flows[flow_id]
        traffic_type = flow_config['traffic_type']
        logging.info(f"开始生成业务流 {flow_id} ({traffic_type.name})")

        # 根据业务类型调用相应的生成方法
        if traffic_type == TrafficType.CBR:
            yield self.env.process(self._generate_cbr_traffic(flow_id))
        elif traffic_type == TrafficType.VBR:
            yield self.env.process(self._generate_vbr_traffic(flow_id))
        elif traffic_type == TrafficType.BURST:
            yield self.env.process(self._generate_burst_traffic(flow_id))
        elif traffic_type == TrafficType.PERIODIC:
            yield self.env.process(self._generate_periodic_traffic(flow_id))
        elif traffic_type == TrafficType.POISSON:
            yield self.env.process(self._generate_poisson_traffic(flow_id))
        elif traffic_type == TrafficType.PARETO:
            yield self.env.process(self._generate_pareto_traffic(flow_id))
        elif traffic_type == TrafficType.CUSTOM:
            yield self.env.process(self._generate_custom_traffic(flow_id))
        else:
            logging.error(f"未知业务类型: {traffic_type}")

        # 完成生成
        logging.info(f"业务流 {flow_id} 生成完成")

        # 移除活跃生成器
        if flow_id in self.active_generators:
            del self.active_generators[flow_id]

        # 更新统计
        if flow_id in self.traffic_stats:
            self.traffic_stats[flow_id].end_time = self.env.now

    def _generate_cbr_traffic(self, flow_id):
        """
        生成恒定比特率(CBR)业务

        Args:
            flow_id: 业务流ID
        """
        flow_config = self.traffic_flows[flow_id]
        source_id = flow_config['source_id']
        dest_id = flow_config['dest_id']
        num_packets = flow_config['num_packets']
        packet_size = flow_config['packet_size']
        data_rate = flow_config['data_rate']  # Mbps
        priority = flow_config['priority']

        # 计算包间隔(ns)
        interval_ns = self._calculate_packet_interval(packet_size, data_rate)

        logging.info(f"CBR业务流 {flow_id}: 间隔={interval_ns}us, 包数={num_packets}")

        # 生成数据包
        for i in range(num_packets):
            success = yield self.env.process(
                self._create_and_send_packet(
                    source_id, dest_id, packet_size, flow_id, priority, i + 1
                )
            )

            # 如果发送失败，可以实现重试逻辑
            if not success:
                logging.warning(f"数据包 {i + 1}/{num_packets} 发送失败")

            # 等待下一个包间隔
            yield self.env.timeout(interval_ns)

            # 检查流是否应该终止(如果设置了duration)
            if self._should_terminate_flow(flow_id):
                logging.info(f"CBR业务流 {flow_id} 达到持续时间限制，提前终止")
                break

    def _generate_vbr_traffic(self, flow_id):
        """
        生成可变比特率(VBR)业务

        Args:
            flow_id: 业务流ID
        """
        flow_config = self.traffic_flows[flow_id]
        source_id = flow_config['source_id']
        dest_id = flow_config['dest_id']
        num_packets = flow_config['num_packets']
        packet_size = flow_config['packet_size']
        base_data_rate = flow_config['data_rate']  # Mbps
        priority = flow_config['priority']
        params = flow_config['params']

        # 获取VBR特定参数
        peak_rate = params.get('peak_rate', base_data_rate * 1)  # 峰值速率
        min_rate = params.get('min_rate', base_data_rate * 0.5)  # 最小速率

        logging.info(f"VBR业务流 {flow_id}: 基准速率={base_data_rate}Mbps, 峰值={peak_rate}Mbps, 包数={num_packets}")

        # 生成数据包
        for i in range(num_packets):
            # 随机变化当前速率
            current_rate = random.uniform(min_rate, peak_rate)

            # 计算当前包间隔
            interval_us = self._calculate_packet_interval(packet_size, current_rate)
            logging.info(f"VBR业务流 {flow_id}: 包 {i + 1}/{num_packets}, 速率={current_rate}Mbps, 间隔={interval_us}us")
            success = yield self.env.process(
                self._create_and_send_packet(
                    source_id, dest_id, packet_size, flow_id, priority, i + 1,
                    metadata={'current_rate': current_rate}
                )
            )

            # 等待下一个包间隔
            yield self.env.timeout(interval_us)

            # 检查流是否应该终止
            if self._should_terminate_flow(flow_id):
                logging.info(f"VBR业务流 {flow_id} 达到持续时间限制，提前终止")
                break

    def _generate_burst_traffic(self, flow_id):
        """
        生成突发业务

        Args:
            flow_id: 业务流ID
        """
        flow_config = self.traffic_flows[flow_id]
        source_id = flow_config['source_id']
        dest_id = flow_config['dest_id']
        num_packets = flow_config['num_packets']
        packet_size = flow_config['packet_size']
        priority = flow_config['priority']
        params = flow_config['params']

        # 获取突发特定参数
        burst_size = params.get('burst_size', 10)  # 每次突发的包数
        num_bursts = params.get('num_bursts', num_packets // burst_size + (1 if num_packets % burst_size else 0))
        burst_interval = params.get('burst_interval', 100000)  # 突发间隔(ns)
        packet_interval = params.get('packet_interval', 1000)  # 包间隔(ns)

        logging.info(f"突发业务流 {flow_id}: 突发数={num_bursts}, 每突发包数={burst_size}, 突发间隔={burst_interval}ns")

        packets_sent = 0

        # 生成突发
        for burst in range(num_bursts):
            # 计算当前突发大小(最后一个可能更小)
            current_burst_size = min(burst_size, num_packets - packets_sent)

            if current_burst_size <= 0:
                break

            logging.info(f"业务流 {flow_id}: 生成第{burst + 1}个突发, 大小={current_burst_size}")

            # 生成当前突发的所有数据包
            for i in range(current_burst_size):
                packet_id = packets_sent + i + 1

                success = yield self.env.process(
                    self._create_and_send_packet(
                        source_id, dest_id, packet_size, flow_id, priority, packet_id,
                        metadata={'burst_id': burst + 1}
                    )
                )

                # 短暂等待，防止拥塞
                yield self.env.timeout(packet_interval)

            packets_sent += current_burst_size

            # 如果完成了所有包，或者流应该终止，则退出
            if packets_sent >= num_packets or self._should_terminate_flow(flow_id):
                logging.info(f"突发业务流 {flow_id} 已完成或达到持续时间限制")
                break

            # 等待下一个突发
            yield self.env.timeout(burst_interval - current_burst_size * packet_interval)

    def _generate_periodic_traffic(self, flow_id):
        """
        生成周期性业务(适合传感器数据等)

        Args:
            flow_id: 业务流ID
        """
        flow_config = self.traffic_flows[flow_id]
        source_id = flow_config['source_id']
        dest_id = flow_config['dest_id']
        num_packets = flow_config['num_packets']
        packet_size = flow_config['packet_size']
        priority = flow_config['priority']
        params = flow_config['params']

        # 获取周期特定参数
        period = params.get('period', 1e3)  # 周期(us)
        jitter = params.get('jitter', 0.1)  # 抖动(周期的比例)

        logging.info(f"周期性业务流 {flow_id}: 周期={period}us, 抖动={jitter * 100}%, 包数={num_packets}")

        # 生成数据包
        for i in range(num_packets):
            success = yield self.env.process(
                self._create_and_send_packet(
                    source_id, dest_id, packet_size, flow_id, priority, i + 1
                )
            )

            # 计算下一个周期(添加抖动)
            next_period = period * (1 + random.uniform(-jitter, jitter))
            yield self.env.timeout(next_period)

            # 检查流是否应该终止
            if self._should_terminate_flow(flow_id):
                logging.info(f"周期性业务流 {flow_id} 达到持续时间限制，提前终止")
                break

    def _generate_poisson_traffic(self, flow_id):
        """
        生成泊松分布业务(适合建模网络访问)

        Args:
            flow_id: 业务流ID
        """
        flow_config = self.traffic_flows[flow_id]
        source_id = flow_config['source_id']
        dest_id = flow_config['dest_id']
        num_packets = flow_config['num_packets']
        packet_size = flow_config['packet_size']
        priority = flow_config['priority']
        params = flow_config['params']
        logging.info(f"泊松业务流 {flow_id}: 包数={num_packets}, 额外参数={params}")
        # 获取泊松特定参数
        lambd = max(500, params.get('lambda', 500))  # 确保 λ>0
        avg_interval = 1e6 / lambd  # 确保单位是 us

        logging.info(f"泊松业务流 {flow_id}: λ={lambd}, 平均间隔={avg_interval}μs, 包数={num_packets}")

        for i in range(num_packets):
            # 发送数据包
            process = self.env.process(
                self._create_and_send_packet(source_id, dest_id, packet_size, flow_id, priority, i + 1)
            )
            yield process  # 确保 `process` 是 SimPy 进程

            # 计算泊松间隔（单位 us）
            next_interval = max(1, random.expovariate(lambd) * 1e6)  # 防止间隔太短

            # 进行时间等待
            yield self.env.timeout(next_interval)  # 如果 SimPy 需要秒，则用 `next_interval / 1e6`

            # 检查流是否应该终止
            if self._should_terminate_flow(flow_id):
                logging.info(f"泊松业务流 {flow_id} 达到持续时间限制，提前终止")
                break

    def _generate_pareto_traffic(self, flow_id):
        """
        生成帕累托分布业务(适合建模自相似流量)

        Args:
            flow_id: 业务流ID
        """
        flow_config = self.traffic_flows[flow_id]
        source_id = flow_config['source_id']
        dest_id = flow_config['dest_id']
        num_packets = flow_config['num_packets']
        packet_size = flow_config['packet_size']
        priority = flow_config['priority']
        params = flow_config['params']

        # 获取帕累托特定参数
        alpha = params.get('alpha', 1.5)  # 形状参数
        scale = params.get('scale', 100000)  # 尺度参数

        logging.info(f"帕累托业务流 {flow_id}: α={alpha}, 尺度={scale}, 包数={num_packets}")

        # 生成数据包
        for i in range(num_packets):
            success = yield self.env.process(
                self._create_and_send_packet(
                    source_id, dest_id, packet_size, flow_id, priority, i + 1
                )
            )

            # 计算下一个帕累托间隔
            # 帕累托分布：X = scale / U^(1/alpha)，其中U是(0,1)上的均匀分布
            u = random.random()
            next_interval = scale / (u ** (1 / alpha))
            yield self.env.timeout(next_interval)

            # 检查流是否应该终止
            if self._should_terminate_flow(flow_id):
                logging.info(f"帕累托业务流 {flow_id} 达到持续时间限制，提前终止")
                break

    def _generate_custom_traffic(self, flow_id):
        """
        生成自定义业务模式

        Args:
            flow_id: 业务流ID
        """
        flow_config = self.traffic_flows[flow_id]
        source_id = flow_config['source_id']
        dest_id = flow_config['dest_id']
        num_packets = flow_config['num_packets']
        packet_size = flow_config['packet_size']
        priority = flow_config['priority']
        params = flow_config['params']

        # 从参数中获取自定义生成函数
        interval_func = params.get('interval_func', lambda i: 100000)  # 默认100μs间隔
        size_func = params.get('size_func', lambda i: packet_size)  # 默认固定大小

        logging.info(f"自定义业务流 {flow_id}: 包数={num_packets}")

        # 生成数据包
        for i in range(num_packets):
            # 计算当前包大小
            current_size = size_func(i)

            success = yield self.env.process(
                self._create_and_send_packet(
                    source_id, dest_id, current_size, flow_id, priority, i + 1
                )
            )

            # 计算下一个间隔
            next_interval = interval_func(i)
            yield self.env.timeout(next_interval)

            # 检查流是否应该终止
            if self._should_terminate_flow(flow_id):
                logging.info(f"自定义业务流 {flow_id} 达到持续时间限制，提前终止")
                break

    def _create_and_send_packet(self, source_id, dest_id, packet_size, flow_id,
                                priority=PriorityLevel.NORMAL, sequence=0, metadata=None):
        """
        创建并发送单个数据包

        Args:
            source_id: 源节点ID
            dest_id: 目标节点ID
            packet_size: 包大小
            flow_id: 流ID
            priority: 优先级
            sequence: 包序列号
            metadata: 额外元数据

        Returns:
            bool: 成功则返回True
        """
        global GLOBAL_DATA_PACKET_ID

        # 更新全局ID
        GLOBAL_DATA_PACKET_ID += 1

        source_drone = self.simulator.drones[source_id]
        dest_drone = self.simulator.drones[dest_id]

        # 创建数据包
        packet = DataPacket(
            src_drone=source_drone,
            dst_drone=dest_drone,
            creation_time=self.env.now,
            data_packet_id=GLOBAL_DATA_PACKET_ID,
            data_packet_length=packet_size,
            simulator=self.simulator,
            priority=priority.value
        )

        # 设置包属性
        packet.transmission_mode = 0  # 单播
        packet.flow_id = flow_id
        packet.sequence = sequence
        packet.waiting_start_time = self.env.now

        # 更新全局数据包生成统计
        self.simulator.metrics.datapacket_generated_num += 1

        # 添加元数据
        if metadata:
            for key, value in metadata.items():
                setattr(packet, key, value)

        # 更新统计信息
        if flow_id in self.traffic_stats:
            self.traffic_stats[flow_id].total_packets += 1
            self.traffic_stats[flow_id].packet_ids.add(GLOBAL_DATA_PACKET_ID)

        # 尝试添加到队列
        success = False
        retries = 0

        while retries <= self.max_retries:
            if source_drone.transmitting_queue.qsize() < source_drone.max_queue_size:
                source_drone.transmitting_queue.put(packet)
                success = True

                # 记录日志(但对大量包只记录少量日志避免过多输出)
                if sequence % 10 == 0 or sequence < 10:
                    logging.info(
                        f"业务流 {flow_id}: 生成数据包 {sequence}/{self.traffic_flows[flow_id]['num_packets']} "
                        f"(ID: {packet.packet_id}), 队列大小: {source_drone.transmitting_queue.qsize()}")
                break
            else:
                # 队列已满，重试或放弃
                logging.warning(
                    f"业务流 {flow_id}: 源节点 {source_id} 队列已满，重试 {retries + 1}/{self.max_retries + 1}")

                if flow_id in self.traffic_stats:
                    self.traffic_stats[flow_id].queue_overflow_count += 1

                # 在重试前等待一小段时间
                yield self.env.timeout(1000)  # 1μs
                retries += 1

        if not success:
            logging.error(f"业务流 {flow_id}: 数据包 {sequence} 丢弃，源节点 {source_id} 队列持续满")

            if flow_id in self.traffic_stats:
                self.traffic_stats[flow_id].dropped_packets += 1

            # 可以选择将包添加到待发送队列
            backlog = self.backlogged_packets.setdefault(source_id, [])
            backlog.append((packet, flow_id))

            # 限制待发送队列大小
            if len(backlog) > 100:
                backlog.pop(0)  # 移除最旧的包
        else:
            if flow_id in self.traffic_stats:
                self.traffic_stats[flow_id].sent_packets += 1

        return success

    def _calculate_packet_interval(self, packet_size, data_rate):
        """
        计算数据包发送间隔

        Args:
            packet_size: 包大小(bit)
            data_rate: 数据速率(Mbps)

        Returns:
            int: 包间隔(us)
        """
        # 计算传输一个包需要的时间(us)
        bits = packet_size
        interval_us = int((bits / (data_rate * 1e6)) * 1e6)  # 1Mbps = 10^6 bps, 结果单位转为us
        return max(interval_us, 1)  # 至少1us

    def _should_terminate_flow(self, flow_id):
        """
        检查业务流是否应该终止

        Args:
            flow_id: 业务流ID

        Returns:
            bool: 如果应该终止则返回True
        """
        if flow_id not in self.traffic_flows:
            return True

        flow_config = self.traffic_flows[flow_id]

        # 检查持续时间
        if flow_config['duration'] and self.env.now >= flow_config['start_time'] + flow_config['duration']:
            return True

        return False

    def stop_traffic_flow(self, flow_id):
        """
        停止业务流生成

        Args:
            flow_id: 业务流ID

        Returns:
            bool: 成功停止返回True
        """
        if flow_id not in self.active_generators:
            logging.warning(f"业务流 {flow_id} 未在运行")
            return False

        # 尝试中断生成器进程
        try:
            self.active_generators[flow_id].interrupt()
        except:
            pass

        # 移除活跃生成器
        del self.active_generators[flow_id]

        # 更新统计
        if flow_id in self.traffic_stats:
            self.traffic_stats[flow_id].end_time = self.env.now

        logging.info(f"停止业务流 {flow_id}")
        return True

    def create_traffic_requirement(self, source_id, dest_id, num_packets,
                                   delay_requirement, qos_requirement, traffic_type=TrafficType.CBR,
                                   priority=PriorityLevel.NORMAL, start_time=0, packet_size=None,
                                   data_rate=None, simulator=None, extra_params=None, **kwargs):
        """
        创建业务需求消息(用于路由准备)

        Args:
            source_id: 源节点ID
            dest_id: 目标节点ID
            num_packets: 数据包数量
            delay_requirement: 延迟要求(ms)
            qos_requirement: QoS要求(0-1)
            traffic_type: 业务类型
            priority: 优先级
            start_time: 开始时间(ns)
            packet_size: 数据包大小
            data_rate: 数据速率
            simulator: 仿真器实例
            extra_params: 特定业务类型的额外参数字典
            **kwargs: 其他参数

        Returns:
            TrafficRequirement: 业务需求对象
        """
        # 创建基本参数字典
        requirement_params = {
            'source_id': source_id,
            'dest_id': dest_id,
            'num_packets': num_packets,
            'delay_requirement': delay_requirement,
            'qos_requirement': qos_requirement,
            'traffic_type': traffic_type,
            'priority': priority,
            'start_time': start_time,
            'simulator': simulator
        }

        # 添加可选参数
        if packet_size is not None:
            requirement_params['packet_size'] = packet_size
        if data_rate is not None:
            requirement_params['data_rate'] = data_rate

        # 添加其他传入的kwargs参数
        requirement_params.update(kwargs)

        # 创建增强型的业务需求
        requirement = EnhancedTrafficRequirement(**requirement_params)

        # 如果有额外参数，将它们作为属性添加到requirement对象
        if extra_params:
            for key, value in extra_params.items():
                setattr(requirement, key, value)


        # 设置源和目标无人机引用
        requirement.src_drone = self.simulator.drones[source_id]
        requirement.dst_drone = self.simulator.drones[dest_id]
        requirement.creation_time = self.env.now

        # 使用源节点的路由协议计算路径
        source_drone = self.simulator.drones[source_id]
        has_route, requirement, _ = source_drone.routing_protocol.next_hop_selection(requirement)

        if has_route:
            # 有路由，提交业务需求
            self.simulator.traffic_generator.submit_traffic_requirement(requirement)
            logging.info(f"业务需求已提交，将使用路由路径: {requirement.routing_path}")
            # 确保路由路径包含源节点
            if requirement.routing_path:
                # 如果路径不为空但不包含源节点，在开头添加源节点
                if requirement.routing_path[0] != source_id:
                    requirement.routing_path.insert(0, source_id)
                    logging.info(f"在路径开头添加了源节点 {source_id}")
        else:
            logging.warning(f"未找到从 {source_id} 到 {dest_id} 的路由路径")
        # try:
        #     # 尝试使用compute_path方法
        #     requirement.routing_path = self.simulator.drones[source_id].routing_protocol.compute_path(
        #         requirement.source_id,
        #         requirement.dest_id,
        #         0  # 选项参数
        #     )
        # except AttributeError:
        #     # 如果路由协议没有compute_path方法，尝试使用dijkstra
        #     logging.warning(
        #         f"路由协议 {type(self.simulator.drones[source_id].routing_protocol).__name__} 不支持compute_path方法，使用dijkstra")
        #     try:
        #         requirement.routing_path = self.simulator.drones[source_id].routing_protocol.dijkstra(
        #             self.simulator.drones[source_id].routing_protocol.calculate_cost_matrix(),
        #             requirement.source_id,
        #             requirement.dest_id,
        #             0
        #         )
        #     except Exception as e:
        #         logging.error(f"计算路由路径失败: {str(e)}")
        #         requirement.routing_path = []  # 设置为空路径

        logging.info(
            f"Time {self.env.now}: Drone {self.simulator.drones[source_id].identifier} routing path: {requirement.routing_path}")
        # if len(requirement.routing_path) != 0:
        #     requirement.routing_path.pop(0)
        logging.info(f"Time {self.env.now}: Drone {self.simulator.drones[source_id].identifier} routing path: {requirement.routing_path}")
        # 存储需求
        self.requirements[requirement.packet_id] = requirement

        logging.info(
            f"创建业务需求: {source_id}->{dest_id}, 包数: {num_packets}")

        return requirement

    def submit_traffic_requirement(self, requirement):
        """
        提交业务需求到源节点

        Args:
            requirement: 业务需求对象

        Returns:
            bool: 成功则返回True
        """
        source_drone = self.simulator.drones[requirement.source_id]

        # 检查队列
        if source_drone.transmitting_queue.qsize() >= source_drone.max_queue_size:
            logging.warning(f"源节点 {requirement.source_id} 队列已满，业务需求提交失败")
            return False

        # 添加到源节点队列
        source_drone.transmitting_queue.put(requirement)

        logging.info(f"业务需求已提交到源节点 {requirement.source_id} 队列")
        return True

    def generate_traffic_for_requirement(self, requirement_id, delayed_start=True):
        """
        为已提交的业务需求生成实际业务流

        Args:
            requirement_id: 业务需求ID
            delayed_start: 是否延迟启动

        Returns:
            str: 流ID，失败返回None
        """
        if requirement_id not in self.requirements:
            logging.error(f"未找到业务需求 {requirement_id}")
            return None

        req = self.requirements[requirement_id]

        # 设置业务流
        flow_id = self.setup_traffic_flow(
            source_id=req.source_id,
            dest_id=req.dest_id,
            traffic_type=req.traffic_type,
            num_packets=req.num_packets,
            packet_size=req.packet_size if hasattr(req, 'packet_size') else None,
            data_rate=req.data_rate if hasattr(req, 'data_rate') else 1.0,
            start_time=req.start_time,
            priority=req.priority,
            delay_req=req.delay_requirement,
            qos_req=req.qos_requirement
        )

        # 启动业务流
        if delayed_start:
            self.start_traffic_flow(flow_id)
        else:
            self.start_traffic_flow(flow_id, immediate=True)

        logging.info(f"为业务需求 {requirement_id} 创建并启动业务流 {flow_id}")
        return flow_id

    def _monitor_traffic_stats(self):
        """定期监控并更新业务流统计"""
        while True:
            yield self.env.timeout(10000)  # 每10ms更新一次

            for flow_id, stats in self.traffic_stats.items():
                if flow_id not in self.traffic_flows:
                    continue

                # 更新接收统计
                received_packets = 0
                total_delay = 0
                min_delay = float('inf')
                max_delay = 0

                for packet_id in stats.packet_ids:
                    if packet_id in self.simulator.metrics.deliver_time_dict:
                        received_packets += 1
                        delay = self.simulator.metrics.deliver_time_dict[packet_id]
                        throughput = np.mean(self.simulator.metrics.throughput_dict[packet_id])/1e3
                        total_delay += delay
                        min_delay = min(min_delay, delay)
                        max_delay = max(max_delay, delay)

                # 更新统计
                stats.received_packets = received_packets
                if received_packets > 0:
                    stats.avg_delay = total_delay / received_packets
                    stats.min_delay = min_delay if min_delay != float('inf') else 0
                    stats.max_delay = max_delay
                    stats.throughout = throughput

    def generate_random_requirements(self, num_requirements=5, auto_start=True):
        """
        随机生成一批业务需求

        Args:
            num_requirements: 需求数量
            auto_start: 是否自动启动对应的业务流

        Returns:
            list: 生成的业务需求ID列表
        """
        n_drones = self.simulator.n_drones
        requirement_ids = []

        for _ in range(num_requirements):
            # 随机选择源和目标
            source_id = random.randint(0, n_drones - 1)
            dest_id = random.randint(0, n_drones - 1)
            while dest_id == source_id:
                dest_id = random.randint(0, n_drones - 1)

            # 随机选择业务类型
            traffic_type = random.choice(list(TrafficType))

            # 随机生成参数
            num_packets = random.randint(50, 500)
            delay_req = random.uniform(500, 5000)  # 0.5-5秒
            qos_req = random.uniform(0.7, 0.99)
            priority = random.choice(list(PriorityLevel))
            start_time = random.randint(0, 100000)  # 0-100ms

            # 创建需求
            req = self.create_traffic_requirement(
                source_id=source_id,
                dest_id=dest_id,
                num_packets=num_packets,
                delay_req=delay_req,
                qos_req=qos_req,
                traffic_type=traffic_type,
                priority=priority,
                start_time=start_time
            )

            requirement_ids.append(req.packet_id)

            # 提交需求
            self.submit_traffic_requirement(req)

            # 自动启动业务流
            if auto_start:
                self.generate_traffic_for_requirement(req.packet_id)

        return requirement_ids

    def get_traffic_stats(self, flow_id=None):
        """
        获取业务流统计信息

        Args:
            flow_id: 业务流ID，None则返回所有

        Returns:
            dict: 统计信息
        """
        if flow_id:
            if flow_id in self.traffic_stats:
                stats = self.traffic_stats[flow_id]
                # 计算PDR
                pdr = stats.received_packets / stats.total_packets if stats.total_packets > 0 else 0
                # 计算吞吐量
                duration = (stats.end_time or self.env.now) - stats.start_time
                throughout = stats.throughout if stats.throughout > 0 else 0
                logging.info(f"业务流 {flow_id}: PDR={pdr}, 吞吐量={throughout},duration={duration}")
                return {
                    'flow_id': flow_id,
                    'total_packets': stats.total_packets,
                    'sent_packets': stats.sent_packets,
                    'received_packets': stats.received_packets,
                    'dropped_packets': stats.dropped_packets,
                    'pdr': pdr,
                    'throughout': throughout,  # bps
                    'avg_delay': stats.avg_delay / 1e3,  # ms (将us转为ms)
                    'min_delay': stats.min_delay / 1e3,  # ms (将us转为ms)
                    'max_delay': stats.max_delay / 1e3,  # ms (将us转为ms)
                    'start_time': stats.start_time,
                    'end_time': stats.end_time or self.env.now,
                    'queue_overflow_count': stats.queue_overflow_count
                }
            return None
        else:
            # 返回所有流的统计
            result = {}
            for fid in self.traffic_stats:
                result[fid] = self.get_traffic_stats(fid)
            return result

    def process_traffic_requirement(self, packet, src_drone_id):
        """
        处理接收到的业务需求消息（用于MAC层回调）

        Args:
            packet: 业务需求包
            src_drone_id: 源无人机ID

        Returns:
            bool: 成功处理返回True
        """
        if not isinstance(packet, TrafficRequirement):
            return False

        logging.info(f"处理来自无人机 {src_drone_id} 的业务需求消息: {packet.source_id}->{packet.dest_id}")

        # 存储需求
        self.requirements[packet.packet_id] = packet

        # 需要创建路径的情况
        if not packet.routing_path and packet.src_drone and packet.dst_drone:
            # 尝试获取路由路径
            routing_protocol = self.simulator.drones[src_drone_id].routing_protocol

            try:
                # 计算路由路径
                cost_matrix = routing_protocol.calculate_cost_matrix()
                path = routing_protocol.dijkstra(cost_matrix, packet.source_id, packet.dest_id, 0)

                if path:
                    packet.routing_path = path
                    logging.info(f"为业务需求 {packet.packet_id} 计算路由路径: {path}")
            except Exception as e:
                logging.error(f"计算路由路径出错: {str(e)}")

        return True

    # 辅助方法 - 创建多种流配置模板
    @classmethod
    def create_cbr_flow(cls, source_id, dest_id, data_rate=1.0, num_packets=100,
                        packet_size=None, priority=PriorityLevel.NORMAL,
                        start_time=0, duration=None):
        """创建CBR流配置"""
        return {
            'source_id': source_id,
            'dest_id': dest_id,
            'traffic_type': TrafficType.CBR,
            'data_rate': data_rate,
            'num_packets': num_packets,
            'packet_size': config.DATA_PACKET_LENGTH,
            'priority': priority,
            'start_time': start_time,
            'duration': duration
        }

    @classmethod
    def create_vbr_flow(cls, source_id, dest_id, mean_rate=1.0, peak_rate=2.0,
                        num_packets=100, packet_size=None, priority=PriorityLevel.NORMAL,
                        start_time=0, duration=None):
        """创建VBR流配置"""
        return {
            'source_id': source_id,
            'dest_id': dest_id,
            'traffic_type': TrafficType.VBR,
            'data_rate': mean_rate,
            'num_packets': num_packets,
            'packet_size': packet_size or config.DATA_PACKET_LENGTH,
            'priority': priority,
            'start_time': start_time,
            'duration': duration,
            'params': {
                'peak_rate': peak_rate,
                'min_rate': mean_rate * 1
            }
        }

    @classmethod
    def create_burst_flow(cls, source_id, dest_id, burst_size=10, num_bursts=5,
                          packet_size=None, priority=PriorityLevel.NORMAL,
                          start_time=0, duration=None):
        """创建突发流配置"""
        return {
            'source_id': source_id,
            'dest_id': dest_id,
            'traffic_type': TrafficType.BURST,
            'num_packets': burst_size * num_bursts,
            'packet_size': packet_size or config.DATA_PACKET_LENGTH,
            'priority': priority,
            'start_time': start_time,
            'duration': duration,
            'params': {
                'burst_size': burst_size,
                'num_bursts': num_bursts,
                'burst_interval': 100000,  # 100ms
                'packet_interval': 1000  # 1ms
            }
        }

    @classmethod
    def create_poisson_flow(cls, source_id, dest_id, lambd=1.0, num_packets=100,
                            packet_size=None, priority=PriorityLevel.NORMAL,
                            start_time=0, duration=None):


        """创建泊松流配置"""
        return {
            'source_id': source_id,
            'dest_id': dest_id,
            'traffic_type': TrafficType.POISSON,
            'num_packets': num_packets,
            'packet_size': packet_size or config.DATA_PACKET_LENGTH,
            'priority': priority,
            'start_time': start_time,
            'duration': duration,
            'params': {
                'lambda': lambd
            }
        }


def generate_traffic(self, source_id, dest_id, num_packets, packet_interval=2000):
    """与旧版API兼容的数据包生成方法"""
    # 创建CBR业务流配置
    # 注意：packet_interval单位是us，需要转换成Mbps
    bits_per_packet = config.DATA_PACKET_LENGTH
    data_rate = (bits_per_packet / packet_interval) * 1  # Mbps (1us = 10^-6s)

    cbr_config = self.create_cbr_flow(
        source_id=source_id,
        dest_id=dest_id,
        data_rate=data_rate,
        num_packets=num_packets,
        packet_size=config.DATA_PACKET_LENGTH,
        priority=PriorityLevel.NORMAL
    )

    # 设置并启动流
    flow_id = self.setup_traffic_flow(**cbr_config)
    self.start_traffic_flow(flow_id, immediate=True)

    return flow_id

class EnhancedTrafficRequirement(TrafficRequirement):
    """增强版的流量需求类，可以追踪活跃状态"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_active = True  # 标记该流是否处于活跃状态
        self.flow_id = f"flow_{self.source_id}_{self.dest_id}"  # 创建唯一的流ID
        self.creation_timestamp = kwargs.get('simulator').env.now if kwargs.get('simulator') else 0  # 创建时间戳


class BatchTrafficManager:
    """
    业务层批量请求管理器
    用于在业务层合并短时间内的多个业务请求
    """

    def __init__(self, simulator):
        """
        初始化批量业务管理器

        Args:
            simulator: 仿真器实例
        """
        self.simulator = simulator
        self.env = simulator.env
        self.traffic_generator = simulator.traffic_generator

        # 批处理相关属性
        self.pending_requirements = []  # 等待处理的业务需求
        self.batch_interval = 5e5  # 批处理时间间隔(0.5秒)
        self.last_batch_time = 0  # 上次批处理时间
        self.is_batch_scheduled = False  # 是否已调度批处理
        self.batch_count = 0  # 批次计数

        # 启动批处理监控进程
        self.env.process(self._batch_monitor())

    def create_traffic_requirement(self, source_id, dest_id, num_packets,
                                   delay_req, qos_req, **kwargs):
        """
        创建业务需求并加入批处理队列

        Args:
            source_id: 源节点ID
            dest_id: 目标节点ID
            num_packets: 数据包数量
            delay_req: 延迟要求(ms)
            qos_req: QoS要求(0-1)
            **kwargs: 其他参数

        Returns:
            requirement: 创建的业务需求对象
        """
        # 创建业务需求
        requirement = self.traffic_generator.create_traffic_requirement(
            source_id=source_id,
            dest_id=dest_id,
            num_packets=num_packets,
            delay_req=delay_req,
            qos_req=qos_req,
            **kwargs
        )

        # 标记为活跃状态
        requirement.is_active = True
        requirement.flow_id = f"flow_{source_id}_{dest_id}"

        # 将需求添加到待处理列表
        self.pending_requirements.append(requirement)

        logging.info(
            f"业务需求 {requirement.flow_id} 已创建并加入批处理队列 (当前队列: {len(self.pending_requirements)})")

        # 如果是首个业务需求或已超过批处理间隔，立即处理
        current_time = self.env.now
        time_since_last = current_time - self.last_batch_time

        if time_since_last >= self.batch_interval or len(self.pending_requirements) == 1:
            # 直接处理当前批次
            self.env.process(self._process_batch())
        elif not self.is_batch_scheduled:
            # 调度一个延迟的批处理
            self.is_batch_scheduled = True
            remaining_time = self.batch_interval - time_since_last
            logging.info(f"调度延迟批处理，将在 {remaining_time / 1e6:.3f} 秒后执行")
            self.env.process(self._delayed_batch_processing(remaining_time))

        return requirement

    def _delayed_batch_processing(self, delay):
        """延迟执行批处理"""
        try:
            yield self.env.timeout(delay)
            yield self.env.process(self._process_batch())
        except Exception as e:
            logging.error(f"延迟批处理出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_batch_scheduled = False

    def _process_batch(self):
        """处理一批业务需求"""
        yield self.env.timeout(10)  # 添加这一行使其成为生成器
        if not self.pending_requirements:
            logging.info("没有待处理的业务需求")
            return

        try:
            # 记录处理开始时间
            process_start = self.env.now

            # 复制当前待处理列表，然后清空原列表
            current_batch = self.pending_requirements.copy()
            self.pending_requirements = []
            self.batch_count += 1

            batch_id = f"batch_{self.batch_count}_{self.env.now}"

            logging.info(f"开始处理批次 {batch_id}: {len(current_batch)} 个业务需求")

            # 为每个需求计算路由路径
            for req in current_batch:
                if not req.routing_path:
                    try:
                        # 尝试使用compute_path方法
                        req.routing_path = self.simulator.drones[req.source_id].routing_protocol.compute_path(
                            req.source_id,
                            req.dest_id,
                            0  # 选项参数
                        )
                    except Exception as e:
                        logging.warning(f"计算路由路径失败: {e}")
                        # 尝试使用dijkstra
                        try:
                            req.routing_path = self.simulator.drones[req.source_id].routing_protocol.dijkstra(
                                self.simulator.drones[req.source_id].routing_protocol.calculate_cost_matrix(),
                                req.source_id,
                                req.dest_id,
                                0
                            )
                        except Exception as e:
                            logging.error(f"计算路由路径失败: {e}")
                            req.routing_path = []

                # if req.routing_path and len(req.routing_path) > 0 and req.routing_path[0] == req.source_id:
                #     req.routing_path = req.routing_path[1:]  # 移除源节点

                logging.info(f"业务 {req.flow_id} 路由路径: {req.routing_path}")

            # 创建批量业务需求
            batch_requirement = self._create_batch_requirement(current_batch, batch_id)

            # 提交批量业务需求到源节点
            for req in current_batch:
                source_drone = self.simulator.drones[req.source_id]

                # 检查队列容量
                if source_drone.transmitting_queue.qsize() >= source_drone.max_queue_size:
                    logging.warning(f"源节点 {req.source_id} 队列已满，业务需求 {req.flow_id} 提交失败")
                    continue

                # 标记为批处理一部分
                req.batch_id = batch_id

                # 添加到源节点队列
                source_drone.transmitting_queue.put(req)
                logging.info(f"业务需求 {req.flow_id} 已提交到源节点 {req.source_id} 队列")

                # 生成实际业务流量
                flow_id = self.traffic_generator.generate_traffic_for_requirement(req.packet_id)
                logging.info(f"为业务需求 {req.flow_id} 创建并启动业务流 {flow_id}")

            # 更新最后批处理时间
            self.last_batch_time = self.env.now

            # 记录处理时间
            process_duration = self.env.now - process_start
            logging.info(f"批次 {batch_id} 处理完成，耗时 {process_duration / 1e6:.3f} 秒")

        except Exception as e:
            logging.error(f"处理业务批次出错: {e}")
            import traceback
            traceback.print_exc()

    def _create_batch_requirement(self, requirements, batch_id):
        """
        创建一个表示整个批次的需求对象
        用于后续查询和管理

        Args:
            requirements: 批次中的业务需求列表
            batch_id: 批次ID

        Returns:
            BatchRequirement: 批次需求对象
        """
        from dataclasses import dataclass, field

        @dataclass
        class BatchRequirement:
            batch_id: str
            requirements: list
            creation_time: int
            source_nodes: set = field(default_factory=set)
            dest_nodes: set = field(default_factory=set)
            all_nodes: set = field(default_factory=set)
            all_paths: list = field(default_factory=list)

        # 收集批次信息
        batch_req = BatchRequirement(
            batch_id=batch_id,
            requirements=requirements,
            creation_time=self.env.now
        )

        # 收集源节点和目标节点
        for req in requirements:
            batch_req.source_nodes.add(req.source_id)
            batch_req.dest_nodes.add(req.dest_id)

            # 收集路径上的所有节点
            if req.routing_path:
                batch_req.all_paths.append(req.routing_path)
                for node in req.routing_path:
                    batch_req.all_nodes.add(node)

        # 将所有来源和目标节点添加到节点集合
        batch_req.all_nodes.update(batch_req.source_nodes)
        batch_req.all_nodes.update(batch_req.dest_nodes)

        return batch_req

    def _batch_monitor(self):
        """定期检查是否有未处理的批次"""

        while True:
            yield self.env.timeout(self.batch_interval)

            if self.pending_requirements and not self.is_batch_scheduled:
                # 如果有未处理的需求且没有调度批处理，启动处理
                logging.info(f"批处理监控检测到 {len(self.pending_requirements)} 个未处理需求")
                self.env.process(self._process_batch())