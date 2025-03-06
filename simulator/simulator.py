import random
import numpy as np
from phy.channel import Channel
from simulator.metrics import Metrics
from mobility import start_coords
from utils import config
from visualization.scatter import scatter_plot
from simulator.TrafficGenerator import TrafficGenerator
from simulator.TrafficGenerator import TrafficRequirement
import simpy
import logging
from utils.util_function import euclidean_distance
from phy.large_scale_fading import maximum_communication_range
from entities.drone import Drone
logging.getLogger('matlotlib').setLevel(logging.WARNING)
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['STFangsong'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
class Simulator:
    """
    Description: simulation environment

    Attributes:
        env: simpy environment
        total_simulation_time: discrete time steps, in nanosecond
        n_drones: number of the drones
        channel_states: a dictionary, used to describe the channel usage
        channel: wireless channel
        metrics: Metrics class, used to record the network performance
        drones: a list, contains all drone instances

    Author: Zihao Zhou, eezihaozhou@gmail.com
    Created at: 2024/1/11
    Updated at: 2024/8/16
    """

    def __init__(self,
                 seed,
                 env,
                 channel_states,
                 n_drones,
                 total_simulation_time=config.SIM_TIME):

        self.env = env
        self.seed = seed
        self.total_simulation_time = total_simulation_time  # total simulation time (ns)

        self.n_drones = n_drones  # total number of drones in the simulation
        print('Total number of drones is: ', n_drones)
        self.channel_states = channel_states
        self.channel = Channel(self.env)

        self.metrics = Metrics(self)  # use to record the network performance

        start_position = start_coords.get_random_start_point_3d(seed)
        self.position = start_position
        # start_position = start_coords.get_custom_start_point_3d(seed)
        self.drones = []
        for i in range(n_drones):
            if config.HETEROGENEOUS:
                speed = random.randint(5, 60)
            else:
                speed = 10

            # print('UAV: ', i, ' initial location is at: ', start_position[i], ' speed is: ', speed)
            drone = Drone(env=env, node_id=i, coords=start_position[i], speed=speed,
                          inbox=self.channel.create_inbox_for_receiver(i), simulator=self)
            self.drones.append(drone)
        for drone in self.drones:
            self.metrics.energy_consumption[drone.identifier] = 0
        scatter_plot(self)
        # 创建并初始化MAC协议
        from mac.stdma import Stdma  # 导入STDMA
        # MAC协议列表，存储每个无人机的MAC协议实例
        self.mac_protocols = [drone.mac_protocol for drone in self.drones]
        # 创建业务流生成器
        self.traffic_generator = TrafficGenerator(self)
        # self.traffic_manager = SequentialTrafficManager(self)
        self.traffic_manager = ParallelTrafficManager(self)
        # 添加业务流
        self.add_traffic_requirements()
        self.global_neighbor_table = {}  # 格式: {drone_id: set(neighbor_ids)}

        # 添加全局二跳邻居表
        self.global_two_hop_neighbors = {}  # 格式: {drone_id: {neighbor_id: set(two_hop_neighbors)}}

        # 添加周期性更新全局邻居表的进程
        self.env.process(self.update_global_neighbor_table())
        # 在需要生成业务流时:
        self.generate_traffic_requirement(
            source_id=4,
            dest_id=5,
            num_packets=500,
            delay_req=1000,  # ms
            qos_req=0.9,
            start_time=1000000
        )

        # 在需要生成业务流时:
        self.generate_traffic_requirement(
            source_id=2,
            dest_id=7,
            num_packets=500,
            delay_req=2000,  # ms
            qos_req=1,
            start_time=0.5*1e5
        )
        #
        #
        # self.generate_traffic_requirement(
        #     source_id=4,
        #     dest_id=9,
        #     num_packets=100,
        #     delay_req=2000,  # ms
        #     qos_req=1,
        #     start_time=1*1e5,
        # )

        # # 初始化多路径集成
        # if hasattr(config, 'MULTIPATH_ENABLED') and config.MULTIPATH_ENABLED:
        #     from routing.multipath.multipath_integration import MultipathIntegration
        #     self.multipath_integration = MultipathIntegration(self)
        #
        #     # 启动多路径性能监控
        #     self.env.process(self.multipath_integration.monitor_network_performance())
        #
        #     # 启动动态调整进程
        #     self.env.process(self.multipath_integration.adjust_multipath_distribution())
        self.env.process(self.print_global_routing_info())
        self.env.process(self.show_performance())
        self.env.process(self.show_time())

    def update_global_neighbor_table(self):
        """定期更新全局邻居表"""
        while True:
            # 完全重建全局邻居表
            self.global_neighbor_table = {}
            self.global_two_hop_neighbors = {}

            # 计算所有无人机之间的邻居关系
            for i in range(self.n_drones):
                drone1 = self.drones[i]
                neighbors = set()

                for j in range(self.n_drones):
                    if i == j:
                        continue  # 跳过自己

                    drone2 = self.drones[j]
                    distance = euclidean_distance(drone1.coords, drone2.coords)

                    # 如果在通信范围内，则视为邻居
                    if distance <= maximum_communication_range():
                        neighbors.add(j)

                # 更新全局邻居表
                self.global_neighbor_table[i] = neighbors

            # 更新全局二跳邻居表
            for i in range(self.n_drones):
                self.global_two_hop_neighbors[i] = {}
                for neighbor_id in self.global_neighbor_table[i]:
                    # 邻居的邻居就是当前节点的二跳邻居
                    two_hop_set = self.global_neighbor_table[neighbor_id] - {i} - self.global_neighbor_table[i]
                    self.global_two_hop_neighbors[i][neighbor_id] = two_hop_set

            # 每隔较短的时间更新一次，如0.5秒
            yield self.env.timeout(0.5 * 1e6)


    def generate_traffic_requirement(self, source_id, dest_id, num_packets,
                                         delay_req, qos_req, start_time=0):
        """生成业务需求消息"""
        requirement = TrafficRequirement(
            source_id=source_id,
            dest_id=dest_id,
            num_packets=num_packets,
            delay_req=delay_req,
            qos_req=qos_req,
            simulator=self,
        )
        requirement.src_drone = self.drones[source_id]
        requirement.dst_drone = self.drones[dest_id]
        requirement.creation_time = self.env.now

        # yield self.env.timeout(start_time)
        # 发送给源节点的MAC层
        source_drone = self.drones[source_id]
        logging.info(f"Queue size before putting requirement: {source_drone.transmitting_queue.qsize()}")
        source_drone.transmitting_queue.put(requirement)

        # 等待时隙分配完成后再生成实际的业务流
        def generate_actual_traffic():
            # 确保时隙已分配
            yield self.env.timeout(100000)

            # 生成实际的数据包流
            self.traffic_generator.generate_traffic(
                source_id=source_id,
                dest_id=dest_id,
                num_packets=num_packets
            )
        logging.info('Generating traffic requirement from %s to %s', source_id, dest_id)
        self.env.process(generate_actual_traffic())
    def show_time(self):
        while True:
            print('At time: ', self.env.now / 1e6, ' s.')
            yield self.env.timeout(0.5*1e6)  # the simulation process is displayed every 0.5s

    def show_performance(self):
        yield self.env.timeout(self.total_simulation_time - 1)

        scatter_plot(self)

        self.metrics.print_metrics()
        self.metrics.plot_all_metrics()
        # self.metrics.plot_energy_metrics()  # 添加能量消耗图表
        # # 绘制特定指标的详细时间序列图
        # self.metrics.plot_metric_over_time('pdr')  # 数据包投递率
        # self.metrics.plot_metric_over_time('delay')  # 端到端延迟
        self.metrics.plot_metric_over_time('throughput')  # 吞吐量
        # self.metrics.plot_metric_over_time('hop_count')  # 跳数
        # self.metrics.plot_metric_over_time('mac_delay')  # MAC延迟

    def print_global_routing_info(self):
        """打印全局路由信息统计"""
        while True:
            yield self.env.timeout(2 * 1e6)  # 每2秒打印一次

            logging.info("=" * 50)
            logging.info("全局路由状态统计:")

            # 统计所有无人机的路由表覆盖率
            total_routes = 0
            possible_routes = self.n_drones * (self.n_drones - 1)  # 所有可能的路由数

            for drone in self.drones:
                # 检查不同类型的路由协议
                route_count = 0

                # 针对MP_OLSR使用routing_table
                if hasattr(drone.routing_protocol, 'routing_table'):
                    route_count = len(drone.routing_protocol.routing_table)
                # 针对AMLB_OPAR使用path_cache
                elif hasattr(drone.routing_protocol, 'path_cache'):
                    route_count = len(drone.routing_protocol.path_cache)

                total_routes += route_count

                # 打印每个无人机的路由表大小
                logging.info(f"UAV {drone.identifier}: {route_count}/{self.n_drones - 1} 路由 "
                             f"({route_count / (self.n_drones - 1) * 100:.1f}%)")

            # 打印整体路由覆盖率
            coverage = total_routes / possible_routes * 100 if possible_routes > 0 else 0
            logging.info(f"整体路由覆盖率: {coverage:.2f}%")

            # 打印全局邻居表大小
            avg_neighbors = sum(len(neighbors) for neighbors in
                                self.global_neighbor_table.values()) / self.n_drones if self.n_drones > 0 else 0
            logging.info(f"平均邻居数: {avg_neighbors:.2f}")

            logging.info("=" * 50)
    def add_traffic_requirements(self):
        """添加所有需要的业务流"""
        # 添加第一条业务流
        # self.traffic_manager.add_traffic_requirement(
        #     source_id=7,
        #     dest_id=2,
        #     num_packets=100,
        #     delay_req=1000,
        #     qos_req=1,
        # )

        # # 添加第二条业务流
        self.traffic_manager.add_traffic_requirement(
            source_id=4,
            dest_id=9,
            num_packets=500,
            delay_req=1000,
            qos_req=1
        )

        # 启动业务流监控
        self.traffic_manager.start_traffic_monitoring()
        # self.traffic_manager.start_monitoring()

class SequentialTrafficManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.traffic_queue = []
        self.current_traffic = None
        self.completed_packets = set()
        self.current_traffic_start_time = None
        self.packet_id_range = {}  # 记录每个业务流的数据包ID范围

    def add_traffic_requirement(self, source_id, dest_id, num_packets, delay_req, qos_req):
        traffic = {
            'source_id': source_id,
            'dest_id': dest_id,
            'num_packets': num_packets,
            'delay_req': delay_req,
            'qos_req': qos_req,
            'completed': False,
            'start_packet_id': None,
            'end_packet_id': None
        }
        self.traffic_queue.append(traffic)
        logging.info(f'Added traffic requirement: {source_id}->{dest_id}, packets: {num_packets}')

    def start_traffic_monitoring(self):
        return self.simulator.env.process(self._monitor_and_generate())

    def _monitor_and_generate(self):
        yield self.simulator.env.timeout(2e6)  # 等待3秒
        logging.info("路由协议收敛后开始业务监控")
        """监控和生成业务流"""
        while True:
            yield self.simulator.env.timeout(1000)  # 每1ms检查一次
            current_time = self.simulator.env.now

            # 没有当前业务流且队列不为空
            if not self.current_traffic and self.traffic_queue:
                self.current_traffic = self.traffic_queue.pop(0)
                self.current_traffic_start_time = current_time
                self._generate_traffic_requirement()
                logging.info(f'Time {current_time / 1e6:.2f}s: Starting new traffic: '
                             f'{self.current_traffic["source_id"]}->{self.current_traffic["dest_id"]}')

            # 有当前业务流
            elif self.current_traffic:
                # 检查完成状态
                completed = self._check_traffic_completion()
                logging.info(f'Time {current_time / 1e6:.2f}s: Current traffic '
                             f'{self.current_traffic["source_id"]}->{self.current_traffic["dest_id"]} '
                             f'completion check result: {completed}')

                if completed:
                    logging.info(f'Time {current_time / 1e6:.2f}s: Traffic from '
                                 f'{self.current_traffic["source_id"]} to '
                                 f'{self.current_traffic["dest_id"]} completed')

                    # 清空当前业务流
                    self.current_traffic = None

                    # 检查是否还有下一条业务流
                    if self.traffic_queue:
                        next_traffic = self.traffic_queue[0]
                        logging.info(f'Time {current_time / 1e6:.2f}s: Next traffic in queue: '
                                     f'{next_traffic["source_id"]}->{next_traffic["dest_id"]}')
                else:
                    # 如果未完成，记录当前状态
                    self._log_traffic_status()

    def _check_traffic_completion(self):
        """检查当前业务流是否完成"""
        if not self.current_traffic:
            return False

        # 检查是否已经开始生成数据包
        if self.current_traffic.get('start_packet_id') is None:
            logging.info('Traffic has not started generating packets yet')
            return False

        dst_id = self.current_traffic['dest_id']
        start_id = self.current_traffic['start_packet_id']
        end_id = self.current_traffic['end_packet_id']

        # 统计已收到的数据包
        received_packets = sum(
            1 for packet_id in self.simulator.metrics.datapacket_arrived
            if (
                    start_id <= packet_id <= end_id and
                    packet_id not in self.completed_packets and
                    packet_id in self.simulator.metrics.deliver_time_dict
            )
        )

        # 详细记录状态
        logging.info(f'Traffic completion check:')
        logging.info(f'- Expected packets: {self.current_traffic["num_packets"]}')
        logging.info(f'- Packet ID range: {start_id} to {end_id}')
        logging.info(f'- Received packets: {received_packets}')
        logging.info(f'- All arrived packets: {sorted(self.simulator.metrics.datapacket_arrived)}')

        # 检查完成条件
        is_completed = received_packets >= self.current_traffic['num_packets']
        logging.info(f'- Completion status: {"Complete" if is_completed else "Incomplete"}')

        return is_completed

    def _generate_traffic_requirement(self):
        traffic = self.current_traffic
        requirement = TrafficRequirement(
            source_id=traffic['source_id'],
            dest_id=traffic['dest_id'],
            num_packets=traffic['num_packets'],
            delay_req=traffic['delay_req'],
            qos_req=traffic['qos_req'],
            simulator=self.simulator
        )

        requirement.src_drone = self.simulator.drones[traffic['source_id']]
        requirement.dst_drone = self.simulator.drones[traffic['dest_id']]
        requirement.creation_time = self.simulator.env.now

        source_drone = self.simulator.drones[traffic['source_id']]
        source_drone.transmitting_queue.put(requirement)

        # 启动实际的业务流生成
        self.simulator.env.process(self._delayed_traffic_generation())

    def _delayed_traffic_generation(self):
        yield self.simulator.env.timeout(100000)  # 等待时隙分配

        if self.current_traffic:
            # 记录起始数据包ID
            current_max_id = max(
                self.simulator.metrics.datapacket_arrived) if self.simulator.metrics.datapacket_arrived else 0
            self.current_traffic['start_packet_id'] = current_max_id + 1

            # 生成业务流
            self.simulator.traffic_generator.generate_traffic(
                source_id=self.current_traffic['source_id'],
                dest_id=self.current_traffic['dest_id'],
                num_packets=self.current_traffic['num_packets']
            )

            # 记录结束数据包ID
            self.current_traffic['end_packet_id'] = self.current_traffic['start_packet_id'] + self.current_traffic[
                'num_packets'] - 1

            logging.info(f'Generated traffic with packet IDs from {self.current_traffic["start_packet_id"]} '
                         f'to {self.current_traffic["end_packet_id"]}')


    def _log_traffic_status(self):
        """记录当前业务流状态"""
        if self.current_traffic:
            elapsed_time = (self.simulator.env.now - self.current_traffic_start_time) / 1e6
            src = self.current_traffic['source_id']
            dst = self.current_traffic['dest_id']

            logging.info(f'Current traffic status after {elapsed_time:.2f}s:')
            logging.info(f'- Source: {src}, Destination: {dst}')
            if self.current_traffic.get('start_packet_id'):
                start_id = self.current_traffic['start_packet_id']
                end_id = self.current_traffic['end_packet_id']
                received = sum(1 for pid in self.simulator.metrics.datapacket_arrived
                               if start_id <= pid <= end_id)
                logging.info(f'- Received: {received}/{self.current_traffic["num_packets"]} packets')


class ParallelTrafficManager:
    """并行业务流管理器，支持同时发送多条业务流"""

    def __init__(self, simulator):
        self.simulator = simulator
        self.traffic_flows = []  # 存储所有业务流
        self.active_flows = {}  # 当前活跃的业务流 {flow_id: flow_data}
        self.completed_flows = {}  # 已完成的业务流
        self.flow_stats = {}  # 业务流统计信息

        # 生成唯一的流ID
        self.next_flow_id = 1

        # 数据包ID追踪
        self.flow_packet_ranges = {}  # {flow_id: (start_id, end_id)}

    def add_traffic_requirement(self, source_id, dest_id, num_packets, delay_req, qos_req,
                                start_time=None, priority=1, auto_start=False):
        """添加业务流需求

        参数:
            source_id: 源节点ID
            dest_id: 目标节点ID
            num_packets: 数据包数量
            delay_req: 延迟要求(ms)
            qos_req: QoS要求(0-1)
            start_time: 开始时间(ns)，如果为None则等待手动启动
            priority: 优先级(1-3)
            auto_start: 是否自动启动
        """
        flow_id = f"flow_{self.next_flow_id}"
        self.next_flow_id += 1

        flow = {
            'flow_id': flow_id,
            'source_id': source_id,
            'dest_id': dest_id,
            'num_packets': num_packets,
            'delay_req': delay_req,
            'qos_req': qos_req,
            'priority': priority,
            'start_time': start_time,
            'completed': False,
            'start_packet_id': None,
            'end_packet_id': None,
            'packets_sent': 0,
            'packets_received': 0
        }

        self.traffic_flows.append(flow)
        logging.info(f'添加业务流: {flow_id} ({source_id}->{dest_id}, {num_packets}包)')

        # 如果设置了自动启动且指定了开始时间
        if auto_start and start_time is not None:
            self.simulator.env.process(self._delayed_start(flow_id, start_time))

        return flow_id

    def _delayed_start(self, flow_id, delay):
        """延迟启动业务流"""
        yield self.simulator.env.timeout(delay)
        self.start_traffic_flow(flow_id)

    def start_all_flows(self):
        """启动所有尚未启动的业务流"""
        for flow in self.traffic_flows:
            if flow['flow_id'] not in self.active_flows and not flow['completed']:
                self.start_traffic_flow(flow['flow_id'])

    def start_traffic_flow(self, flow_id):
        """启动指定的业务流"""
        # 查找业务流
        flow = None
        for f in self.traffic_flows:
            if f['flow_id'] == flow_id:
                flow = f
                break

        if not flow:
            logging.error(f"找不到业务流: {flow_id}")
            return False

        # 避免重复启动
        if flow_id in self.active_flows:
            logging.warning(f"业务流 {flow_id} 已经在活跃状态")
            return False

        # 标记为活跃
        self.active_flows[flow_id] = flow

        # 创建业务需求消息
        self._generate_traffic_requirement(flow)

        logging.info(f"启动业务流: {flow_id} ({flow['source_id']}->{flow['dest_id']})")
        return True

    def _generate_traffic_requirement(self, flow):
        """生成业务需求消息"""
        requirement = TrafficRequirement(
            source_id=flow['source_id'],
            dest_id=flow['dest_id'],
            num_packets=flow['num_packets'],
            delay_req=flow['delay_req'],
            qos_req=flow['qos_req'],
            simulator=self.simulator
        )

        # 设置源和目标节点
        requirement.src_drone = self.simulator.drones[flow['source_id']]
        requirement.dst_drone = self.simulator.drones[flow['dest_id']]
        requirement.creation_time = self.simulator.env.now

        # 发送到源节点队列
        source_drone = self.simulator.drones[flow['source_id']]
        source_drone.transmitting_queue.put(requirement)

        # 延迟后生成实际数据包
        self.simulator.env.process(self._start_packet_generation(flow))

    def _start_packet_generation(self, flow):
        """开始数据包生成"""
        # 等待时隙分配和路由建立
        yield self.simulator.env.timeout(100000)

        # 获取当前最大的数据包ID
        current_max_id = max(
            self.simulator.metrics.datapacket_arrived) if self.simulator.metrics.datapacket_arrived else 0

        # 计算此流的起始和结束ID
        flow['start_packet_id'] = current_max_id + 1
        flow['end_packet_id'] = flow['start_packet_id'] + flow['num_packets'] - 1

        # 记录流的包ID范围，用于后续检查
        self.flow_packet_ranges[flow['flow_id']] = (flow['start_packet_id'], flow['end_packet_id'])

        # 生成数据包流
        self.simulator.traffic_generator.generate_traffic(
            source_id=flow['source_id'],
            dest_id=flow['dest_id'],
            num_packets=flow['num_packets']
        )

        # 初始化流统计信息
        self.flow_stats[flow['flow_id']] = {
            'start_time': self.simulator.env.now,
            'throughput': 0,
            'delay': 0,
            'pdr': 0
        }

        logging.info(f"业务流 {flow['flow_id']} 生成数据包，ID范围: {flow['start_packet_id']} - {flow['end_packet_id']}")

    def start_traffic_monitoring(self):
        """启动业务流监控"""
        return self.simulator.env.process(self._monitor_traffic_flows())

    def _monitor_traffic_flows(self):
        """监控所有业务流的状态"""
        # 等待路由协议收敛
        yield self.simulator.env.timeout(2 * 1e6)
        logging.info("路由协议收敛后开始业务流监控")

        while True:
            yield self.simulator.env.timeout(500000)  # 每0.5秒检查一次
            current_time = self.simulator.env.now

            # 检查每个活跃流的完成状态
            for flow_id, flow in list(self.active_flows.items()):
                completed = self._check_flow_completion(flow)

                if completed:
                    logging.info(f"业务流 {flow_id} ({flow['source_id']}->{flow['dest_id']}) 已完成!")
                    # 更新统计信息
                    self._update_flow_stats(flow_id)
                    # 从活跃流移到已完成流
                    flow['completed'] = True
                    self.completed_flows[flow_id] = flow
                    del self.active_flows[flow_id]
                else:
                    # 更新进度
                    self._log_flow_status(flow_id)

            # 监控和分析数据
            if self.active_flows or self.completed_flows:
                self._analyze_network_performance()

    def _check_flow_completion(self, flow):
        """检查业务流是否完成"""
        # 检查是否已经开始生成数据包
        if flow.get('start_packet_id') is None:
            return False

        start_id = flow['start_packet_id']
        end_id = flow['end_packet_id']

        # 统计已接收的数据包
        received_packets = sum(
            1 for packet_id in self.simulator.metrics.datapacket_arrived
            if start_id <= packet_id <= end_id
        )

        # 更新接收计数
        flow['packets_received'] = received_packets

        # 判断完成条件 (接收到90%以上的包，或者超过预期数量的80%且5秒内无新增)
        completion_threshold = min(flow['num_packets'], int(flow['num_packets'] * 0.9))

        if received_packets >= completion_threshold:
            return True

        # 检查是否长时间没有新的包到达
        flow_stats = self.flow_stats.get(flow['flow_id'])
        if (flow_stats and
                received_packets >= int(flow['num_packets'] * 0.8) and
                self.simulator.env.now - flow_stats.get('last_packet_time', 0) > 5 * 1e6):
            logging.warning(f"业务流 {flow['flow_id']} 已超过5秒无新包到达，视为完成")
            return True

        return False

    def _log_flow_status(self, flow_id):
        """记录业务流状态"""
        flow = self.active_flows[flow_id]

        if flow.get('start_packet_id') is not None:
            progress = (flow['packets_received'] / flow['num_packets']) * 100
            logging.info(
                f"业务流 {flow_id} 状态: 接收 {flow['packets_received']}/{flow['num_packets']} 包 ({progress:.1f}%)")

            # 更新统计信息
            self._update_flow_stats(flow_id)

    def _update_flow_stats(self, flow_id):
        """更新业务流统计信息"""
        if flow_id not in self.flow_stats:
            return

        flow = None
        if flow_id in self.active_flows:
            flow = self.active_flows[flow_id]
        elif flow_id in self.completed_flows:
            flow = self.completed_flows[flow_id]
        else:
            return

        if flow.get('start_packet_id') is None:
            return

        start_id = flow['start_packet_id']
        end_id = flow['end_packet_id']

        # 找出这个流的所有已接收数据包
        received_packets = [
            packet_id for packet_id in self.simulator.metrics.datapacket_arrived
            if start_id <= packet_id <= end_id
        ]

        if received_packets:
            # 更新最后接收包的时间
            self.flow_stats[flow_id]['last_packet_time'] = self.simulator.env.now

            # 计算平均延迟
            delays = [
                self.simulator.metrics.deliver_time_dict.get(packet_id, 0)
                for packet_id in received_packets
            ]
            avg_delay = sum(delays) / len(delays) if delays else 0

            # 计算吞吐量 (bps)
            elapsed_time = (self.simulator.env.now - self.flow_stats[flow_id]['start_time']) / 1e6  # 秒
            if elapsed_time > 0:
                throughput = (len(received_packets) * flow.get('packet_size', 1024 * 8)) / elapsed_time
            else:
                throughput = 0

            # 计算PDR
            pdr = len(received_packets) / flow['num_packets'] if flow['num_packets'] > 0 else 0

            # 更新统计信息
            self.flow_stats[flow_id].update({
                'packets_received': len(received_packets),
                'delay': avg_delay / 1e3,  # 毫秒
                'throughput': throughput,
                'pdr': pdr
            })

    def _analyze_network_performance(self):
        """分析网络性能"""
        # 计算所有活跃流的平均统计信息
        active_flow_ids = list(self.active_flows.keys())
        completed_flow_ids = list(self.completed_flows.keys())

        if not (active_flow_ids or completed_flow_ids):
            return

        all_flow_stats = []
        for flow_id in active_flow_ids + completed_flow_ids:
            if flow_id in self.flow_stats:
                all_flow_stats.append(self.flow_stats[flow_id])

        if all_flow_stats:
            # 计算平均延迟、吞吐量和PDR
            avg_delay = sum(stats.get('delay', 0) for stats in all_flow_stats) / len(all_flow_stats)
            avg_throughput = sum(stats.get('throughput', 0) for stats in all_flow_stats) / len(all_flow_stats)
            avg_pdr = sum(stats.get('pdr', 0) for stats in all_flow_stats) / len(all_flow_stats)

            # 每隔一段时间打印一次汇总统计
            if self.simulator.env.now % (2 * 1e6) < 1000:  # 大约每2秒打印一次
                logging.info(f"网络性能统计 - 活跃流: {len(active_flow_ids)}, 已完成流: {len(completed_flow_ids)}")
                logging.info(
                    f"平均延迟: {avg_delay:.2f}ms, 平均吞吐量: {avg_throughput:.2f}bps, 平均PDR: {avg_pdr:.2f}")