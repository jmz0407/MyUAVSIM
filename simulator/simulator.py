import random
import numpy as np
from phy.channel import Channel
from simulator.metrics import Metrics
from mobility import start_coords
from utils import config
from visualization.scatter import scatter_plot
# 移除旧的导入，仅保留新版生成器
from simulator.improved_traffic_generator import TrafficGenerator, TrafficType, PriorityLevel, TrafficRequirement
import simpy
import logging
from utils.util_function import euclidean_distance
from phy.large_scale_fading import maximum_communication_range
from entities.drone import Drone

logging.getLogger('matlotlib').setLevel(logging.WARNING)
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['STFangsong']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


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

        self.drones = []
        for i in range(n_drones):
            if config.HETEROGENEOUS:
                speed = random.randint(5, 60)
            else:
                speed = 10

            drone = Drone(env=env, node_id=i, coords=start_position[i], speed=speed,
                          inbox=self.channel.create_inbox_for_receiver(i), simulator=self)
            self.drones.append(drone)

        for drone in self.drones:
            self.metrics.energy_consumption[drone.identifier] = 0
        scatter_plot(self)

        # 创建并初始化MAC协议
        from mac.stdma import Stdma
        self.mac_protocols = [drone.mac_protocol for drone in self.drones]

        # 创建改进的业务流生成器
        self.traffic_generator = TrafficGenerator(self)

        # 添加全局邻居表和二跳邻居表
        self.global_neighbor_table = {}  # 格式: {drone_id: set(neighbor_ids)}
        self.global_two_hop_neighbors = {}  # 格式: {drone_id: {neighbor_id: set(two_hop_neighbors)}}

        # 启动各种进程
        self.env.process(self.update_global_neighbor_table())
        self.env.process(self.print_global_routing_info())
        self.env.process(self.show_performance())
        self.env.process(self.show_time())
        # self.env.process(self.drones[0].routing_protocol.visualize_network())

        # 设置业务流 - 在初始化末尾运行
        self.env.process(self.setup_demo_traffic_flows())

    def generate_traffic_requirement(self, source_id, dest_id, num_packets,
                                     delay_req, qos_req, start_time=0, priority=None,
                                     traffic_type=None):
        """
        生成业务需求消息 - 与旧API兼容的方法
        """
        # 设置默认值
        if traffic_type is None:
            traffic_type = TrafficType.CBR

        if priority is None:
            priority = PriorityLevel.NORMAL

        # 创建业务需求
        requirement = self.traffic_generator.create_traffic_requirement(
            source_id=source_id,
            dest_id=dest_id,
            num_packets=num_packets,
            delay_req=delay_req,
            qos_req=qos_req,
            traffic_type=traffic_type,
            priority=priority,
            start_time=start_time
        )

        # 设置源和目标无人机
        requirement.src_drone = self.drones[source_id]
        requirement.dst_drone = self.drones[dest_id]

        # 提交业务需求
        self.traffic_generator.submit_traffic_requirement(requirement)


        # 为业务需求生成实际业务流
        self.traffic_generator.generate_traffic_for_requirement(requirement.packet_id)

        return requirement

    def setup_demo_traffic_flows(self):
        """设置演示用的多种业务流"""
        # 等待2秒后开始生成业务流，确保路由已充分建立
        # yield self.env.timeout(2 * 1e6)

        # # 例1: 标准CBR流
        self.generate_traffic_requirement(
            source_id=4,
            dest_id=5,
            num_packets=200,
            delay_req=1000,
            qos_req=0.9,
            start_time=0,
            traffic_type=TrafficType.CBR
        )

        # 例2: 使用不同API创建VBR流 - 每0.5秒后启动一个
        yield self.env.timeout(0.5 * 1e6)

        vbr_config = self.traffic_generator.create_vbr_flow(
            source_id=4,
            dest_id=5,
            num_packets=150,
            mean_rate=2,
            peak_rate=4,
            priority=PriorityLevel.HIGH
        )
        vbr_flow_id = self.traffic_generator.setup_traffic_flow(**vbr_config)
        self.traffic_generator.start_traffic_flow(vbr_flow_id)

        # 例3: 使用直接配置创建突发流 - 再等0.5秒启动
        yield self.env.timeout(0.5 * 1e6)

        burst_flow_id = self.traffic_generator.setup_traffic_flow(
            source_id=4,
            dest_id=5,
            traffic_type=TrafficType.BURST,
            num_packets=100,
            packet_size=2048,  # 使用更大的数据包
            priority=PriorityLevel.CRITICAL,
            params={
                'burst_size': 10,
                'num_bursts': 10,
                'burst_interval': 200000,  # 200ms
                'packet_interval': 1000  # 1ms
            }
        )
        self.traffic_generator.start_traffic_flow(burst_flow_id)

        # 例4: 创建泊松分布流量 - 再等0.5秒启动
        yield self.env.timeout(0.5 * 1e6)

        poisson_flow_id = self.traffic_generator.setup_traffic_flow(
            source_id=6,
            dest_id=7,
            traffic_type=TrafficType.POISSON,
            num_packets=200,
            priority=PriorityLevel.NORMAL,
            params={
                'lambda': 2000.0  # 平均每秒2个包
            }
        )
        self.traffic_generator.start_traffic_flow(poisson_flow_id)

        # 如果有超过8个节点，还可以创建周期性流
        if self.n_drones > 8:
            # 例5: 创建周期性流量 - 再等0.5秒启动
            yield self.env.timeout(0.5 * 1e6)

            periodic_flow_id = self.traffic_generator.setup_traffic_flow(
                source_id=8,
                dest_id=9,
                traffic_type=TrafficType.PERIODIC,
                num_packets=150,
                priority=PriorityLevel.LOW,
                params={
                    'period': 50000,  # 50ms周期
                    'jitter': 0.1  # 10%抖动
                }
            )
            self.traffic_generator.start_traffic_flow(periodic_flow_id)

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

    def show_time(self):
        """显示仿真时间"""
        while True:
            print('At time: ', self.env.now / 1e6, ' s.')
            yield self.env.timeout(0.5 * 1e6)  # 每0.5s显示一次仿真进度

    def show_performance(self):
        """在仿真结束时展示性能"""
        yield self.env.timeout(self.total_simulation_time - 1)

        scatter_plot(self)

        # 获取业务流统计并打印
        flow_stats = self.traffic_generator.get_traffic_stats()
        print("\n业务流统计信息:")
        for flow_id, stats in flow_stats.items():
            if stats:
                print(f"\n流 {flow_id}:")
                print(
                    f"  发送/接收/丢弃: {stats.get('sent_packets', 0)}/{stats.get('received_packets', 0)}/{stats.get('dropped_packets', 0)}")
                print(f"  PDR: {stats.get('pdr', 0) * 100:.2f}%")
                print(f"  吞吐量: {stats.get('throughout', 0):.2f} Kbps")
                print(f"  平均延迟: {stats.get('avg_delay', 0):.2f} ms")

        # 打印全局数据包统计
        print(f"\n全局数据包统计:")
        print(f"  生成的数据包总数: {self.metrics.datapacket_generated_num}")
        print(f"  成功接收的数据包数: {len(self.metrics.datapacket_arrived)}")

        # 安全计算PDR
        if self.metrics.datapacket_generated_num > 0:
            pdr = len(self.metrics.datapacket_arrived) / self.metrics.datapacket_generated_num * 100
            print(f"  全局PDR: {pdr:.2f}%")
        else:
            print(f"  全局PDR: N/A (无生成数据包)")

        # 生成图表
        self.metrics.print_metrics()
        self.metrics.plot_all_metrics()
        self.metrics.plot_metric_over_time('throughput')  # 吞吐量

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