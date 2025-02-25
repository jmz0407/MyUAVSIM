import random
import numpy as np
from phy.channel import Channel
from entities.drone import Drone
from simulator.metrics import Metrics
from mobility import start_coords
from utils import config
from visualization.scatter import scatter_plot
from simulator.TrafficGenerator import TrafficGenerator
from simulator.TrafficGenerator import TrafficRequirement
import simpy
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

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

            print('UAV: ', i, ' initial location is at: ', start_position[i], ' speed is: ', speed)
            drone = Drone(env=env, node_id=i, coords=start_position[i], speed=speed,
                          inbox=self.channel.create_inbox_for_receiver(i), simulator=self)
            self.drones.append(drone)
        for drone in self.drones:
            self.metrics.energy_consumption[drone.identifier] = 0
        # scatter_plot(self)
        # 创建并初始化MAC协议
        from mac.stdma import Stdma  # 导入STDMA
        # MAC协议列表，存储每个无人机的MAC协议实例
        self.mac_protocols = [drone.mac_protocol for drone in self.drones]
        # 创建业务流生成器
        # 创建业务流生成器
        self.traffic_generator = TrafficGenerator(self)
        self.traffic_manager = SequentialTrafficManager(self)

        # 添加业务流
        self.add_traffic_requirements()
        # 在需要生成业务流时:
        # self.generate_traffic_requirement(
        #     source_id=4,
        #     dest_id=5,
        #     num_packets=10,
        #     delay_req=1000,  # ms
        #     qos_req=0.9,
        #     start_time=1000000
        # )

        # # 在需要生成业务流时:
        # self.generate_traffic_requirement(
        #     source_id=2,
        #     dest_id=7,
        #     num_packets=500,
        #     delay_req=2000,  # ms
        #     qos_req=1,
        #     start_time=0.5*1e5
        # )

        #
        # self.generate_traffic_requirement(
        #     source_id=4,
        #     dest_id=9,
        #     num_packets=100,
        #     delay_req=2000,  # ms
        #     qos_req=1,
        #     start_time=1*1e5,
        # )



        self.env.process(self.show_performance())
        self.env.process(self.show_time())
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

        # scatter_plot(self)

        self.metrics.print_metrics()
        self.metrics.plot_all_metrics()
        self.metrics.plot_energy_metrics()  # 添加能量消耗图表

    def add_traffic_requirements(self):
        """添加所有需要的业务流"""
        # 添加第一条业务流
        self.traffic_manager.add_traffic_requirement(
            source_id=2,
            dest_id=7,
            num_packets=100,
            delay_req=2000,
            qos_req=1
        )

        # 添加第二条业务流
        self.traffic_manager.add_traffic_requirement(
            source_id=4,
            dest_id=9,
            num_packets=50,
            delay_req=2000,
            qos_req=1
        )

        # 启动业务流监控
        self.traffic_manager.start_traffic_monitoring()


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