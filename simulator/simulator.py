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
        # start_position = start_coords.get_custom_start_point_3d(seed)
        self.drones = []
        for i in range(n_drones):
            if config.HETEROGENEOUS:
                speed = random.randint(5, 60)
            else:
                speed = 0.1

            print('UAV: ', i, ' initial location is at: ', start_position[i], ' speed is: ', speed)
            drone = Drone(env=env, node_id=i, coords=start_position[i], speed=speed,
                          inbox=self.channel.create_inbox_for_receiver(i), simulator=self)
            self.drones.append(drone)

        scatter_plot(self)
        # 创建业务流生成器
        # 创建业务流生成器
        self.traffic_generator = TrafficGenerator(self)

        # 生成业务流：从节点0到节点1生成10个数据包，间隔10微秒
        # self.traffic_generator.generate_traffic(
        #     source_id=9,
        #     dest_id=12,
        #     num_packets=10,
        #     packet_interval=10000
        # )
        # 在需要生成业务流时:
        self.generate_traffic_requirement(
            source_id=12,
            dest_id=9,
            num_packets=10,
            delay_req=1000,  # ms
            qos_req=0.9
        )

        self.generate_traffic_requirement(
            source_id=4,
            dest_id=10,
            num_packets=10,
            delay_req=1000,  # ms
            qos_req=0.9
        )

        self.env.process(self.show_performance())
        self.env.process(self.show_time())
    def generate_traffic_requirement(self, source_id, dest_id, num_packets,
                                         delay_req, qos_req):
        """生成业务需求消息"""
        requirement = TrafficRequirement(
            source_id=source_id,
            dest_id=dest_id,
            num_packets=num_packets,
            delay_req=delay_req,
            qos_req=qos_req
        )
        requirement.creation_time = self.env.now

        # 发送给源节点的MAC层
        source_drone = self.drones[source_id]
        source_drone.transmitting_queue.put(requirement)

        # 等待时隙分配完成后再生成实际的业务流
        def generate_actual_traffic():
            # 确保时隙已分配
            yield self.env.timeout(500000)

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
