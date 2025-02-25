import copy
import logging
import math
import numpy as np
from entities.packet import DataPacket
from topology.virtual_force.vf_packet import VfPacket
from utils import config
from utils.util_function import euclidean_distance
from phy.large_scale_fading import maximum_communication_range
from simulator.TrafficGenerator import TrafficRequirement

# config logging
logging.basicConfig(filename='running_log.log',
                    filemode='w',  # there are two modes: 'a' and 'w'
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=config.LOGGING_LEVEL
                    )


class Opar:
    """
    Main procedure of Opar (v3.0)

    Attributes:
        simulator: the simulation platform that contains everything
        my_drone: the drone that installed the routing protocol
        cost: cost matrix, used to record the cost of all links
        best_obj: the minimum objective function value under all iterations
        best_path: optimal routing path corresponding to "best_obj"
        w1: weight of the first term in objective function
        w2: weight of the second term in objective function
        max_comm_range: maximum communication range corresponding to the snr threshold

    References:
        [1] M. Gharib, F. Afghah and E. Bentley, "OPAR: Optimized Predictive and Adaptive Routing for Cooperative UAV
            Networks," in IEEE Conference on Computer Communications Workshops, PP. 1-6, 2021.
        [2] M. Gharib, F. Afghah and E. Bentley, "LB-OPAR: Load Balanced Optimized Predictive and Adaptive Routing for
            Cooperative UAV Networks," Ad hoc Networks, vol. 132, pp. 102878, 2022.

    Author: Zihao Zhou, eezihaozhou@gmail.com
    Created at: 2024/3/19
    Updated at: 2024/9/8
    """

    def __init__(self, simulator, my_drone):
        self.simulator = simulator
        self.my_drone = my_drone
        self.cost = None
        self.best_obj = 0
        self.best_path = None

        self.w1 = 0.5
        self.w2 = 0.5

        self.max_comm_range = maximum_communication_range()
        self.simulator.env.process(self.check_waiting_list())

    def calculate_cost_matrix(self):
        """优化的成本矩阵计算，结合多个因素"""
        n_drones = self.simulator.n_drones
        cost = np.full((n_drones, n_drones), np.inf)

        for i in range(n_drones):
            for j in range(i + 1, n_drones):
                drone1 = self.simulator.drones[i]
                drone2 = self.simulator.drones[j]

                distance = euclidean_distance(drone1.coords, drone2.coords)

                if distance < self.max_comm_range:
                    # 基础距离成本
                    distance_cost = distance / self.max_comm_range

                    # 队列负载成本
                    queue_cost = drone2.transmitting_queue.qsize() / drone2.max_queue_size

                    # 链路稳定性成本
                    lifetime = link_lifetime_predictor(drone1, drone2, self.max_comm_range)
                    stability_cost = 1.0 / (1.0 + lifetime)

                    # 能量成本
                    energy_cost = 1.0 - (drone2.residual_energy / config.INITIAL_ENERGY)

                    # 综合评估
                    total_cost = (
                            0.3 * distance_cost +
                            0.2 * queue_cost +
                            0.3 * stability_cost +
                            0.2 * energy_cost
                    )

                    cost[i, j] = cost[j, i] = total_cost

        return cost

    def dijkstra(self, cost, src_id, dst_id, minimum_link_lifetime):
        """
        Dijkstra's algorithm to find the shortest path
        :param cost: cost matrix
        :param src_id: source node id
        :param dst_id: destination node id
        :param minimum_link_lifetime: used to determine which edges cannot be considered in this iteration
        :return: routing path that has the minimum total cost
        """

        distance_list = [np.inf for _ in range(self.simulator.n_drones)]
        distance_list[src_id] = 0

        prev_list = [-1 for _ in range(self.simulator.n_drones)]
        prev_list[src_id] = -2

        visited_list = [False for _ in range(self.simulator.n_drones)]

        for i in range(self.simulator.n_drones):
            unvisited_list = [(index, value) for index, value in enumerate(distance_list) if not visited_list[index]]
            min_distance_node, _ = min(unvisited_list, key=lambda x: x[1])

            visited_list[min_distance_node] = True

            for j in range(self.simulator.n_drones):
                drone1 = self.simulator.drones[min_distance_node]
                drone2 = self.simulator.drones[j]

                if (visited_list[j] is False) and (cost[min_distance_node, j] != np.inf):
                    delta_temp = link_lifetime_predictor(drone1, drone2, self.max_comm_range)

                    if delta_temp <= minimum_link_lifetime:
                        cost[min_distance_node, j] = np.inf
                        cost[j, min_distance_node] = np.inf

                    alt = distance_list[min_distance_node] + cost[min_distance_node, j]
                    if alt < distance_list[j]:
                        distance_list[j] = alt
                        prev_list[j] = min_distance_node

        # path construction
        current_node = dst_id
        path = [dst_id]

        while current_node != -2:
            current_node = prev_list[current_node]

            if current_node != -1:
                path.insert(0, current_node)
            else:
                path = []
                break

        return path

    def next_hop_selection(self, packet):
        enquire = False
        has_route = True

        def next_hop_selection(self, packet):
            # 记录路由控制包
            if not isinstance(packet, DataPacket):
                self.simulator.metrics.control_packet_num += 1
        if packet.src_drone is self.my_drone:  # if it is the source, optimization should be executed
            self.cost = self.calculate_cost_matrix()
            temp_cost = self.cost
            src_drone = self.my_drone  # packet.src_drone
            dst_drone = packet.dst_drone  # get the destination of the data packet

            path = self.dijkstra(temp_cost, src_drone.identifier, dst_drone.identifier, 0)

            if len(path) != 0:
                path.pop(0)

                total_cost = 0
                t = 0
                minimum_link_lifetime = 1e11

                for link in range(len(path)-1):
                    drone1 = self.simulator.drones[path[link]]
                    drone2 = self.simulator.drones[path[link+1]]
                    link_cost = self.cost[path[link], path[link+1]]
                    total_cost += link_cost

                    delta_t = link_lifetime_predictor(drone1, drone2, self.max_comm_range)

                    if 1/delta_t > t:
                        t = delta_t

                    if delta_t < minimum_link_lifetime:
                        minimum_link_lifetime = delta_t

                # calculate the objective function
                obj = self.w1 * total_cost + self.w2 * t
                self.best_obj = obj
                self.best_path = path
            else:
                minimum_link_lifetime = None
                self.best_path = [src_drone.identifier, src_drone.identifier]

            while len(path) != 0:
                path = self.dijkstra(temp_cost, src_drone.identifier, dst_drone.identifier, minimum_link_lifetime)

                if len(path) != 0:
                    path.pop(0)

                    total_cost = 0
                    t = 0
                    minimum_link_lifetime = 1e11

                    for link in range(len(path) - 1):
                        drone1 = self.simulator.drones[path[link]]
                        drone2 = self.simulator.drones[path[link + 1]]
                        link_cost = self.cost[path[link], path[link + 1]]
                        total_cost += link_cost

                        delta_t = link_lifetime_predictor(drone1, drone2, self.max_comm_range)

                        if 1 / delta_t > t:
                            t = delta_t

                        if delta_t < minimum_link_lifetime:
                            minimum_link_lifetime = delta_t

                    # calculate the objective function
                    obj = self.w1 * total_cost + self.w2 * t

                    if obj < self.best_obj:
                        self.best_obj = obj
                        self.best_path = path

            self.best_path.pop(0)  # remove myself
            packet.routing_path = self.best_path
            best_next_hop_id = self.best_path[0]
            logging.info('my_drone: %s routing Path: %s', self.my_drone.identifier, packet.routing_path)
        else:  # for relay nodes, no additional calculations are required
            routing_path = packet.routing_path
            #业务路径
            if len(routing_path) > 1:
                routing_path.pop(0)
                packet.routing_path = routing_path
                best_next_hop_id = routing_path[0]
            else:
                # if it is passed to itself, it'll try to find the path again the next time the packet is sent
                best_next_hop_id = self.my_drone.identifier

        if best_next_hop_id is self.my_drone.identifier:
            has_route = False  # no available next hop
        else:
            packet.next_hop_id = best_next_hop_id  # it has an available next hop drone

        return has_route, packet, enquire

    def packet_reception(self, packet, src_drone_id):
        """
        Packet reception at network layer, simplified for TDMA without handling ACK.
        :param packet: the received packet
        :param src_drone_id: previous hop
        :return: None
        """
        current_time = self.simulator.env.now
        # logging.info('~~~ Packet: %s is received by UAV: %s at time: %s',
        #              packet.packet_id, self.my_drone.identifier, current_time)

        if isinstance(packet, DataPacket):
            packet_copy = copy.copy(packet)
            logging.info('~~~ DataPacket: %s is received by UAV: %s at time: %s',
                         packet_copy.packet_id, self.my_drone.identifier, current_time)

            # 如果数据包的目标是当前 UAV，则进行处理
            if packet_copy.dst_drone.identifier == self.my_drone.identifier:

                latency = self.simulator.env.now - packet_copy.creation_time  # in us
                self.simulator.metrics.deliver_time_dict[packet_copy.packet_id] = latency
                self.simulator.metrics.throughput_dict[packet_copy.packet_id] = config.DATA_PACKET_LENGTH / (latency / 1e6)
                self.simulator.metrics.hop_cnt_dict[packet_copy.packet_id] = packet_copy.get_current_ttl()
                self.simulator.metrics.datapacket_arrived.add(packet_copy.packet_id)
                logging.info('Latency: %s us, Throughput: %s', latency,
                             self.simulator.metrics.throughput_dict[packet_copy.packet_id])

                # 记录该数据包的相关信息
                logging.info('DataPacket: %s delivered to UAV: %s', packet_copy.packet_id, self.my_drone.identifier)
            else:
                # 如果目标不是当前 UAV，将包放入队列中等待
                if self.my_drone.transmitting_queue.qsize() < self.my_drone.max_queue_size:
                    self.my_drone.transmitting_queue.put(packet_copy)
                else:
                    logging.info('Queue full, DataPacket: %s cannot be added to the queue.', packet_copy.packet_id)

        elif isinstance(packet, VfPacket):
            logging.info('At time %s, UAV: %s receives a VF packet from UAV: %s, packet id: %s',
                         current_time, self.my_drone.identifier, src_drone_id, packet.packet_id)

            # 更新邻居表
            self.my_drone.motion_controller.neighbor_table.add_neighbor(packet, current_time)

            # 如果是 hello 消息，发送一个 ack
            if packet.msg_type == 'hello':
                config.GL_ID_VF_PACKET += 1
                ack_packet = VfPacket(src_drone=self.my_drone,
                                      creation_time=current_time,
                                      id_hello_packet=config.GL_ID_VF_PACKET,
                                      hello_packet_length=config.HELLO_PACKET_LENGTH,
                                      simulator=self.simulator)
                ack_packet.msg_type = 'ack'
                self.my_drone.transmitting_queue.put(ack_packet)


    def check_waiting_list(self):
        while True:
            if not self.my_drone.sleep:
                yield self.simulator.env.timeout(0.6 * 1e6)
                for waiting_pkd in self.my_drone.waiting_list:
                    if self.simulator.env.now < waiting_pkd.creation_time + waiting_pkd.deadline:
                        self.my_drone.waiting_list.remove(waiting_pkd)
                    else:
                        best_next_hop_id = self.next_hop_selection(waiting_pkd)
                        if best_next_hop_id != self.my_drone.identifier:
                            self.my_drone.transmitting_queue.put(waiting_pkd)
                            self.my_drone.waiting_list.remove(waiting_pkd)
                        else:
                            pass
            else:
                break


def link_lifetime_predictor(drone1, drone2, max_comm_range):
    coords1 = drone1.coords
    coords2 = drone2.coords
    velocity1 = drone1.velocity
    velocity2 = drone2.velocity

    x1 = (velocity1[0] - velocity2[0]) ** 2
    x2 = (velocity1[1] - velocity2[1]) ** 2
    x3 = (velocity1[2] - velocity2[2]) ** 2

    y1 = 2*(velocity1[0] - velocity2[0])*(coords1[0] - coords2[0])
    y2 = 2*(velocity1[1] - velocity2[1])*(coords1[1] - coords2[1])
    y3 = 2*(velocity1[2] - velocity2[2])*(coords1[2] - coords2[2])

    z1 = (coords1[0] - coords2[0]) ** 2
    z2 = (coords1[1] - coords2[1]) ** 2
    z3 = (coords1[2] - coords2[2]) ** 2

    A = x1 + x2 + x3

    B = y1 + y2 + y3
    C = (z1 + z2 + z3) - max_comm_range ** 2

    delta_t_1 = (-B + math.sqrt(B ** 2 - 4 * A * C)) / (2 * A)
    delta_t_2 = (-B - math.sqrt(B ** 2 - 4 * A * C)) / (2 * A)

    delta_t = max(delta_t_1, delta_t_2)

    return delta_t
