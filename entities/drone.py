import simpy
import numpy as np
import random
import queue
from entities.packet import DataPacket
from routing.mp_olsr.olsr import OlsrHelloPacket,OlsrTcPacket
from routing.multipath.amlb_opar import AMLB_OPAR
from routing.AMLBR.adaptive_multipath.AMLBR import AMLBR
from routing.AMLBR.manager.routing_manager import RoutingManager
from routing.AMLBR.manager.gat_enhanced_routing_manager import GATEnhancedRoutingManager
from routing.AMLBR.adaptive_multipath.MP_AMLBR import MP_AMLBR
# from mac.self_tdma import Stdma
import simpy
import logging
import numpy as np
import random
import math
import queue
from entities.packet import DataPacket
from routing.mp_olsr.olsr import OlsrHelloPacket,OlsrTcPacket
from routing.mp_olsr.direct_olsr import DirectOlsr
from routing.dsdv.dsdv import Dsdv
from routing.gpsr.gpsr import Gpsr
from routing.grad.grad import Grad
from routing.opar.opar import Opar
from routing.mp_dsr.dsr import GlobalDSR
from routing.mp_dsr.mp_dsr import GlobalMPDSR
from routing.multipath.amlb_opar import AMLB_OPAR
from routing.multipath.mp_amlb_opar import MP_AMLB_OPAR
from routing.multipath.mp_gat_opar import MP_GAT_OPAR
from routing.opar.new_opar import NewOpar
from routing.mp_olsr.mp_olsr import MP_OLSR
from routing.q_routing.q_routing import QRouting
from routing.mp_olsr.olsr import Olsr
from mac.csma_ca import CsmaCa
from mac.reuse_slot_tdma import Tdma
from mac.BasicStdma import BasicStdma
from mac.GraphColoringStdma import GraphColoringStdma
# from mac.self_tdma import Stdma
from mac.stdma import Stdma
from mac.FPRP import FPRP
from mac.pure_aloha import PureAloha
from mobility.gauss_markov_3d import GaussMarkov3D
from mobility.random_walk_3d import RandomWalk3D
from mobility.random_waypoint_3d import RandomWaypoint3D
from topology.virtual_force.vf_motion_control import VfMotionController
from energy.energy_model import EnergyModel
from utils import config
from utils.util_function import has_intersection
from phy.large_scale_fading import *
from mac.deep_seek_stdma import DeepSeekStdma
from simulator.TrafficGenerator import TrafficRequirement
from mac.tra_tdma import Tra_Tdma
from mac.stdma_test import Stdma_Test
from routing.opar.last_opar import LastOpar
from mac.stdma import Stdma
from mobility.gauss_markov_3d import GaussMarkov3D
from mobility.random_walk_3d import RandomWalk3D
from topology.virtual_force.vf_motion_control import VfMotionController
from energy.energy_model import EnergyModel
from phy.large_scale_fading import *
from simulator.TrafficGenerator import TrafficRequirement

# config logging
logging.basicConfig(filename='running_log.log',
                    filemode='w',  # there are two modes: 'a' and 'w'
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=config.LOGGING_LEVEL
                    )

GLOBAL_DATA_PACKET_ID = 0


class MultiPathQueue:
    """支持多路径和广播的队列结构"""

    def __init__(self, max_size):
        self.max_size = max_size
        self.queues = {}  # 每个目的地的队列
        self.broadcast_queue = queue.Queue()  # 广播报文队列
        self.total_size = 0

    def put(self, packet):
        """将数据包放入对应队列"""
        if self.total_size >= self.max_size:
            return False

        # 处理广播报文 (VfPacket 等)
        if not hasattr(packet, 'dst_drone'):
            self.broadcast_queue.put(packet)
            self.total_size += 1
            return True

        # 处理单播数据包
        dst_id = packet.dst_drone.identifier
        if dst_id not in self.queues:
            self.queues[dst_id] = queue.Queue()

        self.queues[dst_id].put(packet)
        self.total_size += 1
        return True

    def get(self):
        """获取下一个要发送的数据包"""
        # 优先处理广播队列
        if not self.broadcast_queue.empty():
            self.total_size -= 1
            return self.broadcast_queue.get()

        # 轮询目的地队列
        for dst_id, q in self.queues.items():
            if not q.empty():
                self.total_size -= 1
                return q.get()
        return None

    def qsize(self):
        """返回总队列大小"""
        return self.total_size

    def empty(self):
        """检查是否所有队列都为空"""
        return self.total_size == 0
class Drone:
    """
    Drone implementation

    Drones in the simulation are served as routers. Each drone can be selected as a potential source node, destination
    and relaying node. Each drone needs to install the corresponding routing module, MAC module, mobility module and
    energy module, etc. At the same time, each drone also has its own queue and can only send one packet at a time, so
    subsequent data packets need queuing for queue resources, which is used to reflect the queue delay in the drone
    network

    Attributes:
        simulator: the simulation platform that contains everything
        env: simulation environment created by simpy
        identifier: used to uniquely represent a drone
        coords: the 3-D position of the drone
        start_coords: the initial position of drone
        direction: current direction of the drone
        pitch: current pitch of the drone
        speed: current speed of the drone
        velocity: velocity components in three directions
        direction_mean: mean direction
        pitch_mean: mean pitch
        velocity_mean: mean velocity
        inbox: a "Store" in simpy, used to receive the packets from other drones (calculate SINR)
        buffer: used to describe the queuing delay of sending packet
        transmitting_queue: when the next hop node receives the packet, it should first temporarily store the packet in
                    "transmitting_queue" instead of immediately yield "packet_coming" process. It can prevent the buffer
                    resource of the previous hop node from being occupied all the time
        waiting_list: for reactive routing protocol, if there is no available next hop, it will put the data packet into
                      "waiting_list". Once the routing information bound for a destination is obtained, drone will get
                      the data packets related to this destination, and put them into "transmitting_queue"
        mac_protocol: installed mac protocol (CSMA/CA, ALOHA, etc.)
        mac_process_dict: a dictionary, used to store the mac_process that is launched each time
        mac_process_finish: a dictionary, used to indicate the completion of the process
        mac_process_count: used to distinguish between different "mac_send" processes
        enable_blocking: describe whether the process of waiting for an ACK blocks the delivery of subsequent packets
                         1: stop-and-wait protocol; 0: sliding window (need further implemented)
        routing_protocol: routing protocol installed (GPSR, DSDV, etc.)
        mobility_model: mobility model installed (3-D Gauss-markov, 3-D random waypoint, etc.)
        energy_model: energy consumption model installed
        residual_energy: the residual energy of drone in Joule
        sleep: if the drone is in a "sleep" state, it cannot perform packet sending and receiving operations.

    Author: Zihao Zhou, eezihaozhou@gmail.com
    Created at: 2024/1/11
    Updated at: 2024/9/18
    """

    def __init__(self,
                 env,
                 node_id,
                 coords,
                 speed,
                 inbox,
                 simulator):
        self.simulator = simulator
        self.env = env
        self.identifier = node_id
        self.coords = coords
        self.start_coords = coords

        random.seed(2024 + self.identifier)
        self.direction = random.uniform(0, 2 * np.pi)

        random.seed(2025 + self.identifier)
        self.rng_drone = random.Random(self.identifier + self.simulator.seed)
        self.pitch = random.uniform(-0.05, 0.05)
        self.speed = 20
        self.velocity = [self.speed * math.cos(self.direction) * math.cos(self.pitch),
                         self.speed * math.sin(self.direction) * math.cos(self.pitch),
                         self.speed * math.sin(self.pitch)]

        self.direction_mean = self.direction
        self.pitch_mean = self.pitch
        self.velocity_mean = self.speed

        self.inbox = inbox

        self.buffer = simpy.Resource(env, capacity=1)
        self.max_queue_size = config.MAX_QUEUE_SIZE
        self.transmitting_queue = queue.Queue()  # queue in the real sense
        self.waiting_list = []

        # self.mac_protocol = CsmaCa(self)
        # self.mac_protocol = Tdma(self)  # Use TDMA protocol instead of CSMA/CA
        self.mac_protocol = BasicStdma(self)
        # self.mac_protocol = GraphColoringStdma(self)
        # self.mac_protocol = Stdma(self)
        # self.mac_protocol = Stdma_Test(self)
        # self.mac_protocol = FPRP(self)
        # self.mac_protocol = Tra_Tdma(self)
        self.mac_process_dict = dict()
        self.mac_process_finish = dict()
        self.mac_process_count = 0
        self.enable_blocking = 0 # enable "stop-and-wait" protocol
        self.neighbor_table = None  # 新增邻居表属性


        # self.routing_protocol = Opar(self.simulator, self)
        # self.routing_protocol = MP_GAT_OPAR(simulator, self)
        # self.routing_protocol = LastOpar(self.simulator, self)
        # self.routing_protocol = AMLB_OPAR(self.simulator, self)
        # self.routing_protocol = AMLBR(self.simulator, self)
        # self.routing_protocol = RoutingManager(self.simulator, self)
        # self.routing_protocol = GATEnhancedRoutingManager(self.simulator, self)
        # self.routing_protocol = MP_AMLBR(self.simulator, self)
        # self.routing_protocol = MP_AMLB_OPAR(self.simulator, self)
        # self.routing_protocol = NewOpar(self.simulator, self)
        # self.routing_protocol = MPDSR(self.simulator, self)
        self.routing_protocol = MP_OLSR(self.simulator, self)
        # self.routing_protocol = Olsr(self.simulator, self)
        # self.routing_protocol = DirectOlsr(self.simulator, self)
        # self.routing_protocol = GlobalDSR(self.simulator, self)
        # self.routing_protocol = GlobalMPDSR(self.simulator, self)
        self.mobility_model = RandomWalk3D(self)
        self.mobility_model = GaussMarkov3D(self)
        self.motion_controller = VfMotionController(self)

        self.energy_model = EnergyModel()
        self.residual_energy = config.INITIAL_ENERGY
        self.sleep = False

        # self.env.process(self.generate_data_packet())
        # 添加多路径支持的属性
        self.path_queues = {}  # 每个目的地的多路径队列
        self.max_paths_per_dest = 3  # 每个目的地的最大路径数
        self.current_path_index = {}  # 当前使用的路径索引

        # 修改队列结构
        self.transmitting_queue = MultiPathQueue(self.max_queue_size)

        self.env.process(self.feed_packet())
        self.env.process(self.energy_monitor())
        self.env.process(self.receive())


    # def generate_data_packet(self, traffic_pattern='Poisson'):
    #     """
    #     生成数据包，支持批量业务流生成
    #     Args:
    #         traffic_pattern: 流量模型，支持 'Poisson' 和 'Uniform'
    #     """
    #     global GLOBAL_DATA_PACKET_ID
    #
    #     while True:
    #         if not self.sleep:
    #             # 生成大批量数据的时间间隔
    #             if traffic_pattern == 'Uniform':
    #                 yield self.env.timeout(random.randint(2000000, 2500000))  # 更长的间隔
    #             elif traffic_pattern == 'Poisson':
    #                 rate = 1  # 降低生成频率
    #                 yield self.env.timeout(round(random.expovariate(rate) * 1e6))
    #
    #             # 随机选择目标节点
    #             all_candidate_list = [i for i in range(config.NUMBER_OF_DRONES)]
    #             all_candidate_list.remove(self.identifier)
    #             dst_id = random.choice(all_candidate_list)
    #             destination = self.simulator.drones[dst_id]
    #
    #             # 一次性生成一批数据包(10-20个)
    #             batch_size = random.randint(10, 11)
    #             logging.info(f'UAV: {self.identifier} starts generating batch traffic '
    #                          f'(size: {batch_size}) to dst: {dst_id} at: {self.env.now}')
    #
    #             for i in range(batch_size):
    #                 GLOBAL_DATA_PACKET_ID += 1
    #
    #                 # 创建数据包
    #                 pkd = DataPacket(self,
    #                                  dst_drone=destination,
    #                                  creation_time=self.env.now,
    #                                  data_packet_id=GLOBAL_DATA_PACKET_ID,
    #                                  data_packet_length=config.DATA_PACKET_LENGTH,
    #                                  simulator=self.simulator,
    #                                  priority=1)  # 高优先级数据包
    #
    #                 pkd.transmission_mode = 0  # unicast
    #                 self.simulator.metrics.datapacket_generated_num += 1
    #
    #                 logging.info(f'UAV: {self.identifier} generates packet {i + 1}/{batch_size} '
    #                              f'(id: {pkd.packet_id}, dst: {destination.identifier}) at: {self.env.now}')
    #
    #                 # 加入发送队列
    #                 pkd.waiting_start_time = self.env.now
    #                 if self.transmitting_queue.qsize() < self.max_queue_size:
    #                     self.transmitting_queue.put(pkd)
    #                     logging.info(f'UAV: {self.identifier}, queue size: {self.transmitting_queue.qsize()}')
    #                 else:
    #                     logging.warning(f'UAV: {self.identifier} queue full, batch packet dropped')
    #                     break  # 如果队列满了就停止生成
    #
    #                 # 同一批次的包之间添加小延迟
    #                 yield self.env.timeout(2000)  # 2ms间隔
    #         else:
    #             break
    # def generate_data_packet(self, traffic_pattern='Poisson'):
    #     """
    #     生成一次业务流数据包
    #     Args:
    #         traffic_pattern: 流量模型，支持 'Poisson' 和 'Uniform'，
    #                         虽然只生成一次，但仍保留此参数以保持接口一致性
    #     """
    #     global GLOBAL_DATA_PACKET_ID
    #
    #     if not self.sleep:
    #         # 随机选择目标节点
    #         all_candidate_list = [i for i in range(config.NUMBER_OF_DRONES)]
    #         all_candidate_list.remove(self.identifier)
    #         dst_id = random.choice(all_candidate_list)
    #         destination = self.simulator.drones[dst_id]
    #
    #         # 生成一批数据包(10-20个)
    #         batch_size = random.randint(10, 11)
    #         logging.info(f'UAV: {self.identifier} starts generating batch traffic '
    #                      f'(size: {batch_size}) to dst: {dst_id} at: {self.env.now}')
    #
    #         for i in range(batch_size):
    #             GLOBAL_DATA_PACKET_ID += 1
    #
    #             # 创建数据包
    #             pkd = DataPacket(self,
    #                              dst_drone=destination,
    #                              creation_time=self.env.now,
    #                              data_packet_id=GLOBAL_DATA_PACKET_ID,
    #                              data_packet_length=config.DATA_PACKET_LENGTH,
    #                              simulator=self.simulator,
    #                              priority=1)  # 高优先级数据包
    #
    #             pkd.transmission_mode = 0  # unicast
    #             self.simulator.metrics.datapacket_generated_num += 1
    #
    #             logging.info(f'UAV: {self.identifier} generates packet {i + 1}/{batch_size} '
    #                          f'(id: {pkd.packet_id}, dst: {destination.identifier}) at: {self.env.now}')
    #
    #             # 加入发送队列
    #             pkd.waiting_start_time = self.env.now
    #             if self.transmitting_queue.qsize() < self.max_queue_size:
    #                 self.transmitting_queue.put(pkd)
    #                 logging.info(f'UAV: {self.identifier}, queue size: {self.transmitting_queue.qsize()}')
    #             else:
    #                 logging.warning(f'UAV: {self.identifier} queue full, batch packet dropped')
    #                 break  # 如果队列满了就停止生成
    #
    #             # 同一批次的包之间添加小延迟
    #             yield self.env.timeout(2000)  # 2ms间隔
    def monitor_performance(self):
        """监控网络性能指标"""
        while True:
            yield self.env.timeout(config.MONITOR_INTERVAL)

            # 计算端到端延迟
            delays = self.simulator.metrics.end_to_end_delay
            avg_delay = sum(delays) / len(delays) if delays else 0

            # 计算吞吐量
            throughput = self.simulator.metrics.calculate_throughput()

            # 计算数据包投递率
            pdr = self.simulator.metrics.calculate_pdr()

            # 记录性能指标
            self.simulator.metrics.record_performance(
                avg_delay=avg_delay,
                throughput=throughput,
                pdr=pdr
            )
    def generate_data_packet(self, traffic_pattern='Poisson'):
        """
        Generate one data packet, it should be noted that only when the current packet has been sent can the next
        packet be started. When the drone generates a data packet, it will first put it into the "transmitting_queue",
        the drone reads a data packet from the head of the queue every very short time through "feed_packet()" function.
        :param traffic_pattern: characterize the time interval between generating data packets
        :return: none
        """

        global GLOBAL_DATA_PACKET_ID

        while True:
            if not self.sleep:
                if traffic_pattern == 'Uniform':
                    # the drone generates a data packet every 0.5s with jitter
                    yield self.env.timeout(self.rng_drone.randint(500000, 505000))
                elif traffic_pattern == 'Poisson':
                    """
                    the process of generating data packets by nodes follows Poisson distribution, thus the generation 
                    interval of data packets follows exponential distribution
                    """

                    rate = 2  # on average, how many packets are generated in 1s
                    yield self.env.timeout(round(self.rng_drone.expovariate(rate) * 1e6))

                GLOBAL_DATA_PACKET_ID += 1  # data packet id

                # randomly choose a destination
                all_candidate_list = [i for i in range(config.NUMBER_OF_DRONES)]
                all_candidate_list.remove(self.identifier)
                dst_id = self.rng_drone.choice(all_candidate_list)
                destination = self.simulator.drones[dst_id]  # obtain the destination drone

                pkd = DataPacket(self,
                                 dst_drone=destination,
                                 creation_time=self.env.now,
                                 data_packet_id=GLOBAL_DATA_PACKET_ID,
                                 data_packet_length=config.DATA_PACKET_LENGTH,
                                 simulator=self.simulator)
                pkd.transmission_mode = 0  # the default transmission mode of data packet is "unicast" (0)

                self.simulator.metrics.datapacket_generated_num += 1

                logging.info('------> UAV: %s generates a data packet (id: %s, dst: %s) at: %s, qsize is: %s',
                             self.identifier, pkd.packet_id, destination.identifier, self.env.now,
                             self.transmitting_queue.qsize())

                pkd.waiting_start_time = self.env.now

                if self.transmitting_queue.qsize() < self.max_queue_size:
                    self.transmitting_queue.put(pkd)
                else:
                    # the drone has no more room for new packets
                    pass
            else:  # cannot generate packets if "my_drone" is in sleep state
                break

    def energy_monitor(self):
        """监控能量消耗"""
        while True:
            yield self.env.timeout(1 * 1e5)  # 每0.1秒更新一次

            # 更新能量统计
            self.simulator.metrics.update_energy_consumption(
                self.identifier,
                self.residual_energy
            )

            if self.residual_energy <= config.ENERGY_THRESHOLD:
                self.sleep = True
    def blocking(self):
        """
        Simulate the head-of-line blocking problem, but without waiting for an ACK,
        we simply check if there's any ongoing transmission task.
        :return: none
        """

        if self.enable_blocking:
            # Check if there's a transmission task that is currently being processed.
            # If there is a transmission task in progress, we block the next packet.
            if self.mac_protocol.current_transmission is not None:
                flag = True  # Indicates that a packet is currently being transmitted, block new packet
            else:
                flag = False  # No transmission is in progress, allow new packet
        else:
            flag = False  # If blocking is not enabled, do not block

        return flag


    def feed_packet(self):
        """
        It should be noted that this function is designed for those packets which need to compete for wireless channel

        Firstly, all packets received or generated will be put into the "transmitting_queue", every very short
        time, the drone will read the packet in the head of the "transmitting_queue". Then the drone will check
        if the packet is expired (exceed its maximum lifetime in the network), check the type of packet:
        1) data packet: check if the data packet exceeds its maximum re-transmission attempts. If the above inspection
           passes, routing protocol is executed to determine the next hop drone. If next hop is found, then this data
           packet is ready to transmit, otherwise, it will be put into the "waiting_queue".
        2) control packet: no need to determine next hop, so it will directly start waiting for buffer

        :return: none
        """

        while True:
            if not self.sleep:  # if drone still has enough energy to relay packets
                yield self.env.timeout(10)  # for speed up the simulation

                if not self.blocking():
                    if not self.transmitting_queue.empty():
                        logging.info('UAV: %s has packets in the queue, start to process' % self.identifier)
                        logging.info('UAV: %s, queue size: %s' % (self.identifier, self.transmitting_queue.qsize()))
                        packet = self.transmitting_queue.get()  # get the packet at the head of the queue
                        logging.info(f"Time {self.env.now}: Drone {self.identifier} got {type(packet).__name__}")
                        if isinstance(packet, TrafficRequirement):
                            logging.info(f"Time {self.env.now}: Drone {self.identifier} processing TrafficRequirement")
                            try:
                                # 尝试使用compute_path方法
                                packet.routing_path = self.routing_protocol.compute_path(
                                    packet.source_id,
                                    packet.dest_id,
                                    0  # 选项参数
                                )
                            except AttributeError:
                                # 如果路由协议没有compute_path方法，尝试使用dijkstra
                                logging.warning(
                                    f"路由协议 {type(self.routing_protocol).__name__} 不支持compute_path方法，使用dijkstra")
                                try:
                                    packet.routing_path = self.routing_protocol.dijkstra(
                                        self.routing_protocol.calculate_cost_matrix(),
                                        packet.source_id,
                                        packet.dest_id,
                                        0
                                    )
                                except Exception as e:
                                    logging.error(f"计算路由路径失败: {str(e)}")
                                    packet.routing_path = []  # 设置为空路径

                            logging.info(
                                f"Time {self.env.now}: Drone {self.identifier} routing path: {packet.routing_path}")
                            if len(packet.routing_path) != 0:
                                packet.routing_path.pop(0)
                            logging.info(f"Time {self.env.now}: Drone {self.identifier} routing path: {packet.routing_path}")
                        if self.env.now < packet.creation_time + packet.deadline:  # this packet has not expired
                            if isinstance(packet, DataPacket):
                                if packet.number_retransmission_attempt[self.identifier] < config.MAX_RETRANSMISSION_ATTEMPT:
                                    # it should be noted that "final_packet" may be the data packet itself or a control
                                    # packet, depending on whether the routing protocol can find an appropriate next hop
                                    has_route, final_packet, enquire = self.routing_protocol.next_hop_selection(packet)

                                    if has_route:
                                        logging.info('UAV: %s obtain the next hop: %s of data packet (id: %s)',
                                                     self.identifier, packet.next_hop_id, packet.packet_id)

                                        # in this case, the "final_packet" is actually the data packet
                                        yield self.env.process(self.packet_coming(final_packet))
                                    else:
                                        self.waiting_list.append(packet)
                                        self.remove_from_queue(packet)

                                        if enquire:
                                            # in this case, the "final_packet" is actually the control packet
                                            yield self.env.process(self.packet_coming(final_packet))

                            else:  # control packet but not ack
                                yield self.env.process(self.packet_coming(packet))
                                logging.info(f'{type(packet).__name__}')
                        else:
                            pass  # means dropping this data packet for expiration
            else:  # this drone runs out of energy
                break  # it is important to break the while loop
    def get_neighbors(self):
        """通过运动控制器获取邻居表"""
        if hasattr(self, 'motion_controller'):
            return self.motion_controller.neighbor_table.neighbor_table.keys()
        return []
    def packet_coming(self, pkd):
        """
        When drone has a packet ready to transmit, yield it.

        The requirement of "ready" is:
            1) this packet is a control packet, or
            2) the valid next hop of this data packet is obtained
        :param pkd: packet that waits to enter the buffer of drone
        :return: none
        """

        if not self.sleep:
            logging.info(f"Time {self.env.now}: Drone {self.identifier} starting packet_coming for {type(pkd).__name__}")
            arrival_time = self.env.now
            logging.info('Packet: %s waiting for UAV: %s buffer resource at: %s',
                         pkd.packet_id, self.identifier, arrival_time)
            with self.buffer.request() as request:
                yield request  # wait to enter to buffer

                logging.info('Packet: %s has been added to the buffer at: %s of UAV: %s, waiting time is: %s',
                             pkd.packet_id, self.env.now, self.identifier, self.env.now - arrival_time)

                pkd.number_retransmission_attempt[self.identifier] += 1

                if pkd.number_retransmission_attempt[self.identifier] == 1:
                    pkd.time_transmitted_at_last_hop = self.env.now

                logging.info('Re-transmission times of pkd: %s at UAV: %s is: %s',
                             pkd.packet_id, self.identifier, pkd.number_retransmission_attempt[self.identifier])

                # every time the drone initiates a data packet transmission, "mac_process_count" will be increased by 1
                self.mac_process_count += 1
                logging.info(f'传输{type(pkd).__name__}')
                key = 'mac_send' + str(self.identifier) + '_' + str(pkd.packet_id)
                mac_process = self.env.process(self.mac_protocol.mac_send(pkd))
                self.mac_process_dict[key] = mac_process
                self.mac_process_finish[key] = 0

                yield mac_process
        else:
            pass

    # def energy_monitor(self):
    #     while True:
    #         yield self.env.timeout(1 * 1e5)  # report residual energy every 0.1s
    #         if self.residual_energy <= config.ENERGY_THRESHOLD:
    #             self.sleep = True
    #             # print('UAV: ', self.identifier, ' run out of energy at: ', self.env.now)

    def remove_from_queue(self, data_pkd):
        """
        After receiving the ack packet, drone should remove the data packet that has been acked from its queue
        :param data_pkd: the acked data packet
        :return: none
        """
        temp_queue = queue.Queue()

        while not self.transmitting_queue.empty():
            pkd_entry = self.transmitting_queue.get()
            if pkd_entry != data_pkd:
                temp_queue.put(pkd_entry)

        while not temp_queue.empty():
            self.transmitting_queue.put(temp_queue.get())

    def receive(self):
        """确保所有控制包不受SINR影响，只有数据包受SINR影响"""
        while True:
            if not self.sleep:
                self.update_inbox()
                flag, all_drones_send_to_me, time_span, potential_packet = self.trigger()

                if flag:
                    # 用于存储数据包的发送者和包
                    data_packet_senders = []
                    data_packets = []

                    # 先处理所有控制包，不受SINR影响
                    for i, (sender, packet) in enumerate(zip(all_drones_send_to_me, potential_packet)):
                        # 检查是否是控制包（OLSR控制包或具有msg_type属性的包）
                        is_control_packet = (isinstance(packet, OlsrHelloPacket) or
                                             isinstance(packet, OlsrTcPacket) or
                                             hasattr(packet, 'msg_type'))

                        if is_control_packet:
                            if packet.get_current_ttl() < config.MAX_TTL:
                                # 直接处理控制包，不受SINR影响
                                if isinstance(packet, OlsrHelloPacket) or isinstance(packet, OlsrTcPacket):
                                    # logging.info('OLSR控制消息 %s 从UAV: %s 接收到 UAV: %s (不受SINR影响)',
                                    #              packet.packet_id, sender, self.identifier)
                                    self.routing_protocol.packet_reception(packet, sender)
                                elif hasattr(packet, 'msg_type'):
                                    # logging.info('控制消息 %s (类型: %s) 从UAV: %s 接收到 UAV: %s (不受SINR影响)',
                                    #              packet.packet_id, packet.msg_type, sender, self.identifier)
                                    # 如果有其他控制包处理逻辑，可以在这里添加
                                    # 例如：self.mac_protocol.handle_control_packet(packet, sender)
                                    self.routing_protocol.packet_reception(packet, sender)
                            else:
                                logging.info(f'控制包 {packet.packet_id} 因超过最大TTL而丢弃')
                        else:
                            # 收集数据包，稍后进行SINR检查
                            data_packet_senders.append(sender)
                            data_packets.append(packet)

                    # 如果有数据包，进行SINR检查
                    if data_packets:
                        # 获取数据包发送者列表
                        transmitting_node_list = list(set(data_packet_senders))

                        # 计算SINR
                        sinr_list = stdma_sinr_calculator(self, data_packet_senders, transmitting_node_list)

                        if sinr_list:
                            max_sinr = max(sinr_list)
                            if max_sinr >= config.SNR_THRESHOLD:
                                # 找到SINR最高的数据包
                                which_one = sinr_list.index(max_sinr)
                                pkd = data_packets[which_one]
                                sender = data_packet_senders[which_one]

                                if pkd.get_current_ttl() < config.MAX_TTL:
                                    # 处理数据包
                                    logging.info(
                                        f'数据包 {pkd.packet_id} 从UAV: {sender} 接收到 UAV: {self.identifier}')
                                    self.routing_protocol.packet_reception(pkd, sender)
                                else:
                                    logging.info(f'数据包 {pkd.packet_id} 因超过最大TTL而丢弃')
                            else:
                                # 记录碰撞
                                self.simulator.metrics.collision_num += len(sinr_list)

                                # 打印丢包信息
                                for i, (sinr, packet) in enumerate(zip(sinr_list, data_packets)):
                                    sender = data_packet_senders[i]
                                    logging.warning(
                                        f'干扰丢包: 数据包 {packet.packet_id} 从UAV: {sender} 到 UAV: {self.identifier}, '
                                        f'SINR: {sinr:.2f}dB < 阈值({config.SNR_THRESHOLD}dB), '
                                        f'同时传输节点: {transmitting_node_list}'
                                    )

                                    if isinstance(packet, DataPacket):
                                        logging.warning(
                                            f'干扰包详情: 源={packet.src_drone.identifier}, '
                                            f'目的={packet.dst_drone.identifier}, '
                                            f'下一跳={packet.next_hop_id if hasattr(packet, "next_hop_id") else "None"}, '
                                            f'路径={packet.routing_path if hasattr(packet, "routing_path") else "None"}'
                                        )

                yield self.env.timeout(5)
            else:
                break
    def update_inbox(self):
        """
        Clear the packets that have been processed.
                                           ↓ (current time step)
                              |==========|←- (current incoming packet p1)
                       |==========|←- (packet p2 that has been processed, but also can affect p1, so reserve it)
        |==========|←- (packet p3 that has been processed, no impact on p1, can be deleted)
        --------------------------------------------------------> time
        :return:
        """

        max_transmission_time = (config.DATA_PACKET_LENGTH / config.BIT_RATE) * 1e6  # for a single data packet
        for item in self.inbox:
            insertion_time = item[1]  # the moment that this packet begins to be sent to the channel
            received = item[3]  # used to indicate if this packet has been processed (1: processed, 0: unprocessed)
            if insertion_time + 2 * max_transmission_time < self.env.now:  # no impact on the current packet
                if received:
                    self.inbox.remove(item)

    def trigger(self):
        """
        Detects whether the drone has received a complete data packet
        :return:
        1. flag: bool variable, "1" means a complete data packet has been received by this drone and vice versa
        2. all_drones_send_to_me: a list, including all the sender of the complete data packets received
        3. time_span, a list, the element inside is the time interval in which the received complete data packet is
           transmitted in the channel
        4. potential_packet, a list, including all the instances of the received complete data packet
        """

        flag = 0  # used to indicate if I receive a complete packet
        all_drones_send_to_me = []
        time_span = []
        potential_packet = []

        for item in self.inbox:
            packet = item[0]  # not sure yet whether it has been completely transmitted
            insertion_time = item[1]  # transmission start time
            transmitter = item[2]
            processed = item[3]  # indicate if this packet has been processed
            transmitting_time = packet.packet_length / config.BIT_RATE * 1e6  # expected transmission time

            if not processed:  # this packet has not been processed yet
                if self.env.now >= insertion_time + transmitting_time:  # it has been transmitted completely
                    flag = 1
                    all_drones_send_to_me.append(transmitter)
                    time_span.append([insertion_time, insertion_time + transmitting_time])
                    potential_packet.append(packet)
                    item[3] = 1
                else:
                    pass
            else:
                pass

        return flag, all_drones_send_to_me, time_span, potential_packet
