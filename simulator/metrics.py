import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


class Metrics:
    """
    Tools for statistics of network performance

    1. Packet Delivery Ratio (PDR): is the ratio of number of packets received at the destinations to the number
       of packets sent from the sources
    2. Average end-to-end (E2E) delay: is the time a packet takes to route from a source to its destination through
       the network. It is the time the data packet reaches the destination minus the time the data packet was generated
       in the source node
    3. Routing Load: is calculated as the ratio between the numbers of control Packets transmitted
       to the number of packets actually received. NRL can reflect the average number of control packets required to
       successfully transmit a data packet and reflect the efficiency of the routing protocol
    4. Throughput: it can be defined as a measure of how fast the data is sent from its source to its intended
       destination without loss. In our simulation, each time the destination receives a data packet, the throughput is
       calculated and finally averaged
    5. Hop count: used to record the number of router output ports through which the packet should pass.

    References:
        [1] Rani. N, Sharma. P, Sharma. P., "Performance Comparison of Various Routing Protocols in Different Mobility
            Models," in arXiv preprint arXiv:1209.5507, 2012.
        [2] Gulati M K, Kumar K. "Performance Comparison of Mobile Ad Hoc Network Routing Protocols," International
            Journal of Computer Networks & Communications. vol. 6, no. 2, pp. 127, 2014.

    Author: Zihao Zhou, eezihaozhou@gmail.com
    Created at: 2024/1/11
    Updated at: 2024/8/31
    """

    def __init__(self, simulator):
        self.simulator = simulator

        self.control_packet_num = 0

        self.datapacket_generated = set()  # all data packets generated
        self.datapacket_arrived = set()  # all data packets that arrives the destination
        self.datapacket_generated_num = 0

        self.delivery_time = []
        self.deliver_time_dict = defaultdict()

        self.throughput = []
        self.throughput_dict = defaultdict()

        self.hop_cnt = []
        self.hop_cnt_dict = defaultdict()

        self.mac_delay = []

        self.collision_num = 0

    # def print_metrics(self):
    #     # calculate the average end-to-end delay
    #     for key in self.deliver_time_dict.keys():
    #         self.delivery_time.append(self.deliver_time_dict[key])
    #
    #     for key2 in self.throughput_dict.keys():
    #         self.throughput.append(self.throughput_dict[key2])
    #
    #     for key3 in self.hop_cnt_dict.keys():
    #         self.hop_cnt.append(self.hop_cnt_dict[key3])
    #
    #     e2e_delay = np.mean(self.delivery_time) / 1e3  # unit: ms
    #
    #     pdr = len(self.datapacket_arrived) / self.datapacket_generated_num * 100  # in %
    #
    #     rl = self.control_packet_num / len(self.datapacket_arrived)
    #
    #     throughput = np.mean(self.throughput) / 1e3  # in Kbps
    #
    #     hop_cnt = np.mean(self.hop_cnt)
    #
    #     average_mac_delay = np.mean(self.mac_delay)
    #
    #     print('Totally send: ', self.datapacket_generated_num, ' data packets')
    #     print('Packet delivery ratio is: ', pdr, '%')
    #     print('Average end-to-end delay is: ', e2e_delay, 'ms')
    #     print('Routing load is: ', rl)
    #     print('Average throughput is: ', throughput, 'Kbps')
    #     print('Average hop count is: ', hop_cnt)
    #     print('Collision num is: ', self.collision_num)
    #     print('Average mac delay is: ', average_mac_delay, 'ms')

    # def print_metrics(self):
    #     # 计算平均端到端延迟
    #     for key in self.deliver_time_dict.keys():
    #         self.delivery_time.append(self.deliver_time_dict[key])
    #
    #     for key2 in self.throughput_dict.keys():
    #         self.throughput.append(self.throughput_dict[key2])
    #
    #     print('hop_cnt_dict: ', self.hop_cnt_dict)
    #     for key3 in self.hop_cnt_dict.keys():
    #         self.hop_cnt.append(self.hop_cnt_dict[key3])
    #     print('delivery_time: ', self.delivery_time)
    #     # 计算平均端到端延迟（单位：毫秒）
    #     e2e_delay = np.mean(self.delivery_time) / 1e3  # 单位：ms
    #
    #     # 计算Packet Delivery Ratio（PDR）
    #     print('Received packets: ', len(self.datapacket_arrived))
    #     if self.datapacket_generated_num > 0:
    #         pdr = len(self.datapacket_arrived) / self.datapacket_generated_num * 100  # 百分比
    #     else:
    #         pdr = 0  # 如果没有生成数据包，则PDR为0%
    #
    #     # 计算Routing Load（RL），避免除以零
    #     print('Control packets sent: ', self.control_packet_num)
    #     if len(self.datapacket_arrived) > 0:
    #         rl = self.control_packet_num / len(self.datapacket_arrived)
    #     else:
    #         rl = 0  # 如果没有接收到数据包，则RL为0
    #
    #     # 计算吞吐量（单位：Kbps）
    #     throughput = np.mean(self.throughput) / 1e3  # 单位：Kbps
    #
    #     # 计算平均跳数
    #     hop_cnt = np.mean(self.hop_cnt)
    #
    #     # 计算平均MAC延迟
    #     average_mac_delay = np.mean(self.mac_delay)
    #
    #     # 打印统计信息
    #     print('Totally sent: ', self.datapacket_generated_num, ' data packets')
    #     print('Packet delivery ratio is: ', pdr, '%')
    #     print('Average end-to-end delay is: ', e2e_delay, 'ms')
    #     print('Routing load is: ', rl)
    #     print('Average throughput is: ', throughput, 'Kbps')
    #     print('Average hop count is: ', hop_cnt)
    #     print('Collision num is: ', self.collision_num)
    #     print('Average mac delay is: ', average_mac_delay, 'ms')

    def print_metrics(self):
        """Calculate and print network performance metrics"""

        print("\nDebugging empty variables:")
        print("-" * 50)

        # 清空数组，防止重复添加
        self.delivery_time = []
        self.throughput = []
        self.hop_cnt = []

        # 从字典转移数据到数组
        for key, value in self.deliver_time_dict.items():
            self.delivery_time.append(value)

        for key, value in self.throughput_dict.items():
            self.throughput.append(value)

        for key, value in self.hop_cnt_dict.items():
            self.hop_cnt.append(value)

        # 打印调试信息
        print(f"delivery_time_dict size: {len(self.deliver_time_dict)}")
        print(f"delivery_time array size: {len(self.delivery_time)}")
        if len(self.delivery_time) == 0:
            print("Warning: delivery_time array is empty!")

        print(f"\nthroughput_dict size: {len(self.throughput_dict)}")
        print(f"throughput array size: {len(self.throughput)}")
        if len(self.throughput) == 0:
            print("Warning: throughput array is empty!")

        print(f"\nhop_cnt_dict size: {len(self.hop_cnt_dict)}")
        print(f"hop_cnt array size: {len(self.hop_cnt)}")
        if len(self.hop_cnt) == 0:
            print("Warning: hop_cnt array is empty!")

        print(f"\nmac_delay array size: {len(self.mac_delay)}")
        if len(self.mac_delay) == 0:
            print("Warning: mac_delay array is empty!")

        print(f"\nTotal packets generated: {self.datapacket_generated_num}")
        print(f"Packets arrived: {len(self.datapacket_arrived)}")
        print(f"Control packets: {self.control_packet_num}")

        print("\nCalculating metrics:")
        print("-" * 50)

        # 计算性能指标
        e2e_delay = 0
        if len(self.delivery_time) > 0:
            e2e_delay = np.mean(self.delivery_time) / 1e3

        pdr = 0
        if self.datapacket_generated_num > 0:
            pdr = len(self.datapacket_arrived) / self.datapacket_generated_num * 100

        rl = 0
        if len(self.datapacket_arrived) > 0:
            rl = self.control_packet_num / len(self.datapacket_arrived)

        throughput = 0
        if len(self.throughput) > 0:
            throughput = np.mean(self.throughput) / 1e3

        hop_cnt = 0
        if len(self.hop_cnt) > 0:
            hop_cnt = np.mean(self.hop_cnt)

        mac_delay = 0
        if len(self.mac_delay) > 0:
            mac_delay = np.mean(self.mac_delay)

        print('\nFinal Performance Metrics:')
        print('-' * 50)
        print(f'Total packets sent: {self.datapacket_generated_num}')
        print(f'Packet delivery ratio: {pdr:.2f}%')
        print(f'Average end-to-end delay: {e2e_delay:.2f} ms')
        print(f'Routing load: {rl:.2f}')
        print(f'Average throughput: {throughput:.2f} Kbps')
        print(f'Average hop count: {hop_cnt:.2f}')
        print(f'Total collisions: {self.collision_num}')
        print(f'Average MAC delay: {mac_delay:.2f} ms')