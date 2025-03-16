import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from utils import config
import matplotlib.pyplot as plt
import numpy as np
from simulator.TemporalMetrics import TemporalMetrics
import matplotlib
import logging
matplotlib.font_manager._log.setLevel(logging.WARNING)
plt.rcParams['font.sans-serif']=['STFangsong'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
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
        self.end_to_end_delay = []

        # 添加能量消耗监控
        self.energy_consumption = {}  # 记录每个无人机的能量消耗
        self.energy_history = defaultdict(list)  # 记录能量消耗历史
        self.energy_timestamps = defaultdict(list)  # 记录对应的时间戳
        # 初始化每个无人机的能量监控
        # 添加时间序列指标收集器
        self.temporal_metrics = TemporalMetrics(simulator, self)

        # 启动性能监控进程
        if simulator and hasattr(simulator, 'env'):
            self.simulator.env.process(self.periodic_metrics_record())
        self.collision_num = 0

    def update_energy_consumption(self, drone_id, current_energy):
        """更新无人机的能量消耗"""
        initial_energy = config.INITIAL_ENERGY
        energy_consumed = initial_energy - current_energy
        self.energy_consumption[drone_id] = energy_consumed

        # 记录历史数据
        self.energy_history[drone_id].append(energy_consumed)
        self.energy_timestamps[drone_id].append(self.simulator.env.now / 1e6)  # 转换为秒
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
        # 添加能量消耗统计
        print('\nEnergy Consumption Statistics:')
        print('-' * 50)
        total_energy = sum(self.energy_consumption.values())
        avg_energy = total_energy / len(self.energy_consumption)
        max_energy = max(self.energy_consumption.values())
        min_energy = min(self.energy_consumption.values())

        print(f'Total network energy consumption: {total_energy:.2f} J')
        print(f'Average energy consumption per UAV: {avg_energy:.2f} J')
        print(f'Maximum energy consumption: {max_energy:.2f} J')
        print(f'Minimum energy consumption: {min_energy:.2f} J')
        print(f'Energy consumption variance: '
              f'{np.var(list(self.energy_consumption.values())):.2f} J²')
    def reset(self):
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

    def calculate_throughput(self):
        """
        Calculate the throughput of the network
        """
        throughput = 0
        for key, value in self.throughput_dict.items():
            throughput += value
        return throughput

    def calculate_pdr(self, simulator):
        """计算数据包交付率"""
        if simulator.metrics.datapacket_generated_num > 0:
            return len(simulator.metrics.datapacket_arrived) / simulator.metrics.datapacket_generated_num
        return 0

    def periodic_metrics_record(self):
        """定期记录指标，用于绘制时间序列图"""
        # 每0.5秒记录一次指标
        while True:
            yield self.simulator.env.timeout(0.5 * 1e6)
            self.temporal_metrics.snapshot()

    def record_packet_reception(self, packet_id, latency, hop_count):
        """记录数据包接收信息，支持时间序列分析"""
        self.datapacket_arrived.add(packet_id)
        self.deliver_time_dict[packet_id] = latency
        self.hop_cnt_dict[packet_id] = hop_count

        # 更新时间序列指标
        self.temporal_metrics.snapshot()

    def record_multipath_metrics(self, flow_id, path_id, delay, success=True):
        """记录多路径传输指标"""
        if not hasattr(self, 'multipath_metrics'):
            self.multipath_metrics = {
                'path_usage': defaultdict(lambda: defaultdict(int)),
                'path_delays': defaultdict(lambda: defaultdict(list)),
                'path_success': defaultdict(lambda: defaultdict(int)),
            }

        # 记录路径使用次数
        self.multipath_metrics['path_usage'][flow_id][path_id] += 1

        # 记录延迟
        self.multipath_metrics['path_delays'][flow_id][path_id].append(delay)

        # 记录成功或失败
        if success:
            self.multipath_metrics['path_success'][flow_id][path_id] += 1

    def plot_energy_metrics(self):
        """绘制能量消耗相关的图表"""
        plt.figure(figsize=(15, 10))

        # 1. 能量消耗随时间变化图
        plt.subplot(2, 1, 1)
        for drone_id in self.energy_history:
            plt.plot(
                self.energy_timestamps[drone_id],
                self.energy_history[drone_id],
                label=f'UAV-{drone_id}'
            )
        plt.title('Energy Consumption Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy Consumed (J)')
        plt.grid(True)
        plt.legend()

        # 2. 最终能量消耗柱状图
        plt.subplot(2, 1, 2)
        drone_ids = list(self.energy_consumption.keys())
        energy_values = list(self.energy_consumption.values())

        plt.bar(drone_ids, energy_values)
        plt.title('Total Energy Consumption per UAV')
        plt.xlabel('UAV ID')
        plt.ylabel('Energy Consumed (J)')
        plt.grid(True)

        # 添加数值标签
        for i, v in enumerate(energy_values):
            plt.text(i, v, f'{v:.1f}J', ha='center', va='bottom')

        plt.tight_layout()
        plt.figure()
        plt.show()

        # 使用时间序列能量图
        self.temporal_metrics.plot_energy_over_time()

    def plot_all_metrics(self):
        """
        一次性绘制所有网络性能指标（吞吐量、端到端时延、跳数、MAC延迟）的折线图
        所有图将显示在同一窗口内，横轴是时间（time）。
        """

        # 将 delivery_time(单位ns) 转换为 ms(毫秒)
        e2e_delay_array = np.array(self.delivery_time) / 1e3  # 转成 ms
        throughput_array = np.array(self.throughput)  # 假设单位是 bps，或者别的单位
        hop_count_array = np.array(self.hop_cnt)  # 跳数
        mac_delay_array = np.array(self.mac_delay)  # MAC 层延迟 (单位自定)

        # 获取最小的数组长度，以确保x和y的维度一致
        min_length = min(len(e2e_delay_array), len(throughput_array), len(hop_count_array), len(mac_delay_array))

        if min_length > 0:
            # 截取数据到最小长度
            e2e_delay_array = e2e_delay_array[:min_length]
            throughput_array = throughput_array[:min_length]
            hop_count_array = hop_count_array[:min_length]
            mac_delay_array = mac_delay_array[:min_length]

            # 创建时间轴：假设每个仿真步骤代表 1 秒
            # 使用仿真环境的时间（如果有的话），否则简单使用步长索引
            time_steps = np.arange(min_length)  # 使用最小长度作为时间步的长度

            # 如果你有环境时间（如 self.env.now），可以替换这行代码
            # time_steps = self.env.now + np.arange(min_length)

            # 创建图形窗口并设置大小
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # 2行2列，4个子图

            # --- 1) 端到端时延折线图 ---
            axs[0, 0].plot(time_steps, e2e_delay_array, marker='o', color='b', label='End-to-End Delay')
            axs[0, 0].set_title('End-to-End Delay over Time')
            axs[0, 0].set_xlabel('Time (s)')  # 横轴是时间（秒）
            axs[0, 0].set_ylabel('Delay (ms)')
            axs[0, 0].grid(True)
            axs[0, 0].legend()

            # --- 2) 吞吐量折线图 ---
            axs[0, 1].plot(time_steps, throughput_array, marker='x', color='g', label='Throughput')
            axs[0, 1].set_title('Throughput over Time')
            axs[0, 1].set_xlabel('Time (s)')  # 横轴是时间（秒）
            axs[0, 1].set_ylabel('Throughput (bps)')
            axs[0, 1].grid(True)
            axs[0, 1].legend()

            # --- 3) 跳数（Hop Count）折线图 ---
            axs[1, 0].plot(time_steps, hop_count_array, marker='s', color='r', label='Hop Count')
            axs[1, 0].set_title('Hop Count over Time')
            axs[1, 0].set_xlabel('Time (s)')  # 横轴是时间（秒）
            axs[1, 0].set_ylabel('Hop Count')
            axs[1, 0].grid(True)
            axs[1, 0].legend()

            # --- 4) MAC 延迟折线图 ---
            axs[1, 1].plot(time_steps, mac_delay_array, marker='^', color='m', label='MAC Delay')
            axs[1, 1].set_title('MAC Delay over Time')
            axs[1, 1].set_xlabel('Time (s)')  # 横轴是时间（秒）
            axs[1, 1].set_ylabel('Delay (us)')
            axs[1, 1].grid(True)
            axs[1, 1].legend()

            # 调整布局以避免子图重叠
            plt.tight_layout()
            plt.figure()
            plt.show()

        # 使用时间序列指标绘制更丰富的图表
        self.temporal_metrics.plot_all_metrics()
        plt.savefig('all_metrics_over_time.png', dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形以释放内存

    def plot_metric_over_time(self, metric_name):
        """
        绘制单个指标随时间变化的详细图表

        Args:
            metric_name: 指标名称，例如 'pdr', 'delay', 'throughput', 等
        """
        self.temporal_metrics.plot_metric_comparison(metric_name)


