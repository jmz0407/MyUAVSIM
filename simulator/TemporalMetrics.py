import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from utils import config


class TemporalMetrics:
    """
    用于记录仿真期间各项指标随时间变化的增强型指标收集器
    支持滑动平均，更好地展示趋势

    功能：
    1. 跟踪所有指标随时间的变化
    2. 支持定期自动采样
    3. 提供可视化工具，包括滑动平均
    4. 兼容现有 Metrics 类结构
    """

    def __init__(self, simulator, parent_metrics):
        """
        初始化时间序列指标收集器

        Args:
            simulator: 仿真器实例
            parent_metrics: 父级Metrics实例
        """
        self.simulator = simulator
        self.parent_metrics = parent_metrics

        # 按时间采样的指标数据
        self.time_points = []

        # 数据包指标随时间的变化
        self.pdr_values = []  # 数据包投递率
        self.e2e_delay_values = []  # 端到端延迟
        self.throughput_values = []  # 吞吐量
        self.hop_count_values = []  # 跳数
        self.routing_load_values = []  # 路由负载
        self.mac_delay_values = []  # MAC延迟
        self.collision_values = []  # 碰撞次数

        # 能量消耗随时间的变化
        self.energy_timestamps = defaultdict(list)  # 能量记录的时间戳
        self.energy_values = defaultdict(list)  # 无人机能量消耗历史

        # 用于吞吐量计算的临时变量
        self.last_packet_count = 0
        self.last_time_point = 0

        # 采样间隔（秒）
        self.sampling_interval = 0.5  # 默认每0.5秒采样一次

        # 滑动平均窗口大小
        self.window_size = 5  # 默认5个点的窗口

        # 启动自动采样进程
        if simulator and hasattr(simulator, 'env'):
            self.simulator.env.process(self._automatic_sampling())

        # # 配置matplotlib避免PyCharm显示问题
        # self.configure_matplotlib_for_pycharm()

    def configure_matplotlib_for_pycharm(self):
        """配置matplotlib以避免PyCharm显示问题"""
        import matplotlib
        import os

        # 检测是否在PyCharm中运行
        in_pycharm = 'PYCHARM_HOSTED' in os.environ or 'JUPYTER_HOSTED' in os.environ

        if in_pycharm:
            # 尝试使用TkAgg后端（通常更稳定）
            try:
                matplotlib.use('TkAgg')
            except:
                # 如果失败，回退到Agg（非交互式）
                matplotlib.use('Agg')
                print("警告: 使用非交互式后端，图表将保存到文件而不显示")

        # 设置全局保存图表的参数
        matplotlib.rcParams['savefig.dpi'] = 300
        matplotlib.rcParams['figure.autolayout'] = True

    def moving_average(self, data, window_size=None):
        """
        计算滑动平均

        Args:
            data: 输入数据列表
            window_size: 窗口大小，None表示使用默认值

        Returns:
            滑动平均后的数据列表
        """
        if window_size is None:
            window_size = self.window_size

        # 如果数据点不足，返回原始数据
        if len(data) < window_size:
            return data

        # 计算滑动平均
        weights = np.ones(window_size) / window_size
        ma_data = np.convolve(data, weights, mode='valid')

        # 填充前面的数据点（保持长度一致）
        padding = [data[0]] * (len(data) - len(ma_data))
        ma_data = np.concatenate([padding, ma_data])

        return ma_data

    def snapshot(self, current_time=None):
        """
        采集当前时刻所有指标的快照

        Args:
            current_time: 当前时间（秒），如果未提供则从模拟器获取
        """
        if current_time is None:
            current_time = self.simulator.env.now / 1e6  # 转换为秒

        # 记录时间点
        self.time_points.append(current_time)

        # 计算数据包投递率(PDR)
        if self.parent_metrics.datapacket_generated_num > 0:
            pdr = len(self.parent_metrics.datapacket_arrived) / self.parent_metrics.datapacket_generated_num * 100
        else:
            pdr = 0
        self.pdr_values.append(pdr)

        # 计算端到端延迟
        recent_delays = []
        for key, value in self.parent_metrics.deliver_time_dict.items():
            if key in self.parent_metrics.datapacket_arrived:
                recent_delays.append(value)

        if recent_delays:
            e2e_delay = np.mean(recent_delays) / 1e3  # 转换为毫秒
        else:
            e2e_delay = 0 if not self.e2e_delay_values else self.e2e_delay_values[-1]
        self.e2e_delay_values.append(e2e_delay)

        # 计算吞吐量（每秒成功传输的比特数）
        current_packet_count = len(self.parent_metrics.datapacket_arrived)
        time_diff = current_time - self.last_time_point if self.last_time_point > 0 else self.sampling_interval

        if time_diff > 0:
            packet_diff = current_packet_count - self.last_packet_count
            throughput = (packet_diff * config.DATA_PACKET_LENGTH) / (time_diff * 1e3)  # 转换为Kbps
        else:
            throughput = 0

        self.throughput_values.append(throughput)
        self.last_packet_count = current_packet_count
        self.last_time_point = current_time

        # 计算跳数
        recent_hops = []
        for key, value in self.parent_metrics.hop_cnt_dict.items():
            if key in self.parent_metrics.datapacket_arrived:
                recent_hops.append(value)

        if recent_hops:
            hop_count = np.mean(recent_hops)
        else:
            hop_count = 0 if not self.hop_count_values else self.hop_count_values[-1]
        self.hop_count_values.append(hop_count)

        # 计算路由负载（控制包与数据包的比率）
        if len(self.parent_metrics.datapacket_arrived) > 0:
            routing_load = self.parent_metrics.control_packet_num / len(self.parent_metrics.datapacket_arrived)
        else:
            routing_load = 0
        self.routing_load_values.append(routing_load)

        # 计算MAC延迟
        if self.parent_metrics.mac_delay:
            mac_delay = np.mean(self.parent_metrics.mac_delay)
        else:
            mac_delay = 0 if not self.mac_delay_values else self.mac_delay_values[-1]
        self.mac_delay_values.append(mac_delay)

        # 记录碰撞次数
        self.collision_values.append(self.parent_metrics.collision_num)

    def update_energy(self, drone_id, energy_consumed, timestamp=None):
        """
        更新无人机能量消耗记录

        Args:
            drone_id: 无人机ID
            energy_consumed: 已消耗的能量
            timestamp: 时间戳（秒），如果未提供则从模拟器获取
        """
        if timestamp is None:
            timestamp = self.simulator.env.now / 1e6  # 转换为秒

        self.energy_timestamps[drone_id].append(timestamp)
        self.energy_values[drone_id].append(energy_consumed)

    def _automatic_sampling(self):
        """自动周期性采样进程"""
        while True:
            # 采集一次快照
            self.snapshot()

            # 等待下一个采样周期
            yield self.simulator.env.timeout(self.sampling_interval * 1e6)  # 转换为微秒

    def plot_all_metrics(self):
        """绘制所有指标随时间变化的图表，包括滑动平均"""
        if not self.time_points:
            print("没有可用的时间序列数据")
            return

        # 创建一个有6个子图的图表
        fig, axs = plt.subplots(3, 2, figsize=(15, 18))
        axs = axs.flatten()

        # 计算滑动平均
        pdr_ma = self.moving_average(self.pdr_values)
        e2e_delay_ma = self.moving_average(self.e2e_delay_values)
        throughput_ma = self.moving_average(self.throughput_values)
        hop_count_ma = self.moving_average(self.hop_count_values)
        routing_load_ma = self.moving_average(self.routing_load_values)
        mac_delay_ma = self.moving_average(self.mac_delay_values)

        # 1. PDR随时间变化
        axs[0].plot(self.time_points, self.pdr_values, 'b-', linewidth=1.5, marker='o', markersize=3, alpha=0.6,
                    label='原始数据')
        axs[0].plot(self.time_points, pdr_ma, 'r-', linewidth=2.5, label=f'滑动平均 (窗口={self.window_size})')
        axs[0].set_title('数据包投递率(PDR)随时间变化')
        axs[0].set_xlabel('仿真时间 (秒)')
        axs[0].set_ylabel('PDR (%)')
        axs[0].grid(True)
        axs[0].legend()

        # 2. 端到端延迟随时间变化
        axs[1].plot(self.time_points, self.e2e_delay_values, 'g-', linewidth=1.5, marker='s', markersize=3, alpha=0.6,
                    label='原始数据')
        axs[1].plot(self.time_points, e2e_delay_ma, 'r-', linewidth=2.5, label=f'滑动平均 (窗口={self.window_size})')
        axs[1].set_title('端到端延迟随时间变化')
        axs[1].set_xlabel('仿真时间 (秒)')
        axs[1].set_ylabel('延迟 (毫秒)')
        axs[1].grid(True)
        axs[1].legend()

        # 3. 吞吐量随时间变化
        axs[2].plot(self.time_points, self.throughput_values, 'r-', linewidth=1.5, marker='^', markersize=3, alpha=0.6,
                    label='原始数据')
        axs[2].plot(self.time_points, throughput_ma, 'b-', linewidth=2.5, label=f'滑动平均 (窗口={self.window_size})')
        axs[2].set_title('吞吐量随时间变化')
        axs[2].set_xlabel('仿真时间 (秒)')
        axs[2].set_ylabel('吞吐量 (Kbps)')
        axs[2].grid(True)
        axs[2].legend()

        # 4. 跳数随时间变化
        axs[3].plot(self.time_points, self.hop_count_values, 'm-', linewidth=1.5, marker='d', markersize=3, alpha=0.6,
                    label='原始数据')
        axs[3].plot(self.time_points, hop_count_ma, 'r-', linewidth=2.5, label=f'滑动平均 (窗口={self.window_size})')
        axs[3].set_title('平均跳数随时间变化')
        axs[3].set_xlabel('仿真时间 (秒)')
        axs[3].set_ylabel('跳数')
        axs[3].grid(True)
        axs[3].legend()

        # 5. 路由负载随时间变化
        axs[4].plot(self.time_points, self.routing_load_values, 'c-', linewidth=1.5, marker='*', markersize=5,
                    alpha=0.6, label='原始数据')
        axs[4].plot(self.time_points, routing_load_ma, 'r-', linewidth=2.5, label=f'滑动平均 (窗口={self.window_size})')
        axs[4].set_title('路由负载随时间变化')
        axs[4].set_xlabel('仿真时间 (秒)')
        axs[4].set_ylabel('控制包/数据包')
        axs[4].grid(True)
        axs[4].legend()

        # 6. 碰撞次数和MAC延迟
        ax6_1 = axs[5]
        ax6_2 = ax6_1.twinx()

        # 计算碰撞增量
        collision_increments = [self.collision_values[0]] + [self.collision_values[i] - self.collision_values[i - 1] for
                                                             i in range(1, len(self.collision_values))]
        collision_increments_ma = self.moving_average(collision_increments)

        # 碰撞次数（柱状图）
        width = 0.4
        if len(self.time_points) > 30:  # 避免柱状图过密
            sample_interval = len(self.time_points) // 30
            sampled_times = self.time_points[::sample_interval]
            sampled_collisions = collision_increments[::sample_interval]

            ax6_1.bar(sampled_times, sampled_collisions, width=width, color='orange', alpha=0.7, label='碰撞增量')
        else:
            ax6_1.bar(self.time_points, collision_increments, width=width, color='orange', alpha=0.7, label='碰撞增量')

        # MAC延迟（线图）
        ax6_2.plot(self.time_points, self.mac_delay_values, 'k-', linewidth=1.5, marker='x', markersize=4, alpha=0.6,
                   label='MAC延迟')
        ax6_2.plot(self.time_points, mac_delay_ma, 'r-', linewidth=2.5, label=f'MAC延迟滑动平均')

        ax6_1.set_title('碰撞次数和MAC延迟随时间变化')
        ax6_1.set_xlabel('仿真时间 (秒)')
        ax6_1.set_ylabel('碰撞增量')
        ax6_2.set_ylabel('MAC延迟 (毫秒)')
        ax6_1.legend(loc='upper left')
        ax6_2.legend(loc='upper right')
        ax6_1.grid(True)

        # 调整布局
        plt.tight_layout()

        # 保存图表并尝试显示
        plt.savefig('all_metrics_over_time.png', dpi=300, bbox_inches='tight')
        try:
            plt.show()
        except Exception as e:
            print(f"无法显示图表，已保存到文件: all_metrics_over_time.png")
            print(f"错误信息: {e}")
        finally:
            plt.close()

    def plot_energy_over_time(self):
        """绘制能量消耗随时间变化的图表，包括滑动平均"""
        if not self.energy_values:
            print("没有可用的能量数据")
            return

        plt.figure(figsize=(12, 8))

        # 为每个无人机绘制能量消耗曲线
        for drone_id in self.energy_values:
            timestamps = self.energy_timestamps[drone_id]
            energy_values = self.energy_values[drone_id]

            if timestamps and energy_values:
                # 原始数据
                plt.plot(timestamps, energy_values, linestyle='-', linewidth=1.5, marker='.',
                         markersize=4, alpha=0.6, label=f'无人机-{drone_id} (原始)')

                # 如果数据点足够多，计算滑动平均
                if len(energy_values) >= self.window_size:
                    energy_ma = self.moving_average(energy_values)
                    plt.plot(timestamps, energy_ma, linestyle='-', linewidth=2.5,
                             label=f'无人机-{drone_id} (滑动平均)')

        plt.title('无人机能量消耗随时间变化')
        plt.xlabel('仿真时间 (秒)')
        plt.ylabel('能量消耗 (J)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # 保存图表并尝试显示
        plt.savefig('energy_consumption_over_time.png', dpi=300)
        try:
            plt.show()
        except Exception as e:
            print(f"无法显示图表，已保存到文件: energy_consumption_over_time.png")
            print(f"错误信息: {e}")
        finally:
            plt.close()

    def plot_metric_comparison(self, metric_name):
        """
        绘制指定指标随时间变化的详细图表，包括滑动平均

        Args:
            metric_name: 指标名称，可选值：'pdr', 'delay', 'throughput',
                         'hop_count', 'routing_load', 'mac_delay', 'collisions'
        """
        if not self.time_points:
            print("没有可用的时间序列数据")
            return

        plt.figure(figsize=(12, 7))

        # 根据指标名称选择数据和标题
        if metric_name == 'pdr':
            plt.plot(self.time_points, self.pdr_values, 'b-', linewidth=1.5, marker='o', markersize=4, alpha=0.6,
                     label='原始数据')
            plt.title('数据包投递率(PDR)随时间变化')
            plt.ylabel('PDR (%)')

            # 添加滑动平均线
            pdr_ma = self.moving_average(self.pdr_values)
            plt.plot(self.time_points, pdr_ma, 'r-', linewidth=2.5,
                     label=f'滑动平均 (窗口={self.window_size})')

            # 添加趋势线
            if len(self.time_points) > 5:
                z = np.polyfit(self.time_points, self.pdr_values, 1)
                p = np.poly1d(z)
                plt.plot(self.time_points, p(self.time_points), "g--", linewidth=1.5,
                         label=f'趋势线 (斜率: {z[0]:.4f})')

            plt.legend()

        elif metric_name == 'delay':
            plt.plot(self.time_points, self.e2e_delay_values, 'g-', linewidth=1.5, marker='s', markersize=4, alpha=0.6,
                     label='原始数据')
            plt.title('端到端延迟随时间变化')
            plt.ylabel('延迟 (毫秒)')

            # 添加滑动平均线
            delay_ma = self.moving_average(self.e2e_delay_values)
            plt.plot(self.time_points, delay_ma, 'r-', linewidth=2.5,
                     label=f'滑动平均 (窗口={self.window_size})')

            # 高亮显示延迟峰值
            threshold = np.mean(self.e2e_delay_values) + np.std(self.e2e_delay_values)
            peak_indices = [i for i, v in enumerate(self.e2e_delay_values) if v > threshold]

            if peak_indices:
                peak_times = [self.time_points[i] for i in peak_indices]
                peak_values = [self.e2e_delay_values[i] for i in peak_indices]
                plt.scatter(peak_times, peak_values, color='red', s=100, marker='*', zorder=5, label='延迟峰值')

            plt.legend()

        elif metric_name == 'throughput':
            plt.plot(self.time_points, self.throughput_values, 'r-', linewidth=1.5, marker='^', markersize=4, alpha=0.6,
                     label='原始数据')
            plt.title('吞吐量随时间变化')
            plt.ylabel('吞吐量 (Kbps)')

            # 添加滑动平均线
            throughput_ma = self.moving_average(self.throughput_values)
            plt.plot(self.time_points, throughput_ma, 'b-', linewidth=2.5,
                     label=f'滑动平均 (窗口={self.window_size})')

            # 添加统计信息
            avg_throughput = np.mean(self.throughput_values)
            max_throughput = np.max(self.throughput_values)
            plt.axhline(y=avg_throughput, color='g', linestyle='--', label=f'平均值: {avg_throughput:.2f} Kbps')
            plt.axhline(y=max_throughput, color='y', linestyle='--', label=f'最大值: {max_throughput:.2f} Kbps')

            plt.legend()

        elif metric_name == 'hop_count':
            plt.plot(self.time_points, self.hop_count_values, 'm-', linewidth=1.5, marker='d', markersize=4, alpha=0.6,
                     label='原始数据')
            plt.title('平均跳数随时间变化')
            plt.ylabel('跳数')

            # 添加滑动平均线
            hop_count_ma = self.moving_average(self.hop_count_values)
            plt.plot(self.time_points, hop_count_ma, 'r-', linewidth=2.5,
                     label=f'滑动平均 (窗口={self.window_size})')

            plt.legend()

        elif metric_name == 'routing_load':
            plt.plot(self.time_points, self.routing_load_values, 'c-', linewidth=1.5, marker='*', markersize=5,
                     alpha=0.6, label='原始数据')
            plt.title('路由负载随时间变化')
            plt.ylabel('控制包/数据包')

            # 添加滑动平均线
            routing_load_ma = self.moving_average(self.routing_load_values)
            plt.plot(self.time_points, routing_load_ma, 'r-', linewidth=2.5,
                     label=f'滑动平均 (窗口={self.window_size})')

            plt.legend()

        elif metric_name == 'mac_delay':
            plt.plot(self.time_points, self.mac_delay_values, 'k-', linewidth=1.5, marker='x', markersize=4, alpha=0.6,
                     label='原始数据')
            plt.title('MAC延迟随时间变化')
            plt.ylabel('MAC延迟 (毫秒)')

            # 添加滑动平均线
            mac_delay_ma = self.moving_average(self.mac_delay_values)
            plt.plot(self.time_points, mac_delay_ma, 'r-', linewidth=2.5,
                     label=f'滑动平均 (窗口={self.window_size})')

            plt.legend()

        elif metric_name == 'collisions':
            # 计算碰撞增量
            collision_increments = [self.collision_values[0]] + [self.collision_values[i] - self.collision_values[i - 1]
                                                                 for i in range(1, len(self.collision_values))]

            # 原始累积碰撞次数
            plt.plot(self.time_points, self.collision_values, 'orange', linewidth=1.5, marker='.', markersize=4,
                     alpha=0.6, label='累积碰撞次数')

            # 使用第二个Y轴显示碰撞率
            ax2 = plt.gca().twinx()

            # 计算碰撞率
            collision_rates = [0]
            for i in range(1, len(self.collision_values)):
                time_diff = self.time_points[i] - self.time_points[i - 1]
                collision_diff = self.collision_values[i] - self.collision_values[i - 1]
                rate = collision_diff / time_diff if time_diff > 0 else 0
                collision_rates.append(rate)

            # 绘制碰撞率与滑动平均
            ax2.plot(self.time_points, collision_rates, 'b-', linewidth=1.5, alpha=0.6, label='碰撞率')
            collision_rates_ma = self.moving_average(collision_rates)
            ax2.plot(self.time_points, collision_rates_ma, 'r-', linewidth=2.5, label=f'碰撞率滑动平均')

            plt.title('碰撞次数随时间变化')
            plt.ylabel('累积碰撞次数')
            ax2.set_ylabel('碰撞率 (次/秒)')

            # 合并两个图例
            lines, labels = plt.gca().get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='upper left')

        plt.xlabel('仿真时间 (秒)')
        plt.grid(True)
        plt.tight_layout()

        # 保存图表并尝试显示
        plt.savefig(f'{metric_name}_over_time.png', dpi=300)
        try:
            plt.show()
        except Exception as e:
            print(f"无法显示图表，已保存到文件: {metric_name}_over_time.png")
            print(f"错误信息: {e}")
        finally:
            plt.close()

    def set_moving_average_window(self, window_size):
        """
        设置滑动平均窗口大小

        Args:
            window_size: 整数窗口大小，至少为2
        """
        if window_size < 2:
            print("警告: 窗口大小应至少为2，使用默认值2")
            self.window_size = 2
        else:
            self.window_size = window_size
        print(f"滑动平均窗口大小设置为: {self.window_size}")