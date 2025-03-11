import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib.ticker import FuncFormatter
import time
import threading
import os


class NetworkThroughputMonitor:
    """
    实时监控和显示网络吞吐量的变化
    """

    def __init__(self, simulator, update_interval=1.0):
        """
        初始化网络吞吐量监控器

        Args:
            simulator: 模拟器实例
            update_interval: 更新间隔（秒）
        """
        self.simulator = simulator
        self.update_interval = update_interval
        self.metrics = simulator.metrics

        # 数据存储
        self.timestamps = []
        self.network_throughput = []
        self.node_throughputs = {}
        self.link_throughputs = {}

        # 监控选项
        self.running = False
        self.max_data_points = 300  # 最多显示的数据点数量

        # 图形设置
        self.fig = None
        self.axes = None
        self.lines = {}
        self.animation = None

        # 创建输出目录
        os.makedirs('throughput_data', exist_ok=True)

    def start(self):
        """开始监控网络吞吐量"""
        if self.running:
            print("监控器已经在运行")
            return

        self.running = True

        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

        # 设置图形
        self._setup_figure()

        # 开始动画
        self.animation = animation.FuncAnimation(
            self.fig, self._update_figure, interval=self.update_interval * 1000, blit=False)

        plt.show(block=False)

        print("网络吞吐量监控器已启动")

    def stop(self):
        """停止监控网络吞吐量"""
        self.running = False
        if self.animation:
            self.animation.event_source.stop()

        # 保存记录的数据
        self._save_data()

        print("网络吞吐量监控器已停止")

    def _monitoring_loop(self):
        """监控线程主循环"""
        while self.running:
            current_time = self.simulator.env.now / 1e6  # 转换为秒

            # 记录网络吞吐量
            self.timestamps.append(current_time)
            self.network_throughput.append(self.metrics.network_throughput_value / 1e3)  # 转为Kbps

            # 记录节点吞吐量
            for node_id, throughput in self.metrics.node_throughput_value.items():
                if node_id not in self.node_throughputs:
                    self.node_throughputs[node_id] = []
                self.node_throughputs[node_id].append(throughput / 1e3)  # 转为Kbps

            # 记录链路吞吐量
            for link, throughput in self.metrics.link_throughput_value.items():
                if link not in self.link_throughputs:
                    self.link_throughputs[link] = []
                self.link_throughputs[link].append(throughput / 1e3)  # 转为Kbps

            # 限制数据点数量
            if len(self.timestamps) > self.max_data_points:
                self.timestamps = self.timestamps[-self.max_data_points:]
                self.network_throughput = self.network_throughput[-self.max_data_points:]

                for node_id in self.node_throughputs:
                    self.node_throughputs[node_id] = self.node_throughputs[node_id][-self.max_data_points:]

                for link in self.link_throughputs:
                    self.link_throughputs[link] = self.link_throughputs[link][-self.max_data_points:]

            # 等待下一个更新间隔
            time.sleep(self.update_interval)

    def _setup_figure(self):
        """设置图形窗口"""
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title('实时网络吞吐量监控')

        # 创建子图
        self.axes = {}
        self.axes['network'] = self.fig.add_subplot(2, 1, 1)
        self.axes['nodes'] = self.fig.add_subplot(2, 1, 2)

        # 设置网络吞吐量图
        self.axes['network'].set_title('网络总体吞吐量')
        self.axes['network'].set_xlabel('仿真时间（秒）')
        self.axes['network'].set_ylabel('吞吐量 (Kbps)')
        self.axes['network'].grid(True)

        # 设置节点吞吐量图
        self.axes['nodes'].set_title('主要节点吞吐量')
        self.axes['nodes'].set_xlabel('仿真时间（秒）')
        self.axes['nodes'].set_ylabel('吞吐量 (Kbps)')
        self.axes['nodes'].grid(True)

        # 初始化线条
        self.lines['network'], = self.axes['network'].plot([], [], 'b-', lw=2, label='网络吞吐量')
        self.axes['network'].legend()

        # 节点线条将在更新时动态添加
        self.lines['nodes'] = {}

        plt.tight_layout()

    def _update_figure(self, frame):
        """更新图形"""
        if not self.running or len(self.timestamps) < 2:
            return

        # 更新网络吞吐量图
        self.lines['network'].set_data(self.timestamps, self.network_throughput)
        self.axes['network'].relim()
        self.axes['network'].autoscale_view()

        # 获取吞吐量最高的几个节点
        top_nodes = sorted(
            [(node_id, vals[-1] if vals else 0) for node_id, vals in self.node_throughputs.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]  # 显示前5个节点

        # 更新或添加节点线条
        for node_id, _ in top_nodes:
            if node_id not in self.lines['nodes']:
                # 添加新节点线条
                line, = self.axes['nodes'].plot(
                    [], [], marker='.', markersize=8,
                    label=f'节点 {node_id}'
                )
                self.lines['nodes'][node_id] = line

            # 更新线条数据
            if len(self.timestamps) == len(self.node_throughputs[node_id]):
                self.lines['nodes'][node_id].set_data(
                    self.timestamps, self.node_throughputs[node_id])

        # 移除不在前五的节点线条
        current_top_nodes = [node_id for node_id, _ in top_nodes]
        for node_id in list(self.lines['nodes'].keys()):
            if node_id not in current_top_nodes:
                self.lines['nodes'][node_id].remove()
                del self.lines['nodes'][node_id]

        # 自动调整坐标轴
        self.axes['nodes'].relim()
        self.axes['nodes'].autoscale_view()
        self.axes['nodes'].legend()

        return list(self.lines['nodes'].values()) + [self.lines['network']]

    def _save_data(self):
        """保存记录的数据"""
        # 保存网络吞吐量数据
        np.savetxt(
            'throughput_data/network_throughput.csv',
            np.column_stack((self.timestamps, self.network_throughput)),
            delimiter=',',
            header='time,throughput_kbps',
            comments=''
        )

        # 保存节点吞吐量数据
        for node_id, values in self.node_throughputs.items():
            if len(values) == len(self.timestamps):
                np.savetxt(
                    f'throughput_data/node_{node_id}_throughput.csv',
                    np.column_stack((self.timestamps, values)),
                    delimiter=',',
                    header='time,throughput_kbps',
                    comments=''
                )

        # 生成更详细的吞吐量报告图
        self._generate_detailed_report()

    def _generate_detailed_report(self):
        """生成详细的吞吐量报告图"""
        # 生成网络吞吐量随时间变化的高分辨率图
        plt.figure(figsize=(15, 10))

        # 绘制网络吞吐量
        plt.subplot(2, 1, 1)
        plt.plot(self.timestamps, self.network_throughput, 'b-', linewidth=2)
        plt.title('网络吞吐量随时间变化', fontsize=14)
        plt.xlabel('仿真时间（秒）', fontsize=12)
        plt.ylabel('吞吐量 (Kbps)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)

        # 添加平均值线
        avg_throughput = np.mean(self.network_throughput)
        plt.axhline(y=avg_throughput, color='r', linestyle='--',
                    label=f'平均值: {avg_throughput:.2f} Kbps')
        plt.legend()

        # 绘制节点吞吐量
        plt.subplot(2, 1, 2)
        cmap = plt.cm.get_cmap('tab10', 10)  # 使用Tab10色图最多10种颜色

        # 排序节点吞吐量
        sorted_nodes = sorted(
            [(node_id, np.mean(values)) for node_id, values in self.node_throughputs.items()
             if len(values) == len(self.timestamps)],
            key=lambda x: x[1],
            reverse=True
        )

        # 绘制每个节点的吞吐量曲线
        for i, (node_id, _) in enumerate(sorted_nodes[:10]):  # 最多显示10个节点
            values = self.node_throughputs[node_id]
            color = cmap(i % 10)
            plt.plot(self.timestamps, values,
                     color=color, linewidth=1.5,
                     label=f'节点 {node_id}: {np.mean(values):.2f} Kbps')

        plt.title('主要节点吞吐量随时间变化', fontsize=14)
        plt.xlabel('仿真时间（秒）', fontsize=12)
        plt.ylabel('吞吐量 (Kbps)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

        plt.tight_layout()
        plt.savefig('throughput_data/detailed_throughput_report.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 生成链路吞吐量热图
        self._generate_link_heatmap()

    def _generate_link_heatmap(self):
        """生成链路吞吐量热图"""
        # 如果链路数量太少，则跳过
        active_links = [link for link, values in self.link_throughputs.items()
                        if len(values) > 0 and np.mean(values) > 0]

        if len(active_links) < 2:
            return

        # 计算每条链路的平均吞吐量
        link_avg_throughput = {link: np.mean(values)
                               for link, values in self.link_throughputs.items()
                               if len(values) > 0}

        # 获取所有节点ID
        all_nodes = set()
        for src, dst in link_avg_throughput.keys():
            all_nodes.add(src)
            all_nodes.add(dst)

        nodes_list = sorted(list(all_nodes))
        n = len(nodes_list)

        # 创建邻接矩阵
        adjacency_matrix = np.zeros((n, n))

        # 节点ID到矩阵索引的映射
        node_indices = {node: i for i, node in enumerate(nodes_list)}

        # 填充邻接矩阵
        for (src, dst), throughput in link_avg_throughput.items():
            if src in node_indices and dst in node_indices:
                i, j = node_indices[src], node_indices[dst]
                adjacency_matrix[i, j] = throughput

        # 绘制热力图
        plt.figure(figsize=(12, 10))
        plt.imshow(adjacency_matrix, cmap='hot', interpolation='nearest')

        # 设置坐标轴标签
        plt.xticks(range(n), nodes_list)
        plt.yticks(range(n), nodes_list)

        # 添加颜色条和标题
        cbar = plt.colorbar()
        cbar.set_label('平均吞吐量 (Kbps)')
        plt.title('链路吞吐量热力图', fontsize=14)
        plt.xlabel('目标节点', fontsize=12)
        plt.ylabel('源节点', fontsize=12)

        # 添加数值标签
        for i in range(n):
            for j in range(n):
                if adjacency_matrix[i, j] > 0:
                    plt.text(j, i, f'{adjacency_matrix[i, j]:.1f}',
                             ha='center', va='center',
                             color='white' if adjacency_matrix[i, j] > np.max(adjacency_matrix) / 2 else 'black')

        plt.tight_layout()
        plt.savefig('throughput_data/link_throughput_heatmap.png', dpi=300)
        plt.close()

    def save_snapshot(self, filename=None):
        """保存当前网络吞吐量快照"""
        if not filename:
            timestamp = int(time.time())
            filename = f'throughput_data/throughput_snapshot_{timestamp}.png'

        if self.fig:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"网络吞吐量快照已保存至: {filename}")