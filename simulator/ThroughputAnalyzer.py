import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob


class ThroughputAnalyzer:
    """
    分析网络吞吐量数据并生成详细报告
    """

    def __init__(self, data_dir='throughput_data'):
        self.data_dir = data_dir
        self.network_data = None
        self.node_data = {}
        self.output_dir = os.path.join(data_dir, 'analysis')

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        """加载保存的吞吐量数据"""
        # 加载网络吞吐量数据
        network_file = os.path.join(self.data_dir, 'network_throughput.csv')
        if os.path.exists(network_file):
            self.network_data = pd.read_csv(network_file)
            print(f"已加载网络吞吐量数据: {len(self.network_data)} 条记录")

        # 加载节点吞吐量数据
        node_files = glob.glob(os.path.join(self.data_dir, 'node_*_throughput.csv'))
        for file in node_files:
            node_id = int(file.split('node_')[1].split('_')[0])
            self.node_data[node_id] = pd.read_csv(file)
            print(f"已加载节点 {node_id} 吞吐量数据: {len(self.node_data[node_id])} 条记录")

    def analyze_network_throughput(self):
        """分析网络吞吐量数据"""
        if self.network_data is None or len(self.network_data) < 2:
            print("网络吞吐量数据不足，无法分析")
            return

        # 计算基本统计量
        avg_throughput = self.network_data['throughput_kbps'].mean()
        max_throughput = self.network_data['throughput_kbps'].max()
        min_throughput = self.network_data['throughput_kbps'].min()
        std_throughput = self.network_data['throughput_kbps'].std()

        # 计算变异系数（标准差/平均值）
        cv = std_throughput / avg_throughput if avg_throughput > 0 else 0

        # 生成时间序列图
        plt.figure(figsize=(15, 10))

        # 绘制网络吞吐量时间序列
        plt.subplot(2, 1, 1)
        plt.plot(self.network_data['time'], self.network_data['throughput_kbps'],
                 'b-', linewidth=2)

        # 添加平均值线和波动范围
        plt.axhline(y=avg_throughput, color='r', linestyle='--',
                    label=f'平均值: {avg_throughput:.2f} Kbps')
        plt.fill_between(self.network_data['time'],
                         avg_throughput - std_throughput,
                         avg_throughput + std_throughput,
                         color='r', alpha=0.2, label=f'标准差: ±{std_throughput:.2f} Kbps')

        # 添加平滑曲线（移动平均）
        window_size = max(5, len(self.network_data) // 20)  # 动态窗口大小
        self.network_data['smooth'] = self.network_data['throughput_kbps'].rolling(
            window=window_size, min_periods=1).mean()
        plt.plot(self.network_data['time'], self.network_data['smooth'],
                 'g-', linewidth=1.5, label='移动平均')

        plt.title('网络吞吐量随时间变化', fontsize=14)
        plt.xlabel('仿真时间（秒）', fontsize=12)
        plt.ylabel('吞吐量 (Kbps)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # 绘制直方图和核密度估计
        plt.subplot(2, 1, 2)
        self.network_data['throughput_kbps'].hist(bins=30, alpha=0.6, color='b')

        # 添加核密度估计
        self.network_data['throughput_kbps'].plot.kde(color='r', linewidth=2)

        # 添加统计信息
        plt.axvline(x=avg_throughput, color='r', linestyle='--',
                    label=f'平均值: {avg_throughput:.2f} Kbps')
        plt.axvline(x=max_throughput, color='g', linestyle='-.',
                    label=f'最大值: {max_throughput:.2f} Kbps')
        plt.axvline(x=min_throughput, color='orange', linestyle='-.',
                    label=f'最小值: {min_throughput:.2f} Kbps')

        plt.title(f'网络吞吐量分布 (变异系数: {cv:.3f})', fontsize=14)
        plt.xlabel('吞吐量 (Kbps)', fontsize=12)
        plt.ylabel('频率', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'network_throughput_analysis.png'), dpi=300)
        plt.close()

        # 保存统计结果
        stats = {
            '平均吞吐量 (Kbps)': avg_throughput,
            '最大吞吐量 (Kbps)': max_throughput,
            '最小吞吐量 (Kbps)': min_throughput,
            '标准差 (Kbps)': std_throughput,
            '变异系数': cv
        }

        with open(os.path.join(self.output_dir, 'network_throughput_stats.txt'), 'w') as f:
            f.write("网络吞吐量统计分析\n")
            f.write("-" * 40 + "\n")
            for key, value in stats.items():
                f.write(f"{key}: {value:.4f}\n")

        print(f"网络吞吐量分析完成，结果已保存至 {self.output_dir}")
        return stats

    def analyze_node_throughput(self):
        """分析节点吞吐量数据"""
        if not self.node_data:
            print("没有节点吞吐量数据，无法分析")
            return

        # 计算每个节点的平均吞吐量
        node_avg_throughput = {node_id: data['throughput_kbps'].mean()
                               for node_id, data in self.node_data.items()}

        # 排序节点（按平均吞吐量降序）
        sorted_nodes = sorted(node_avg_throughput.items(), key=lambda x: x[1], reverse=True)

        # 绘制节点吞吐量对比图
        plt.figure(figsize=(12, 8))

        # 绘制条形图
        node_ids = [node_id for node_id, _ in sorted_nodes]
        throughputs = [throughput for _, throughput in sorted_nodes]

        bars = plt.bar(node_ids, throughputs, color='skyblue', edgecolor='blue')

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 5,
                     f'{height:.1f}',
                     ha='center', va='bottom', fontsize=10)

        plt.title('各节点平均吞吐量对比', fontsize=14)
        plt.xlabel('节点ID', fontsize=12)
        plt.ylabel('平均吞吐量 (Kbps)', fontsize=12)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'node_throughput_comparison.png'), dpi=300)
        plt.close()

        # 绘制前几个节点的时间序列
        plt.figure(figsize=(15, 10))

        # 选择吞吐量最高的几个节点
        top_nodes = sorted_nodes[:min(5, len(sorted_nodes))]

        # 绘制每个节点的时间序列
        for node_id, avg_throughput in top_nodes:
            data = self.node_data[node_id]
            plt.plot(data['time'], data['throughput_kbps'],
                     label=f'节点 {node_id}: {avg_throughput:.2f} Kbps')

        plt.title('主要节点吞吐量随时间变化', fontsize=14)
        plt.xlabel('仿真时间（秒）', fontsize=12)
        plt.ylabel('吞吐量 (Kbps)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'top_nodes_throughput.png'), dpi=300)
        plt.close()

        # 保存节点吞吐量统计
        with open(os.path.join(self.output_dir, 'node_throughput_stats.txt'), 'w') as f:
            f.write("节点吞吐量统计分析\n")
            f.write("-" * 40 + "\n")
            f.write("按平均吞吐量排序：\n\n")

            for i, (node_id, avg_throughput) in enumerate(sorted_nodes):
                data = self.node_data[node_id]
                max_throughput = data['throughput_kbps'].max()
                min_throughput = data['throughput_kbps'].min()
                std_throughput = data['throughput_kbps'].std()

                f.write(f"#{i + 1} 节点 {node_id}:\n")
                f.write(f"  平均吞吐量: {avg_throughput:.2f} Kbps\n")
                f.write(f"  最大吞吐量: {max_throughput:.2f} Kbps\n")
                f.write(f"  最小吞吐量: {min_throughput:.2f} Kbps\n")
                f.write(f"  标准差: {std_throughput:.2f} Kbps\n")
                f.write(f"  变异系数: {std_throughput / avg_throughput if avg_throughput > 0 else 0:.3f}\n\n")

        print(f"节点吞吐量分析完成，结果已保存至 {self.output_dir}")

    def run_analysis(self):
        """运行完整分析"""
        self.load_data()
        self.analyze_network_throughput()
        self.analyze_node_throughput()
        print("吞吐量分析完成！")


# 使用示例
if __name__ == "__main__":
    analyzer = ThroughputAnalyzer()
    analyzer.run_analysis()