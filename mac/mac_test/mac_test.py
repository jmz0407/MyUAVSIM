#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import simpy
import random

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入仿真组件
from utils import config
from simulator.simulator import Simulator
from entities.drone import Drone
from mac.stdma import Stdma
from mac.tra_tdma import Tra_Tdma
from mac.BasicStdma import BasicStdma
from simulator.improved_traffic_generator import TrafficType, PriorityLevel
from mac.muti_stdma import MutiStdma
from mac.DQN_stdma import DQN_Stdma

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='mac_protocol_test.log',
    filemode='w'
)

# # 创建控制台处理器
# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# console.setFormatter(formatter)
# logging.getLogger('').addHandler(console)


class ProtocolTester:
    """MAC协议性能测试框架"""

    def __init__(self, output_dir="test_results"):
        """初始化测试框架

        Args:
            output_dir: 结果输出目录
        """
        self.output_dir = output_dir
        self.results = {}
        self.protocols = {
            "PPO-STDMA": Stdma,
            "DQN-STDMA": DQN_Stdma,
            "TraTDMA": Tra_Tdma,
            "BasicSTDMA": BasicStdma,

        }

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 当前测试时间戳
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def setup_simulator(self, protocol_class, num_drones=10, seed=42, sim_time=7 * 1e6):
        """设置仿真器

        Args:
            protocol_class: MAC协议类
            num_drones: 无人机数量
            seed: 随机种子
            sim_time: 仿真时间(μs)

        Returns:
            Simulator: 配置好的仿真器实例
        """
        # 保存原始配置
        original_num_drones = config.NUMBER_OF_DRONES
        original_sim_time = config.SIM_TIME

        # 修改配置
        config.NUMBER_OF_DRONES = num_drones
        config.SIM_TIME = sim_time

        # 创建SimPy环境
        env = simpy.Environment()
        channel_states = {}

        # 创建仿真器
        simulator = Simulator(
            seed=seed,
            env=env,
            channel_states=channel_states,
            n_drones=num_drones,
            total_simulation_time=sim_time
        )

        # 将每个无人机的MAC协议替换为指定协议
        for drone in simulator.drones:
            drone.mac_protocol = protocol_class(drone)

        # 恢复原始配置
        config.NUMBER_OF_DRONES = original_num_drones
        config.SIM_TIME = original_sim_time

        return simulator

    def configure_traffic_flows(self, simulator, traffic_pattern, num_flows):
        """配置业务流

        先发送业务需求给节点，等待网络稳定后再实际发送业务流

        Args:
            simulator: 仿真器实例
            traffic_pattern: 业务模式
            num_flows: 业务流数量

        Returns:
            list: 已创建的业务流ID列表
        """
        flow_ids = []
        n_drones = simulator.n_drones

        # 设置随机种子，保证可重复性
        random.seed(42)

        # 确保流数量不超过可能的源-目标对数量
        max_flows = min(num_flows, n_drones * (n_drones - 1) // 2)

        # 已使用的源目标对
        used_pairs = set()

        logging.info(f"准备创建 {max_flows} 条业务流 (模式: {traffic_pattern})")

        # 直接使用setup_traffic_flow和start_traffic_flow的模式，而不是先创建需求
        for i in range(max_flows):
            # 选择源节点和目标节点（确保不重复）
            while True:
                source_id = random.randint(0, n_drones - 1)
                dest_id = random.randint(0, n_drones - 1)

                # 确保源和目标不同
                if source_id != dest_id and (source_id, dest_id) not in used_pairs:
                    used_pairs.add((source_id, dest_id))
                    break

            logging.info(f"创建业务流 {i + 1}/{max_flows}: {source_id} -> {dest_id}, 模式: {traffic_pattern}")

            # 为不同业务模式设置参数
            if traffic_pattern == "CBR":
                # 恒定比特率业务
                flow_config = {
                    'source_id': source_id,
                    'dest_id': dest_id,
                    'traffic_type': TrafficType.CBR,
                    'num_packets': 200,
                    'data_rate': 2.0,  # 2Mbps
                    'packet_size': config.DATA_PACKET_LENGTH,
                    'priority': PriorityLevel.NORMAL,
                    'delay_req': 100,  # 较低延迟要求 (ms)
                    'qos_req': 0.95  # 较高可靠性
                }

            elif traffic_pattern == "VBR":
                # 可变比特率业务
                flow_config = {
                    'source_id': source_id,
                    'dest_id': dest_id,
                    'traffic_type': TrafficType.VBR,
                    'num_packets': 200,
                    'data_rate': 1.5,  # 平均1.5Mbps
                    'packet_size': config.DATA_PACKET_LENGTH,
                    'priority': PriorityLevel.NORMAL,
                    'delay_req': 200,  # 中等延迟要求
                    'qos_req': 0.9,  # 可靠性
                    'params': {
                        'peak_rate': 3.0,  # 峰值3Mbps
                        'min_rate': 0.5  # 最小0.5Mbps
                    }
                }

            elif traffic_pattern == "BURST":
                # 突发业务
                flow_config = {
                    'source_id': source_id,
                    'dest_id': dest_id,
                    'traffic_type': TrafficType.BURST,
                    'num_packets': 200,
                    'packet_size': config.DATA_PACKET_LENGTH,
                    'priority': PriorityLevel.HIGH,
                    'delay_req': 80,  # 低延迟
                    'qos_req': 0.99,  # 高可靠性
                    'params': {
                        'burst_size': 10,
                        'num_bursts': 20,
                        'burst_interval': 100000,  # 100ms
                        'packet_interval': 1000  # 1ms
                    }
                }

            elif traffic_pattern == "PERIODIC":
                # 周期性业务
                flow_config = {
                    'source_id': source_id,
                    'dest_id': dest_id,
                    'traffic_type': TrafficType.PERIODIC,
                    'num_packets': 200,
                    'packet_size': config.DATA_PACKET_LENGTH,
                    'priority': PriorityLevel.NORMAL,
                    'delay_req': 500,  # 高延迟容忍
                    'qos_req': 0.8,  # 较低可靠性
                    'params': {
                        'period': 20000,  # 20ms
                        'jitter': 0.1  # 10%抖动
                    }
                }

            else:
                # 默认使用CBR
                flow_config = {
                    'source_id': source_id,
                    'dest_id': dest_id,
                    'traffic_type': TrafficType.CBR,
                    'num_packets': 200,
                    'data_rate': 2.0,
                    'packet_size': config.DATA_PACKET_LENGTH,
                    'priority': PriorityLevel.NORMAL,
                    'delay_req': 100,
                    'qos_req': 0.9
                }

            # 设置业务流
            try:
                # 注意：我们跳过了先创建和提交需求的步骤，直接设置并启动流量
                flow_id = simulator.traffic_generator.setup_traffic_flow(**flow_config)

                # 设置开始延迟，让网络有足够时间准备
                # 每个流的延迟略有不同，避免所有流同时开始
                start_delay = 1000000 + i * 50000  # 1秒 + 每流增加50ms

                # 启动业务流（延迟启动）
                simulator.env.process(self._delayed_start_flow(simulator, flow_id, start_delay))

                flow_ids.append(flow_id)
                logging.info(f"业务流 {flow_id} 已配置，将在 {start_delay / 1e6:.2f} 秒后启动")
            except Exception as e:
                logging.error(f"配置业务流失败: {e}")

        logging.info(f"已配置 {len(flow_ids)} 条业务流，它们将在网络准备就绪后开始传输")
        return flow_ids

    def _delayed_start_flow(self, simulator, flow_id, delay):
        """辅助方法：延迟启动业务流"""
        yield simulator.env.timeout(delay)
        simulator.traffic_generator.start_traffic_flow(flow_id)
        logging.info(f"业务流 {flow_id} 已开始传输")

    def run_single_test(self, protocol_name, num_drones=10, traffic_pattern="CBR",
                        num_flows=3, seed=42, sim_time=7 * 1e6):
        """运行单次测试

        Args:
            protocol_name: 协议名称
            num_drones: 无人机数量
            traffic_pattern: 业务模式
            num_flows: 业务流数量
            seed: 随机种子
            sim_time: 仿真时间(μs)

        Returns:
            dict: 测试结果
        """
        protocol_class = self.protocols[protocol_name]

        logging.info(f"开始测试 {protocol_name}, {num_drones}架无人机, {traffic_pattern}业务模式, {num_flows}条流")

        # 设置仿真器
        simulator = self.setup_simulator(
            protocol_class=protocol_class,
            num_drones=num_drones,
            seed=seed,
            sim_time=sim_time
        )

        # 配置业务流
        flow_ids = self.configure_traffic_flows(simulator, traffic_pattern, num_flows)
        logging.info(f"已配置 {len(flow_ids)} 条业务流")

        # 记录测试开始时间
        start_time = time.time()

        # 运行仿真
        simulator.env.run(until=sim_time)

        # 记录测试耗时
        elapsed_time = time.time() - start_time

        # 收集指标
        metrics = simulator.metrics

        # 计算平均值
        e2e_delay = np.mean(metrics.delivery_time) / 1e3 if metrics.delivery_time else 0  # ms
        pdr = (
                    len(metrics.datapacket_arrived) / metrics.datapacket_generated_num * 100) if metrics.datapacket_generated_num > 0 else 0
        throughput = np.mean(metrics.throughput) / 1e3 if metrics.throughput else 0  # kbps
        hop_cnt = np.mean(metrics.hop_cnt) if metrics.hop_cnt else 0
        mac_delay = np.mean(metrics.mac_delay) if metrics.mac_delay else 0  # ms
        collision_num = metrics.collision_num

        # 计算能量消耗统计
        energy_values = list(metrics.energy_consumption.values())
        avg_energy = np.mean(energy_values) if energy_values else 0
        max_energy = np.max(energy_values) if energy_values else 0
        energy_std = np.std(energy_values) if energy_values else 0

        # 收集结果
        result = {
            "protocol": protocol_name,
            "num_drones": num_drones,
            "traffic_pattern": traffic_pattern,
            "num_flows": num_flows,
            "sim_time_us": sim_time,
            "real_time_s": elapsed_time,
            "e2e_delay_ms": e2e_delay,
            "pdr_percent": pdr,
            "throughput_kbps": throughput,
            "hop_count": hop_cnt,
            "mac_delay_ms": mac_delay,
            "collision_num": collision_num,
            "avg_energy_j": avg_energy,
            "max_energy_j": max_energy,
            "energy_std_j": energy_std,
            "packets_generated": metrics.datapacket_generated_num,
            "packets_delivered": len(metrics.datapacket_arrived)
        }

        logging.info(f"测试完成: {protocol_name}, PDR={pdr:.2f}%, 延迟={e2e_delay:.2f}ms, 吞吐量={throughput:.2f}kbps")

        return result

    def run_comparison_tests(self, drone_counts=[10, 20, 30], traffic_patterns=["CBR"],
                             flow_counts=[1, 1, 1], seeds=[42], sim_time=7 * 1e6):
        """运行比较测试

        Args:
            drone_counts: 要测试的无人机数量列表
            traffic_patterns: 要测试的业务模式列表
            flow_counts: 要测试的业务流数量列表
            seeds: 随机种子列表
            sim_time: 仿真时间(μs)
        """
        all_results = []

        # 创建测试组合
        test_configs = []
        for protocol in self.protocols.keys():
            for num_drones in drone_counts:
                for traffic in traffic_patterns:
                    for flows in flow_counts:
                        for seed in seeds:
                            test_configs.append({
                                "protocol": protocol,
                                "num_drones": num_drones,
                                "traffic": traffic,
                                "flows": flows,
                                "seed": seed
                            })

        # 运行所有测试
        total_tests = len(test_configs)
        for i, config in enumerate(test_configs):
            logging.info(f"运行测试 {i + 1}/{total_tests}: {config}")

            result = self.run_single_test(
                protocol_name=config["protocol"],
                num_drones=config["num_drones"],
                traffic_pattern=config["traffic"],
                num_flows=config["flows"],
                seed=config["seed"],
                sim_time=sim_time
            )

            all_results.append(result)

            # 保存中间结果
            self.save_results(all_results, f"interim_results_{i + 1}")

        # 保存最终结果
        self.results = all_results
        self.save_results(all_results, "final_results")

        # 生成报告
        self.generate_report()

    def save_results(self, results, filename):
        """保存测试结果

        Args:
            results: 测试结果列表
            filename: 输出文件名
        """
        # 转换为DataFrame
        df = pd.DataFrame(results)

        # 保存为CSV
        csv_path = os.path.join(self.output_dir, f"{filename}.csv")
        df.to_csv(csv_path, index=False)

        logging.info(f"结果已保存至 {csv_path}")

    def generate_report(self):
        """生成测试报告"""
        if not self.results:
            logging.error("没有可用的测试结果")
            return

        # 转换为DataFrame
        df = pd.DataFrame(self.results)

        # 创建报告目录
        report_dir = os.path.join(self.output_dir, f"report_{self.timestamp}")
        os.makedirs(report_dir, exist_ok=True)

        # 保存汇总结果
        summary_path = os.path.join(report_dir, "summary.csv")
        df.to_csv(summary_path, index=False)

        # 生成性能对比图表
        self._generate_performance_charts(df, report_dir)

        # 生成HTML报告
        self._generate_html_report(df, report_dir)

        logging.info(f"报告已生成至 {report_dir}")

    def _generate_performance_charts(self, df, output_dir):
        """生成性能对比图表

        Args:
            df: 结果DataFrame
            output_dir: 输出目录
        """
        # 设置图表样式
        plt.style.use('ggplot')

        # 所有协议
        protocols = sorted(df['protocol'].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(protocols)))

        # 1. 不同无人机数量下的端到端延迟
        self._create_comparison_chart(
            df, 'num_drones', 'e2e_delay_ms',
            '无人机数量', '平均端到端延迟 (ms)',
            '不同无人机数量下的端到端延迟对比',
            os.path.join(output_dir, 'e2e_delay_vs_drones.png'),
            protocols, colors
        )

        # 2. 不同无人机数量下的数据包投递率
        self._create_comparison_chart(
            df, 'num_drones', 'pdr_percent',
            '无人机数量', '数据包投递率 (%)',
            '不同无人机数量下的数据包投递率对比',
            os.path.join(output_dir, 'pdr_vs_drones.png'),
            protocols, colors
        )

        # 3. 不同无人机数量下的吞吐量
        self._create_comparison_chart(
            df, 'num_drones', 'throughput_kbps',
            '无人机数量', '吞吐量 (kbps)',
            '不同无人机数量下的吞吐量对比',
            os.path.join(output_dir, 'throughput_vs_drones.png'),
            protocols, colors
        )

        # 4. 不同无人机数量下的MAC层延迟
        self._create_comparison_chart(
            df, 'num_drones', 'mac_delay_ms',
            '无人机数量', 'MAC层延迟 (ms)',
            '不同无人机数量下的MAC层延迟对比',
            os.path.join(output_dir, 'mac_delay_vs_drones.png'),
            protocols, colors
        )

        # 5. 不同无人机数量下的平均能量消耗
        self._create_comparison_chart(
            df, 'num_drones', 'avg_energy_j',
            '无人机数量', '平均能量消耗 (J)',
            '不同无人机数量下的平均能量消耗对比',
            os.path.join(output_dir, 'energy_vs_drones.png'),
            protocols, colors
        )

        # 6. 不同无人机数量下的碰撞次数
        self._create_comparison_chart(
            df, 'num_drones', 'collision_num',
            '无人机数量', '碰撞次数',
            '不同无人机数量下的碰撞次数对比',
            os.path.join(output_dir, 'collisions_vs_drones.png'),
            protocols, colors
        )

        # 7. 蜘蛛图比较不同协议的综合性能
        self._create_radar_chart(
            df, protocols,
            os.path.join(output_dir, 'protocol_radar_comparison.png')
        )

        # 8. 不同业务模式下的对比图
        if len(df['traffic_pattern'].unique()) > 1:
            for metric, title, ylabel in [
                ('e2e_delay_ms', '不同业务流下的端到端延迟对比', '平均端到端延迟 (ms)'),
                ('pdr_percent', '不同业务流下的数据包投递率对比', '数据包投递率 (%)'),
                ('throughput_kbps', '不同业务流下的吞吐量对比', '吞吐量 (kbps)')
            ]:
                self._create_traffic_comparison_chart(
                    df, metric, title, ylabel,
                    os.path.join(output_dir, f'{metric}_vs_traffic.png'),
                    protocols, colors
                )

    def _create_comparison_chart(self, df, x_col, y_col, x_label, y_label,
                                 title, output_path, protocols, colors):
        """创建比较图表

        Args:
            df: 数据DataFrame
            x_col: X轴列名
            y_col: Y轴列名
            x_label: X轴标签
            y_label: Y轴标签
            title: 图表标题
            output_path: 输出路径
            protocols: 协议列表
            colors: 颜色列表
        """
        plt.figure(figsize=(10, 6))

        for i, protocol in enumerate(protocols):
            protocol_df = df[df['protocol'] == protocol]

            # 按x_col分组并计算y_col的均值
            grouped = protocol_df.groupby(x_col)[y_col].mean().reset_index()

            plt.plot(grouped[x_col], grouped[y_col], 'o-',
                     color=colors[i], label=protocol, linewidth=2, markersize=8)

        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)

        # 保存图表
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _create_traffic_comparison_chart(self, df, metric, title, ylabel,
                                         output_path, protocols, colors):
        """创建不同业务模式下的性能对比图

        Args:
            df: 数据DataFrame
            metric: 要比较的指标
            title: 图表标题
            ylabel: Y轴标签
            output_path: 输出路径
            protocols: 协议列表
            colors: 颜色列表
        """
        plt.figure(figsize=(12, 8))

        traffic_patterns = sorted(df['traffic_pattern'].unique())
        n_patterns = len(traffic_patterns)

        # 计算条形图的位置
        bar_width = 0.8 / len(protocols)
        r = np.arange(n_patterns)

        for i, protocol in enumerate(protocols):
            protocol_df = df[df['protocol'] == protocol]

            values = []
            for pattern in traffic_patterns:
                pattern_df = protocol_df[protocol_df['traffic_pattern'] == pattern]
                values.append(pattern_df[metric].mean())

            # 计算当前协议的条形图位置
            bar_positions = [pos + bar_width * (i - len(protocols) / 2 + 0.5) for pos in r]

            plt.bar(bar_positions, values, width=bar_width, color=colors[i],
                    label=protocol, edgecolor='white', alpha=0.8)

        # 设置图表属性
        plt.xlabel('业务流模式', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14)
        plt.xticks(r, traffic_patterns)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.3, axis='y')

        # 保存图表
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _create_radar_chart(self, df, protocols, output_path):
        """创建雷达图比较不同协议的综合性能

        Args:
            df: 数据DataFrame
            protocols: 协议列表
            output_path: 输出路径
        """
        # 定义要比较的指标
        metrics = [
            ('pdr_percent', 'PDR (%)', True),  # 名称、标签、是否更高更好
            ('throughput_kbps', '吞吐量 (kbps)', True),
            ('e2e_delay_ms', '端到端延迟 (ms)', False),
            ('mac_delay_ms', 'MAC延迟 (ms)', False),
            ('avg_energy_j', '能量消耗 (J)', False),
            ('collision_num', '碰撞次数', False)
        ]

        # 计算每个协议在每个指标上的平均值
        protocol_metrics = {}

        for protocol in protocols:
            protocol_df = df[df['protocol'] == protocol]
            metrics_values = {}

            for metric_name, _, higher_better in metrics:
                avg_value = protocol_df[metric_name].mean()
                metrics_values[metric_name] = avg_value

            protocol_metrics[protocol] = metrics_values

        # 归一化指标值
        normalized_metrics = {}

        for metric_name, _, higher_better in metrics:
            all_values = [protocol_metrics[p][metric_name] for p in protocols]
            min_val = min(all_values)
            max_val = max(all_values)

            # 避免除以零
            range_val = max_val - min_val
            if range_val == 0:
                range_val = 1

            for protocol in protocols:
                if protocol not in normalized_metrics:
                    normalized_metrics[protocol] = {}

                value = protocol_metrics[protocol][metric_name]

                # 归一化到0-1之间，并根据指标性质调整（更高更好或更低更好）
                if higher_better:
                    norm_value = (value - min_val) / range_val
                else:
                    norm_value = 1 - (value - min_val) / range_val

                normalized_metrics[protocol][metric_name] = norm_value

        # 创建雷达图
        num_metrics = len(metrics)
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形

        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

        # 绘制每个协议的雷达图
        for i, protocol in enumerate(protocols):
            values = [normalized_metrics[protocol][m[0]] for m in metrics]
            values += values[:1]  # 闭合图形

            ax.plot(angles, values, 'o-', linewidth=2, label=protocol, color=plt.cm.tab10(i))
            ax.fill(angles, values, alpha=0.1, color=plt.cm.tab10(i))

        # 设置标签
        ax.set_thetagrids(np.degrees(angles[:-1]), [m[1] for m in metrics])

        # 设置刻度
        ax.set_rlabel_position(0)
        ax.set_rticks([0.25, 0.5, 0.75, 1])
        ax.set_yticklabels(["0.25", "0.5", "0.75", "1"], color="grey", size=8)

        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title("MAC协议性能雷达图对比", size=14, y=1.1)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_html_report(self, df, output_dir):
        """生成HTML报告

        Args:
            df: 结果DataFrame
            output_dir: 输出目录
        """
        # 创建HTML文件
        html_path = os.path.join(output_dir, "report.html")

        # 所有协议
        protocols = sorted(df['protocol'].unique())

        # 所有无人机数量
        drone_counts = sorted(df['num_drones'].unique())

        # 所有业务模式
        traffic_patterns = sorted(df['traffic_pattern'].unique())

        # 生成HTML内容
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write('''
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>MAC协议性能测试报告</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                        margin: 0;
                        padding: 20px;
                        color: #333;
                    }
                    h1, h2, h3 {
                        color: #2c3e50;
                    }
                    table {
                        border-collapse: collapse;
                        width: 100%;
                        margin-bottom: 20px;
                    }
                    th, td {
                        text-align: left;
                        padding: 12px;
                        border-bottom: 1px solid #ddd;
                    }
                    th {
                        background-color: #f2f2f2;
                    }
                    tr:hover {
                        background-color: #f5f5f5;
                    }
                    .chart-container {
                        display: flex;
                        flex-wrap: wrap;
                        justify-content: space-around;
                        margin: 20px 0;
                    }
                    .chart {
                        margin: 10px;
                        box-shadow: 0 0 10px rgba(0,0,0,0.1);
                    }
                    .highlight {
                        font-weight: bold;
                        color: #1a73e8;
                    }
                    .summary-card {
                        background-color: #f9f9f9;
                        border-radius: 8px;
                        padding: 15px;
                        margin-bottom: 20px;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    }
                    .recommendation {
                        background-color: #e8f5e9;
                        padding: 15px;
                        border-radius: 8px;
                        margin-top: 20px;
                    }
                    .traffic-section {
                        margin-top: 30px;
                        border-top: 1px solid #ddd;
                        padding-top: 20px;
                    }
                </style>
            </head>
            <body>
                <h1>MAC协议性能测试报告</h1>
                <p>生成时间: ''' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '''</p>

                <div class="summary-card">
                    <h2>测试概览</h2>
                    <p>本报告比较了以下MAC协议在无人机网络中的性能表现：</p>
                    <ul>
                        ''' + ''.join([f'<li>{p}</li>' for p in protocols]) + '''
                    </ul>
                    <p>测试条件：</p>
                    <ul>
                        <li>无人机数量: ''' + ', '.join(map(str, drone_counts)) + '''</li>
                        <li>业务模式: ''' + ', '.join(traffic_patterns) + '''</li>
                        <li>仿真时间: ''' + str(int(df['sim_time_us'].iloc[0] / 1e6)) + ''' 秒</li>
                    </ul>
                </div>

                <h2>性能结果摘要</h2>
            ''')

            # 生成摘要表格
            f.write('''
                <table>
                    <tr>
                        <th>协议</th>
                        <th>平均端到端延迟 (ms)</th>
                        <th>平均数据包投递率 (%)</th>
                        <th>平均吞吐量 (kbps)</th>
                        <th>平均MAC延迟 (ms)</th>
                        <th>平均能量消耗 (J)</th>
                        <th>平均碰撞次数</th>
                    </tr>
            ''')

            for protocol in protocols:
                protocol_df = df[df['protocol'] == protocol]

                e2e_delay = protocol_df['e2e_delay_ms'].mean()
                pdr = protocol_df['pdr_percent'].mean()
                throughput = protocol_df['throughput_kbps'].mean()
                mac_delay = protocol_df['mac_delay_ms'].mean()
                energy = protocol_df['avg_energy_j'].mean()
                collisions = protocol_df['collision_num'].mean()

                f.write(f'''
                    <tr>
                        <td>{protocol}</td>
                        <td>{e2e_delay:.2f}</td>
                        <td>{pdr:.2f}</td>
                        <td>{throughput:.2f}</td>
                        <td>{mac_delay:.2f}</td>
                        <td>{energy:.2f}</td>
                        <td>{collisions:.0f}</td>
                    </tr>
                ''')

            f.write('</table>')

            # 添加图表部分
            f.write('''
                <h2>性能对比图表</h2>

                <div class="chart-container">
                    <div class="chart">
                        <h3>端到端延迟对比</h3>
                        <img src="e2e_delay_vs_drones.png" alt="端到端延迟对比" width="400">
                    </div>
                    <div class="chart">
                        <h3>数据包投递率对比</h3>
                        <img src="pdr_vs_drones.png" alt="数据包投递率对比" width="400">
                    </div>
                </div>

                <div class="chart-container">
                    <div class="chart">
                        <h3>吞吐量对比</h3>
                        <img src="throughput_vs_drones.png" alt="吞吐量对比" width="400">
                    </div>
                    <div class="chart">
                        <h3>MAC层延迟对比</h3>
                        <img src="mac_delay_vs_drones.png" alt="MAC层延迟对比" width="400">
                    </div>
                </div>

                <div class="chart-container">
                    <div class="chart">
                        <h3>能量消耗对比</h3>
                        <img src="energy_vs_drones.png" alt="能量消耗对比" width="400">
                    </div>
                    <div class="chart">
                        <h3>碰撞次数对比</h3>
                        <img src="collisions_vs_drones.png" alt="碰撞次数对比" width="400">
                    </div>
                </div>

                <div class="chart-container">
                    <div class="chart">
                        <h3>协议综合性能雷达图</h3>
                        <img src="protocol_radar_comparison.png" alt="协议综合性能雷达图" width="600">
                    </div>
                </div>
            ''')

            # 如果有多种业务模式，添加业务模式比较图表
            if len(traffic_patterns) > 1:
                f.write('''
                    <div class="traffic-section">
                        <h2>不同业务模式下的性能比较</h2>

                        <div class="chart-container">
                            <div class="chart">
                                <h3>不同业务流下的端到端延迟</h3>
                                <img src="e2e_delay_ms_vs_traffic.png" alt="不同业务流下的端到端延迟" width="500">
                            </div>
                            <div class="chart">
                                <h3>不同业务流下的数据包投递率</h3>
                                <img src="pdr_percent_vs_traffic.png" alt="不同业务流下的数据包投递率" width="500">
                            </div>
                        </div>

                        <div class="chart-container">
                            <div class="chart">
                                <h3>不同业务流下的吞吐量</h3>
                                <img src="throughput_kbps_vs_traffic.png" alt="不同业务流下的吞吐量" width="500">
                            </div>
                        </div>
                    </div>
                ''')

            # 分析和结论
            # 找出整体表现最好的协议
            protocol_scores = {}
            for protocol in protocols:
                protocol_df = df[df['protocol'] == protocol]

                # 计算归一化分数 (更高的PDR和吞吐量更好，更低的延迟和能耗更好)
                pdr_score = protocol_df['pdr_percent'].mean() / df['pdr_percent'].max() if df[
                                                                                               'pdr_percent'].max() > 0 else 0
                throughput_score = protocol_df['throughput_kbps'].mean() / df['throughput_kbps'].max() if df[
                                                                                                              'throughput_kbps'].max() > 0 else 0

                delay_min = df['e2e_delay_ms'].min()
                delay_max = df['e2e_delay_ms'].max()
                delay_range = delay_max - delay_min
                delay_score = 1 - (
                        (protocol_df['e2e_delay_ms'].mean() - delay_min) / delay_range) if delay_range > 0 else 0.5

                mac_delay_min = df['mac_delay_ms'].min()
                mac_delay_max = df['mac_delay_ms'].max()
                mac_delay_range = mac_delay_max - mac_delay_min
                mac_delay_score = 1 - ((protocol_df[
                                            'mac_delay_ms'].mean() - mac_delay_min) / mac_delay_range) if mac_delay_range > 0 else 0.5

                energy_min = df['avg_energy_j'].min()
                energy_max = df['avg_energy_j'].max()
                energy_range = energy_max - energy_min
                energy_score = 1 - ((protocol_df[
                                         'avg_energy_j'].mean() - energy_min) / energy_range) if energy_range > 0 else 0.5

                collision_min = df['collision_num'].min()
                collision_max = df['collision_num'].max()
                collision_range = collision_max - collision_min
                collision_score = 1 - ((protocol_df[
                                            'collision_num'].mean() - collision_min) / collision_range) if collision_range > 0 else 0.5

                # 总分 (加权)
                total_score = (pdr_score * 0.25 +
                               throughput_score * 0.2 +
                               delay_score * 0.2 +
                               mac_delay_score * 0.15 +
                               energy_score * 0.1 +
                               collision_score * 0.1)

                protocol_scores[protocol] = total_score

            # 找出最佳协议
            best_protocol = max(protocol_scores, key=protocol_scores.get)

            # 找出每个指标的最佳协议
            best_pdr_protocol = df.groupby('protocol')['pdr_percent'].mean().idxmax()
            best_throughput_protocol = df.groupby('protocol')['throughput_kbps'].mean().idxmax()
            best_delay_protocol = df.groupby('protocol')['e2e_delay_ms'].mean().idxmin()
            best_mac_delay_protocol = df.groupby('protocol')['mac_delay_ms'].mean().idxmin()
            best_energy_protocol = df.groupby('protocol')['avg_energy_j'].mean().idxmin()
            best_collision_protocol = df.groupby('protocol')['collision_num'].mean().idxmin()

            # 添加分析和建议
            f.write(f'''
                <h2>分析与结论</h2>

                <div class="summary-card">
                    <h3>综合表现评估</h3>
                    <p>经过综合评分，<span class="highlight">{best_protocol}</span> 协议在测试场景中整体表现最佳。</p>

                    <h3>各指标最佳表现</h3>
                    <ul>
                        <li>最高数据包投递率: <span class="highlight">{best_pdr_protocol}</span></li>
                        <li>最高吞吐量: <span class="highlight">{best_throughput_protocol}</span></li>
                        <li>最低端到端延迟: <span class="highlight">{best_delay_protocol}</span></li>
                        <li>最低MAC层延迟: <span class="highlight">{best_mac_delay_protocol}</span></li>
                        <li>最低能量消耗: <span class="highlight">{best_energy_protocol}</span></li>
                        <li>最少碰撞次数: <span class="highlight">{best_collision_protocol}</span></li>
                    </ul>
                </div>

                <div class="recommendation">
                    <h3>协议选择建议</h3>
                    <p>基于测试结果，我们建议:</p>
                    <ul>
            ''')

            # 根据不同场景提供建议
            protocols_by_density = {}
            for density in df['num_drones'].unique():
                density_df = df[df['num_drones'] == density]
                protocols_scores = {}

                for protocol in protocols:
                    protocol_density_df = density_df[density_df['protocol'] == protocol]
                    if len(protocol_density_df) > 0:
                        # 同样的评分逻辑
                        pdr_score = protocol_density_df['pdr_percent'].mean() / density_df['pdr_percent'].max() if \
                            density_df['pdr_percent'].max() > 0 else 0
                        throughput_score = protocol_density_df['throughput_kbps'].mean() / density_df[
                            'throughput_kbps'].max() if density_df['throughput_kbps'].max() > 0 else 0

                        delay_min = density_df['e2e_delay_ms'].min()
                        delay_max = density_df['e2e_delay_ms'].max()
                        delay_range = delay_max - delay_min
                        delay_score = 1 - ((protocol_density_df[
                                                'e2e_delay_ms'].mean() - delay_min) / delay_range) if delay_range > 0 else 0.5

                        total_score = (pdr_score * 0.4 + throughput_score * 0.3 + delay_score * 0.3)
                        protocols_scores[protocol] = total_score

                if protocols_scores:
                    protocols_by_density[density] = max(protocols_scores, key=protocols_scores.get)

            # 添加密度相关建议
            for density, best_proto in sorted(protocols_by_density.items()):
                if density <= 10:
                    density_desc = "小规模网络"
                elif density <= 20:
                    density_desc = "中等规模网络"
                else:
                    density_desc = "大规模网络"

                f.write(
                    f'<li>对于{density_desc} ({density}架无人机): 建议使用 <span class="highlight">{best_proto}</span></li>')

            # 添加业务类型相关建议
            if len(traffic_patterns) > 1:
                f.write('<li>不同业务模式建议:</li><ul>')
                for traffic in traffic_patterns:
                    traffic_df = df[df['traffic_pattern'] == traffic]
                    best_traffic_protocol = traffic_df.groupby('protocol')['pdr_percent'].mean().idxmax()
                    f.write(
                        f'<li>对于{traffic}业务模式: 建议使用 <span class="highlight">{best_traffic_protocol}</span></li>')
                f.write('</ul>')

            f.write('''
                    </ul>
                </div>

                <h2>详细测试数据</h2>
            ''')

            # 添加详细数据表格
            f.write('''
                <table>
                    <tr>
                        <th>协议</th>
                        <th>无人机数量</th>
                        <th>业务模式</th>
                        <th>业务流数量</th>
                        <th>端到端延迟 (ms)</th>
                        <th>数据包投递率 (%)</th>
                        <th>吞吐量 (kbps)</th>
                        <th>跳数</th>
                        <th>MAC延迟 (ms)</th>
                        <th>碰撞次数</th>
                        <th>平均能耗 (J)</th>
                    </tr>
            ''')

            # 为每个测试添加一行
            for _, row in df.iterrows():
                f.write(f'''
                    <tr>
                        <td>{row['protocol']}</td>
                        <td>{row['num_drones']}</td>
                        <td>{row['traffic_pattern']}</td>
                        <td>{row['num_flows']}</td>
                        <td>{row['e2e_delay_ms']:.2f}</td>
                        <td>{row['pdr_percent']:.2f}</td>
                        <td>{row['throughput_kbps']:.2f}</td>
                        <td>{row['hop_count']:.2f}</td>
                        <td>{row['mac_delay_ms']:.2f}</td>
                        <td>{row['collision_num']:.0f}</td>
                        <td>{row['avg_energy_j']:.2f}</td>
                    </tr>
                ''')

            f.write('''
                </table>

                <footer>
                    <p>注: 本报告通过自动化测试生成，实际部署时应考虑具体应用场景进行最终决策。</p>
                </footer>
            </body>
            </html>
            ''')

        logging.info(f"HTML报告已生成: {html_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MAC协议性能测试工具")
    parser.add_argument("--protocols", nargs='+', default=["PPO-STDMA", "DQN-STDMA", "TraTDMA", "BasicSTDMA"],
                        help="要测试的MAC协议")
    parser.add_argument("--drone-counts", nargs='+', type=int, default=[30],
                        help="要测试的无人机数量")
    parser.add_argument("--traffic-patterns", nargs='+', default=["CBR", "POISSON", "BURST", "VBR", "PERIODIC"],
                        help="要测试的业务模式")
    parser.add_argument("--flow-counts", nargs='+', type=int, default=[1],
                        help="要测试的业务流数量")
    parser.add_argument("--seeds", nargs='+', type=int, default=[42],
                        help="随机种子")
    parser.add_argument("--sim-time", type=float, default=5 * 1e6,
                        help="仿真时间(μs)")
    parser.add_argument("--output-dir", default="test_results",
                        help="结果输出目录")

    args = parser.parse_args()

    logging.info("启动MAC协议性能测试")

    # 初始化测试器
    tester = ProtocolTester(output_dir=args.output_dir)

    # 过滤可用协议
    available_protocols = {}
    for protocol_name in args.protocols:
        if protocol_name in tester.protocols:
            available_protocols[protocol_name] = tester.protocols[protocol_name]
        else:
            logging.warning(f"未知协议: {protocol_name}")

    if not available_protocols:
        logging.error("没有可用的协议进行测试")
        return

    tester.protocols = available_protocols

    # 运行比较测试
    tester.run_comparison_tests(
        drone_counts=args.drone_counts,
        traffic_patterns=args.traffic_patterns,
        flow_counts=args.flow_counts,
        seeds=args.seeds,
        sim_time=args.sim_time
    )

    logging.info("测试完成")


if __name__ == "__main__":
    main()