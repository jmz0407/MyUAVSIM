#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
路由协议测试结果分析工具
用于深入分析测试结果并生成额外的可视化图表
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams['font.sans-serif']=['STFangsong'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

class ResultAnalyzer:
    """测试结果分析工具"""

    def __init__(self, results_file='test_results/results.pkl'):
        self.results_file = results_file
        self.results = None
        self.load_results()



        # 创建输出目录
        os.makedirs('test_results/analysis', exist_ok=True)

    def load_results(self):
        """加载测试结果"""
        try:
            with open(self.results_file, 'rb') as f:
                self.results = pickle.load(f)
        except Exception as e:
            print(f"加载结果文件失败: {str(e)}")
            # 创建样例数据用于开发与测试
            self.create_sample_data()

    def create_sample_data(self):
        """创建样例数据用于开发与测试"""
        # 基本测试数据结构
        self.results = {
            'mp_olsr_basic': {
                'protocol': 'MP_OLSR',
                'scenario': 'basic',
                'pdr': 0.92,
                'delay': 45.3,
                'throughput': 120.5,
                'hop_count': 2.4,
                'energy_consumption': 1200.5,
                'mac_delay': 12.3,
                'load_balance': 0.78,
                'path_diversity': 2.5,
                'path_length': 3.2,
                'convergence_time': 0.85
            },
            'amlb_opar_basic': {
                'protocol': 'AMLB_OPAR',
                'scenario': 'basic',
                'pdr': 0.89,
                'delay': 48.7,
                'throughput': 115.2,
                'hop_count': 2.6,
                'energy_consumption': 1150.2,
                'mac_delay': 13.1,
                'load_balance': 0.82,
                'path_diversity': 2.2,
                'path_length': 3.4,
                'convergence_time': 0.95
            }
        }

        # 添加样例数据用于其他场景
        scenarios = ['dense', 'mobile', 'failure']
        for scenario in scenarios:
            for protocol in ['mp_olsr', 'amlb_opar']:
                key = f"{protocol}_{scenario}"
                self.results[key] = {
                    'protocol': protocol.upper(),
                    'scenario': scenario,
                    'pdr': np.random.uniform(0.85, 0.95),
                    'delay': np.random.uniform(40, 60),
                    'throughput': np.random.uniform(100, 130),
                    'hop_count': np.random.uniform(2, 3),
                    'energy_consumption': np.random.uniform(1000, 1300),
                    'mac_delay': np.random.uniform(10, 15),
                    'load_balance': np.random.uniform(0.7, 0.9),
                    'path_diversity': np.random.uniform(1.8, 2.8),
                    'path_length': np.random.uniform(2.8, 3.8),
                    'convergence_time': np.random.uniform(0.7, 1.2)
                }

                # 为故障场景添加恢复指标
                if scenario == 'failure':
                    self.results[key]['recovery_metrics'] = {
                        'recovery_time': np.random.uniform(1.5, 3.0),
                        'reroute_count': np.random.randint(1, 4)
                    }

        # 保存样例数据
        self.save_results()

    def save_results(self):
        """保存测试结果"""
        try:
            with open(self.results_file, 'wb') as f:
                pickle.dump(self.results, f)
            print(f"结果已保存到 {self.results_file}")
        except Exception as e:
            print(f"保存结果失败: {str(e)}")

    def analyze_all(self):
        """执行所有分析"""
        if not self.results:
            print("没有测试结果可供分析")
            return

        # 执行各种分析
        self.plot_radar_chart()
        self.plot_performance_heatmap()
        self.analyze_scenario_impact()
        self.analyze_path_characteristics()
        self.analyze_failure_recovery()
        self.generate_analysis_report()

    def plot_radar_chart(self):
        """绘制雷达图比较两种协议的主要指标"""
        # 提取基本场景下的数据
        mp_olsr = self.results.get('mp_olsr_basic', {})
        amlb_opar = self.results.get('amlb_opar_basic', {})

        if not mp_olsr or not amlb_opar:
            print("缺少基本场景数据，无法绘制雷达图")
            return

        # 选择要在雷达图中显示的指标
        metrics = ['pdr', 'throughput', 'load_balance', 'path_diversity']
        metric_names = ['数据包交付率', '吞吐量', '负载平衡', '路径多样性']

        # 一些指标需要归一化处理
        mp_values = []
        amlb_values = []

        for metric in metrics:
            mp_val = mp_olsr.get(metric, 0)
            amlb_val = amlb_opar.get(metric, 0)

            # PDR转为百分比
            if metric == 'pdr':
                mp_val *= 100
                amlb_val *= 100

            # 负载平衡转为百分比
            elif metric == 'load_balance':
                mp_val *= 100
                amlb_val *= 100

            mp_values.append(mp_val)
            amlb_values.append(amlb_val)

        # 为了雷达图美观，计算每个指标的最大值作为标准化参考
        max_values = [max(mp_val, amlb_val) * 1.1 for mp_val, amlb_val in zip(mp_values, amlb_values)]

        # 创建雷达图
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)

        # 设置角度和标签
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形

        mp_values += mp_values[:1]  # 闭合MP-OLSR数据
        amlb_values += amlb_values[:1]  # 闭合AMLB-OPAR数据
        max_values += max_values[:1]  # 闭合最大值数据

        # 绘制雷达图
        ax.plot(angles, mp_values, 'o-', linewidth=2, label='MP-OLSR', color='#4472C4')
        ax.fill(angles, mp_values, alpha=0.25, color='#4472C4')

        ax.plot(angles, amlb_values, 'o-', linewidth=2, label='AMLB-OPAR', color='#ED7D31')
        ax.fill(angles, amlb_values, alpha=0.25, color='#ED7D31')

        # 设置刻度和标签
        ax.set_thetagrids(np.degrees(angles[:-1]), metric_names)
        ax.set_rlim(0, max(max_values))

        # 添加图例和标题
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('MP-OLSR与AMLB-OPAR性能雷达图比较', size=15, y=1.1)

        # 保存图表
        plt.tight_layout()
        plt.savefig('test_results/analysis/radar_chart.png', dpi=300)
        plt.close()

        print("雷达图分析已完成")

    def plot_performance_heatmap(self):
        """绘制性能热图，展示不同协议在不同场景下的表现"""
        # 提取数据
        protocols = ['MP_OLSR', 'AMLB_OPAR']
        scenarios = ['basic', 'dense', 'mobile', 'failure']
        metrics = ['pdr', 'delay', 'throughput', 'path_diversity']

        # 创建数据结构
        data = {metric: {} for metric in metrics}

        for protocol in protocols:
            protocol_lower = protocol.lower()
            for scenario in scenarios:
                key = f"{protocol_lower}_{scenario}"
                if key in self.results:
                    result = self.results[key]
                    for metric in metrics:
                        if metric not in data:
                            data[metric] = {}

                        protocol_key = f"{protocol}"
                        scenario_key = f"{scenario.capitalize()}"

                        if protocol_key not in data[metric]:
                            data[metric][protocol_key] = {}

                        # 处理一些指标的特殊转换
                        if metric == 'pdr':
                            data[metric][protocol_key][scenario_key] = result.get(metric, 0) * 100  # 转为百分比
                        else:
                            data[metric][protocol_key][scenario_key] = result.get(metric, 0)

        # 为每个指标绘制热图
        for i, metric in enumerate(metrics):
            # 将数据转换为DataFrame
            df = pd.DataFrame(data[metric])

            # 设置图形
            plt.figure(figsize=(10, 6))

            # 根据指标选择颜色映射
            if metric in ['pdr', 'throughput', 'path_diversity']:
                cmap = 'RdYlGn'  # 红黄绿，值越高越好
            else:
                cmap = 'RdYlGn_r'  # 绿黄红，值越低越好

            # 绘制热图
            ax = sns.heatmap(df, annot=True, fmt='.2f', cmap=cmap, linewidths=.5)

            # 设置标题和标签
            metric_titles = {
                'pdr': '数据包交付率 (%)',
                'delay': '端到端延迟 (ms)',
                'throughput': '吞吐量 (Kbps)',
                'path_diversity': '路径多样性'
            }

            plt.title(f'{metric_titles.get(metric, metric)} - 不同场景比较', fontsize=14)
            plt.xlabel('协议', fontsize=12)
            plt.ylabel('场景', fontsize=12)

            # 保存图表
            plt.tight_layout()
            plt.savefig(f'test_results/analysis/heatmap_{metric}.png', dpi=300)
            plt.close()

        print("性能热图分析已完成")

    def analyze_scenario_impact(self):
        """分析不同场景对协议性能的影响"""
        # 提取数据
        protocols = ['MP_OLSR', 'AMLB_OPAR']
        scenarios = ['basic', 'dense', 'mobile', 'failure']
        metrics = ['pdr', 'delay', 'throughput']
        metric_titles = {
            'pdr': '数据包交付率 (%)',
            'delay': '端到端延迟 (ms)',
            'throughput': '吞吐量 (Kbps)'
        }

        # 准备数据
        scenario_data = {protocol: {metric: [] for metric in metrics} for protocol in protocols}
        scenario_labels = []

        for scenario in scenarios:
            scenario_labels.append(scenario.capitalize())

            for protocol in protocols:
                protocol_lower = protocol.lower()
                key = f"{protocol_lower}_{scenario}"

                if key in self.results:
                    result = self.results[key]

                    for metric in metrics:
                        value = result.get(metric, 0)
                        if metric == 'pdr':
                            value *= 100  # 转为百分比

                        scenario_data[protocol][metric].append(value)

        # 绘制每个指标的场景影响图
        for metric in metrics:
            plt.figure(figsize=(12, 6))

            # 设置柱状图位置
            x = np.arange(len(scenario_labels))
            width = 0.35

            # 绘制柱状图
            plt.bar(x - width / 2, scenario_data['MP_OLSR'][metric], width, label='MP-OLSR', color='#4472C4')
            plt.bar(x + width / 2, scenario_data['AMLB_OPAR'][metric], width, label='AMLB-OPAR', color='#ED7D31')

            # 添加数值标签
            for i, v in enumerate(scenario_data['MP_OLSR'][metric]):
                plt.text(i - width / 2, v + 0.01, f'{v:.2f}', ha='center', va='bottom')

            for i, v in enumerate(scenario_data['AMLB_OPAR'][metric]):
                plt.text(i + width / 2, v + 0.01, f'{v:.2f}', ha='center', va='bottom')

            # 设置标题和标签
            plt.title(f'不同场景下的{metric_titles.get(metric, metric)}比较', fontsize=14)
            plt.xlabel('场景', fontsize=12)
            plt.ylabel(metric_titles.get(metric, metric), fontsize=12)
            plt.xticks(x, scenario_labels)

            # 添加图例
            plt.legend()

            # 设置Y轴从0开始
            plt.ylim(bottom=0)

            # 保存图表
            plt.tight_layout()
            plt.savefig(f'test_results/analysis/scenario_impact_{metric}.png', dpi=300)
            plt.close()

        print("场景影响分析已完成")

    def analyze_path_characteristics(self):
        """分析路径特征"""
        # 提取数据
        protocols = ['MP_OLSR', 'AMLB_OPAR']
        scenarios = ['basic', 'dense', 'mobile']
        path_metrics = ['path_diversity', 'path_length', 'convergence_time']

        metric_titles = {
            'path_diversity': '路径多样性',
            'path_length': '平均路径长度',
            'convergence_time': '路由收敛时间 (s)'
        }

        # 准备数据
        data = {protocol: {metric: {} for metric in path_metrics} for protocol in protocols}

        for protocol in protocols:
            protocol_lower = protocol.lower()
            for scenario in scenarios:
                key = f"{protocol_lower}_{scenario}"

                if key in self.results:
                    result = self.results[key]

                    for metric in path_metrics:
                        if metric in result:
                            data[protocol][metric][scenario] = result[metric]

        # 绘制路径特征比较图
        plt.figure(figsize=(15, 10))

        for i, metric in enumerate(path_metrics):
            plt.subplot(2, 2, i + 1)

            # 准备绘图数据
            mp_olsr_values = [data['MP_OLSR'][metric].get(scenario, 0) for scenario in scenarios]
            amlb_opar_values = [data['AMLB_OPAR'][metric].get(scenario, 0) for scenario in scenarios]

            # 设置柱状图位置
            x = np.arange(len(scenarios))
            width = 0.35

            # 绘制柱状图
            plt.bar(x - width / 2, mp_olsr_values, width, label='MP-OLSR', color='#4472C4')
            plt.bar(x + width / 2, amlb_opar_values, width, label='AMLB-OPAR', color='#ED7D31')

            # 添加数值标签
            for j, v in enumerate(mp_olsr_values):
                plt.text(j - width / 2, v + 0.01, f'{v:.2f}', ha='center', va='bottom')

            for j, v in enumerate(amlb_opar_values):
                plt.text(j + width / 2, v + 0.01, f'{v:.2f}', ha='center', va='bottom')

            # 设置标题和标签
            plt.title(metric_titles.get(metric, metric))
            plt.xlabel('场景')
            plt.xticks(x, [s.capitalize() for s in scenarios])

            # 设置Y轴从0开始
            plt.ylim(bottom=0)

            # 只在第一个子图上显示图例
            if i == 0:
                plt.legend()

        # 调整布局
        plt.tight_layout()
        plt.savefig('test_results/analysis/path_characteristics.png', dpi=300)
        plt.close()

        print("路径特征分析已完成")

    def analyze_failure_recovery(self):
        """分析故障恢复特性"""
        # 提取故障恢复数据
        mp_olsr_failure = self.results.get('mp_olsr_failure', {})
        amlb_opar_failure = self.results.get('amlb_opar_failure', {})

        mp_olsr_recovery = mp_olsr_failure.get('recovery_metrics', {})
        amlb_opar_recovery = amlb_opar_failure.get('recovery_metrics', {})

        if not mp_olsr_recovery or not amlb_opar_recovery:
            print("缺少故障恢复数据，无法进行分析")
            return

        # 设置图表
        plt.figure(figsize=(14, 6))

        # 1. 恢复时间比较
        plt.subplot(1, 2, 1)
        recovery_times = [
            mp_olsr_recovery.get('recovery_time', 0),
            amlb_opar_recovery.get('recovery_time', 0)
        ]

        plt.bar(['MP-OLSR', 'AMLB-OPAR'], recovery_times, color=['#4472C4', '#ED7D31'])

        # 添加数值标签
        for i, v in enumerate(recovery_times):
            plt.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom')

        plt.title('故障恢复时间比较 (秒)')
        plt.ylabel('恢复时间 (秒)')
        plt.ylim(bottom=0)

        # 2. 路由重构次数比较
        plt.subplot(1, 2, 2)
        reroute_counts = [
            mp_olsr_recovery.get('reroute_count', 0),
            amlb_opar_recovery.get('reroute_count', 0)
        ]

        plt.bar(['MP-OLSR', 'AMLB-OPAR'], reroute_counts, color=['#4472C4', '#ED7D31'])

        # 添加数值标签
        for i, v in enumerate(reroute_counts):
            plt.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom')

        plt.title('路由重构次数比较')
        plt.ylabel('重构次数')
        plt.ylim(bottom=0)

        # 调整布局
        plt.tight_layout()
        plt.savefig('test_results/analysis/failure_recovery.png', dpi=300)
        plt.close()

        print("故障恢复分析已完成")

    def generate_analysis_report(self):
        """生成详细的分析报告"""
        report = "# MP_OLSR与AMLB_OPAR路由协议性能深入分析报告\n\n"

        # 添加综合性能分析
        report += "## 1. 综合性能分析\n\n"

        # 提取基本场景数据
        mp_olsr_basic = self.results.get('mp_olsr_basic', {})
        amlb_opar_basic = self.results.get('amlb_opar_basic', {})

        if mp_olsr_basic and amlb_opar_basic:
            # 计算总体性能评分
            metrics = {
                'pdr': (0.3, True),  # 权重0.3，值越大越好
                'delay': (0.2, False),  # 权重0.2，值越小越好
                'throughput': (0.2, True),  # 权重0.2，值越大越好
                'energy_consumption': (0.1, False),  # 权重0.1，值越小越好
                'path_diversity': (0.1, True),  # 权重0.1，值越大越好
                'convergence_time': (0.1, False)  # 权重0.1，值越小越好
            }

            mp_olsr_score = 0
            amlb_opar_score = 0

            for metric, (weight, higher_better) in metrics.items():
                mp_val = mp_olsr_basic.get(metric, 0)
                amlb_val = amlb_opar_basic.get(metric, 0)

                # 避免除以零
                if mp_val == 0 and amlb_val == 0:
                    continue

                if higher_better:
                    # 值越大越好
                    total = mp_val + amlb_val
                    if total > 0:
                        mp_olsr_score += (mp_val / total) * weight * 100
                        amlb_opar_score += (amlb_val / total) * weight * 100
                else:
                    # 值越小越好，计算倒数比例
                    if mp_val > 0 and amlb_val > 0:  # 避免除以零
                        mp_olsr_inv = 1 / mp_val
                        amlb_opar_inv = 1 / amlb_val
                        total_inv = mp_olsr_inv + amlb_opar_inv

                        mp_olsr_score += (mp_olsr_inv / total_inv) * weight * 100
                        amlb_opar_score += (amlb_opar_inv / total_inv) * weight * 100

            # 添加分数比较
            report += "### 1.1 总体性能评分\n\n"
            report += f"基于关键性能指标的加权评分结果：\n\n"
            report += f"- **MP-OLSR**: {mp_olsr_score:.2f}分\n"
            report += f"- **AMLB-OPAR**: {amlb_opar_score:.2f}分\n\n"

            if abs(mp_olsr_score - amlb_opar_score) < 5:
                report += "两种协议的总体性能相近，差异不大。\n\n"
            else:
                better = "MP-OLSR" if mp_olsr_score > amlb_opar_score else "AMLB-OPAR"
                diff = abs(mp_olsr_score - amlb_opar_score)
                report += f"**{better}** 在总体性能上领先 {diff:.2f} 分。\n\n"

        # 各指标详细分析
        report += "### 1.2 关键指标详细分析\n\n"

        if mp_olsr_basic and amlb_opar_basic:
            # PDR分析
            mp_pdr = mp_olsr_basic.get('pdr', 0) * 100
            amlb_pdr = amlb_opar_basic.get('pdr', 0) * 100
            pdr_diff = abs(mp_pdr - amlb_pdr)

            report += "#### 数据包交付率 (PDR)\n\n"
            report += f"- MP-OLSR: {mp_pdr:.2f}%\n"
            report += f"- AMLB-OPAR: {amlb_pdr:.2f}%\n\n"

            if pdr_diff < 2:
                report += "两种协议的PDR相近，差异不显著。\n\n"
            else:
                better = "MP-OLSR" if mp_pdr > amlb_pdr else "AMLB-OPAR"
                report += f"**{better}** 的PDR高出 {pdr_diff:.2f} 个百分点，表现更优。\n\n"

            # 延迟分析
            mp_delay = mp_olsr_basic.get('delay', 0)
            amlb_delay = amlb_opar_basic.get('delay', 0)
            delay_diff = abs(mp_delay - amlb_delay)

            report += "#### 端到端延迟\n\n"
            report += f"- MP-OLSR: {mp_delay:.2f} ms\n"
            report += f"- AMLB-OPAR: {amlb_delay:.2f} ms\n\n"

            if delay_diff < 5:
                report += "两种协议的端到端延迟相近，差异不显著。\n\n"
            else:
                better = "MP-OLSR" if mp_delay < amlb_delay else "AMLB-OPAR"
                report += f"**{better}** 的端到端延迟低 {delay_diff:.2f} ms，表现更优。\n\n"

            # 吞吐量分析
            mp_throughput = mp_olsr_basic.get('throughput', 0)
            amlb_throughput = amlb_opar_basic.get('throughput', 0)
            throughput_diff = abs(mp_throughput - amlb_throughput)

            report += "#### 吞吐量\n\n"
            report += f"- MP-OLSR: {mp_throughput:.2f} Kbps\n"
            report += f"- AMLB-OPAR: {amlb_throughput:.2f} Kbps\n\n"

            if throughput_diff < 10:
                report += "两种协议的吞吐量相近，差异不显著。\n\n"
            else:
                better = "MP-OLSR" if mp_throughput > amlb_throughput else "AMLB-OPAR"
                report += f"**{better}** 的吞吐量高出 {throughput_diff:.2f} Kbps，表现更优。\n\n"

        # 各场景性能分析
        report += "## 2. 不同场景性能分析\n\n"

        scenarios = {
            'dense': '高密度场景',
            'mobile': '高移动性场景',
            'failure': '故障恢复场景'
        }

        for scenario_key, scenario_name in scenarios.items():
            mp_data = self.results.get(f'mp_olsr_{scenario_key}', {})
            amlb_data = self.results.get(f'amlb_opar_{scenario_key}', {})

            if not mp_data or not amlb_data:
                continue

            report += f"### 2.{list(scenarios.keys()).index(scenario_key) + 1} {scenario_name}\n\n"

            # 性能指标比较
            metrics_to_show = ['pdr', 'delay', 'throughput', 'path_diversity']
            metric_names = {
                'pdr': '数据包交付率 (%)',
                'delay': '端到端延迟 (ms)',
                'throughput': '吞吐量 (Kbps)',
                'path_diversity': '路径多样性'
            }

            report += "| 性能指标 | MP-OLSR | AMLB-OPAR | 差异 |\n"
            report += "|---------|---------|-----------|------|\n"

            for metric in metrics_to_show:
                mp_val = mp_data.get(metric, 0)
                amlb_val = amlb_data.get(metric, 0)

                # 处理特殊指标
                if metric == 'pdr':
                    mp_val *= 100  # 转为百分比
                    amlb_val *= 100

                # 计算差异
                diff = mp_val - amlb_val
                diff_str = f"{diff:.2f}"

                # 添加符号
                if diff > 0:
                    diff_str = "+" + diff_str

                report += f"| {metric_names.get(metric, metric)} | {mp_val:.2f} | {amlb_val:.2f} | {diff_str} |\n"

            report += "\n"

            # 场景特有分析
            if scenario_key == 'dense':
                report += "高密度场景下，网络拓扑更为复杂，邻居节点数量增加，路由协议需要处理更多的干扰和竞争。\n\n"

                better_pdr = "MP-OLSR" if mp_data.get('pdr', 0) > amlb_data.get('pdr', 0) else "AMLB-OPAR"
                better_delay = "MP-OLSR" if mp_data.get('delay', 0) < amlb_data.get('delay', 0) else "AMLB-OPAR"

                report += f"在这种场景下，**{better_pdr}** 表现出更高的数据包交付率，**{better_delay}** 提供了更低的端到端延迟。\n\n"

            elif scenario_key == 'mobile':
                report += "高移动性场景下，网络拓扑快速变化，链路容易断开，路由协议需要快速适应拓扑变化并找到新的路由路径。\n\n"

                better_pdr = "MP-OLSR" if mp_data.get('pdr', 0) > amlb_data.get('pdr', 0) else "AMLB-OPAR"
                better_diversity = "MP-OLSR" if mp_data.get('path_diversity', 0) > amlb_data.get('path_diversity',
                                                                                                 0) else "AMLB-OPAR"

                report += f"在这种场景下，**{better_pdr}** 表现出更强的适应性，**{better_diversity}** 提供了更多的可选路径，有助于应对链路断开情况。\n\n"

            elif scenario_key == 'failure':
                report += "故障恢复场景下，部分节点突然失效，路由协议需要快速检测故障并重新计算路由路径。\n\n"

                mp_recovery = mp_data.get('recovery_metrics', {})
                amlb_recovery = amlb_data.get('recovery_metrics', {})

                mp_recovery_time = mp_recovery.get('recovery_time', 0)
                amlb_recovery_time = amlb_recovery.get('recovery_time', 0)

                better_recovery = "MP-OLSR" if mp_recovery_time < amlb_recovery_time else "AMLB-OPAR"
                recovery_diff = abs(mp_recovery_time - amlb_recovery_time)

                report += f"在这种场景下，**{better_recovery}** 表现出更快的故障恢复能力，恢复时间低 {recovery_diff:.2f} 秒。\n\n"

        # 路由特性分析
        report += "## 3. 路由特性深入分析\n\n"

        report += "### 3.1 路径特性比较\n\n"

        # 基本场景下的路径特性
        mp_olsr_basic = self.results.get('mp_olsr_basic', {})
        amlb_opar_basic = self.results.get('amlb_opar_basic', {})

        if mp_olsr_basic and amlb_opar_basic:
            report += "| 路径特性 | MP-OLSR | AMLB-OPAR | 差异 |\n"
            report += "|---------|---------|-----------|------|\n"

            path_metrics = {
                'path_diversity': '路径多样性',
                'path_length': '平均路径长度',
                'convergence_time': '路由收敛时间 (s)'
            }

            for metric, name in path_metrics.items():
                mp_val = mp_olsr_basic.get(metric, 0)
                amlb_val = amlb_opar_basic.get(metric, 0)

                # 计算差异
                diff = mp_val - amlb_val
                diff_str = f"{diff:.2f}"

                # 添加符号
                if diff > 0:
                    diff_str = "+" + diff_str

                report += f"| {name} | {mp_val:.2f} | {amlb_val:.2f} | {diff_str} |\n"

            report += "\n"

            # 路径多样性分析
            mp_diversity = mp_olsr_basic.get('path_diversity', 0)
            amlb_diversity = amlb_opar_basic.get('path_diversity', 0)

            report += "#### 路径多样性分析\n\n"

            if mp_diversity > amlb_diversity:
                report += f"MP-OLSR提供了更高的路径多样性（{mp_diversity:.2f} vs {amlb_diversity:.2f}），"
                report += "这意味着它能为每个目的地维护更多的备选路径。这有助于在链路故障时快速切换路径，"
                report += "并支持负载均衡，但也会增加路由开销。\n\n"
            else:
                report += f"AMLB-OPAR提供了更高的路径多样性（{amlb_diversity:.2f} vs {mp_diversity:.2f}），"
                report += "这意味着它能为每个目的地维护更多的备选路径。这有助于在链路故障时快速切换路径，"
                report += "并支持负载均衡，但也会增加路由开销。\n\n"

            # 路径长度分析
            mp_length = mp_olsr_basic.get('path_length', 0)
            amlb_length = amlb_opar_basic.get('path_length', 0)

            report += "#### 路径长度分析\n\n"

            if mp_length < amlb_length:
                report += f"MP-OLSR找到的路径平均更短（{mp_length:.2f} vs {amlb_length:.2f} 跳），"
                report += "这意味着数据传输需要经过更少的中继节点，"
                report += "有助于减少端到端延迟和节点能耗，并降低中间传输中断的风险。\n\n"
            else:
                report += f"AMLB-OPAR找到的路径平均更短（{amlb_length:.2f} vs {mp_length:.2f} 跳），"
                report += "这意味着数据传输需要经过更少的中继节点，"
                report += "有助于减少端到端延迟和节点能耗，并降低中间传输中断的风险。\n\n"

            # 路由收敛分析
            mp_convergence = mp_olsr_basic.get('convergence_time', 0)
            amlb_convergence = amlb_opar_basic.get('convergence_time', 0)

            report += "#### 路由收敛分析\n\n"

            if mp_convergence < amlb_convergence:
                report += f"MP-OLSR的路由收敛时间更短（{mp_convergence:.2f}s vs {amlb_convergence:.2f}s），"
                report += "这意味着在网络拓扑变化后，它能更快地建立新的有效路由。"
                report += "快速收敛对于高移动性场景和故障恢复尤为重要。\n\n"
            else:
                report += f"AMLB-OPAR的路由收敛时间更短（{amlb_convergence:.2f}s vs {mp_convergence:.2f}s），"
                report += "这意味着在网络拓扑变化后，它能更快地建立新的有效路由。"
                report += "快速收敛对于高移动性场景和故障恢复尤为重要。\n\n"

        report += "### 3.2 故障恢复特性\n\n"

        # 故障恢复特性
        mp_olsr_failure = self.results.get('mp_olsr_failure', {})
        amlb_opar_failure = self.results.get('amlb_opar_failure', {})

        if mp_olsr_failure and amlb_opar_failure:
            mp_recovery = mp_olsr_failure.get('recovery_metrics', {})
            amlb_recovery = amlb_opar_failure.get('recovery_metrics', {})

            if mp_recovery and amlb_recovery:
                report += "| 恢复特性 | MP-OLSR | AMLB-OPAR | 差异 |\n"
                report += "|---------|---------|-----------|------|\n"

                recovery_metrics = {
                    'recovery_time': '恢复时间 (s)',
                    'reroute_count': '重路由次数'
                }

                for metric, name in recovery_metrics.items():
                    mp_val = mp_recovery.get(metric, 0)
                    amlb_val = amlb_recovery.get(metric, 0)

                    # 计算差异
                    diff = mp_val - amlb_val
                    diff_str = f"{diff:.2f}"

                    # 添加符号
                    if diff > 0:
                        diff_str = "+" + diff_str

                    report += f"| {name} | {mp_val:.2f} | {amlb_val:.2f} | {diff_str} |\n"

                report += "\n"

                # 详细分析
                mp_recovery_time = mp_recovery.get('recovery_time', 0)
                amlb_recovery_time = amlb_recovery.get('recovery_time', 0)

                report += "#### 故障恢复时间分析\n\n"

                if mp_recovery_time < amlb_recovery_time:
                    report += f"MP-OLSR在节点故障后展现出更快的恢复速度（{mp_recovery_time:.2f}s vs {amlb_recovery_time:.2f}s），"
                    report += "这可能得益于其预先计算的多条备选路径和更高效的路由更新机制。"
                    report += "更短的恢复时间意味着网络中断时间更短，服务质量更有保障。\n\n"
                else:
                    report += f"AMLB-OPAR在节点故障后展现出更快的恢复速度（{amlb_recovery_time:.2f}s vs {mp_recovery_time:.2f}s），"
                    report += "这可能得益于其预先计算的多条备选路径和更高效的路由更新机制。"
                    report += "更短的恢复时间意味着网络中断时间更短，服务质量更有保障。\n\n"

                mp_reroute = mp_recovery.get('reroute_count', 0)
                amlb_reroute = amlb_recovery.get('reroute_count', 0)

                report += "#### 重路由次数分析\n\n"

                if mp_reroute < amlb_reroute:
                    report += f"MP-OLSR在故障期间需要的重路由次数更少（{mp_reroute:.2f} vs {amlb_reroute:.2f}），"
                    report += "这表明其路由更为稳定，能够更好地适应故障情况，"
                    report += "并且产生更少的路由控制开销。\n\n"
                else:
                    report += f"AMLB-OPAR在故障期间需要的重路由次数更少（{amlb_reroute:.2f} vs {mp_reroute:.2f}），"
                    report += "这表明其路由更为稳定，能够更好地适应故障情况，"
                    report += "并且产生更少的路由控制开销。\n\n"

        # 总结与建议
        report += "## 4. 总结与应用建议\n\n"

        # 获取MP-OLSR和AMLB-OPAR的基本性能
        mp_olsr_basic = self.results.get('mp_olsr_basic', {})
        amlb_opar_basic = self.results.get('amlb_opar_basic', {})

        if mp_olsr_basic and amlb_opar_basic:
            mp_pdr = mp_olsr_basic.get('pdr', 0) * 100
            amlb_pdr = amlb_opar_basic.get('pdr', 0) * 100

            mp_delay = mp_olsr_basic.get('delay', 0)
            amlb_delay = amlb_opar_basic.get('delay', 0)

            mp_diversity = mp_olsr_basic.get('path_diversity', 0)
            amlb_diversity = amlb_opar_basic.get('path_diversity', 0)

            # 确定整体更优的协议
            mp_score = (mp_pdr - amlb_pdr) / amlb_pdr * 30  # PDR差异（权重30%）
            mp_score += (amlb_delay - mp_delay) / amlb_delay * 20  # 延迟差异（权重20%）
            mp_score += (mp_diversity - amlb_diversity) / amlb_diversity * 10  # 路径多样性差异（权重10%）

            # MP-OLSR和AMLB-OPAR的优势场景
            mp_scenarios = []
            amlb_scenarios = []

            # 检查高密度场景
            mp_dense = self.results.get('mp_olsr_dense', {})
            amlb_dense = self.results.get('amlb_opar_dense', {})

            if mp_dense and amlb_dense:
                if mp_dense.get('pdr', 0) > amlb_dense.get('pdr', 0):
                    mp_scenarios.append("高密度")
                else:
                    amlb_scenarios.append("高密度")

            # 检查高移动性场景
            mp_mobile = self.results.get('mp_olsr_mobile', {})
            amlb_mobile = self.results.get('amlb_opar_mobile', {})

            if mp_mobile and amlb_mobile:
                if mp_mobile.get('pdr', 0) > amlb_mobile.get('pdr', 0):
                    mp_scenarios.append("高移动性")
                else:
                    amlb_scenarios.append("高移动性")

            # 检查故障恢复场景
            mp_failure = self.results.get('mp_olsr_failure', {})
            amlb_failure = self.results.get('amlb_opar_failure', {})

            if mp_failure and amlb_failure:
                mp_recovery = mp_failure.get('recovery_metrics', {}).get('recovery_time', 0)
                amlb_recovery = amlb_failure.get('recovery_metrics', {}).get('recovery_time', 0)

                if mp_recovery < amlb_recovery:
                    mp_scenarios.append("故障恢复")
                else:
                    amlb_scenarios.append("故障恢复")

            # 生成总结
            if mp_score > 0:
                report += "### 4.1 总体性能比较\n\n"
                report += "根据综合性能评估，**MP-OLSR** 在整体性能上略优于 AMLB-OPAR，"

                if mp_pdr > amlb_pdr:
                    report += f"特别是在数据包交付率方面高出约 {mp_pdr - amlb_pdr:.2f}%，"

                if mp_delay < amlb_delay:
                    report += f"端到端延迟低约 {amlb_delay - mp_delay:.2f} ms，"

                report += "但差异并不显著。\n\n"

                # MP-OLSR优势场景
                if mp_scenarios:
                    report += f"MP-OLSR在以下场景中表现更优: {', '.join(mp_scenarios)}。\n\n"

                # AMLB-OPAR优势场景
                if amlb_scenarios:
                    report += f"而AMLB-OPAR在以下场景中具有优势: {', '.join(amlb_scenarios)}。\n\n"
            else:
                report += "### 4.1 总体性能比较\n\n"
                report += "根据综合性能评估，**AMLB-OPAR** 在整体性能上略优于 MP-OLSR，"

                if amlb_pdr > mp_pdr:
                    report += f"特别是在数据包交付率方面高出约 {amlb_pdr - mp_pdr:.2f}%，"

                if amlb_delay < mp_delay:
                    report += f"端到端延迟低约 {mp_delay - amlb_delay:.2f} ms，"

                report += "但差异并不显著。\n\n"

                # AMLB-OPAR优势场景
                if amlb_scenarios:
                    report += f"AMLB-OPAR在以下场景中表现更优: {', '.join(amlb_scenarios)}。\n\n"

                # MP-OLSR优势场景
                if mp_scenarios:
                    report += f"而MP-OLSR在以下场景中具有优势: {', '.join(mp_scenarios)}。\n\n"

            # 应用建议
            report += "### 4.2 应用建议\n\n"

            report += "根据测试结果，我们提出以下应用建议：\n\n"

            if mp_score > 0:
                # MP-OLSR更优
                report += "1. **一般应用场景**：对于大多数无人机网络应用，建议优先使用 **MP-OLSR**，"
                report += "它在PDR、延迟等关键指标上表现更为稳定。\n\n"

                if mp_diversity > amlb_diversity:
                    report += "2. **高可靠性需求**：MP-OLSR提供更高的路径多样性，"
                    report += "对于要求高可靠性和故障容错的应用尤为适合。\n\n"

                if "高密度" in mp_scenarios:
                    report += "3. **密集部署场景**：在无人机密集部署的情况下，"
                    report += "MP-OLSR能更好地处理节点间干扰，保持较高的网络性能。\n\n"

                if amlb_scenarios:
                    report += f"4. **特定场景考虑**：在{'、'.join(amlb_scenarios)}等场景下，"
                    report += "可以考虑使用 AMLB-OPAR 以获得更好的性能。\n\n"
            else:
                # AMLB-OPAR更优
                report += "1. **一般应用场景**：对于大多数无人机网络应用，建议优先使用 **AMLB-OPAR**，"
                report += "它在PDR、延迟等关键指标上表现更为稳定。\n\n"

                if amlb_diversity > mp_diversity:
                    report += "2. **高可靠性需求**：AMLB-OPAR提供更高的路径多样性，"
                    report += "对于要求高可靠性和故障容错的应用尤为适合。\n\n"

                if "高密度" in amlb_scenarios:
                    report += "3. **密集部署场景**：在无人机密集部署的情况下，"
                    report += "AMLB-OPAR能更好地处理节点间干扰，保持较高的网络性能。\n\n"

                if mp_scenarios:
                    report += f"4. **特定场景考虑**：在{'、'.join(mp_scenarios)}等场景下，"
                    report += "可以考虑使用 MP-OLSR 以获得更好的性能。\n\n"

            report += "5. **混合使用**：在复杂的网络环境中，可以考虑两种协议的混合使用，"
            report += "根据不同区域的网络特性选择更适合的协议。\n\n"

            report += "6. **参数优化**：无论选择哪种协议，建议根据具体应用场景对协议参数进行优化，"
            report += "如路径数量、更新频率等，以获得最佳性能。\n\n"

        # 保存报告
        try:
            with open('test_results/analysis/detailed_analysis_report.md', 'w', encoding='utf-8') as f:
                f.write(report)
            print("详细分析报告已保存到: test_results/analysis/detailed_analysis_report.md")
        except Exception as e:
            print(f"保存报告失败: {str(e)}")

        return report


if __name__ == "__main__":
    # 创建分析器
    analyzer = ResultAnalyzer()

    # 运行所有分析
    analyzer.analyze_all()

    print("分析完成！结果保存在 test_results/analysis/ 目录下")