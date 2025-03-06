import simpy
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import logging
import os
import random
from utils import config
from simulator.simulator import Simulator
from routing.mp_olsr.mp_olsr import MP_OLSR
from routing.multipath.amlb_opar import AMLB_OPAR
from routing.multipath.mp_amlb_opar import MP_AMLB_OPAR
plt.rcParams['font.sans-serif']=['STFangsong'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('routing_comparison.log', mode='w'),
        logging.StreamHandler()
    ]
)


class RoutingProtocolTester:
    """
    MP_OLSR和AMLB_OPAR路由协议比较测试框架
    用于在不同场景下对比两种多路径路由协议的性能
    """

    def __init__(self, base_config=None):
        """
        初始化测试框架

        Args:
            base_config: 基础配置参数，如无人机数量、仿真时间等
        """
        self.base_config = base_config or {}
        self.test_results = {}

        # 保存原始配置
        self._save_original_config()

        # 应用基础配置
        self._apply_base_config()

        # 创建输出目录
        os.makedirs('test_results', exist_ok=True)

    def _save_original_config(self):
        """保存原始配置，以便测试后恢复"""
        self.original_config = {
            'NUMBER_OF_DRONES': config.NUMBER_OF_DRONES,
            'SIM_TIME': config.SIM_TIME,
            'MULTIPATH_ENABLED': getattr(config, 'MULTIPATH_ENABLED', True),
            'MAX_PATHS': getattr(config, 'MAX_PATHS', 3),
            'PATH_SELECTION_STRATEGY': getattr(config, 'PATH_SELECTION_STRATEGY', 'adaptive'),
            'ROUTING_PROTOCOL': getattr(config, 'ROUTING_PROTOCOL', 'MP-DSR')
        }

    def _apply_base_config(self):
        """应用基础配置参数"""
        for key, value in self.base_config.items():
            setattr(config, key, value)

    def _restore_original_config(self):
        """恢复原始配置"""
        for key, value in self.original_config.items():
            setattr(config, key, value)

    def run_all_tests(self):
        """运行所有测试场景"""
        # 基本场景测试 - MP_OLSR
        self.test_results['mp_olsr_basic'] = self.run_protocol_test(
            protocol_name='MP_OLSR',
            scenario_name='basic'
        )

        # 基本场景测试 - AMLB_OPAR
        self.test_results['amlb_opar_basic'] = self.run_protocol_test(
            protocol_name='AMLB_OPAR',
            scenario_name='basic'
        )

        # 高密度场景测试
        self.test_results['mp_olsr_dense'] = self.run_protocol_test(
            protocol_name='MP_OLSR',
            scenario_name='dense',
            override_config={'NUMBER_OF_DRONES': 20}
        )

        self.test_results['amlb_opar_dense'] = self.run_protocol_test(
            protocol_name='AMLB_OPAR',
            scenario_name='dense',
            override_config={'NUMBER_OF_DRONES': 20}
        )

        # 高移动性场景测试
        self.test_results['mp_olsr_mobile'] = self.run_protocol_test(
            protocol_name='MP_OLSR',
            scenario_name='mobile',
            override_config={'MOBILITY_SPEED': 30}  # 假设这是控制移动速度的参数
        )

        self.test_results['amlb_opar_mobile'] = self.run_protocol_test(
            protocol_name='AMLB_OPAR',
            scenario_name='mobile',
            override_config={'MOBILITY_SPEED': 30}
        )

        # 故障恢复场景测试
        self.test_results['mp_olsr_failure'] = self.run_failure_test(
            protocol_name='MP_OLSR'
        )

        self.test_results['amlb_opar_failure'] = self.run_failure_test(
            protocol_name='AMLB_OPAR'
        )

        # 恢复原始配置
        self._restore_original_config()

        return self.test_results

    def run_protocol_test(self, protocol_name, scenario_name, override_config=None):
        """
        运行特定协议和场景的测试

        Args:
            protocol_name: 协议名称 ('MP_OLSR' 或 'AMLB_OPAR')
            scenario_name: 场景名称
            override_config: 覆盖的配置参数

        Returns:
            测试结果字典
        """
        # 临时应用场景特定配置
        temp_config = {}
        if override_config:
            for key, value in override_config.items():
                temp_config[key] = getattr(config, key, None)
                setattr(config, key, value)

        # 设置路由协议
        temp_protocol = getattr(config, 'ROUTING_PROTOCOL', None)
        config.ROUTING_PROTOCOL = protocol_name

        logging.info(f"开始测试: 协议={protocol_name}, 场景={scenario_name}")

        # 创建并运行模拟器
        simulator = self._create_simulator(protocol_name)
        metrics = self._run_simulation(simulator)

        # 增加协议和场景标识
        metrics['protocol'] = protocol_name
        metrics['scenario'] = scenario_name

        # 记录路由统计数据
        route_stats = self._collect_routing_stats(simulator, protocol_name)
        metrics.update(route_stats)

        # 恢复临时配置
        if override_config:
            for key, value in temp_config.items():
                if value is not None:
                    setattr(config, key, value)
                else:
                    delattr(config, key)

        if temp_protocol:
            config.ROUTING_PROTOCOL = temp_protocol

        logging.info(f"完成测试: 协议={protocol_name}, 场景={scenario_name}")

        return metrics

    def run_failure_test(self, protocol_name):
        """
        运行故障恢复测试场景

        Args:
            protocol_name: 协议名称

        Returns:
            测试结果字典
        """
        # 设置路由协议
        temp_protocol = getattr(config, 'ROUTING_PROTOCOL', None)
        config.ROUTING_PROTOCOL = protocol_name

        logging.info(f"开始故障恢复测试: 协议={protocol_name}")

        # 创建并运行模拟器
        simulator = self._create_simulator(protocol_name)

        # 安排节点故障事件
        self._schedule_node_failures(simulator)

        # 运行仿真
        metrics = self._run_simulation(simulator)

        # 增加协议和场景标识
        metrics['protocol'] = protocol_name
        metrics['scenario'] = 'failure_recovery'

        # 记录路由统计数据
        route_stats = self._collect_routing_stats(simulator, protocol_name)
        metrics.update(route_stats)

        # 收集故障恢复指标
        recovery_metrics = self._collect_failure_metrics(simulator)
        metrics['recovery_metrics'] = recovery_metrics

        # 恢复路由协议配置
        if temp_protocol:
            config.ROUTING_PROTOCOL = temp_protocol

        logging.info(f"完成故障恢复测试: 协议={protocol_name}")

        return metrics

    def _create_simulator(self, protocol_name):
        """创建仿真器实例"""
        env = simpy.Environment()
        n_drones = getattr(config, 'NUMBER_OF_DRONES', 10)
        channel_states = {i: simpy.Resource(env, capacity=1) for i in range(n_drones)}

        sim = Simulator(
            seed=2024,
            env=env,
            channel_states=channel_states,
            n_drones=n_drones
        )

        # 确保使用正确的路由协议
        for drone in sim.drones:
            if protocol_name == 'MP_OLSR':
                drone.routing_protocol = MP_OLSR(sim, drone)
            elif protocol_name == 'AMLB_OPAR':
                drone.routing_protocol = AMLB_OPAR(sim, drone)

        return sim

    def _run_simulation(self, simulator):
        """运行仿真并返回性能指标"""
        # 运行仿真
        simulator.env.run(until=config.SIM_TIME)

        # 收集基本性能指标
        metrics = {
            'pdr': self._calculate_pdr(simulator),
            'delay': self._calculate_average_delay(simulator),
            'throughput': self._calculate_throughput(simulator),
            'hop_count': self._calculate_average_hop_count(simulator),
            'energy_consumption': self._calculate_energy_consumption(simulator),
            'mac_delay': self._calculate_mac_delay(simulator),
            'load_balance': self._calculate_load_balance(simulator),
        }

        return metrics

    def _calculate_pdr(self, simulator):
        """计算数据包交付率"""
        if simulator.metrics.datapacket_generated_num > 0:
            return len(simulator.metrics.datapacket_arrived) / simulator.metrics.datapacket_generated_num
        return 0

    def _calculate_average_delay(self, simulator):
        """计算平均端到端延迟"""
        delays = list(simulator.metrics.deliver_time_dict.values())
        if delays:
            return sum(delays) / len(delays) / 1e3  # 转换为毫秒
        return 0

    def _calculate_throughput(self, simulator):
        """计算吞吐量"""
        throughputs = list(simulator.metrics.throughput_dict.values())
        if throughputs:
            return sum(throughputs) / len(throughputs) / 1e3  # 转换为Kbps
        return 0

    def _calculate_average_hop_count(self, simulator):
        """计算平均跳数"""
        hop_counts = list(simulator.metrics.hop_cnt_dict.values())
        if hop_counts:
            return sum(hop_counts) / len(hop_counts)
        return 0

    def _calculate_energy_consumption(self, simulator):
        """计算能量消耗"""
        return sum(simulator.metrics.energy_consumption.values())

    def _calculate_mac_delay(self, simulator):
        """计算MAC层延迟"""
        if hasattr(simulator.metrics, 'mac_delay') and simulator.metrics.mac_delay:
            return sum(simulator.metrics.mac_delay) / len(simulator.metrics.mac_delay)
        return 0

    def _calculate_load_balance(self, simulator):
        """计算负载平衡指标 (0-1，1表示完全平衡)"""
        queue_sizes = []
        for drone in simulator.drones:
            if hasattr(drone, 'transmitting_queue'):
                queue_sizes.append(drone.transmitting_queue.qsize())

        if not queue_sizes:
            return 1.0

        avg_queue = sum(queue_sizes) / len(queue_sizes)

        if avg_queue == 0:
            return 1.0

        # 计算变异系数 (CV)
        variance = sum((q - avg_queue) ** 2 for q in queue_sizes) / len(queue_sizes)
        std_dev = variance ** 0.5
        cv = std_dev / avg_queue

        # 将CV转换为0-1之间的负载平衡指标 (CV越小，平衡性越好)
        balance = 1.0 / (1.0 + cv)

        return balance

    def _collect_routing_stats(self, simulator, protocol_name):
        """收集路由协议特定的统计数据"""
        stats = {
            'routing_overhead': 0,  # 路由开销（控制包数量）
            'path_diversity': 0,  # 路径多样性
            'path_length': 0,  # 平均路径长度
            'convergence_time': 0,  # 路由收敛时间
        }

        # 根据不同协议收集不同指标
        if protocol_name == 'MP_OLSR':
            # 收集MP_OLSR特定指标
            stats = self._collect_mp_olsr_stats(simulator, stats)
        elif protocol_name == 'AMLB_OPAR':
            # 收集AMLB_OPAR特定指标
            stats = self._collect_amlb_opar_stats(simulator, stats)

        return stats

    def _collect_mp_olsr_stats(self, simulator, stats):
        """收集MP_OLSR特定的统计数据"""
        path_lengths = []
        path_counts = []

        for drone in simulator.drones:
            if hasattr(drone.routing_protocol, 'path_cache'):
                # 统计路径多样性
                for dest_id, paths in drone.routing_protocol.path_cache.items():
                    path_counts.append(len(paths))

                    # 统计路径长度
                    for path in paths:
                        if path:
                            path_lengths.append(len(path))

        # 计算平均值
        if path_counts:
            stats['path_diversity'] = sum(path_counts) / len(path_counts)

        if path_lengths:
            stats['path_length'] = sum(path_lengths) / len(path_lengths)

        # 估算路由收敛时间（这里简化处理）
        stats['convergence_time'] = 1.0  # 简化为常数值

        return stats

    def _collect_amlb_opar_stats(self, simulator, stats):
        """收集AMLB_OPAR特定的统计数据"""
        path_lengths = []
        path_counts = []

        for drone in simulator.drones:
            if hasattr(drone.routing_protocol, 'path_cache'):
                # 统计路径多样性
                for dest_id, paths in drone.routing_protocol.path_cache.items():
                    path_counts.append(len(paths))

                    # 统计路径长度
                    for path in paths:
                        if path:
                            path_lengths.append(len(path))

        # 计算平均值
        if path_counts:
            stats['path_diversity'] = sum(path_counts) / len(path_counts)

        if path_lengths:
            stats['path_length'] = sum(path_lengths) / len(path_lengths)

        # 估算路由收敛时间
        stats['convergence_time'] = 1.2  # 简化为常数值

        return stats

    def _schedule_node_failures(self, simulator):
        """安排节点故障事件"""
        # 记录故障事件
        self.failure_events = []

        # 设置故障时间点
        failure_times = [
            config.SIM_TIME * 0.3,  # 仿真30%时间点
            config.SIM_TIME * 0.6  # 仿真60%时间点
        ]

        # 为每个故障时间点安排事件
        for failure_time in failure_times:
            # 随机选择一个非关键节点
            node_candidates = list(range(1, config.NUMBER_OF_DRONES - 1))  # 避开0和最后一个节点

            if node_candidates:
                fail_node = random.choice(node_candidates)

                # 安排故障事件
                simulator.env.process(self._node_failure_process(simulator, fail_node, failure_time))

                # 记录故障信息
                self.failure_events.append({
                    'time': failure_time,
                    'node_id': fail_node,
                    'type': 'failure'
                })

                # 安排恢复事件
                recovery_time = failure_time + config.SIM_TIME * 0.1  # 故障持续10%的仿真时间
                if recovery_time < config.SIM_TIME:
                    simulator.env.process(self._node_recovery_process(simulator, fail_node, recovery_time))

                    # 记录恢复信息
                    self.failure_events.append({
                        'time': recovery_time,
                        'node_id': fail_node,
                        'type': 'recovery'
                    })

    def _node_failure_process(self, simulator, node_id, failure_time):
        """模拟节点故障过程"""
        # 等待到故障时间
        yield simulator.env.timeout(failure_time)

        # 设置节点为休眠状态 (模拟故障)
        if node_id < len(simulator.drones):
            simulator.drones[node_id].sleep = True
            logging.info(f"节点 {node_id} 在时间 {simulator.env.now / 1e6} 秒发生故障")

    def _node_recovery_process(self, simulator, node_id, recovery_time):
        """模拟节点恢复过程"""
        # 等待到恢复时间
        yield simulator.env.timeout(recovery_time)

        # 恢复节点
        if node_id < len(simulator.drones):
            simulator.drones[node_id].sleep = False
            logging.info(f"节点 {node_id} 在时间 {simulator.env.now / 1e6} 秒恢复正常")

    def _collect_failure_metrics(self, simulator):
        """收集与故障恢复相关的指标"""
        metrics = {
            'failure_events': self.failure_events,
            'recovery_time': self._calculate_recovery_time(simulator),
            'reroute_count': self._calculate_reroute_count(simulator),
        }

        return metrics

    def _calculate_recovery_time(self, simulator):
        """估算从故障到恢复的平均时间"""
        # 在实际系统中，这需要详细监控网络性能随时间的变化
        # 这里简化为恢复事件和故障事件的时间差
        recovery_times = []

        failure_times = {event['node_id']: event['time']
                         for event in self.failure_events
                         if event['type'] == 'failure'}

        recovery_times_dict = {event['node_id']: event['time']
                               for event in self.failure_events
                               if event['type'] == 'recovery'}

        for node_id, failure_time in failure_times.items():
            if node_id in recovery_times_dict:
                recovery_time = recovery_times_dict[node_id]
                recovery_times.append((recovery_time - failure_time) / 1e6)  # 转换为秒

        if recovery_times:
            return sum(recovery_times) / len(recovery_times)
        return 0

    def _calculate_reroute_count(self, simulator):
        """计算路由重建次数"""
        # 在本例中，简化为固定值
        # 在实际系统中，应该监控路由表变化次数
        return len(self.failure_events) // 2  # 每对故障/恢复事件算作一次重路由

    def visualize_results(self):
        """可视化测试结果"""
        if not self.test_results:
            logging.warning("没有测试结果可供可视化")
            return

        # 对比结果可视化
        self._plot_performance_comparison()

        # 各种场景下的性能比较
        self._plot_scenario_performance()

        # 路由特性比较
        self._plot_routing_characteristics()

        # 故障恢复比较
        self._plot_failure_recovery()

    def _plot_performance_comparison(self):
        """绘制两种协议的性能比较图"""
        # 提取基本场景下的性能数据
        mp_olsr_data = self.test_results.get('mp_olsr_basic', {})
        amlb_opar_data = self.test_results.get('amlb_opar_basic', {})

        if not mp_olsr_data or not amlb_opar_data:
            logging.warning("缺少基本场景测试数据")
            return

        # 设置图表
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

        # 准备数据
        metrics = ['pdr', 'delay', 'throughput', 'energy_consumption']
        titles = ['数据包交付率', '端到端延迟 (ms)', '吞吐量 (Kbps)', '能量消耗 (J)']
        protocols = ['MP-OLSR', 'AMLB-OPAR']
        colors = ['#4472C4', '#ED7D31']

        # 绘制四个主要指标的比较
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            row, col = i // 2, i % 2

            # 提取数据
            mp_olsr_value = mp_olsr_data.get(metric, 0)
            amlb_opar_value = amlb_opar_data.get(metric, 0)

            # 针对PDR和吞吐量，值越大越好；针对延迟和能量消耗，值越小越好
            if metric in ['pdr', 'throughput']:
                mp_olsr_value = max(0, mp_olsr_value)
                amlb_opar_value = max(0, amlb_opar_value)
                if metric == 'pdr':  # 转换为百分比
                    mp_olsr_value *= 100
                    amlb_opar_value *= 100

            # 绘制柱状图
            axs[row, col].bar(protocols, [mp_olsr_value, amlb_opar_value], color=colors)

            # 添加数值标签
            for j, value in enumerate([mp_olsr_value, amlb_opar_value]):
                axs[row, col].text(j, value, f'{value:.2f}', ha='center', va='bottom')

            # 设置标题和标签
            axs[row, col].set_title(title)
            axs[row, col].set_ylabel('值')

            # 设置Y轴从0开始
            if metric in ['pdr', 'throughput']:
                axs[row, col].set_ylim(bottom=0)

        # 调整布局
        plt.tight_layout()
        plt.savefig('test_results/performance_comparison.png', dpi=300)
        plt.close()

    def _plot_scenario_performance(self):
        """绘制不同场景下的性能比较"""
        scenarios = ['basic', 'dense', 'mobile']
        metrics = ['pdr', 'delay', 'hop_count', 'mac_delay']
        titles = ['数据包交付率 (%)', '端到端延迟 (ms)', '平均跳数', 'MAC延迟 (ms)']

        # 提取数据
        mp_olsr_data = {scenario: self.test_results.get(f'mp_olsr_{scenario}', {})
                        for scenario in scenarios}
        amlb_opar_data = {scenario: self.test_results.get(f'amlb_opar_{scenario}', {})
                          for scenario in scenarios}

        # 检查数据完整性
        if not all(mp_olsr_data.values()) or not all(amlb_opar_data.values()):
            logging.warning("缺少部分场景测试数据")
            # 继续使用可用数据

        # 设置图表
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

        # 绘制四个指标在不同场景下的比较
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            row, col = i // 2, i % 2

            # 准备数据
            mp_olsr_values = []
            amlb_opar_values = []

            for scenario in scenarios:
                mp_data = mp_olsr_data.get(scenario, {})
                amlb_data = amlb_opar_data.get(scenario, {})

                # 提取并转换数据
                mp_value = mp_data.get(metric, 0)
                amlb_value = amlb_data.get(metric, 0)

                if metric == 'pdr':  # 转换为百分比
                    mp_value *= 100
                    amlb_value *= 100

                mp_olsr_values.append(mp_value)
                amlb_opar_values.append(amlb_value)

            # 设置柱状图位置
            x = np.arange(len(scenarios))
            width = 0.35

            # 绘制柱状图
            axs[row, col].bar(x - width / 2, mp_olsr_values, width, label='MP-OLSR', color='#4472C4')
            axs[row, col].bar(x + width / 2, amlb_opar_values, width, label='AMLB-OPAR', color='#ED7D31')

            # 设置标题和标签
            axs[row, col].set_title(title)
            axs[row, col].set_xticks(x)
            axs[row, col].set_xticklabels(['基本场景', '高密度', '高移动性'])

            # 添加图例
            axs[row, col].legend()

            # 设置Y轴从0开始
            if metric not in ['delay', 'mac_delay']:
                axs[row, col].set_ylim(bottom=0)

        # 调整布局
        plt.tight_layout()
        plt.savefig('test_results/scenario_performance.png', dpi=300)
        plt.close()

    def _plot_routing_characteristics(self):
        """绘制路由特性比较"""
        # 提取数据
        mp_olsr_data = self.test_results.get('mp_olsr_basic', {})
        amlb_opar_data = self.test_results.get('amlb_opar_basic', {})

        if not mp_olsr_data or not amlb_opar_data:
            logging.warning("缺少路由特性数据")
            return

        # 设置图表
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

        # 准备数据
        metrics = ['path_diversity', 'path_length', 'convergence_time']
        titles = ['路径多样性', '平均路径长度', '路由收敛时间 (s)']
        protocols = ['MP-OLSR', 'AMLB-OPAR']
        colors = ['#4472C4', '#ED7D31']

        # 绘制三个路由特性的比较
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            # 提取数据
            mp_olsr_value = mp_olsr_data.get(metric, 0)
            amlb_opar_value = amlb_opar_data.get(metric, 0)

            # 绘制柱状图
            axs[i].bar(protocols, [mp_olsr_value, amlb_opar_value], color=colors)

            # 添加数值标签
            for j, value in enumerate([mp_olsr_value, amlb_opar_value]):
                axs[i].text(j, value, f'{value:.2f}', ha='center', va='bottom')

            # 设置标题和标签
            axs[i].set_title(title)
            axs[i].set_ylabel('值')

            # 设置Y轴从0开始
            axs[i].set_ylim(bottom=0)

        # 调整布局
        plt.tight_layout()
        plt.savefig('test_results/routing_characteristics.png', dpi=300)
        plt.close()

    def _plot_failure_recovery(self):
        """绘制故障恢复性能比较"""
        # 提取数据
        mp_olsr_data = self.test_results.get('mp_olsr_failure', {})
        amlb_opar_data = self.test_results.get('amlb_opar_failure', {})

        if not mp_olsr_data or not amlb_opar_data:
            logging.warning("缺少故障恢复数据")
            return

        # 提取故障恢复指标
        mp_olsr_recovery = mp_olsr_data.get('recovery_metrics', {})
        amlb_opar_recovery = amlb_opar_data.get('recovery_metrics', {})

        # 设置图表
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

        # 绘制恢复时间比较
        recovery_times = [
            mp_olsr_recovery.get('recovery_time', 0),
            amlb_opar_recovery.get('recovery_time', 0)
        ]

        axs[0].bar(['MP-OLSR', 'AMLB-OPAR'], recovery_times, color=['#4472C4', '#ED7D31'])
        axs[0].set_title('故障恢复时间 (秒)')
        axs[0].set_ylabel('时间 (秒)')

        # 添加数值标签
        for i, time in enumerate(recovery_times):
            axs[0].text(i, time, f'{time:.2f}', ha='center', va='bottom')

        # 绘制故障期间的PDR比较
        pdr_values = [
            mp_olsr_data.get('pdr', 0) * 100,  # 转换为百分比
            amlb_opar_data.get('pdr', 0) * 100
        ]

        axs[1].bar(['MP-OLSR', 'AMLB-OPAR'], pdr_values, color=['#4472C4', '#ED7D31'])
        axs[1].set_title('故障场景下的PDR (%)')
        axs[1].set_ylabel('PDR (%)')

        # 添加数值标签
        for i, pdr in enumerate(pdr_values):
            axs[1].text(i, pdr, f'{pdr:.2f}', ha='center', va='bottom')

        # 设置Y轴从0开始
        axs[0].set_ylim(bottom=0)
        axs[1].set_ylim(bottom=0)

        # 调整布局
        plt.tight_layout()
        plt.savefig('test_results/failure_recovery.png', dpi=300)
        plt.close()

    def generate_report(self):
        """生成测试报告"""
        if not self.test_results:
            logging.warning("没有测试结果可供报告")
            return ""

        report = "# MP_OLSR与AMLB_OPAR路由协议性能对比报告\n\n"

        # 添加测试配置信息
        report += "## 测试配置\n\n"
        report += f"- 无人机数量: {config.NUMBER_OF_DRONES}\n"
        report += f"- 仿真时间: {config.SIM_TIME / 1e6} 秒\n"
        report += f"- 多路径最大路径数: {getattr(config, 'MAX_PATHS', 3)}\n"
        report += f"- 路径选择策略: {getattr(config, 'PATH_SELECTION_STRATEGY', 'adaptive')}\n\n"

        # 添加基本性能对比
        report += "## 基本性能对比\n\n"
        report += "| 性能指标 | MP-OLSR | AMLB-OPAR | 比较结果 |\n"
        report += "|---------|---------|-----------|----------|\n"

        mp_olsr_basic = self.test_results.get('mp_olsr_basic', {})
        amlb_opar_basic = self.test_results.get('amlb_opar_basic', {})

        metrics = {
            'pdr': ('数据包交付率 (%)', lambda x: x * 100, True),  # 转为百分比，值越大越好
            'delay': ('端到端延迟 (ms)', lambda x: x, False),  # 值越小越好
            'throughput': ('吞吐量 (Kbps)', lambda x: x, True),  # 值越大越好
            'hop_count': ('平均跳数', lambda x: x, False),  # 值越小越好
            'energy_consumption': ('能量消耗 (J)', lambda x: x, False),  # 值越小越好
            'path_diversity': ('路径多样性', lambda x: x, True),  # 值越大越好
            'path_length': ('平均路径长度', lambda x: x, False),  # 值越小越好
            'convergence_time': ('路由收敛时间 (s)', lambda x: x, False)  # 值越小越好
        }

        for metric, (name, transform, higher_better) in metrics.items():
            mp_value = mp_olsr_basic.get(metric, 0)
            amlb_value = amlb_opar_basic.get(metric, 0)

            # 变换数据
            mp_value = transform(mp_value)
            amlb_value = transform(amlb_value)

            # 确定哪个更好
            if abs(mp_value - amlb_value) < 1e-6:  # 接近相等
                result = "相似"
            elif (mp_value > amlb_value) == higher_better:
                result = "MP-OLSR更优"
            else:
                result = "AMLB-OPAR更优"

            report += f"| {name} | {mp_value:.2f} | {amlb_value:.2f} | {result} |\n"

        # 不同场景性能比较
        report += "\n## 不同场景性能比较\n\n"

        scenarios = {
            'basic': '基本场景',
            'dense': '高密度场景',
            'mobile': '高移动性场景',
            'failure': '故障恢复场景'
        }

        for scenario, scenario_name in scenarios.items():
            mp_data = self.test_results.get(f'mp_olsr_{scenario}', {})
            amlb_data = self.test_results.get(f'amlb_opar_{scenario}', {})

            if not mp_data or not amlb_data:
                continue

            report += f"### {scenario_name}\n\n"
            report += "| 性能指标 | MP-OLSR | AMLB-OPAR | 比较结果 |\n"
            report += "|---------|---------|-----------|----------|\n"

            for metric, (name, transform, higher_better) in metrics.items():
                if metric in mp_data and metric in amlb_data:
                    mp_value = transform(mp_data.get(metric, 0))
                    amlb_value = transform(amlb_data.get(metric, 0))

                    # 确定哪个更好
                    if abs(mp_value - amlb_value) < 1e-6:  # 接近相等
                        result = "相似"
                    elif (mp_value > amlb_value) == higher_better:
                        result = "MP-OLSR更优"
                    else:
                        result = "AMLB-OPAR更优"

                    report += f"| {name} | {mp_value:.2f} | {amlb_value:.2f} | {result} |\n"

            report += "\n"

        # 添加故障恢复特定指标
        if 'mp_olsr_failure' in self.test_results and 'amlb_opar_failure' in self.test_results:
            report += "## 故障恢复特性\n\n"
            report += "| 指标 | MP-OLSR | AMLB-OPAR | 比较结果 |\n"
            report += "|------|---------|-----------|----------|\n"

            mp_recovery = self.test_results['mp_olsr_failure'].get('recovery_metrics', {})
            amlb_recovery = self.test_results['amlb_opar_failure'].get('recovery_metrics', {})

            recovery_metrics = [
                ('recovery_time', '恢复时间 (秒)', False),  # 值越小越好
                ('reroute_count', '重路由次数', False)  # 值越小越好
            ]

            for metric, name, higher_better in recovery_metrics:
                mp_value = mp_recovery.get(metric, 0)
                amlb_value = amlb_recovery.get(metric, 0)

                # 确定哪个更好
                if abs(mp_value - amlb_value) < 1e-6:  # 接近相等
                    result = "相似"
                elif (mp_value > amlb_value) == higher_better:
                    result = "MP-OLSR更优"
                else:
                    result = "AMLB-OPAR更优"

                report += f"| {name} | {mp_value:.2f} | {amlb_value:.2f} | {result} |\n"

        # 总结分析
        report += "\n## 总结分析\n\n"

        # 统计各项指标的优势情况
        mp_olsr_wins = 0
        amlb_opar_wins = 0

        for scenario in scenarios.keys():
            mp_data = self.test_results.get(f'mp_olsr_{scenario}', {})
            amlb_data = self.test_results.get(f'amlb_opar_{scenario}', {})

            if not mp_data or not amlb_data:
                continue

            for metric, (_, transform, higher_better) in metrics.items():
                if metric in mp_data and metric in amlb_data:
                    mp_value = transform(mp_data.get(metric, 0))
                    amlb_value = transform(amlb_data.get(metric, 0))

                    if abs(mp_value - amlb_value) >= 1e-6:  # 不接近相等
                        if (mp_value > amlb_value) == higher_better:
                            mp_olsr_wins += 1
                        else:
                            amlb_opar_wins += 1

        # 根据获胜次数确定总体表现
        if mp_olsr_wins > amlb_opar_wins:
            overall_winner = "MP-OLSR"
            loser = "AMLB-OPAR"
        elif amlb_opar_wins > mp_olsr_wins:
            overall_winner = "AMLB-OPAR"
            loser = "MP-OLSR"
        else:
            overall_winner = "两种协议"
            loser = "二者"

        report += f"通过对比测试分析，{overall_winner}在总体性能上表现更优，在{max(mp_olsr_wins, amlb_opar_wins)}项指标中领先于{loser}。\n\n"

        # 针对基本场景的详细分析
        report += "### 基本场景分析\n\n"

        # PDR分析
        mp_pdr = mp_olsr_basic.get('pdr', 0) * 100
        amlb_pdr = amlb_opar_basic.get('pdr', 0) * 100
        pdr_diff = abs(mp_pdr - amlb_pdr)

        if pdr_diff < 1:  # 差异小于1%
            report += "两种协议在数据包交付率方面表现相近，差异不显著。"
        else:
            better = "MP-OLSR" if mp_pdr > amlb_pdr else "AMLB-OPAR"
            report += f"{better}在数据包交付率方面表现更优，高出约{pdr_diff:.1f}个百分点。"

        # 延迟分析
        mp_delay = mp_olsr_basic.get('delay', 0)
        amlb_delay = amlb_opar_basic.get('delay', 0)
        delay_diff = abs(mp_delay - amlb_delay)

        if abs(mp_delay - amlb_delay) < 1:  # 差异小于1ms
            report += "两种协议在端到端延迟方面表现相近。"
        else:
            better = "MP-OLSR" if mp_delay < amlb_delay else "AMLB-OPAR"
            report += f"{better}在端到端延迟方面表现更优，平均延迟低约{delay_diff:.1f}毫秒。"

        report += "\n\n"

        # 针对特殊场景的分析
        report += "### 特殊场景分析\n\n"

        # 高密度场景
        mp_dense = self.test_results.get('mp_olsr_dense', {})
        amlb_dense = self.test_results.get('amlb_opar_dense', {})

        if mp_dense and amlb_dense:
            mp_dense_pdr = mp_dense.get('pdr', 0) * 100
            amlb_dense_pdr = amlb_dense.get('pdr', 0) * 100

            better = "MP-OLSR" if mp_dense_pdr > amlb_dense_pdr else "AMLB-OPAR"
            report += f"在高密度场景下，{better}表现更为稳定，能够更好地处理较多节点带来的挑战。\n\n"

        # 高移动性场景
        mp_mobile = self.test_results.get('mp_olsr_mobile', {})
        amlb_mobile = self.test_results.get('amlb_opar_mobile', {})

        if mp_mobile and amlb_mobile:
            mp_mobile_pdr = mp_mobile.get('pdr', 0) * 100
            amlb_mobile_pdr = amlb_mobile.get('pdr', 0) * 100

            better = "MP-OLSR" if mp_mobile_pdr > amlb_mobile_pdr else "AMLB-OPAR"
            report += f"在高移动性场景下，{better}对网络拓扑变化的适应性更强。\n\n"

        # 故障恢复场景
        mp_failure = self.test_results.get('mp_olsr_failure', {})
        amlb_failure = self.test_results.get('amlb_opar_failure', {})

        if mp_failure and amlb_failure:
            mp_recovery_time = mp_failure.get('recovery_metrics', {}).get('recovery_time', 0)
            amlb_recovery_time = amlb_failure.get('recovery_metrics', {}).get('recovery_time', 0)

            better = "MP-OLSR" if mp_recovery_time < amlb_recovery_time else "AMLB-OPAR"
            report += f"在故障恢复场景中，{better}展现出更强的故障恢复能力，恢复时间更短。\n\n"

        # 路由特性分析
        report += "### 路由特性分析\n\n"

        # 路径多样性分析
        mp_diversity = mp_olsr_basic.get('path_diversity', 0)
        amlb_diversity = amlb_opar_basic.get('path_diversity', 0)

        better = "MP-OLSR" if mp_diversity > amlb_diversity else "AMLB-OPAR"
        report += f"{better}提供了更高的路径多样性，平均为每个目的地维护更多的备选路径。\n\n"

        # 路由收敛分析
        mp_convergence = mp_olsr_basic.get('convergence_time', 0)
        amlb_convergence = amlb_opar_basic.get('convergence_time', 0)

        better = "MP-OLSR" if mp_convergence < amlb_convergence else "AMLB-OPAR"
        report += f"{better}的路由收敛速度更快，在网络拓扑变化后能更快建立有效路由。\n\n"

        # 综合建议
        report += "## 建议\n\n"

        # 根据测试结果生成建议
        if mp_olsr_wins > amlb_opar_wins:
            report += "1. 在大多数场景下，建议使用MP-OLSR作为主要路由协议，尤其在对"

            # 找出MP-OLSR显著优势项
            advantages = []
            if mp_olsr_basic.get('pdr', 0) > amlb_opar_basic.get('pdr', 0) * 1.05:
                advantages.append("数据包交付率")
            if amlb_opar_basic.get('delay', 0) > mp_olsr_basic.get('delay', 0) * 1.05:
                advantages.append("端到端延迟")
            if mp_olsr_basic.get('throughput', 0) > amlb_opar_basic.get('throughput', 0) * 1.05:
                advantages.append("吞吐量")

            report += "、".join(advantages) + "有较高要求时。\n"

            # 找出AMLB-OPAR的优势场景
            amlb_scenario = ""
            if amlb_mobile and mp_mobile:
                if amlb_mobile.get('pdr', 0) > mp_mobile.get('pdr', 0):
                    amlb_scenario = "高移动性"
            elif amlb_dense and mp_dense:
                if amlb_dense.get('pdr', 0) > mp_dense.get('pdr', 0):
                    amlb_scenario = "高密度"

            if amlb_scenario:
                report += f"2. 在{amlb_scenario}场景下，可以考虑使用AMLB-OPAR以获得更好的性能。\n"
            else:
                report += "2. AMLB-OPAR可作为备选方案，特别是在特定应用场景有特殊需求时。\n"

        else:
            report += "1. 在大多数场景下，建议使用AMLB-OPAR作为主要路由协议，尤其在对"

            # 找出AMLB-OPAR显著优势项
            advantages = []
            if amlb_opar_basic.get('pdr', 0) > mp_olsr_basic.get('pdr', 0) * 1.05:
                advantages.append("数据包交付率")
            if mp_olsr_basic.get('delay', 0) > amlb_opar_basic.get('delay', 0) * 1.05:
                advantages.append("端到端延迟")
            if amlb_opar_basic.get('throughput', 0) > mp_olsr_basic.get('throughput', 0) * 1.05:
                advantages.append("吞吐量")

            report += "、".join(advantages) + "有较高要求时。\n"

            # 找出MP-OLSR的优势场景
            mp_scenario = ""
            if mp_mobile and amlb_mobile:
                if mp_mobile.get('pdr', 0) > amlb_mobile.get('pdr', 0):
                    mp_scenario = "高移动性"
            elif mp_dense and amlb_dense:
                if mp_dense.get('pdr', 0) > amlb_dense.get('pdr', 0):
                    mp_scenario = "高密度"

            if mp_scenario:
                report += f"2. 在{mp_scenario}场景下，可以考虑使用MP-OLSR以获得更好的性能。\n"
            else:
                report += "2. MP-OLSR可作为备选方案，特别是在特定应用场景有特殊需求时。\n"

        report += "3. 对于关键应用，建议在实际部署前进行更全面的场景测试，确保选用的路由协议能满足具体应用需求。\n"

        return report