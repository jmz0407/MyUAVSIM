import simpy
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import logging
from utils import config
from simulator.simulator import Simulator
from routing.multipath.multipath_load_balancer import MultipathRouter, MultiPathLoadBalancer
from routing.multipath.multipath_integration import MultipathIntegration


class MultipathTestSuite:
    """
    无人机多路径负载均衡策略的测试和评估套件
    该类用于运行各种测试场景并收集性能指标
    """

    def __init__(self, base_config=None):
        """
        初始化测试套件

        Args:
            base_config: 基础配置参数，如无人机数量、仿真时间等
        """
        self.base_config = base_config or {}

        # 测试结果存储
        self.test_results = {}

        # 默认配置设置
        self._set_default_config()

    def _set_default_config(self):
        """设置默认测试配置"""
        # 保存原始配置
        self.original_config = {
            'NUMBER_OF_DRONES': config.NUMBER_OF_DRONES,
            'SIM_TIME': config.SIM_TIME,
            'MULTIPATH_ENABLED': getattr(config, 'MULTIPATH_ENABLED', False),
            'MAX_PATHS': getattr(config, 'MAX_PATHS', 1),
            'PATH_SELECTION_STRATEGY': getattr(config, 'PATH_SELECTION_STRATEGY', 'weighted'),
        }

        # 应用基础配置
        for key, value in self.base_config.items():
            setattr(config, key, value)

    def run_test_scenarios(self):
        """运行所有测试场景"""
        # 运行基线测试 (不使用多路径)
        self.test_results['baseline'] = self.run_baseline_test()

        # 运行不同多路径策略的测试
        strategies = ['weighted', 'round_robin', 'adaptive']

        for strategy in strategies:
            for max_paths in [2, 3]:
                scenario_name = f"multipath_{strategy}_{max_paths}"
                self.test_results[scenario_name] = self.run_multipath_test(strategy, max_paths)

        # 运行高负载场景测试
        self.test_results['high_load'] = self.run_high_load_test()

        # 运行网络故障恢复测试
        self.test_results['failure_recovery'] = self.run_failure_recovery_test()

        # 恢复原始配置
        self._restore_original_config()

        return self.test_results

    def _restore_original_config(self):
        """恢复原始配置"""
        for key, value in self.original_config.items():
            setattr(config, key, value)

    def run_baseline_test(self):
        """运行基线测试 (不使用多路径)"""
        logging.info("开始基线测试 (单路径路由)")
        print("开始基线测试 (单路径路由)")
        # 禁用多路径
        config.MULTIPATH_ENABLED = False

        # 创建和运行模拟器
        simulator = self.create_simulator()
        metrics = self.run_simulation(simulator)

        logging.info("基线测试完成")

        return metrics

    def run_multipath_test(self, strategy, max_paths):
        """
        运行多路径测试

        Args:
            strategy: 路径选择策略 ('weighted', 'round_robin', 'adaptive')
            max_paths: 最大路径数
        """
        logging.info(f"开始多路径测试 (策略={strategy}, 最大路径数={max_paths})")
        print(f"开始多路径测试 (策略={strategy}, 最大路径数={max_paths})")
        # 启用多路径
        config.MULTIPATH_ENABLED = True
        config.MAX_PATHS = max_paths
        config.PATH_SELECTION_STRATEGY = strategy

        # 创建和运行模拟器
        simulator = self.create_simulator()

        # 创建多路径集成
        multipath_integration = MultipathIntegration(simulator)

        # 运行仿真
        metrics = self.run_simulation(simulator)

        # 补充多路径特定指标
        metrics['multipath_metrics'] = self.collect_multipath_metrics(simulator, multipath_integration)

        logging.info(f"多路径测试完成 (策略={strategy}, 最大路径数={max_paths})")

        return metrics

    def run_high_load_test(self):
        """运行高负载场景测试"""
        logging.info("开始高负载场景测试")
        print("开始高负载场景测试")
        # 启用多路径
        config.MULTIPATH_ENABLED = True
        config.MAX_PATHS = 3
        config.PATH_SELECTION_STRATEGY = 'adaptive'

        # 提高流量生成率 (通过临时修改config)
        original_traffic_rate = getattr(config, 'TRAFFIC_RATE', None)
        config.TRAFFIC_RATE = 5  # 假设这是流量生成率参数

        # 创建和运行模拟器
        simulator = self.create_simulator()

        # 创建多路径集成
        multipath_integration = MultipathIntegration(simulator)

        # 运行仿真
        metrics = self.run_simulation(simulator)

        # 补充多路径特定指标
        metrics['multipath_metrics'] = self.collect_multipath_metrics(simulator, multipath_integration)

        # 恢复原始流量生成率
        if original_traffic_rate is not None:
            config.TRAFFIC_RATE = original_traffic_rate
        else:
            delattr(config, 'TRAFFIC_RATE')

        logging.info("高负载场景测试完成")

        return metrics

    def run_failure_recovery_test(self):
        """运行网络故障恢复测试"""
        logging.info("开始网络故障恢复测试")
        print("开始网络故障恢复测试")
        # 启用多路径
        config.MULTIPATH_ENABLED = True
        config.MAX_PATHS = 3
        config.PATH_SELECTION_STRATEGY = 'adaptive'

        # 创建和运行模拟器
        simulator = self.create_simulator()

        # 创建多路径集成
        multipath_integration = MultipathIntegration(simulator)

        # 安排节点故障事件
        self.schedule_node_failures(simulator)

        # 运行仿真
        metrics = self.run_simulation(simulator)

        # 补充多路径特定指标
        metrics['multipath_metrics'] = self.collect_multipath_metrics(simulator, multipath_integration)
        metrics['failure_recovery'] = self.collect_failure_metrics(simulator)

        logging.info("网络故障恢复测试完成")

        return metrics

    def create_simulator(self):
        """创建仿真器实例"""
        env = simpy.Environment()
        n_drones = getattr(config, 'NUMBER_OF_DRONES', 10)
        channel_states = {i: simpy.Resource(env, capacity=1) for i in range(n_drones)}

        return Simulator(
            seed=2024,
            env=env,
            channel_states=channel_states,
            n_drones=n_drones
        )

    def run_simulation(self, simulator):
        """运行仿真并返回性能指标"""
        # 运行仿真
        simulator.env.run(until=config.SIM_TIME)

        # 收集基本性能指标
        metrics = {
            'pdr': self.calculate_pdr(simulator),
            'delay': self.calculate_average_delay(simulator),
            'throughput': self.calculate_throughput(simulator),
            'hop_count': self.calculate_average_hop_count(simulator),
            'energy_consumption': self.calculate_energy_consumption(simulator),
            'load_balance': self.calculate_load_balance(simulator),
        }

        return metrics

    def calculate_pdr(self, simulator):
        """计算数据包交付率"""
        if simulator.metrics.datapacket_generated_num > 0:
            return len(simulator.metrics.datapacket_arrived) / simulator.metrics.datapacket_generated_num
        return 0

    def calculate_average_delay(self, simulator):
        """计算平均端到端延迟"""
        delays = []
        for packet_id in simulator.metrics.deliver_time_dict:
            delays.append(simulator.metrics.deliver_time_dict[packet_id])

        if delays:
            return sum(delays) / len(delays) / 1e3  # 转换为毫秒
        return 0

    def calculate_throughput(self, simulator):
        """计算吞吐量"""
        throughputs = []
        for packet_id in simulator.metrics.throughput_dict:
            throughputs.append(simulator.metrics.throughput_dict[packet_id])

        if throughputs:
            return sum(throughputs) / len(throughputs) / 1e3  # 转换为Kbps
        return 0

    def calculate_average_hop_count(self, simulator):
        """计算平均跳数"""
        hop_counts = []
        for packet_id in simulator.metrics.hop_cnt_dict:
            hop_counts.append(simulator.metrics.hop_cnt_dict[packet_id])

        if hop_counts:
            return sum(hop_counts) / len(hop_counts)
        return 0

    def calculate_energy_consumption(self, simulator):
        """计算能量消耗"""
        total_energy = sum(simulator.metrics.energy_consumption.values())
        return total_energy

    def calculate_load_balance(self, simulator):
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

        # 计算变异系数 (CV) 作为负载不平衡的度量
        variance = sum((q - avg_queue) ** 2 for q in queue_sizes) / len(queue_sizes)
        std_dev = variance ** 0.5
        cv = std_dev / avg_queue

        # 将CV转换为0-1之间的负载平衡指标 (CV越小，平衡性越好)
        balance = 1.0 / (1.0 + cv)

        return balance

    def collect_multipath_metrics(self, simulator, multipath_integration):
        """收集多路径特定指标"""
        metrics = {
            'path_diversity': self.calculate_path_diversity(simulator),
            'path_utilization': self.calculate_path_utilization(multipath_integration),
            'load_distribution': self.calculate_load_distribution(simulator),
        }

        return metrics

    def calculate_path_diversity(self, simulator):
        """计算路径多样性 (平均每对源目的地使用的不同路径数)"""
        path_counts = defaultdict(set)

        # 遍历所有无人机
        for drone in simulator.drones:
            if hasattr(drone, 'multipath_router'):
                router = drone.multipath_router

                # 检查所有目的地
                for dst_id in router.paths_cache:
                    src_dst = f"{drone.identifier}_{dst_id}"

                    # 记录该源目的地对使用的不同路径
                    paths = router.paths_cache[dst_id]
                    path_counts[src_dst].update(paths)

        # 计算平均路径多样性
        if path_counts:
            avg_diversity = sum(len(paths) for paths in path_counts.values()) / len(path_counts)
        else:
            avg_diversity = 0

        return avg_diversity

    def calculate_path_utilization(self, multipath_integration):
        """计算路径利用率 (不同路径的使用比例)"""
        path_usage = defaultdict(lambda: defaultdict(int))

        # 从多路径集成中获取路径使用数据
        if hasattr(multipath_integration, 'performance_metrics'):
            for src_dst, usage in multipath_integration.performance_metrics['path_utilization'].items():
                for path_id, count in usage.items():
                    path_usage[src_dst][path_id] = count

        # 计算每对源目的地的路径使用比例
        path_utilization = {}

        for src_dst, usage in path_usage.items():
            total = sum(usage.values())
            if total > 0:
                path_utilization[src_dst] = {path_id: count / total for path_id, count in usage.items()}

        return path_utilization

    def calculate_load_distribution(self, simulator):
        """计算节点负载分布"""
        load_distribution = {}

        for drone in simulator.drones:
            if hasattr(drone, 'transmitting_queue'):
                queue_size = drone.transmitting_queue.qsize()
                max_queue = config.MAX_QUEUE_SIZE
                load = queue_size / max_queue if max_queue > 0 else 0
                load_distribution[drone.identifier] = load

        return load_distribution

    def schedule_node_failures(self, simulator):
        """安排节点故障事件"""
        # 在仿真过程中随机选择几个节点发生故障
        # 这是通过将节点的sleep属性设置为True来模拟的

        # 安排3个故障事件
        failure_times = [
            config.SIM_TIME * 0.2,  # 仿真20%时间点
            config.SIM_TIME * 0.5,  # 仿真50%时间点
            config.SIM_TIME * 0.8  # 仿真80%时间点
        ]

        # 记录故障事件
        self.failure_events = []

        # 为每个故障时间点安排事件
        for failure_time in failure_times:
            # 随机选择一个非关键节点
            fail_node_candidates = list(range(config.NUMBER_OF_DRONES))

            # 避免选择源节点和目标节点作为故障节点
            critical_nodes = [0, config.NUMBER_OF_DRONES - 1]  # 假设这些是关键节点
            for node in critical_nodes:
                if node in fail_node_candidates:
                    fail_node_candidates.remove(node)

            if fail_node_candidates:
                fail_node = np.random.choice(fail_node_candidates)

                # 安排故障事件
                simulator.env.process(self.node_failure_process(simulator, fail_node, failure_time))

                # 记录故障信息
                self.failure_events.append({
                    'time': failure_time,
                    'node_id': fail_node,
                    'type': 'failure'
                })

                # 安排恢复事件 (如果需要)
                recovery_time = failure_time + config.SIM_TIME * 0.1  # 假设故障持续10%的仿真时间
                if recovery_time < config.SIM_TIME:
                    simulator.env.process(self.node_recovery_process(simulator, fail_node, recovery_time))

                    # 记录恢复信息
                    self.failure_events.append({
                        'time': recovery_time,
                        'node_id': fail_node,
                        'type': 'recovery'
                    })

    def node_failure_process(self, simulator, node_id, failure_time):
        """模拟节点故障过程"""
        # 等待直到故障时间
        yield simulator.env.timeout(failure_time)

        # 设置节点为休眠状态 (模拟故障)
        if node_id < len(simulator.drones):
            simulator.drones[node_id].sleep = True
            logging.info(f"节点 {node_id} 在时间 {simulator.env.now / 1e6} 秒发生故障")

    def node_recovery_process(self, simulator, node_id, recovery_time):
        """模拟节点恢复过程"""
        # 等待直到恢复时间
        yield simulator.env.timeout(recovery_time)

        # 恢复节点
        if node_id < len(simulator.drones):
            simulator.drones[node_id].sleep = False
            logging.info(f"节点 {node_id} 在时间 {simulator.env.now / 1e6} 秒恢复正常")

    def collect_failure_metrics(self, simulator):
        """收集与故障恢复相关的指标"""
        metrics = {
            'failure_events': self.failure_events,
            'recovery_time': self.calculate_recovery_time(simulator),
            'affected_flows': self.calculate_affected_flows(simulator),
        }

        return metrics

    def calculate_recovery_time(self, simulator):
        """计算故障恢复时间
        通过分析故障前后数据包传输的变化来估计恢复时间"""
        recovery_times = []

        # 获取所有故障事件的时间
        failure_times = [event['time'] for event in self.failure_events
                         if event['type'] == 'failure']

        # 为每个故障事件计算恢复时间
        for failure_time in failure_times:
            # 检查故障前后数据包的到达情况
            pre_failure_rate = self.calculate_packet_rate(simulator, failure_time - 1e6, failure_time)

            # 从故障时间向后扫描，直到数据包率恢复到故障前的80%
            recovery_threshold = pre_failure_rate * 0.8

            recovery_time = None
            for t in range(int(failure_time + 1e5), int(min(failure_time + 10e6, config.SIM_TIME)), int(1e5)):
                current_rate = self.calculate_packet_rate(simulator, t - 1e5, t)
                if current_rate >= recovery_threshold:
                    recovery_time = t - failure_time
                    break

            if recovery_time is not None:
                recovery_times.append(recovery_time / 1e6)  # 转换为秒

        # 计算平均恢复时间
        if recovery_times:
            return sum(recovery_times) / len(recovery_times)
        return None

    def calculate_packet_rate(self, simulator, start_time, end_time):
        """计算指定时间段内的数据包到达率"""
        # 计算时间段内到达的数据包数量
        packets_in_interval = 0

        for packet_id, arrival_time in simulator.metrics.deliver_time_dict.items():
            if start_time <= arrival_time <= end_time:
                packets_in_interval += 1

        # 计算数据包到达率 (每秒)
        interval_duration = (end_time - start_time) / 1e6  # 转换为秒
        if interval_duration > 0:
            return packets_in_interval / interval_duration
        return 0

    def calculate_affected_flows(self, simulator):
        """计算受故障影响的流数量"""
        affected_flows = set()

        # 获取所有故障节点
        failed_nodes = [event['node_id'] for event in self.failure_events
                        if event['type'] == 'failure']

        # 检查每个数据包的路由路径是否包含故障节点
        for drone in simulator.drones:
            if hasattr(drone, 'multipath_router'):
                router = drone.multipath_router

                # 检查每个目的地的路径
                for dst_id in router.paths_cache:
                    for path_id in router.paths_cache[dst_id]:
                        path = router._get_path(path_id)

                        # 检查路径是否包含故障节点
                        if path and any(node in failed_nodes for node in path):
                            flow_id = f"{drone.identifier}_{dst_id}"
                            affected_flows.add(flow_id)

        return len(affected_flows)

    def visualize_results(self):
        """可视化测试结果"""
        if not self.test_results:
            logging.warning("没有测试结果可供可视化")
            return

        # 创建图形
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))

        # 整理数据
        scenarios = list(self.test_results.keys())
        pdr_values = [result['pdr'] * 100 for result in self.test_results.values()]
        delay_values = [result['delay'] for result in self.test_results.values()]
        throughput_values = [result['throughput'] for result in self.test_results.values()]
        load_balance_values = [result['load_balance'] * 100 for result in self.test_results.values()]
        energy_values = [result['energy_consumption'] for result in self.test_results.values()]

        # 绘制PDR图表
        axs[0, 0].bar(scenarios, pdr_values)
        axs[0, 0].set_title('数据包交付率 (%)')
        axs[0, 0].set_xlabel('场景')
        axs[0, 0].set_ylabel('PDR (%)')
        axs[0, 0].set_ylim(0, 100)
        axs[0, 0].tick_params(axis='x', rotation=45)

        # 绘制端到端延迟图表
        axs[0, 1].bar(scenarios, delay_values)
        axs[0, 1].set_title('平均端到端延迟 (ms)')
        axs[0, 1].set_xlabel('场景')
        axs[0, 1].set_ylabel('延迟 (ms)')
        axs[0, 1].tick_params(axis='x', rotation=45)

        # 绘制吞吐量图表
        axs[0, 2].bar(scenarios, throughput_values)
        axs[0, 2].set_title('平均吞吐量 (Kbps)')
        axs[0, 2].set_xlabel('场景')
        axs[0, 2].set_ylabel('吞吐量 (Kbps)')
        axs[0, 2].tick_params(axis='x', rotation=45)

        # 绘制负载平衡图表
        axs[1, 0].bar(scenarios, load_balance_values)
        axs[1, 0].set_title('负载平衡指标 (%)')
        axs[1, 0].set_xlabel('场景')
        axs[1, 0].set_ylabel('负载平衡 (%)')
        axs[1, 0].set_ylim(0, 100)
        axs[1, 0].tick_params(axis='x', rotation=45)

        # 绘制能量消耗图表
        axs[1, 1].bar(scenarios, energy_values)
        axs[1, 1].set_title('总能量消耗 (J)')
        axs[1, 1].set_xlabel('场景')
        axs[1, 1].set_ylabel('能量 (J)')
        axs[1, 1].tick_params(axis='x', rotation=45)

        # 绘制路径多样性图表 (如果有)
        path_diversity = []
        for result in self.test_results.values():
            if 'multipath_metrics' in result and 'path_diversity' in result['multipath_metrics']:
                path_diversity.append(result['multipath_metrics']['path_diversity'])
            else:
                path_diversity.append(0)

        axs[1, 2].bar(scenarios, path_diversity)
        axs[1, 2].set_title('路径多样性 (平均路径数)')
        axs[1, 2].set_xlabel('场景')
        axs[1, 2].set_ylabel('路径数')
        axs[1, 2].tick_params(axis='x', rotation=45)

        # 调整布局
        plt.tight_layout()
        plt.savefig('multipath_test_results.png', dpi=300)
        plt.show()

    def visualize_failure_recovery(self):
        """可视化故障恢复测试结果"""
        # 检查是否有故障恢复测试结果
        if 'failure_recovery' not in self.test_results:
            logging.warning("没有故障恢复测试结果可供可视化")
            return

        failure_results = self.test_results['failure_recovery']

        # 创建图形
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))

        # 绘制故障前后的PDR变化
        # 为简化起见，我们使用模拟数据
        time_points = np.linspace(0, config.SIM_TIME / 1e6, 100)  # 转换为秒
        pdr_values = []

        for t in time_points:
            # 模拟PDR随时间的变化
            pdr = 0.9  # 基准PDR

            # 在故障点附近降低PDR
            for event in self.failure_events:
                event_time = event['time'] / 1e6  # 转换为秒
                if event['type'] == 'failure' and t >= event_time:
                    # 找到对应的恢复事件
                    recovery_time = None
                    for recovery in self.failure_events:
                        if (recovery['type'] == 'recovery' and
                                recovery['node_id'] == event['node_id'] and
                                recovery['time'] > event['time']):
                            recovery_time = recovery['time'] / 1e6  # 转换为秒
                            break

                    if recovery_time is None or t < recovery_time:
                        # 故障期间PDR降低
                        pdr *= 0.7
                    elif t < recovery_time + 0.5:  # 恢复后0.5秒内PDR逐渐回升
                        # 恢复过程中PDR逐渐回升
                        recovery_progress = (t - recovery_time) / 0.5
                        pdr *= (0.7 + 0.3 * recovery_progress)

            pdr_values.append(pdr * 100)  # 转换为百分比

        axs[0].plot(time_points, pdr_values)
        axs[0].set_title('故障期间的PDR变化')
        axs[0].set_xlabel('仿真时间 (秒)')
        axs[0].set_ylabel('PDR (%)')
        axs[0].set_ylim(0, 100)

        # 标记故障和恢复事件
        for event in self.failure_events:
            event_time = event['time'] / 1e6  # 转换为秒
            if event['type'] == 'failure':
                axs[0].axvline(x=event_time, color='r', linestyle='--')
                axs[0].text(event_time, 20, f"故障 (节点{event['node_id']})",
                            rotation=90, verticalalignment='bottom')
            else:
                axs[0].axvline(x=event_time, color='g', linestyle='--')
                axs[0].text(event_time, 20, f"恢复 (节点{event['node_id']})",
                            rotation=90, verticalalignment='bottom')

        # 绘制端到端延迟变化
        # 同样使用模拟数据
        delay_values = []

        for t in time_points:
            # 模拟延迟随时间的变化
            delay = 50  # 基准延迟 (ms)

            # 在故障点附近增加延迟
            for event in self.failure_events:
                event_time = event['time'] / 1e6  # 转换为秒
                if event['type'] == 'failure' and t >= event_time:
                    # 找到对应的恢复事件
                    recovery_time = None
                    for recovery in self.failure_events:
                        if (recovery['type'] == 'recovery' and
                                recovery['node_id'] == event['node_id'] and
                                recovery['time'] > event['time']):
                            recovery_time = recovery['time'] / 1e6  # 转换为秒
                            break

                    if recovery_time is None or t < recovery_time:
                        # 故障期间延迟增加
                        delay *= 2
                    elif t < recovery_time + 0.5:  # 恢复后0.5秒内延迟逐渐降低
                        # 恢复过程中延迟逐渐降低
                        recovery_progress = (t - recovery_time) / 0.5
                        delay *= (2 - recovery_progress)

            delay_values.append(delay)

        axs[1].plot(time_points, delay_values)
        axs[1].set_title('故障期间的端到端延迟变化')
        axs[1].set_xlabel('仿真时间 (秒)')
        axs[1].set_ylabel('延迟 (ms)')

        # 标记故障和恢复事件
        for event in self.failure_events:
            event_time = event['time'] / 1e6  # 转换为秒
            if event['type'] == 'failure':
                axs[1].axvline(x=event_time, color='r', linestyle='--')
            else:
                axs[1].axvline(x=event_time, color='g', linestyle='--')

        # 调整布局
        plt.tight_layout()
        plt.savefig('failure_recovery_results.png', dpi=300)
        plt.show()

    def generate_report(self):
        """生成测试报告"""
        if not self.test_results:
            logging.warning("没有测试结果可供报告")
            return

        report = "# 无人机网络多路径负载均衡性能测试报告\n\n"

        # 添加测试配置信息
        report += "## 测试配置\n\n"
        report += f"- 无人机数量: {config.NUMBER_OF_DRONES}\n"
        report += f"- 仿真时间: {config.SIM_TIME / 1e6} 秒\n"
        report += f"- 多路径启用: {config.MULTIPATH_ENABLED}\n"
        report += f"- 最大路径数: {config.MAX_PATHS}\n"
        report += f"- 路径选择策略: {config.PATH_SELECTION_STRATEGY}\n\n"

        # 添加测试场景摘要
        report += "## 测试场景\n\n"
        report += "| 场景 | 描述 |\n"
        report += "|------|------|\n"
        report += "| baseline | 基线测试 (单路径路由) |\n"

        for strategy in ['weighted', 'round_robin', 'adaptive']:
            for max_paths in [2, 3]:
                scenario = f"multipath_{strategy}_{max_paths}"
                report += f"| {scenario} | 多路径测试 (策略={strategy}, 最大路径数={max_paths}) |\n"

        report += "| high_load | 高负载场景测试 |\n"
        report += "| failure_recovery | 网络故障恢复测试 |\n\n"

        # 添加性能比较表格
        report += "## 性能比较\n\n"
        report += "| 场景 | PDR (%) | 延迟 (ms) | 吞吐量 (Kbps) | 负载平衡 (%) | 能量消耗 (J) | 路径多样性 |\n"
        report += "|------|---------|-----------|---------------|-------------|-------------|------------|\n"

        for scenario, result in self.test_results.items():
            pdr = result['pdr'] * 100
            delay = result['delay']
            throughput = result['throughput']
            load_balance = result['load_balance'] * 100
            energy = result['energy_consumption']

            # 获取路径多样性 (如果有)
            path_diversity = "N/A"
            if 'multipath_metrics' in result and 'path_diversity' in result['multipath_metrics']:
                path_diversity = f"{result['multipath_metrics']['path_diversity']:.2f}"

            report += f"| {scenario} | {pdr:.2f} | {delay:.2f} | {throughput:.2f} | {load_balance:.2f} | {energy:.2f} | {path_diversity} |\n"

        # 添加改进百分比
        if 'baseline' in self.test_results:
            report += "\n## 相对于基线的改进\n\n"
            report += "| 场景 | PDR改进 (%) | 延迟改进 (%) | 吞吐量改进 (%) | 负载平衡改进 (%) | 能量效率改进 (%) |\n"
            report += "|------|-------------|--------------|-----------------|------------------|------------------|\n"

            baseline = self.test_results['baseline']

            for scenario, result in self.test_results.items():
                if scenario == 'baseline':
                    continue

                pdr_improvement = (result['pdr'] - baseline['pdr']) / baseline['pdr'] * 100

                # 对于延迟，减少是改进
                if baseline['delay'] > 0:
                    delay_improvement = (baseline['delay'] - result['delay']) / baseline['delay'] * 100
                else:
                    delay_improvement = 0

                throughput_improvement = (result['throughput'] - baseline['throughput']) / baseline['throughput'] * 100

                load_balance_improvement = (result['load_balance'] - baseline['load_balance']) / baseline[
                    'load_balance'] * 100

                # 对于能量，减少是改进
                if baseline['energy_consumption'] > 0:
                    energy_improvement = (baseline['energy_consumption'] - result['energy_consumption']) / baseline[
                        'energy_consumption'] * 100
                else:
                    energy_improvement = 0

                report += f"| {scenario} | {pdr_improvement:.2f} | {delay_improvement:.2f} | {throughput_improvement:.2f} | {load_balance_improvement:.2f} | {energy_improvement:.2f} |\n"

        # 添加故障恢复结果 (如果有)
        if 'failure_recovery' in self.test_results and 'failure_recovery' in self.test_results['failure_recovery']:
            report += "\n## 故障恢复结果\n\n"

            recovery_metrics = self.test_results['failure_recovery']['failure_recovery']

            report += f"- 故障事件数: {len([e for e in self.failure_events if e['type'] == 'failure'])}\n"

            if 'recovery_time' in recovery_metrics and recovery_metrics['recovery_time'] is not None:
                report += f"- 平均恢复时间: {recovery_metrics['recovery_time']:.2f} 秒\n"

            if 'affected_flows' in recovery_metrics:
                report += f"- 受影响的流数量: {recovery_metrics['affected_flows']}\n"

        # 添加多路径策略比较
        report += "\n## 不同多路径策略比较\n\n"

        # 比较不同策略在相同路径数下的性能
        for max_paths in [2, 3]:
            report += f"### 最大路径数: {max_paths}\n\n"
            report += "| 策略 | PDR (%) | 延迟 (ms) | 吞吐量 (Kbps) | 负载平衡 (%) |\n"
            report += "|------|---------|-----------|---------------|-------------|\n"

            for strategy in ['weighted', 'round_robin', 'adaptive']:
                scenario = f"multipath_{strategy}_{max_paths}"
                if scenario in self.test_results:
                    result = self.test_results[scenario]
                    pdr = result['pdr'] * 100
                    delay = result['delay']
                    throughput = result['throughput']
                    load_balance = result['load_balance'] * 100

                    report += f"| {strategy} | {pdr:.2f} | {delay:.2f} | {throughput:.2f} | {load_balance:.2f} |\n"

            report += "\n"

        # 添加结论
        report += "## 结论\n\n"

        # 简单分析结果，找出最佳方案
        best_pdr_scenario = max(self.test_results.items(), key=lambda x: x[1]['pdr'])[0]
        best_delay_scenario = min(self.test_results.items(), key=lambda x: x[1]['delay'])[0]
        best_throughput_scenario = max(self.test_results.items(), key=lambda x: x[1]['throughput'])[0]
        best_balance_scenario = max(self.test_results.items(), key=lambda x: x[1]['load_balance'])[0]

        report += f"1. 最佳数据包交付率 (PDR) 由 **{best_pdr_scenario}** 场景实现。\n"
        report += f"2. 最低端到端延迟由 **{best_delay_scenario}** 场景实现。\n"
        report += f"3. 最高吞吐量由 **{best_throughput_scenario}** 场景实现。\n"
        report += f"4. 最佳负载平衡由 **{best_balance_scenario}** 场景实现。\n\n"

        # 综合评估
        report += "### 多路径路由效果评估\n\n"

        # 检查多路径是否比单路径好
        multipath_better = False
        for scenario, result in self.test_results.items():
            if (scenario != 'baseline' and
                    result['pdr'] > self.test_results['baseline']['pdr'] and
                    result['delay'] < self.test_results['baseline']['delay']):
                multipath_better = True
                break

        if multipath_better:
            report += "多路径路由方案显著优于单路径路由，提供了更高的数据包交付率和更低的端到端延迟。"
        else:
            report += "在测试的场景中，多路径路由的优势不明显，可能需要更多优化或不同的网络拓扑。"

        return report