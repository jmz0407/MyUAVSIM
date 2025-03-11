import simpy
import logging
import numpy as np
import matplotlib.pyplot as plt
from utils import config
from simulator.simulator import Simulator
from routing.AMLBR.manager.routing_manager import RoutingManager


def run_routing_test(test_config, duration=10e6, scenario="default"):
    """
    运行路由协议测试

    参数:
        test_config: 测试配置参数
        duration: 测试持续时间(微秒)
        scenario: 测试场景名称
    """
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=f'routing_test_{scenario}.log',
        filemode='w'
    )

    # 应用测试配置
    apply_test_config(test_config)

    # 创建模拟环境
    env = simpy.Environment()
    channel_states = {i: simpy.Resource(env, capacity=1) for i in range(config.NUMBER_OF_DRONES)}

    # 创建模拟器
    sim = Simulator(
        seed=test_config.get('seed', 2024),
        env=env,
        channel_states=channel_states,
        n_drones=config.NUMBER_OF_DRONES
    )

    # 替换默认路由协议为路由管理器
    for drone in sim.drones:
        drone.routing_protocol = RoutingManager(sim, drone)

    # 添加性能监控
    sim.env.process(monitor_performance(sim))

    # 添加测试流量
    add_test_traffic(sim, test_config.get('traffic_pattern', []))

    # 运行模拟
    logging.info(f"开始 {scenario} 场景测试, 持续时间: {duration / 1e6} 秒")
    sim.env.run(until=duration)

    # 收集结果
    results = collect_results(sim)

    # 打印结果摘要
    print_results_summary(results, scenario)

    # 生成性能图表
    plot_performance_charts(results, scenario)

    return results


def apply_test_config(test_config):
    """应用测试配置参数"""
    # 设置无人机数量
    if 'num_drones' in test_config:
        config.NUMBER_OF_DRONES = test_config['num_drones']

    # 设置物理层参数
    if 'phy_params' in test_config:
        phy_params = test_config['phy_params']
        if 'transmitting_power' in phy_params:
            config.TRANSMITTING_POWER = phy_params['transmitting_power']
        if 'snr_threshold' in phy_params:
            config.SNR_THRESHOLD = phy_params['snr_threshold']

    # 设置MAC层参数
    if 'mac_params' in test_config:
        mac_params = test_config['mac_params']
        if 'slot_duration' in mac_params:
            config.SLOT_DURATION = mac_params['slot_duration']

    # 设置路由参数
    if 'routing_params' in test_config:
        routing_params = test_config['routing_params']
        if 'max_paths' in routing_params:
            config.MAX_PATHS = routing_params['max_paths']
        if 'path_selection_strategy' in routing_params:
            config.PATH_SELECTION_STRATEGY = routing_params['path_selection_strategy']

    # 设置能量参数
    if 'energy_params' in test_config:
        energy_params = test_config['energy_params']
        if 'initial_energy' in energy_params:
            config.INITIAL_ENERGY = energy_params['initial_energy']


def monitor_performance(simulator):
    """监控网络性能"""
    interval = 1 * 1e6  # 每1秒记录一次

    while True:
        yield simulator.env.timeout(interval)

        # 计算当前性能指标
        current_time = simulator.env.now / 1e6  # 转换为秒

        # 延迟
        delays = simulator.metrics.end_to_end_delay
        avg_delay = sum(delays) / len(delays) if delays else 0

        # 吞吐量
        throughput = simulator.metrics.calculate_throughput()

        # 数据包投递率
        pdr = simulator.metrics.calculate_pdr()

        # 记录性能指标
        simulator.metrics.record_performance(
            avg_delay=avg_delay,
            throughput=throughput,
            pdr=pdr,
            time=current_time
        )

        # 打印当前状态
        logging.info(f"时间: {current_time:.1f}s - 延迟: {avg_delay / 1000:.2f}ms, "
                     f"吞吐量: {throughput:.2f}bps, PDR: {pdr * 100:.2f}%")


def add_test_traffic(simulator, traffic_pattern):
    """添加测试流量"""
    if not traffic_pattern:
        return

    # 创建流量生成进程
    for pattern in traffic_pattern:
        simulator.env.process(generate_traffic(simulator, pattern))


def generate_traffic(simulator, pattern):
    """根据模式生成流量"""
    # 解析流量参数
    src_id = pattern.get('source', 0)
    dst_id = pattern.get('destination', 1)
    start_time = pattern.get('start_time', 0)
    num_packets = pattern.get('num_packets', 100)
    interval = pattern.get('interval', 50000)  # 默认50ms
    priority = pattern.get('priority', 0)

    # 等待开始时间
    yield simulator.env.timeout(start_time)

    logging.info(f"开始生成从 {src_id} 到 {dst_id} 的流量，"
                 f"数据包数: {num_packets}, 优先级: {priority}")

    # 生成流量
    simulator.traffic_generator.generate_traffic(
        source_id=src_id,
        dest_id=dst_id,
        num_packets=num_packets,
        packet_interval=interval
    )


def collect_results(simulator):
    """收集模拟结果"""
    metrics = simulator.metrics

    # 提取性能数据
    results = {
        'end_to_end_delay': metrics.end_to_end_delay,
        'throughput': metrics.throughput_records,
        'pdr': metrics.pdr_records,
        'energy_consumption': {drone_id: metrics.energy_consumption[drone_id]
                               for drone_id in metrics.energy_consumption},
        'hop_counts': metrics.hop_cnt_dict,
        'collision_num': metrics.collision_num,
        'time_series': {
            'time': metrics.time_records,
            'delay': metrics.delay_records,
            'throughput': metrics.throughput_records,
            'pdr': metrics.pdr_records
        },
        'packet_stats': {
            'generated': metrics.datapacket_generated_num,
            'arrived': len(metrics.datapacket_arrived)
        }
    }

    return results


def print_results_summary(results, scenario):
    """打印结果摘要"""
    print(f"\n{'=' * 20} {scenario} 场景测试结果 {'=' * 20}")

    # 计算平均延迟
    delays = results['end_to_end_delay']
    avg_delay = sum(delays) / len(delays) if delays else 0
    print(f"平均端到端延迟: {avg_delay / 1000:.2f} ms")

    # 计算平均吞吐量
    throughputs = results['throughput']
    avg_throughput = sum(throughputs) / len(throughputs) if throughputs else 0
    print(f"平均吞吐量: {avg_throughput:.2f} bps")

    # 计算平均PDR
    pdrs = results['pdr']
    avg_pdr = sum(pdrs) / len(pdrs) if pdrs else 0
    print(f"平均数据包投递率: {avg_pdr * 100:.2f}%")

    # 计算平均跳数
    hop_counts = list(results['hop_counts'].values())
    avg_hops = sum(hop_counts) / len(hop_counts) if hop_counts else 0
    print(f"平均跳数: {avg_hops:.2f}")

    # 碰撞统计
    print(f"总碰撞次数: {results['collision_num']}")

    # 数据包统计
    generated = results['packet_stats']['generated']
    arrived = results['packet_stats']['arrived']
    print(f"总生成数据包: {generated}, 成功投递: {arrived}, "
          f"总体成功率: {arrived / generated * 100:.2f}% 如果生成不为0")

    # 能量消耗
    total_energy = sum(results['energy_consumption'].values())
    print(f"总能量消耗: {total_energy:.2f} J")
    print(f"{'=' * 50}")


def plot_performance_charts(results, scenario):
    """生成性能图表"""
    time_data = results['time_series']['time']
    delay_data = [d / 1000 for d in results['time_series']['delay']]  # 转换为毫秒
    throughput_data = results['time_series']['throughput']
    pdr_data = [p * 100 for p in results['time_series']['pdr']]  # 转换为百分比

    # 创建画布
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle(f"{scenario} 场景性能指标", fontsize=16)

    # 延迟图
    axes[0].plot(time_data, delay_data, 'b-', marker='o', markersize=4)
    axes[0].set_title('端到端延迟')
    axes[0].set_xlabel('时间 (秒)')
    axes[0].set_ylabel('延迟 (ms)')
    axes[0].grid(True)

    # 吞吐量图
    axes[1].plot(time_data, throughput_data, 'g-', marker='s', markersize=4)
    axes[1].set_title('网络吞吐量')
    axes[1].set_xlabel('时间 (秒)')
    axes[1].set_ylabel('吞吐量 (bps)')
    axes[1].grid(True)

    # PDR图
    axes[2].plot(time_data, pdr_data, 'r-', marker='^', markersize=4)
    axes[2].set_title('数据包投递率')
    axes[2].set_xlabel('时间 (秒)')
    axes[2].set_ylabel('PDR (%)')
    axes[2].set_ylim([0, 105])
    axes[2].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'performance_{scenario}.png', dpi=300)
    plt.close()

    # 绘制能量消耗图
    plt.figure(figsize=(10, 6))
    drone_ids = sorted(results['energy_consumption'].keys())
    energy_values = [results['energy_consumption'][drone_id] for drone_id in drone_ids]

    plt.bar(drone_ids, energy_values, color='orange')
    plt.title(f"{scenario} 场景能量消耗")
    plt.xlabel('无人机ID')
    plt.ylabel('能量消耗 (J)')
    plt.grid(True, axis='y')
    plt.savefig(f'energy_{scenario}.png', dpi=300)
    plt.close()


def compare_scenarios(results_dict):
    """比较不同场景的性能差异"""
    scenarios = list(results_dict.keys())
    metrics = ['avg_delay', 'avg_throughput', 'avg_pdr', 'energy_consumption']

    # 计算平均指标
    comparison_data = {}
    for scenario, results in results_dict.items():
        comparison_data[scenario] = {
            'avg_delay': sum(results['end_to_end_delay']) / len(results['end_to_end_delay']) if results[
                'end_to_end_delay'] else 0,
            'avg_throughput': sum(results['throughput']) / len(results['throughput']) if results['throughput'] else 0,
            'avg_pdr': sum(results['pdr']) / len(results['pdr']) if results['pdr'] else 0,
            'energy_consumption': sum(results['energy_consumption'].values())
        }

    # 创建比较图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('不同场景性能比较', fontsize=16)

    # 延迟比较
    axes[0, 0].bar(scenarios, [comparison_data[s]['avg_delay'] / 1000 for s in scenarios], color='skyblue')
    axes[0, 0].set_title('平均端到端延迟')
    axes[0, 0].set_ylabel('延迟 (ms)')
    axes[0, 0].grid(True, axis='y')

    # 吞吐量比较
    axes[0, 1].bar(scenarios, [comparison_data[s]['avg_throughput'] for s in scenarios], color='green')
    axes[0, 1].set_title('平均吞吐量')
    axes[0, 1].set_ylabel('吞吐量 (bps)')
    axes[0, 1].grid(True, axis='y')

    # PDR比较
    axes[1, 0].bar(scenarios, [comparison_data[s]['avg_pdr'] * 100 for s in scenarios], color='red')
    axes[1, 0].set_title('平均数据包投递率')
    axes[1, 0].set_ylabel('PDR (%)')
    axes[1, 0].set_ylim([0, 105])
    axes[1, 0].grid(True, axis='y')

    # 能量消耗比较
    axes[1, 1].bar(scenarios, [comparison_data[s]['energy_consumption'] for s in scenarios], color='orange')
    axes[1, 1].set_title('总能量消耗')
    axes[1, 1].set_ylabel('能量 (J)')
    axes[1, 1].grid(True, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('scenarios_comparison.png', dpi=300)
    plt.close()

    # 打印比较结果
    print("\n不同场景性能比较:")
    for metric in metrics:
        print(f"\n{metric}:")
        for scenario in scenarios:
            value = comparison_data[scenario][metric]
            if metric == 'avg_delay':
                print(f"  {scenario}: {value / 1000:.2f} ms")
            elif metric == 'avg_pdr':
                print(f"  {scenario}: {value * 100:.2f}%")
            else:
                print(f"  {scenario}: {value:.2f}")


if __name__ == "__main__":
    # 定义测试场景配置
    test_scenarios = {
        "AMLB-OPAR": {
            'num_drones': 10,
            'seed': 2024,
            'phy_params': {
                'transmitting_power': 0.1,
                'snr_threshold': 6
            },
            'routing_params': {
                'max_paths': 3,
                'path_selection_strategy': 'adaptive'
            },
            'traffic_pattern': [
                {
                    'source': 0,
                    'destination': 5,
                    'start_time': 1 * 1e6,  # 1秒后开始
                    'num_packets': 100,
                    'interval': 50000,  # 50ms间隔
                    'priority': 1  # 中优先级
                },
                {
                    'source': 2,
                    'destination': 7,
                    'start_time': 2 * 1e6,  # 2秒后开始
                    'num_packets': 100,
                    'interval': 100000,  # 100ms间隔
                    'priority': 0  # 低优先级
                },
                {
                    'source': 4,
                    'destination': 9,
                    'start_time': 5 * 1e6,  # 5秒后开始
                    'num_packets': 50,
                    'interval': 20000,  # 20ms间隔
                    'priority': 2  # 高优先级
                }
            ]
        },
        "Single-Path": {
            'num_drones': 10,
            'seed': 2024,
            'routing_params': {
                'max_paths': 1,
                'path_selection_strategy': 'best_quality'
            },
            'traffic_pattern': [
                {
                    'source': 0,
                    'destination': 5,
                    'start_time': 1 * 1e6,
                    'num_packets': 100,
                    'interval': 50000,
                    'priority': 1
                },
                {
                    'source': 2,
                    'destination': 7,
                    'start_time': 2 * 1e6,
                    'num_packets': 100,
                    'interval': 100000,
                    'priority': 0
                },
                {
                    'source': 4,
                    'destination': 9,
                    'start_time': 5 * 1e6,
                    'num_packets': 50,
                    'interval': 20000,
                    'priority': 2
                }
            ]
        },
        "High-Density": {
            'num_drones': 10,  # 更多无人机
            'seed': 2024,
            'routing_params': {
                'max_paths': 3,
                'path_selection_strategy': 'adaptive'
            },
            'traffic_pattern': [
                {
                    'source': 0,
                    'destination': 10,
                    'start_time': 1 * 1e6,
                    'num_packets': 100,
                    'interval': 50000,
                    'priority': 1
                },
                {
                    'source': 5,
                    'destination': 14,
                    'start_time': 2 * 1e6,
                    'num_packets': 100,
                    'interval': 100000,
                    'priority': 0
                },
                {
                    'source': 3,
                    'destination': 12,
                    'start_time': 3 * 1e6,
                    'num_packets': 100,
                    'interval': 50000,
                    'priority': 1
                },
                {
                    'source': 2,
                    'destination': 8,
                    'start_time': 4 * 1e6,
                    'num_packets': 50,
                    'interval': 20000,
                    'priority': 2
                }
            ]
        },
        "Energy-Limited": {
            'num_drones': 10,
            'seed': 2024,
            'energy_params': {
                'initial_energy': 5 * 1e3  # 降低初始能量
            },
            'routing_params': {
                'max_paths': 3,
                'path_selection_strategy': 'adaptive'
            },
            'traffic_pattern': [
                {
                    'source': 0,
                    'destination': 5,
                    'start_time': 1 * 1e6,
                    'num_packets': 200,  # 增加包数量以消耗更多能量
                    'interval': 30000,
                    'priority': 1
                },
                {
                    'source': 2,
                    'destination': 7,
                    'start_time': 2 * 1e6,
                    'num_packets': 200,
                    'interval': 30000,
                    'priority': 0
                }
            ]
        }
    }

    # 运行所有测试场景
    results_dict = {}
    for scenario, config in test_scenarios.items():
        print(f"\n运行 {scenario} 场景测试...")
        results = run_routing_test(config, duration=20e6, scenario=scenario)
        results_dict[scenario] = results

    # 比较不同场景的性能
    compare_scenarios(results_dict)

    print("\n所有测试完成，详细结果已保存到对应的日志和图表文件中。")