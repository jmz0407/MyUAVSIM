# compare_protocols.py

import simpy
import matplotlib.pyplot as plt
import numpy as np
from utils import config
from simulator.simulator import Simulator


def run_simulation(protocol_name):
    """运行单次仿真并返回性能指标"""
    # 设置协议
    config.ROUTING_PROTOCOL = protocol_name

    # 创建仿真环境
    env = simpy.Environment()
    channel_states = {i: simpy.Resource(env, capacity=1) for i in range(config.NUMBER_OF_DRONES)}

    sim = Simulator(
        seed=2024,
        env=env,
        channel_states=channel_states,
        n_drones=config.NUMBER_OF_DRONES
    )

    # 运行仿真
    env.run(until=config.SIM_TIME)

    # 收集性能指标
    metrics = {
        'protocol': protocol_name,
        'pdr': sim.metrics.calculate_pdr() * 100,  # 转换为百分比
        'delay': np.mean(sim.metrics.delivery_time) / 1e3 if sim.metrics.delivery_time else 0,  # 转换为毫秒
        'throughput': np.mean(sim.metrics.throughput) / 1e3 if sim.metrics.throughput else 0,  # 转换为Kbps
        'hop_count': np.mean(sim.metrics.hop_cnt) if sim.metrics.hop_cnt else 0,
        'energy': sum(sim.metrics.energy_consumption.values())
    }

    return metrics


def compare_protocols():
    """对比不同协议的性能"""
    # 设置基本配置
    config.NUMBER_OF_DRONES = 20
    config.SIM_TIME = 60 * 1e6  # 60秒

    # 运行不同协议的仿真
    opar_metrics = run_simulation('OPAR')
    mp_olsr_metrics = run_simulation('MP-OLSR')
    amlb_opar_metrics = run_simulation('AMLB-OPAR')

    # 打印结果
    print("\n性能对比:")
    print(f"{'指标':<15} {'OPAR':<10} {'MP-OLSR':<10} {'AMLB-OPAR':<10}")
    print("-" * 50)

    metrics_to_compare = ['pdr', 'delay', 'throughput', 'hop_count', 'energy']
    for metric in metrics_to_compare:
        opar_value = opar_metrics[metric]
        mp_olsr_value = mp_olsr_metrics[metric]
        amlb_value = amlb_opar_metrics[metric]

        print(f"{metric:<15} {opar_value:<10.2f} {mp_olsr_value:<10.2f} {amlb_value:<10.2f}")

    # 计算AMLB-OPAR相对于MP-OLSR的改进
    print("\nAMLB-OPAR相对于MP-OLSR的改进百分比:")
    for metric in metrics_to_compare:
        mp_olsr_value = mp_olsr_metrics[metric]
        amlb_value = amlb_opar_metrics[metric]

        # 计算改进百分比
        if mp_olsr_value > 0:
            if metric in ['delay', 'energy', 'hop_count']:  # 这些指标越低越好
                improvement = (mp_olsr_value - amlb_value) / mp_olsr_value * 100
            else:  # PDR, throughput 越高越好
                improvement = (amlb_value - mp_olsr_value) / mp_olsr_value * 100
        else:
            improvement = float('inf')

        print(f"{metric:<15} {improvement:>10.2f}%")

    # 可视化对比
    plot_comparison(opar_metrics, mp_olsr_metrics, amlb_opar_metrics)


def plot_comparison(opar_metrics, mp_olsr_metrics, amlb_opar_metrics):
    """绘制协议性能对比图"""
    metrics = ['PDR (%)', 'Delay (ms)', 'Throughput (Kbps)', 'Hop Count', 'Energy (J)']
    opar_values = [
        opar_metrics['pdr'],
        opar_metrics['delay'],
        opar_metrics['throughput'],
        opar_metrics['hop_count'],
        opar_metrics['energy']
    ]
    mp_olsr_values = [
        mp_olsr_metrics['pdr'],
        mp_olsr_metrics['delay'],
        mp_olsr_metrics['throughput'],
        mp_olsr_metrics['hop_count'],
        mp_olsr_metrics['energy']
    ]
    amlb_values = [
        amlb_opar_metrics['pdr'],
        amlb_opar_metrics['delay'],
        amlb_opar_metrics['throughput'],
        amlb_opar_metrics['hop_count'],
        amlb_opar_metrics['energy']
    ]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width, opar_values, width, label='OPAR', color='green')
    rects2 = ax.bar(x, mp_olsr_values, width, label='MP-OLSR', color='blue')
    rects3 = ax.bar(x + width, amlb_values, width, label='AMLB-OPAR', color='red')

    ax.set_ylabel('Value')
    ax.set_title('协议性能对比')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    # 添加数值标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()
    plt.savefig('protocol_comparison.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    compare_protocols()