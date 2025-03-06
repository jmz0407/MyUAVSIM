# high_load_test.py

import simpy
import matplotlib.pyplot as plt
import numpy as np
from utils import config
from simulator.simulator import Simulator


def run_high_load_test(protocol_name):
    """运行高负载情景测试"""
    # 设置协议
    config.ROUTING_PROTOCOL = protocol_name

    # 增加流量生成率
    original_traffic_rate = getattr(config, 'TRAFFIC_RATE', None)
    config.TRAFFIC_RATE = 5  # 高流量生成率

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

    # 恢复原始流量生成率
    if original_traffic_rate is not None:
        config.TRAFFIC_RATE = original_traffic_rate
    else:
        delattr(config, 'TRAFFIC_RATE')

    # 收集性能指标
    metrics = {
        'protocol': protocol_name,
        'pdr': sim.metrics.calculate_pdr() * 100,
        'delay': np.mean(sim.metrics.delivery_time) / 1e3 if sim.metrics.delivery_time else 0,
        'throughput': np.mean(sim.metrics.throughput) / 1e3 if sim.metrics.throughput else 0,
        'hop_count': np.mean(sim.metrics.hop_cnt) if sim.metrics.hop_cnt else 0,
        'energy': sum(sim.metrics.energy_consumption.values()),
        'collisions': sim.metrics.collision_num
    }

    return metrics


def compare_high_load():
    """对比高负载情景下的协议性能"""
    # 设置基本配置
    config.NUMBER_OF_DRONES = 20
    config.SIM_TIME = 60 * 1e6  # 60秒

    # 运行高负载测试
    mp_olsr_metrics = run_high_load_test('MP-OLSR')
    amlb_opar_metrics = run_high_load_test('AMLB-OPAR')

    # 打印结果
    print("\n高负载情景性能对比:")
    print(f"{'指标':<15} {'MP-OLSR':<10} {'AMLB-OPAR':<10} {'改进':<10}")
    print("-" * 50)

    metrics_to_compare = ['pdr', 'delay', 'throughput', 'hop_count', 'energy', 'collisions']
    for metric in metrics_to_compare:
        mp_olsr_value = mp_olsr_metrics[metric]
        amlb_value = amlb_opar_metrics[metric]

        # 计算改进百分比
        if mp_olsr_value > 0:
            if metric in ['delay', 'energy', 'hop_count', 'collisions']:  # 这些指标越低越好
                improvement = (mp_olsr_value - amlb_value) / mp_olsr_value * 100
            else:  # PDR, throughput 越高越好
                improvement = (amlb_value - mp_olsr_value) / mp_olsr_value * 100
        else:
            improvement = float('inf')

        print(f"{metric:<15} {mp_olsr_value:<10.2f} {amlb_value:<10.2f} {improvement:>10.2f}%")

    # 可视化对比
    plot_high_load_comparison(mp_olsr_metrics, amlb_opar_metrics)


def plot_high_load_comparison(mp_olsr_metrics, amlb_opar_metrics):
    """绘制高负载情景下的协议性能对比图"""
    metrics = ['PDR (%)', 'Delay (ms)', 'Throughput (Kbps)', 'Collisions']

    mp_olsr_values = [
        mp_olsr_metrics['pdr'],
        mp_olsr_metrics['delay'],
        mp_olsr_metrics['throughput'],
        mp_olsr_metrics['collisions']
    ]

    amlb_values = [
        amlb_opar_metrics['pdr'],
        amlb_opar_metrics['delay'],
        amlb_opar_metrics['throughput'],
        amlb_opar_metrics['collisions']
    ]

    # 计算改进百分比
    improvements = []
    for i, metric in enumerate(metrics):
        mp_olsr_value = mp_olsr_values[i]
        amlb_value = amlb_values[i]

        if mp_olsr_value > 0:
            if metric in ['Delay (ms)', 'Collisions']:  # 这些指标越低越好
                improvement = (mp_olsr_value - amlb_value) / mp_olsr_value * 100
            else:  # PDR, Throughput 越高越好
                improvement = (amlb_value - mp_olsr_value) / mp_olsr_value * 100
        else:
            improvement = 0

        improvements.append(improvement)

    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # 第一个子图：性能指标对比
    x = np.arange(len(metrics))
    width = 0.35

    rects1 = ax1.bar(x - width / 2, mp_olsr_values, width, label='MP-OLSR', color='blue')
    rects2 = ax1.bar(x + width / 2, amlb_values, width, label='AMLB-OPAR', color='red')

    ax1.set_ylabel('Value')
    ax1.set_title('高负载情景下的性能指标对比')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()

    # 添加数值标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax1.annotate(f'{height:.2f}',
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    # 第二个子图：改进百分比
    ax2.bar(metrics, improvements, color='green')

    ax2.set_ylabel('改进百分比 (%)')
    ax2.set_title('AMLB-OPAR相对于MP-OLSR的改进')

    # 添加数值标签
    for i, v in enumerate(improvements):
        ax2.text(i, v + 0.5, f"{v:.2f}%", ha='center')

    fig.tight_layout()
    plt.savefig('high_load_comparison.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    compare_high_load()