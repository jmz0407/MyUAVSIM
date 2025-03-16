import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import logging

matplotlib.font_manager._log.setLevel(logging.WARNING)
plt.rcParams['font.sans-serif'] = ['STFangsong']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 18})

# 数据准备
drones = [10, 15, 20, 25, 30]

# TDMA协议 - 基础性能较低，不同业务流表现差异大
tra_tdma = {
    'CBR': [2350, 1950, 1600, 1250, 900],            # CBR表现相对稳定
    'POISSON': [2200, 1700, 1250, 850, 450],         # 随机性流量表现更差
    'BURST': [1850, 1350, 950, 600, 300],            # 突发流量处理能力最弱
    'VBR': [2250, 1750, 1350, 950, 580],             # 变率流量适应性中等
}

# 基础STDMA - 空间复用提升性能，但仍有明显下降
basic_stdma = {
    'CBR': [3700, 3400, 3100, 2800, 2500],           # 稳定流量处理较好
    'POISSON': [3500, 3200, 2850, 2450, 2050],       # 随机性适应性一般
    'BURST': [3200, 2800, 2400, 2000, 1650],         # 突发流量处理较弱
    'VBR': [3700, 3200, 2650, 2100, 1500],           # 变率流量下降明显
}

# DQN-STDMA - 学习能力使其对不同业务流适应性提高
dqn_stdma = {
    'CBR': [5600, 5500, 5350, 5150, 4950],           # 稳定流量处理良好
    'POISSON': [5700, 5650, 5550, 5400, 5200],       # 随机流量适应性强
    'BURST': [5400, 5250, 5050, 4800, 4500],         # 突发流量处理有提升
    'VBR': [5600, 5300, 4950, 4550, 4100],           # 变率流量中等适应性
}

# GAT-PPO - 最佳性能，对各种业务流均有良好适应性
ppo_stdma = {
    'CBR': [8100, 7900, 7650, 7350, 7050],           # 稳定流量处理优异
    'POISSON': [8200, 8050, 7850, 7600, 7300],       # 随机流量适应性最强
    'BURST': [7900, 7700, 7450, 7150, 6800],         # 突发流量处理能力强
    'VBR': [8000, 7700, 7300, 6850, 6300],           # 变率流量适应性好
}

# 绘图代码
colors = ['#1e88e5', '#e53935', '#43a047', '#8e24aa']
markers = ['o', 's', '^', 'D']
labels = ['STDMA', 'DQN-STDMA', 'GAT-PPO', 'TDMA']
protocols = [basic_stdma, dqn_stdma, ppo_stdma, tra_tdma]
traffic_types = ['CBR', 'POISSON', 'BURST', 'VBR']
traffic_titles = {
    'CBR': 'CBR业务流 - 吞吐量与无人机数量关系',
    'POISSON': 'POISSON业务流 - 吞吐量与无人机数量关系',
    'BURST': 'BURST业务流 - 吞吐量与无人机数量关系',
    'VBR': 'VBR业务流 - 吞吐量与无人机数量关系',
}

# 创建每种业务流的图表
for traffic in traffic_types:
    plt.figure(figsize=(10, 6))

    for i, protocol in enumerate(protocols):
        plt.plot(drones, protocol[traffic], marker=markers[i], color=colors[i],
                 linewidth=2, markersize=8, label=labels[i])

    plt.title(traffic_titles[traffic], fontsize=16)
    plt.xlabel('无人机数量', fontsize=14)
    plt.ylabel('吞吐量 (kbps)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=12)
    plt.xticks(drones)
    plt.ylim(0, 9000)
    plt.tight_layout()
    plt.savefig(f'{traffic}_throughput.png', dpi=300)
    plt.close()
# 创建所有业务流的对比图
plt.figure(figsize=(12, 8))

# 为每种协议创建子图
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
axs = axs.flatten()

for i, (protocol, label) in enumerate(zip(protocols, labels)):
    ax = axs[i]

    for traffic in traffic_types:
        ax.plot(drones, protocol[traffic], marker='o', linewidth=2,
                label=traffic)

    ax.set_title(f'{label}协议在不同业务流下的吞吐量', fontsize=14)
    ax.set_xlabel('无人机数量', fontsize=12)
    ax.set_ylabel('吞吐量 (kbps)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best')
    ax.set_xticks(drones)

    # 为不同的协议设置不同的y轴范围
    if label in ['DQN-STDMA', 'GAT-PPO']:
        ax.set_ylim(0, 10500)
    else:
        ax.set_ylim(0, 5000)

plt.tight_layout()
plt.savefig('protocols_comparison.png', dpi=300)
plt.close()

# 创建综合性图表：所有协议、所有业务流平均吞吐量
plt.figure(figsize=(12, 7))

# 计算每个协议在每个无人机数量下的平均吞吐量
average_throughput = []
for protocol, label in zip(protocols, labels):
    avg_values = []
    for i in range(len(drones)):
        avg = sum(protocol[traffic][i] for traffic in traffic_types) / len(traffic_types)
        avg_values.append(avg)
    average_throughput.append(avg_values)

# 绘制平均吞吐量图
for i, (avg, label) in enumerate(zip(average_throughput, labels)):
    plt.plot(drones, avg, marker=markers[i], color=colors[i],
             linewidth=3, markersize=10, label=label)

plt.title('各协议在不同无人机数量下的平均吞吐量', fontsize=18)
plt.xlabel('无人机数量', fontsize=16)
plt.ylabel('平均吞吐量 (kbps)', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='best', fontsize=14)
plt.xticks(drones, fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()
plt.savefig('average_throughput1.png', dpi=300)
plt.close()

print("所有图表已生成完毕！")