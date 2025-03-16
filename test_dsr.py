import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置中文字体支持
plt.rcParams['font.sans-serif']=['STFangsong'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
# # 设置seaborn风格
# sns.set_style("whitegrid")

# 数据准备
network_sizes = [10, 15, 20, 25, 30]

# 时隙复用率数据
reuse_ratio_data = {
    'TDMA': [1.0, 1.0, 1.0, 1.0, 1.0],
    'STDMA': [1.42, 1.58, 1.67, 1.75, 1.82],
    'DQN-STDMA': [1.56, 1.71, 1.85, 1.93, 1.99],
    'GAT-PPO': [1.81, 1.96, 2.12, 2.19, 2.25]
}

# QoS满足率数据
qos_data = {
    'TDMA': [55.2, 68.5, 63.9, 60.1, 57.5],
    'STDMA': [61.8, 75.3, 71.2, 68.4, 65.7],
    'DQN-STDMA': [68.3, 82.7, 79.4, 75.8, 73.2],
    'GAT-PPO': [78.1, 91.2, 88.5, 85.2, 83.1]
}

# 创建三个子图
fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(18, 6))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
methods = list(reuse_ratio_data.keys())
# 定义颜色和标记
colors = ['#1e88e5', '#e53935', '#43a047', '#8e24aa']  # 蓝、红、绿、紫
markers = ['o', 's', '^', 'D']  # 圆形、方形、三角形、菱形
labels = ['STDMA', 'DQN-STDMA', 'GAT-PPO', 'TDMA']
# protocols = [basic_stdma, dqn_stdma, ppo_stdma, tra_tdma]
# 绘制时隙复用率图
for i, method in enumerate(methods):
    ax1.plot(network_sizes, reuse_ratio_data[method], marker='o', color=colors[i], linewidth=2)

ax1.set_title('不同网络规模下的时隙复用率', fontsize=14)
ax1.set_xlabel('网络规模（节点数）', fontsize=12)
ax1.set_ylabel('时隙复用率', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_xticks(network_sizes)


# 绘制QoS满足率图
for i, method in enumerate(methods):
    ax3.plot(network_sizes, qos_data[method], marker='^', color=colors[i], linewidth=2)

ax3.set_title('不同网络规模下的QoS满足率', fontsize=14)
ax3.set_xlabel('网络规模（节点数）', fontsize=12)
ax3.set_ylabel('QoS满足率 (%)', fontsize=12)
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.set_xticks(network_sizes)

# 添加图例（仅需添加一次，放在最后一个子图中）
lines = []
labels = []
for i, method in enumerate(methods):
    line, = ax1.plot([], [], color=colors[i], marker='o', linewidth=2)
    line, = ax1.plot([], [], color=colors[i], marker='o', linewidth=2)
    lines.append(line)
    labels.append(method)

#在每幅图中添加图例
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05),
           fancybox=True, shadow=True, ncol=5, fontsize=12)
ax3.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05),
           fancybox=True, shadow=True, ncol=5, fontsize=12)
# fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05),
#            fancybox=True, shadow=True, ncol=5, fontsize=12)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig('图3-3_不同网络规模下的性能比较.png', dpi=300, bbox_inches='tight')
plt.show()