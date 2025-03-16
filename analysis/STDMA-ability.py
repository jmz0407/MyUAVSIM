import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['STFangSong']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 设置seaborn风格
# sns.set_style("whitegrid")

# 数据准备
network_sizes = [10, 20, 30, 40, 50]

# 时隙复用率数据
reuse_ratio_data = {
    'Fixed-TDMA': [1.0, 1.0, 1.0, 1.0, 1.0],
    'DQN': [1.56, 1.71, 1.85, 1.93, 1.99],
    'MLP-PPO': [1.68, 1.81, 1.93, 2.02, 2.09],
    'GAT-PPO': [1.81, 1.96, 2.12, 2.20, 2.25]
}

# 吞吐量数据
throughput_data = {
    'Fixed-TDMA': [14.2, 18.5, 22.1, 25.3, 28.4],
    'DQN': [22.5, 28.6, 33.8, 38.2, 41.7],
    'MLP-PPO': [24.1, 30.2, 35.9, 40.8, 44.5],
    'GAT-PPO': [27.8, 35.8, 42.3, 47.9, 52.6]
}

# QoS满足率数据
qos_data = {
    'Fixed-TDMA': [55.2, 68.5, 63.9, 60.1, 57.5],
    'DQN': [68.3, 82.7, 79.4, 75.8, 73.2],
    'MLP-PPO': [72.5, 85.9, 82.7, 79.3, 76.8],
    'GAT-PPO': [78.1, 91.2, 88.5, 85.2, 83.1]
}

# 创建三个子图
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
methods = list(reuse_ratio_data.keys())

# 绘制时隙复用率图
for i, method in enumerate(methods):
    ax1.plot(network_sizes, reuse_ratio_data[method], marker='o', color=colors[i], linewidth=2)

ax1.set_title('不同网络规模下的时隙复用率', fontsize=14)
ax1.set_xlabel('网络规模（节点数）', fontsize=12)
ax1.set_ylabel('时隙复用率', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_xticks(network_sizes)

# 绘制吞吐量图
for i, method in enumerate(methods):
    ax2.plot(network_sizes, throughput_data[method], marker='s', color=colors[i], linewidth=2)

ax2.set_title('不同网络规模下的吞吐量', fontsize=14)
ax2.set_xlabel('网络规模（节点数）', fontsize=12)
ax2.set_ylabel('吞吐量 (Mbps)', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.set_xticks(network_sizes)

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
    line, = ax3.plot([], [], color=colors[i], marker='o', linewidth=2)
    lines.append(line)
    labels.append(method)

fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05),
           fancybox=True, shadow=True, ncol=5, fontsize=12)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig('图3-3_不同网络规模下的性能比较.png', dpi=300, bbox_inches='tight')
plt.show()