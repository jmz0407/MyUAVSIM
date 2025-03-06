import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 设置仿宋字体
plt.rcParams['font.sans-serif'] = ['STFangSong']
plt.rcParams['axes.unicode_minus'] = False

# 数据：不同方法在不同网络规模下的吞吐量
network_sizes = [10, 20, 30]
methods = ['TDMA',  'DQN-STDMA', 'F-STDMA', 'PPO-STDMA']

throughput_data = {
    'TDMA': [5.32, 8.45, 10.21],
    'DQN-STDMA': [7.94, 12.65, 15.43],
    'F-STDMA': [7.21, 11.87, 14.26],
    'PPO-STDMA': [10.45, 17.83, 21.56]
}

# 创建图表
plt.figure(figsize=(12, 8))

# 绘制折线图
for method in methods:
    plt.plot(network_sizes, throughput_data[method], 'o-', linewidth=2, markersize=8, label=method)

plt.xlabel('网络规模（节点数）', fontsize=14)
plt.ylabel('吞吐量 (Mbps)', fontsize=14)
plt.title('不同网络规模下各方法的吞吐量对比', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.xticks(network_sizes)

# 为每条线的每个点添加数值标签
for method in methods:
    for i, value in enumerate(throughput_data[method]):
        plt.text(network_sizes[i], value+0.3, f"{value}", ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('throughput_comparison_line.png', dpi=300, bbox_inches='tight')
plt.show()