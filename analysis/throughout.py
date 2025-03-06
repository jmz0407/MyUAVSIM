import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置中文字体支持
plt.rcParams['font.sans-serif']=['STFangsong'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

# # 设置seaborn风格
# sns.set_style("whitegrid")

# 数据准备
network_sizes = [10, 20, 30, 40, 50]

# 吞吐量数据
throughput_data = {
    'F-TDMA': [12.5, 16.8, 19.3, 21.6, 23.4],
    'RA-STDMA': [17.2, 22.9, 25.8, 28.4, 30.5],
    'Greedy-STDMA': [19.8, 26.4, 30.3, 33.7, 36.2],
    'DGC-STDMA': [22.3, 29.7, 34.2, 38.0, 41.1],
    'CO-STDMA': [24.9, 32.8, 38.1, 42.3, 45.9],
    'RL-STDMA': [27.2, 36.9, 43.5, 48.7, 52.4]
}

plt.figure(figsize=(10, 6))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
methods = list(throughput_data.keys())

# 绘制吞吐量图
for i, method in enumerate(methods):
    plt.plot(network_sizes, throughput_data[method], marker='o', color=colors[i], linewidth=2, label=method)

plt.title('不同网络规模下的吞吐量比较', fontsize=14)
plt.xlabel('网络规模（节点数）', fontsize=12)
plt.ylabel('吞吐量 (Mbps)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(network_sizes)
plt.legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig('图4-3_不同网络规模下的吞吐量比较.png', dpi=300, bbox_inches='tight')
plt.show()