import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 设置仿宋字体
plt.rcParams['font.sans-serif'] = ['STFangSong']
plt.rcParams['axes.unicode_minus'] = False

# 数据
node_counts = [10, 20, 30, 50]
methods = ['OLSR', 'MP-OLSR', 'AMLB']

control_overhead = {
    'OLSR': [2.34, 5.87, 11.25, 28.32],
    'MP-OLSR': [3.56, 8.93, 18.76, 46.57],
    'AMLB': [2.87, 6.25, 12.43, 26.85]
}

convergence_time = {
    'OLSR': [1.25, 2.53, 4.65, 8.94],
    'MP-OLSR': [1.73, 3.42, 6.32, 12.67],
    'AMLB': [1.42, 2.68, 4.73, 8.25]
}

# 创建图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# 控制开销
markers = ['o', 's', '^']
for i, method in enumerate(methods):
    ax1.plot(node_counts, control_overhead[method], marker=markers[i], linewidth=2, markersize=8, label=method)

ax1.set_xlabel('节点数', fontsize=14)
ax1.set_ylabel('控制开销 (KB/s)', fontsize=14)
ax1.set_title('不同网络规模下的控制开销', fontsize=16)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(fontsize=12)
ax1.set_xticks(node_counts)

# 收敛时间
for i, method in enumerate(methods):
    ax2.plot(node_counts, convergence_time[method], marker=markers[i], linewidth=2, markersize=8, label=method)

ax2.set_xlabel('节点数', fontsize=14)
ax2.set_ylabel('收敛时间 (s)', fontsize=14)
ax2.set_title('不同网络规模下的收敛时间', fontsize=16)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend(fontsize=12)
ax2.set_xticks(node_counts)

plt.tight_layout()
plt.savefig('scalability_analysis_line.png', dpi=300, bbox_inches='tight')
plt.show()