import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 设置仿宋字体
plt.rcParams['font.sans-serif'] = ['STFangSong']
plt.rcParams['axes.unicode_minus'] = False

# 数据
load_levels = ['低负载', '中负载', '高负载']
methods = ['OLSR', 'MP-OLSR', 'AOMDV', 'DMLR', 'AMLB']

throughput_data = {
    'OLSR': [8.32, 7.14, 4.75],
    'MP-OLSR': [10.87, 9.95, 7.82],
    'AOMDV': [9.73, 8.46, 6.32],
    'DMLR': [12.45, 11.78, 9.84],
    'AMLB': [14.82, 14.53, 13.16]
}

delay_data = {
    'OLSR': [124.5, 186.3, 312.6],
    'MP-OLSR': [98.2, 142.7, 215.4],
    'AOMDV': [112.6, 165.8, 267.3],
    'DMLR': [78.3, 103.5, 157.8],
    'AMLB': [65.7, 84.2, 112.5]
}

# 绘制两个子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# 绘制吞吐量对比图
markers = ['o', 's', '^', 'd', 'v']
for i, method in enumerate(methods):
    ax1.plot(load_levels, throughput_data[method], marker=markers[i], linewidth=2, markersize=8, label=method)

ax1.set_xlabel('负载级别', fontsize=14)
ax1.set_ylabel('吞吐量 (Mbps)', fontsize=14)
ax1.set_title('不同负载级别下的吞吐量对比', fontsize=16)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(fontsize=12)

# 绘制延迟对比图
for i, method in enumerate(methods):
    ax2.plot(load_levels, delay_data[method], marker=markers[i], linewidth=2, markersize=8, label=method)

ax2.set_xlabel('负载级别', fontsize=14)
ax2.set_ylabel('端到端延迟 (ms)', fontsize=14)
ax2.set_title('不同负载级别下的延迟对比', fontsize=16)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend(fontsize=12)

plt.tight_layout()
plt.savefig('high_load_scenario_performance_line.png', dpi=300, bbox_inches='tight')
plt.show()