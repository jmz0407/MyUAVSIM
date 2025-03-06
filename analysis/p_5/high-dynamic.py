import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 设置仿宋字体
plt.rcParams['font.sans-serif'] = ['STFangSong']
plt.rcParams['axes.unicode_minus'] = False

# 数据
speeds = [5, 10, 15, 20]
methods = ['OLSR', 'MP-OLSR', 'AOMDV', 'DMLR', 'AMLB']

pdr_data = {
    'OLSR': [87.3, 74.5, 62.8, 48.3],
    'MP-OLSR': [92.1, 85.6, 76.3, 64.7],
    'AOMDV': [89.5, 79.2, 68.5, 55.2],
    'DMLR': [94.8, 89.3, 81.2, 72.5],
    'AMLB': [96.5, 93.7, 88.2, 81.6]
}

# 绘制折线图
plt.figure(figsize=(12, 8))

markers = ['o', 's', '^', 'd', 'v']
for i, method in enumerate(methods):
    plt.plot(speeds, pdr_data[method], marker=markers[i], linewidth=2, markersize=8, label=method)

plt.xlabel('移动速度 (m/s)', fontsize=14)
plt.ylabel('数据包投递率 (%)', fontsize=14)
plt.title('不同移动速度下的数据包投递率', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.xticks(speeds)
plt.ylim(40, 100)

# 为每条线的末尾添加数值标签
for method in methods:
    last_value = pdr_data[method][-1]
    plt.text(speeds[-1]+0.5, last_value, f"{last_value}%", ha='left', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('high_mobility_scenario_performance_line.png', dpi=300, bbox_inches='tight')
plt.show()