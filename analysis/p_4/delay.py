import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 设置仿宋字体
plt.rcParams['font.sans-serif'] = ['STFangSong']
plt.rcParams['axes.unicode_minus'] = False

# 数据：不同方法对不同业务类型的延迟
traffic_types = ['周期性业务', '突发业务', '流媒体业务', '控制业务']
methods = ['TDMA', 'DQN-STDMA', 'F-STDMA', 'PPO-STDMA']

delay_data = {
    'TDMA': [8.5, 12.3, 14.7, 8.2],
    'DQN-STDMA': [2.3, 10.7, 7.5, 8.4],
    'F-STDMA': [5.9, 10.5, 8.3, 8.1],
    'PPO-STDMA': [1.7, 8.4, 5.9, 8.0]
}

# 创建图表
plt.figure(figsize=(14, 8))

# 绘制折线图
markers = ['o', 's', '^', 'd', 'v']
for i, method in enumerate(methods):
    plt.plot(traffic_types, delay_data[method], marker=markers[i], linewidth=2, markersize=8, label=method)

plt.xlabel('业务类型', fontsize=14)
plt.ylabel('端到端延迟 (ms)', fontsize=14)
plt.title('不同业务类型的端到端延迟对比', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# 为RL-STDMA的每个点添加数值标签（为避免拥挤，只标注最优方法）
for i, value in enumerate(delay_data['PPO-STDMA']):
    plt.text(i, value-3, f"{value}", ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('delay_comparison_line.png', dpi=300, bbox_inches='tight')
plt.show()