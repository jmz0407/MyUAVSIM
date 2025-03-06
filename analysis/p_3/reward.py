import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['STFangSong']
plt.rcParams['axes.unicode_minus'] = False

# 数据
reward_components = ['完整奖励', '仅吞吐量', '仅延迟', '仅能源', '无公平性']
throughput = [15.67, 16.23, 13.45, 12.87, 14.95]
delay = [58.4, 73.1, 51.3, 72.5, 61.2]
energy_efficiency = [18.75, 15.32, 16.41, 20.13, 17.86]
fairness_index = [0.92, 0.68, 0.73, 0.71, 0.62]

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 吞吐量对比
axes[0, 0].bar(reward_components, throughput, color=sns.color_palette("YlGnBu", 5))
axes[0, 0].set_ylabel('吞吐量 (Mbps)')
axes[0, 0].set_title('奖励函数组件对吞吐量的影响')
axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
axes[0, 0].set_ylim(10, 18)
for i, v in enumerate(throughput):
    axes[0, 0].text(i, v + 0.2, f"{v}", ha='center')

# 延迟对比
axes[0, 1].bar(reward_components, delay, color=sns.color_palette("YlOrRd", 5))
axes[0, 1].set_ylabel('端到端延迟 (ms)')
axes[0, 1].set_title('奖励函数组件对延迟的影响')
axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
axes[0, 1].set_ylim(40, 80)
for i, v in enumerate(delay):
    axes[0, 1].text(i, v + 1, f"{v}", ha='center')

# 能源效率对比
axes[1, 0].bar(reward_components, energy_efficiency, color=sns.color_palette("YlGn", 5))
axes[1, 0].set_ylabel('能源效率 (Kb/J)')
axes[1, 0].set_title('奖励函数组件对能源效率的影响')
axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
axes[1, 0].set_ylim(14, 22)
for i, v in enumerate(energy_efficiency):
    axes[1, 0].text(i, v + 0.2, f"{v}", ha='center')

# 公平性指数对比
axes[1, 1].bar(reward_components, fairness_index, color=sns.color_palette("PuBu", 5))
axes[1, 1].set_ylabel('公平性指数')
axes[1, 1].set_title('奖励函数组件对公平性的影响')
axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
axes[1, 1].set_ylim(0.5, 1.0)
for i, v in enumerate(fairness_index):
    axes[1, 1].text(i, v + 0.02, f"{v}", ha='center')

plt.tight_layout()
plt.savefig('reward_component_analysis.png', dpi=300, bbox_inches='tight')
plt.show()