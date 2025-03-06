import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置仿宋字体
plt.rcParams['font.sans-serif'] = ['STFangSong']
plt.rcParams['axes.unicode_minus'] = False

# 数据
algorithms = ['GAT-PPO', 'DQN', 'PPO', 'GAT-DQN']
throughput = [15.67, 14.68, 15.02, 13.22]
energy_efficiency = [18.75, 17.17, 17.72, 15.33]
delay = [58.4, 63.9, 61.5, 72.7]

# 创建图表
fig, ax = plt.subplots(figsize=(12, 8))

# 创建次坐标轴
ax2 = ax.twinx()
ax3 = ax.twinx()
ax3.spines['right'].set_position(('outward', 60))

# 绘制折线图
line1, = ax.plot(algorithms, throughput, 'o-', color='indigo', linewidth=2, markersize=8, label='吞吐量 (Mbps)')
line2, = ax2.plot(algorithms, delay, 's-', color='crimson', linewidth=2, markersize=8, label='端到端延迟 (ms)')
line3, = ax3.plot(algorithms, energy_efficiency, '^-', color='teal', linewidth=2, markersize=8, label='能源效率 (Kb/J)')

# 设置坐标轴标签
ax.set_xlabel('强化学习算法', fontsize=14)
ax.set_ylabel('吞吐量 (Mbps)', fontsize=14, color='indigo')
ax2.set_ylabel('端到端延迟 (ms)', fontsize=14, color='crimson')
ax3.set_ylabel('能源效率 (Kb/J)', fontsize=14, color='teal')

# 设置坐标轴颜色
ax.tick_params(axis='y', colors='indigo')
ax2.tick_params(axis='y', colors='crimson')
ax3.tick_params(axis='y', colors='teal')

# 添加网格线
ax.grid(True, linestyle='--', alpha=0.3)

# 添加图例
lines = [line1, line2, line3]
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc='upper left', fontsize=12)

# 设置标题
plt.title('强化学习算法性能比较', fontsize=16)

# 添加数值标签
for i, v in enumerate(throughput):
    ax.text(i, v+0.1, f"{v}", ha='center', va='bottom', color='indigo', fontsize=10)
for i, v in enumerate(delay):
    ax2.text(i, v+1, f"{v}", ha='center', va='bottom', color='crimson', fontsize=10)
for i, v in enumerate(energy_efficiency):
    ax3.text(i, v+0.1, f"{v}", ha='center', va='bottom', color='teal', fontsize=10)

plt.tight_layout()
plt.savefig('rl_algorithm_comparison_line.png', dpi=300, bbox_inches='tight')
plt.show()