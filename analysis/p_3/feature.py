import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置仿宋字体
plt.rcParams['font.sans-serif'] = ['STFangSong']  # 使用仿宋字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 数据
methods = ['GAT', 'GCN', 'MLP', '原始特征']
throughput = [15.67, 14.47, 13.60, 12.68]
delay = [58.4, 66.7, 74.8, 90.2]
energy_efficiency = [18.75, 16.8, 15.2, 14.1]

# 创建图表
fig, ax = plt.subplots(figsize=(12, 8))

# 创建次坐标轴
ax2 = ax.twinx()
ax3 = ax.twinx()
ax3.spines['right'].set_position(('outward', 60))  # 将第三个y轴向右移动

# 绘制折线图
line1, = ax.plot(methods, throughput, 'o-', color='royalblue', linewidth=2, markersize=8, label='吞吐量 (Mbps)')
line2, = ax2.plot(methods, delay, 's-', color='tomato', linewidth=2, markersize=8, label='端到端延迟 (ms)')
line3, = ax3.plot(methods, energy_efficiency, '^-', color='forestgreen', linewidth=2, markersize=8, label='能源效率 (Kb/J)')

# 设置坐标轴标签
ax.set_xlabel('特征提取器', fontsize=14)
ax.set_ylabel('吞吐量 (Mbps)', fontsize=14, color='royalblue')
ax2.set_ylabel('端到端延迟 (ms)', fontsize=14, color='tomato')
ax3.set_ylabel('能源效率 (Kb/J)', fontsize=14, color='forestgreen')

# 设置坐标轴范围
ax.set_ylim(12, 17)
ax2.set_ylim(50, 100)
ax3.set_ylim(13, 20)

# 设置坐标轴颜色
ax.tick_params(axis='y', colors='royalblue')
ax2.tick_params(axis='y', colors='tomato')
ax3.tick_params(axis='y', colors='forestgreen')

# 添加网格线
ax.grid(True, linestyle='--', alpha=0.3)

# 添加图例
lines = [line1, line2, line3]
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc='upper right', fontsize=12)

# 设置标题
plt.title('特征提取器性能比较', fontsize=16)

# 添加数值标签
for i, v in enumerate(throughput):
    ax.text(i, v+0.1, f"{v}", ha='center', va='bottom', color='royalblue', fontsize=10)
for i, v in enumerate(delay):
    ax2.text(i, v+1, f"{v}", ha='center', va='bottom', color='tomato', fontsize=10)
for i, v in enumerate(energy_efficiency):
    ax3.text(i, v+0.1, f"{v}", ha='center', va='bottom', color='forestgreen', fontsize=10)

plt.tight_layout()
plt.savefig('feature_extractor_comparison_line.png', dpi=300, bbox_inches='tight')
plt.show()