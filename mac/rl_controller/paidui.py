import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

from matplotlib import pyplot as plt
# 设置中文字体（以PingFang SC为例）
plt.rcParams['font.sans-serif']=['STFangsong'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

# 数据准备
protocols = ['GAT-PPO-STDMA', 'DQN-STDMA', 'STDMA', 'TDMA']
queue_lengths = [4.2, 7.8, 12.3, 18.6]  # 队列长度（数据包）
packet_service_time = 6  # 单位：ms
theoretical_delays = [q * packet_service_time for q in queue_lengths]  # 理论排队延迟
measured_delays = [1.45, 3.42, 15.47, 58.96]  # 实测端到端延迟（ms）

# 颜色设置
colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2']
bar_colors = ['#4E79A7', '#59A14F', '#EDC948', '#B07AA1']

# 创建图形和轴
fig = plt.figure(figsize=(12, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.2])

# 第一个子图：队列长度对比
ax1 = plt.subplot(gs[0])
bars = ax1.bar(protocols, queue_lengths, color=bar_colors, alpha=0.8, width=0.6)
ax1.set_ylabel('平均队列长度 (数据包)', fontsize=12)
ax1.set_title('图4-9 不同MAC协议的队列长度', fontsize=14, fontweight='bold')
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# 为柱状图添加数值标签
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
             f'{height}', ha='center', va='bottom', fontsize=10)

# 第二个子图：延迟对比
ax2 = plt.subplot(gs[1])
x = np.arange(len(protocols))
width = 0.35

# 绘制理论排队延迟（柱状图）
bars1 = ax2.bar(x - width / 2, theoretical_delays, width, label='理论排队延迟', color='#59A14F', alpha=0.7)
# 绘制实测端到端延迟（柱状图）
bars2 = ax2.bar(x + width / 2, measured_delays, width, label='实测端到端延迟', color='#E15759', alpha=0.7)

# 为柱状图添加数值标签
for bar in bars1:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2., height + 1,
             f'{height:.1f}', ha='center', va='bottom', fontsize=9)

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2., height + 1,
             f'{height:.2f}', ha='center', va='bottom', fontsize=9)

# 添加折线图，展示队列长度与实测延迟的关系
ax3 = ax2.twinx()
ax3.plot(protocols, measured_delays, 'o-', color='#EDC948', linewidth=2, markersize=8, label='实测延迟曲线')

# 设置x轴标签
ax2.set_xticks(x)
ax2.set_xticklabels(protocols)
ax2.set_xlabel('MAC协议', fontsize=12)

# 设置y轴标签
ax2.set_ylabel('延迟 (ms)', fontsize=12)
ax3.set_ylabel('实测端到端延迟 (ms)', fontsize=12, color='#EDC948')

# 添加图例
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax3.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# 添加网格线
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# 添加注释
plt.figtext(0.12, 0.01, '注：理论排队延迟基于Little定律计算，假设数据包处理时间为6ms。\n'
                        '实测端到端延迟包括传输延迟、传播延迟、处理延迟和排队延迟的综合结果。',
            fontsize=10, ha='left')

plt.tight_layout()
plt.subplots_adjust(bottom=0.1)

# 保存图片
plt.savefig('queue_analysis_chart.png', dpi=100, bbox_inches='tight')

# 显示图表
plt.show()