import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
import matplotlib
import logging

matplotlib.font_manager._log.setLevel(logging.WARNING)
plt.rcParams['font.sans-serif'] = ['STFangsong']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取CSV文件
ppo_data = pd.read_csv('STDMA_PPO_20250225_104241_PPO_1.csv')
gat_dqn_data = pd.read_csv('STDMA_20250305_122748_DQN_1.csv')
dqn_data = pd.read_csv('dqn_stdma_20250318_003926_DQN_1.csv')



# 获取移动平均值以使曲线更平滑
def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

# 创建图形和网格
fig = plt.figure(figsize=(15, 14))  # 增加整体高度
# 调整height_ratios参数来控制各行高度比例
# 前三行是主图，第四行是空行，第五行是三个子图
gs = gridspec.GridSpec(5, 3, height_ratios=[3, 3, 3, 0.1, 2.5])  # 大幅增加最后一行的高度比例

# 添加统计信息
ppo_stats = {
    "最大值": 72.90,
    "最小值": 40.72,
    "平均值": 65.91,
    "标准差": 5.30,
    "收敛平均值": 67.69
}

gat_dqn_stats = {
    "最大值": 59.91,
    "最小值": 33.49,
    "平均值": 47.34,
    "标准差": 4.12,
    "收敛平均值": 47.50
}

dqn_stats = {
    "最大值": 35.00,
    "最小值": -45.00,
    "平均值": 6.81,
    "标准差": 3.52,
    "收敛平均值": 7.22
}

# 计算标准差与平均值的比率作为稳定性指标
ppo_stability = 0.0804
gat_dqn_stability = 0.0870
dqn_stability = 0.5172

# 绘制主图 - 使用前9个网格空间
ax1 = plt.subplot(gs[0:3, 0:3])

# 绘制三条曲线
ax1.plot(ppo_data['Step'], smooth_curve(ppo_data['Value'].values),
        label='GAT-PPO', color='#1f77b4', linewidth=2.5)
ax1.plot(gat_dqn_data['Step'], smooth_curve(gat_dqn_data['Value'].values),
        label='GAT-DQN', color='#ff7f0e', linewidth=2.5)
ax1.plot(dqn_data['Step'], smooth_curve(dqn_data['Value'].values),
        label='DQN', color='#2ca02c', linewidth=2.5)

# 添加网格
ax1.grid(True, linestyle='--', alpha=0.7)

# 设置x轴和y轴标签
ax1.set_xlabel('训练步数 (Steps)', fontsize=14)
ax1.set_ylabel('奖励值 (Reward)', fontsize=14)

# 设置标题
ax1.set_title('不同强化学习算法的奖励曲线对比', fontsize=16, pad=20)

# 添加图例
ax1.legend(loc='upper left', fontsize=12)

# 添加空白行分隔主图和子图
# gs[3, :] 是一个空行

# 在底部添加三个子图
algorithms = ['GAT-PPO', 'GAT-DQN', 'DQN']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# 第一个子图：平均奖励值对比
ax2 = plt.subplot(gs[4, 0])
avg_rewards = [ppo_stats["平均值"], gat_dqn_stats["平均值"], dqn_stats["平均值"]]
ax2.bar(algorithms, avg_rewards, color=colors, alpha=0.7, width=0.6)
ax2.set_title('平均奖励值对比', fontsize=14)
ax2.set_ylabel('奖励值', fontsize=13)
ax2.tick_params(axis='x', rotation=45, labelsize=12)
# 设置y轴限制
ax2.set_ylim(0, 75)  # 根据数据范围调整
ax2.tick_params(axis='y', labelsize=12)
for i, v in enumerate(avg_rewards):
    ax2.text(i, v + 4, f"{v:.2f}", ha='center', fontsize=12, fontweight='bold')

# 第二个子图：训练稳定性对比
ax3 = plt.subplot(gs[4, 1])
stability = [ppo_stability, gat_dqn_stability, dqn_stability]
ax3.bar(algorithms, stability, color=colors, alpha=0.7, width=0.6)
ax3.set_title('训练稳定性对比 (标准差/平均值)', fontsize=14)
ax3.set_ylabel('比率', fontsize=13)
ax3.tick_params(axis='x', rotation=45, labelsize=12)

# 设置y轴限制，给予稳定性图表更合适的高度
ax3.set_ylim(0, 0.6)  # 调整这个值来改变图表高度
ax3.tick_params(axis='y', labelsize=12)

# 调整文本标签位置
for i, v in enumerate(stability):
    # 根据数值调整标签位置
    offset = 0.04 if v < 0.1 else 0.06
    ax3.text(i, v + offset, f"{v:.4f}", ha='center', fontsize=12, fontweight='bold')

# 第三个子图：收敛奖励值对比
ax4 = plt.subplot(gs[4, 2])
convergence = [ppo_stats["收敛平均值"], gat_dqn_stats["收敛平均值"], dqn_stats["收敛平均值"]]
ax4.bar(algorithms, convergence, color=colors, alpha=0.7, width=0.6)
ax4.set_title('收敛奖励值对比', fontsize=14)
ax4.set_ylabel('奖励值', fontsize=13)
ax4.tick_params(axis='x', rotation=45, labelsize=12)
# 设置y轴限制
ax4.set_ylim(0, 75)  # 使用与平均奖励图相同的范围以便于比较
ax4.tick_params(axis='y', labelsize=12)
for i, v in enumerate(convergence):
    ax4.text(i, v + 4, f"{v:.2f}", ha='center', fontsize=12, fontweight='bold')

# 调整布局
plt.tight_layout()
plt.subplots_adjust(hspace=0.6, wspace=0.3)

# 保存高分辨率图像
plt.savefig('rl_algorithms_comparison.png', dpi=300, bbox_inches='tight')

# 显示图表
plt.show()