import matplotlib.pyplot as plt
import numpy as np

# 模拟数据
np.random.seed(42)
steps = np.arange(0, 90000, 1000)

# 创建类似于原图中的曲线
# 设置相同的起点（约为45）
start_point = 45

# GAT数据（红线）
with_gat_reward = np.zeros(len(steps))
with_gat_reward[0] = start_point  # 设置相同的起点
with_gat_reward[1:20] = np.linspace(47, 60, 19) + np.random.normal(scale=1, size=19)
with_gat_reward[20:30] = 60 + 2 * np.sin(np.arange(10) / 2) + np.random.normal(scale=1, size=10)
with_gat_reward[30:40] = np.linspace(60, 65, 10) + np.random.normal(scale=1, size=10)
with_gat_reward[40:50] = 65 + 5 * np.sin(np.arange(10) / 2) + np.random.normal(scale=1, size=10)

# 第一个峰值
with_gat_reward[30:35] = np.array([65, 68, 73, 67, 65])

with_gat_reward[50:60] = np.linspace(65, 60, 10) + np.random.normal(scale=1, size=10)
with_gat_reward[60:70] = 57 + 2 * np.sin(np.arange(10) / 2) + np.random.normal(scale=1, size=10)
with_gat_reward[70:80] = np.linspace(60, 69, 10) + np.random.normal(scale=1, size=10)

# 第三个峰值区域
with_gat_reward[80:90] = np.array([68, 69, 70, 66, 63, 62, 63, 65, 63, 61])

# 第二个峰值
with_gat_reward[57:62] = np.array([60, 65, 73, 65, 60])

# MLP数据（蓝线）
without_gat_reward = np.zeros(len(steps))
without_gat_reward[0] = start_point  # 设置相同的起点
without_gat_reward[1:10] = np.linspace(47, 52, 9) + np.random.normal(scale=1, size=9)
without_gat_reward[10:20] = np.linspace(52, 48, 10) + np.random.normal(scale=1, size=10)
without_gat_reward[20:30] = 50 + 5 * np.sin(np.arange(10) / 1.5) + np.random.normal(scale=1, size=10)
without_gat_reward[30:40] = np.linspace(48, 55, 10) + np.random.normal(scale=1, size=10)
without_gat_reward[40:50] = np.linspace(55, 62, 10) + np.random.normal(scale=1, size=10)
without_gat_reward[50:60] = np.linspace(62, 50, 10) + np.random.normal(scale=1, size=10)
without_gat_reward[60:70] = 52 + 2 * np.sin(np.arange(10) / 2) + np.random.normal(scale=1, size=10)
without_gat_reward[70:80] = np.linspace(50, 45, 10) + np.random.normal(scale=1, size=10)
without_gat_reward[80:] = np.linspace(45, 55, 10) + np.random.normal(scale=1, size=10)

# 创建图表，设置为浅灰色背景
plt.figure(figsize=(10, 6), facecolor='white')
ax = plt.gca()

# 绘制曲线
plt.plot(steps, with_gat_reward, color='red', label='With_GAT')
plt.plot(steps, without_gat_reward, color='navy', label='Without_GAT')

# 设置标题和轴标签
plt.title('GAT vs MLP Learning Curves (Evaluated every 1000 steps)', fontsize=13)
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Average Reward', fontsize=12)

# 添加图例，放在右上角
plt.legend(loc='upper right')

# 设置轴的范围，使其与原图相似
plt.ylim(45, 75)
plt.xlim(0, 90000)

# 保存图片为白底的PNG
plt.savefig('gat_vs_mlp_learning_curves.png', bbox_inches='tight', facecolor='white')

plt.show()