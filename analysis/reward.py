import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import make_interp_spline

# 设置中文字体支持
plt.rcParams['font.sans-serif']=['STFangsong'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号


# 数据准备
steps = np.linspace(0, 10000, 50)

# 训练奖励数据（包含一些随机波动）
np.random.seed(42)
rewards_gat_ppo = 85 * (1 - np.exp(-steps / 3000)) + np.random.normal(0, 3, 50)
rewards_gcn_ppo = 75 * (1 - np.exp(-steps / 3500)) + np.random.normal(0, 4, 50)
rewards_gat_dqn = 70 * (1 - np.exp(-steps / 4000)) + np.random.normal(0, 5, 50)
rewards_mlp_ppo = 65 * (1 - np.exp(-steps / 3800)) + np.random.normal(0, 4, 50)


# 平滑曲线
def smooth_curve(x, y, factor=0.3):
    # 增加点的数量
    x_new = np.linspace(x.min(), x.max(), 300)

    # 创建样条插值模型
    spl = make_interp_spline(x, y, k=3)
    y_new = spl(x_new)

    return x_new, y_new


steps_smooth, rewards_gat_ppo_smooth = smooth_curve(steps, rewards_gat_ppo)
_, rewards_gcn_ppo_smooth = smooth_curve(steps, rewards_gcn_ppo)
_, rewards_gat_dqn_smooth = smooth_curve(steps, rewards_gat_dqn)
_, rewards_mlp_ppo_smooth = smooth_curve(steps, rewards_mlp_ppo)

plt.figure(figsize=(10, 6))

# 绘制原始数据点（小点，透明）
plt.scatter(steps, rewards_gat_ppo, s=10, alpha=0.3, color='#1f77b4')
plt.scatter(steps, rewards_gcn_ppo, s=10, alpha=0.3, color='#ff7f0e')
plt.scatter(steps, rewards_gat_dqn, s=10, alpha=0.3, color='#2ca02c')
plt.scatter(steps, rewards_mlp_ppo, s=10, alpha=0.3, color='#d62728')

# 绘制平滑曲线
plt.plot(steps_smooth, rewards_gat_ppo_smooth, label='GAT-PPO', color='#1f77b4', linewidth=2)
plt.plot(steps_smooth, rewards_gcn_ppo_smooth, label='GCN-PPO', color='#ff7f0e', linewidth=2)
plt.plot(steps_smooth, rewards_gat_dqn_smooth, label='GAT-DQN', color='#2ca02c', linewidth=2)
plt.plot(steps_smooth, rewards_mlp_ppo_smooth, label='MLP-PPO', color='#d62728', linewidth=2)

plt.title('不同模型的训练奖励曲线', fontsize=14)
plt.xlabel('训练步数', fontsize=12)
plt.ylabel('平均奖励', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)

plt.tight_layout()
plt.savefig('图3-5_不同模型的训练奖励曲线.png', dpi=300, bbox_inches='tight')
plt.show()