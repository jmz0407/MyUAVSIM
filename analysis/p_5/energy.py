import matplotlib.pyplot as plt
import numpy as np
import matplotlib
plt.rcParams['font.sans-serif'] = ['STFangSong']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 12  # 设置字体大小

# # 设置图表风格
# plt.style.use('seaborn-v0_8-whitegrid')

# 图表数据
# 不同负载下的系统吞吐量数据
load = np.array([10, 15, 20, 25, 30, 35])
throughput_joint = np.array([10.2, 13.5, 15.8, 18.7, 19.9, 20.3])
throughput_separate = np.array([8.9, 11.3, 13.2, 15.3, 15.8, 16.1])
throughput_traditional = np.array([6.5, 7.8, 8.5, 9.2, 9.0, 8.7])

# 不同网络规模下的端到端延迟数据
nodes = np.array([5, 10, 15, 20, 25, 30])
delay_joint = np.array([42.3, 55.6, 68.4, 78.5, 91.2, 103.8])
delay_separate = np.array([56.8, 74.2, 89.5, 103.7, 118.4, 134.5])
delay_traditional = np.array([87.3, 112.5, 135.7, 156.2, 178.3, 202.7])

# 不同移动速度下的数据包递交率数据
speed = np.array([5, 10, 15, 20, 25, 30])
pdr_joint = np.array([99.2, 98.7, 98.1, 97.3, 96.2, 94.8])
pdr_separate = np.array([97.5, 96.2, 94.7, 92.8, 90.4, 87.9])
pdr_traditional = np.array([94.8, 91.5, 88.9, 85.1, 81.7, 76.3])

# 不同节点密度下的资源利用率数据
density = np.array([5, 10, 15, 20, 25, 30])
utilization_joint = np.array([76.2, 79.5, 81.8, 83.5, 84.7, 85.2])
utilization_separate = np.array([68.3, 71.1, 72.4, 74.2, 75.3, 76.1])
utilization_traditional = np.array([57.4, 59.2, 60.3, 61.8, 62.5, 63.0])

# 创建子图
fig, axs = plt.subplots(2, 2, figsize=(15, 12))
fig.tight_layout(pad=4.0)  # 调整子图间距

# 图5-6：不同负载条件下的系统吞吐量对比
axs[0, 0].plot(load, throughput_joint, 'o-', linewidth=2, markersize=8, label='联合优化 (GAT-PPO-Joint)')
axs[0, 0].plot(load, throughput_separate, 's-', linewidth=2, markersize=8, label='分离优化 (GAT-PPO-MAC + GAT-PPO-NET)')
axs[0, 0].plot(load, throughput_traditional, '^-', linewidth=2, markersize=8, label='传统方法 (STDMA + OLSR)')
axs[0, 0].set_xlabel('网络负载 (每节点包/秒)')
axs[0, 0].set_ylabel('系统吞吐量 (Mbps)')
axs[0, 0].set_title('图5-6 不同负载条件下的系统吞吐量对比')
axs[0, 0].legend()
axs[0, 0].grid(True)

# 图5-7：不同网络规模下的端到端延迟对比
axs[0, 1].plot(nodes, delay_joint, 'o-', linewidth=2, markersize=8, label='联合优化 (GAT-PPO-Joint)')
axs[0, 1].plot(nodes, delay_separate, 's-', linewidth=2, markersize=8, label='分离优化 (GAT-PPO-MAC + GAT-PPO-NET)')
axs[0, 1].plot(nodes, delay_traditional, '^-', linewidth=2, markersize=8, label='传统方法 (STDMA + OLSR)')
axs[0, 1].set_xlabel('网络节点数')
axs[0, 1].set_ylabel('平均端到端延迟 (ms)')
axs[0, 1].set_title('图5-7 不同网络规模下的端到端延迟对比')
axs[0, 1].legend()
axs[0, 1].grid(True)

# 图5-8：不同移动速度下的数据包递交率对比
axs[1, 0].plot(speed, pdr_joint, 'o-', linewidth=2, markersize=8, label='联合优化 (GAT-PPO-Joint)')
axs[1, 0].plot(speed, pdr_separate, 's-', linewidth=2, markersize=8, label='分离优化 (GAT-PPO-MAC + GAT-PPO-NET)')
axs[1, 0].plot(speed, pdr_traditional, '^-', linewidth=2, markersize=8, label='传统方法 (STDMA + OLSR)')
axs[1, 0].set_xlabel('无人机最大移动速度 (m/s)')
axs[1, 0].set_ylabel('数据包递交率 (%)')
axs[1, 0].set_title('图5-8 不同移动速度下的数据包递交率对比')
axs[1, 0].legend()
axs[1, 0].grid(True)

# 图5-9：不同节点密度下的资源利用率对比
axs[1, 1].plot(density, utilization_joint, 'o-', linewidth=2, markersize=8, label='联合优化 (GAT-PPO-Joint)')
axs[1, 1].plot(density, utilization_separate, 's-', linewidth=2, markersize=8, label='分离优化 (GAT-PPO-MAC + GAT-PPO-NET)')
axs[1, 1].plot(density, utilization_traditional, '^-', linewidth=2, markersize=8, label='传统方法 (STDMA + OLSR)')
axs[1, 1].set_xlabel('节点密度 (节点/100m²)')
axs[1, 1].set_ylabel('资源利用率 (%)')
axs[1, 1].set_title('图5-9 不同节点密度下的资源利用率对比')
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.savefig('simulation_charts.png', dpi=300, bbox_inches='tight')
plt.show()

# 额外创建单个图表，更适合论文排版

# 图5-6：不同负载条件下的系统吞吐量对比
plt.figure(figsize=(8, 6))
plt.plot(load, throughput_joint, 'o-', linewidth=2, markersize=8, label='联合优化 (GAT-PPO-Joint)')
plt.plot(load, throughput_separate, 's-', linewidth=2, markersize=8, label='分离优化 (GAT-PPO-MAC + GAT-PPO-NET)')
plt.plot(load, throughput_traditional, '^-', linewidth=2, markersize=8, label='传统方法 (STDMA + OLSR)')
plt.xlabel('网络负载 (每节点包/秒)')
plt.ylabel('系统吞吐量 (Mbps)')
plt.title('图5-6 不同负载条件下的系统吞吐量对比')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('throughput_comparison.png', dpi=300, bbox_inches='tight')

# 图5-7：不同网络规模下的端到端延迟对比
plt.figure(figsize=(8, 6))
plt.plot(nodes, delay_joint, 'o-', linewidth=2, markersize=8, label='联合优化 (GAT-PPO-Joint)')
plt.plot(nodes, delay_separate, 's-', linewidth=2, markersize=8, label='分离优化 (GAT-PPO-MAC + GAT-PPO-NET)')
plt.plot(nodes, delay_traditional, '^-', linewidth=2, markersize=8, label='传统方法 (STDMA + OLSR)')
plt.xlabel('网络节点数')
plt.ylabel('平均端到端延迟 (ms)')
plt.title('图5-7 不同网络规模下的端到端延迟对比')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('delay_comparison.png', dpi=300, bbox_inches='tight')

# 图5-8：不同移动速度下的数据包递交率对比
plt.figure(figsize=(8, 6))
plt.plot(speed, pdr_joint, 'o-', linewidth=2, markersize=8, label='联合优化 (GAT-PPO-Joint)')
plt.plot(speed, pdr_separate, 's-', linewidth=2, markersize=8, label='分离优化 (GAT-PPO-MAC + GAT-PPO-NET)')
plt.plot(speed, pdr_traditional, '^-', linewidth=2, markersize=8, label='传统方法 (STDMA + OLSR)')
plt.xlabel('无人机最大移动速度 (m/s)')
plt.ylabel('数据包递交率 (%)')
plt.title('图5-8 不同移动速度下的数据包递交率对比')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('pdr_comparison.png', dpi=300, bbox_inches='tight')

# 图5-9：不同节点密度下的资源利用率对比
plt.figure(figsize=(8, 6))
plt.plot(density, utilization_joint, 'o-', linewidth=2, markersize=8, label='联合优化 (GAT-PPO-Joint)')
plt.plot(density, utilization_separate, 's-', linewidth=2, markersize=8, label='分离优化 (GAT-PPO-MAC + GAT-PPO-NET)')
plt.plot(density, utilization_traditional, '^-', linewidth=2, markersize=8, label='传统方法 (STDMA + OLSR)')
plt.xlabel('节点密度 (节点/100m²)')
plt.ylabel('资源利用率 (%)')
plt.title('图5-9 不同节点密度下的资源利用率对比')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('utilization_comparison.png', dpi=300, bbox_inches='tight')

# 能量效率对比图
network_load = np.array([10, 15, 20, 25, 30, 35])
energy_joint = np.array([21.5, 24.3, 26.8, 28.3, 29.1, 29.5])
energy_separate = np.array([18.2, 20.5, 22.1, 23.6, 24.0, 24.3])
energy_traditional = np.array([14.1, 15.6, 16.4, 17.1, 17.3, 17.2])

plt.figure(figsize=(8, 6))
plt.plot(network_load, energy_joint, 'o-', linewidth=2, markersize=8, label='联合优化 (GAT-PPO-Joint)')
plt.plot(network_load, energy_separate, 's-', linewidth=2, markersize=8, label='分离优化 (GAT-PPO-MAC + GAT-PPO-NET)')
plt.plot(network_load, energy_traditional, '^-', linewidth=2, markersize=8, label='传统方法 (STDMA + OLSR)')
plt.xlabel('网络负载 (每节点包/秒)')
plt.ylabel('能量效率 (Kbits/J)')
plt.title('图5-10 不同负载条件下的能量效率对比')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('energy_efficiency_comparison.png', dpi=300, bbox_inches='tight')