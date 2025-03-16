import matplotlib.pyplot as plt
import numpy as np
import matplotlib
plt.rcParams['font.sans-serif'] = ['STFangSong']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 15  # 设置字体大小

# 无人机数量
nodes = [10, 15, 20, 25, 30]

# 各协议的吞吐量数据
mpdsr_throughput = [6230.38, 6002.31, 5874.21, 5159.19, 4942.23]  # 替换为你的实际数据
mp_olsr_throughput = [7099.92, 7031.02, 6694.05, 5920.26, 5631.99]  # 替换为你的实际数据
amlbr_throughput = [7973.00, 7594.24,  6955.61, 6849.67, 7099.28]  # 替换为你的实际数据
opar_throughput = [5438.62, 5368.05, 5127.72, 4314.17, 4543.98]  # 替换为你的实际数据

# 创建绘图
plt.figure(figsize=(10, 6))

# 绘制每个协议的折线
plt.plot(nodes, mpdsr_throughput, 's-', label='MP-DSR', linewidth=2)
plt.plot(nodes, mp_olsr_throughput, '^-', label='MP-OLSR', linewidth=2)
plt.plot(nodes, amlbr_throughput, 'D-', label='AMLBR', linewidth=2)
plt.plot(nodes, opar_throughput, 'x-', label='OLSR', linewidth=2)

# 设置图表标题和标签
plt.title('无人机网络路由协议的吞吐量对比', fontsize=14)
plt.xlabel('无人机数量', fontsize=12)
plt.ylabel('吞吐量 (kb/s)', fontsize=12)

# 设置x轴刻度
plt.xticks(nodes)

# 设置y轴范围，可以根据实际数据调整
plt.ylim(2500, 9000)

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.7)

# 添加图例
plt.legend(loc='best', fontsize=10)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('throughput_comparison.png', dpi=300)

# 显示图表
plt.show()