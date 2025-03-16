import matplotlib.pyplot as plt
import numpy as np

import matplotlib
plt.rcParams['font.sans-serif'] = ['STFangSong']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 16  # 设置字体大小

# 时间点(分钟)
time_points = np.array([0, 5, 10, 15, 20, 25, 30])

# 各算法在不同时间点的吞吐量数据(kb/s)
# 基于论文分析内容构建的数据
amlbr = np.array([8300, 8100, 7900, 7600, 7200, 6750, 6300])
mp_olsr = np.array([9500, 9400, 9300, 7800, 6800, 5500, 4900])
mp_dsr = np.array([8200, 8000, 7500, 6800, 6000, 5400, 4800])

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制折线图
plt.plot(time_points, amlbr, 'o-', linewidth=2, markersize=8, label='AMLBR')
plt.plot(time_points, mp_olsr, '^-', linewidth=2, markersize=8, label='MP-OLSR')
plt.plot(time_points, mp_dsr, 'D-', linewidth=2, markersize=8, label='MP-DSR')

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.7)

# 设置坐标轴标签和标题
plt.xlabel('时间 (分钟)', fontsize=14)
plt.ylabel('吞吐量 (kbps)', fontsize=14)
plt.title('长时间测试下的吞吐量变化', fontsize=16)

# 设置x轴刻度
plt.xticks(time_points)

# 设置y轴范围
plt.ylim(3000, 10000)

# 添加图例
plt.legend(loc='best', fontsize=12)

# 优化布局
plt.tight_layout()
# 保存图片
plt.savefig('长时间吞吐量对比分析.png', dpi=300, bbox_inches='tight')
# 显示图形
plt.show()