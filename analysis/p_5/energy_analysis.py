import matplotlib.pyplot as plt
import numpy as np

import matplotlib
plt.rcParams['font.sans-serif'] = ['STFangSong']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 16  # 设置字体大小


# 业务负载(kbps)
loads = np.array([4000, 5000, 6000, 7000, 8000, 9000, 10000])

# 各算法在不同负载下的能耗数据(J)
amlbr = np.array([4200, 5000, 6200, 7600, 9000, 10200, 11000])
olsr = np.array([4500, 6200, 8000, 10000, 12000, 13500, 15000])
mp_olsr = np.array([4300, 5600, 7000, 8600, 10200, 11800, 13000])
mp_dsr = np.array([4400, 5800, 7200, 8800, 10500, 12000, 13000])

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制折线图
plt.plot(loads, amlbr, 'o-', linewidth=2, markersize=8, label='AMLBR')
plt.plot(loads, olsr, 's-', linewidth=2, markersize=8, label='OLSR')
plt.plot(loads, mp_olsr, '^-', linewidth=2, markersize=8, label='MP-OLSR')
plt.plot(loads, mp_dsr, 'D-', linewidth=2, markersize=8, label='MP-DSR')

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.7)

# 设置坐标轴标签和标题
plt.xlabel('业务负载 (kbps)', fontsize=14)
plt.ylabel('能耗 (J)', fontsize=14)
plt.title('不同业务负载下的能耗对比', fontsize=16)

# 设置x轴刻度
plt.xticks(loads)

# 添加图例
plt.legend(loc='best', fontsize=12)

# 优化布局
plt.tight_layout()
# 保存图片
plt.savefig('图5-9_能耗比较.png', dpi=300, bbox_inches='tight')
# 显示图形
plt.show()