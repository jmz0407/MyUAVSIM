import matplotlib.pyplot as plt
import numpy as np


from matplotlib import pyplot as plt
# 设置中文字体（以PingFang SC为例）
plt.rcParams['font.sans-serif']=['STFangsong'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
# 节点数量
nodes = np.array([10, 15, 20, 25, 30])

# 各算法在不同节点数量下的时隙复用率数据
# 根据描述中的信息，构建合理的数据趋势
amlbr = np.array([1.80, 1.95, 2.05, 2.15, 2.25])
olsr = np.array([1.43, 1.51, 1.56, 1.60, 1.65])
mp_olsr = np.array([1.60, 1.68, 1.73, 1.78, 1.82])
mp_dsr = np.array([1.58, 1.63, 1.68, 1.72, 1.75])

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制折线图
plt.plot(nodes, amlbr, 'o-', linewidth=2, markersize=8, label='AMLBR')
plt.plot(nodes, olsr, 's-', linewidth=2, markersize=8, label='OLSR')
plt.plot(nodes, mp_olsr, '^-', linewidth=2, markersize=8, label='MP-OLSR')
plt.plot(nodes, mp_dsr, 'D-', linewidth=2, markersize=8, label='MP-DSR')

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.7)

# 设置坐标轴标签和标题
plt.xlabel('节点数量', fontsize=14)
plt.ylabel('时隙复用率', fontsize=14)
plt.title('不同路由算法的时隙复用率比较', fontsize=16)

# 设置x轴刻度
plt.xticks(nodes)

# 设置y轴范围
plt.ylim(1.0, 2.5)

# 添加图例
plt.legend(loc='best', fontsize=12)

# 优化布局
plt.tight_layout()
# 保存图片
plt.savefig('图5-7_时隙复用率比较.png', dpi=300, bbox_inches='tight')
# 显示图形
plt.show()

