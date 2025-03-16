import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm


from matplotlib import pyplot as plt
# 设置中文字体（以PingFang SC为例）
plt.rcParams['font.sans-serif']=['STFangsong'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

# 数据：不同节点数量下各协议的吞吐量
node_counts = [10, 15, 20, 25, 30]

# 各协议的吞吐量数据(kbps)
amlbr_throughput = [8300, 7800, 7400, 6800, 6050]
mp_olsr_throughput = [9500, 8900, 8250, 7600, 6700]
mp_dsr_throughput = [8100, 7500, 6800, 6100, 5250]
olsr_throughput = [5450, 5250,5015, 4778, 4338]

# 创建图形和坐标轴
plt.figure(figsize=(10, 6))

# 绘制折线图
plt.plot(node_counts, amlbr_throughput, 'o-', linewidth=2.5, markersize=8, label='AMLBR', color='#3498db')
plt.plot(node_counts, mp_olsr_throughput, 's-', linewidth=2.5, markersize=8, label='MP-OLSR', color='#2ecc71')
plt.plot(node_counts, mp_dsr_throughput, '^-', linewidth=2.5, markersize=8, label='MP-DSR', color='#e74c3c')
plt.plot(node_counts, olsr_throughput, 'D-', linewidth=2.5, markersize=8, label='OLSR', color='#f39c12')


# 添加标签和标题
plt.xlabel('无人机节点数量', fontsize=15, fontweight='bold')
plt.ylabel('吞吐量 (kbps)', fontsize=15, fontweight='bold')
plt.title('无人机网络路由协议吞吐量对比', fontsize=16, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper right', fontsize=15)

# 设置X轴刻度
plt.xticks(node_counts)

# 设置Y轴范围
plt.ylim(4000, 10500)


plt.tight_layout()
plt.savefig('throughput_comparison_line.png', dpi=300, bbox_inches='tight')
plt.show()