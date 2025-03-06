import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['STFangsong']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号


# 数据准备
mobility_speeds = [5, 10, 15, 20, 25, 30]
protocols = ['OLSR', 'MP-OLSR', 'MP-DSR', 'AMLBR']

# 数据包投递率数据(PDR %)
pdr_data = {
    'OLSR': [93.2, 89.4, 82.1, 76.8, 72.3, 66.5],
    'MP-OLSR': [95.0, 91.7, 85.8, 81.2, 76.9, 72.4],
    'MP-DSR': [95.7, 92.4, 87.2, 82.7, 78.6, 77.7],
    'AMLBR': [97.2, 94.8, 91.3, 88.6, 84.9, 84.3]
}

plt.figure(figsize=(10, 6))

colors = sns.color_palette('tab10', len(protocols))
markers = ['o', 's', '^', 'D', '*', 'p', 'X']

# 绘制PDR图
for i, protocol in enumerate(protocols):
    plt.plot(mobility_speeds, pdr_data[protocol], marker=markers[i],
             color=colors[i], linewidth=2, label=protocol)

plt.title('不同移动速度下的数据包投递率比较', fontsize=14)
plt.xlabel('移动速度 (m/s)', fontsize=12)
plt.ylabel('数据包投递率 (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(mobility_speeds)
plt.ylim(60, 100)
plt.legend(loc='lower left', fontsize=10)

# 添加高亮区域标注高移动性场景
plt.axvspan(20, 30, alpha=0.15, color='gray', label='高移动性场景')
plt.text(25, 62, '高移动性场景', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('图5-4_不同协议的数据包投递率比较.png', dpi=300, bbox_inches='tight')
plt.show()