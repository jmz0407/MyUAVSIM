import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['STFangsong']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 数据准备
methods = ['F-TDMA', 'RA-STDMA', 'DGC-STDMA', 'CO-STDMA', 'RL-STDMA']
priorities = ['高优先级', '中优先级', '低优先级']

# QoS满足率数据
qos_data = np.array([
    [65.3, 52.8, 38.7],  # F-TDMA
    [74.2, 61.5, 45.3],  # RA-STDMA
    [83.5, 72.6, 56.4],  # DGC-STDMA
    [87.3, 76.9, 61.2],  # CO-STDMA
    [94.5, 85.3, 76.3]  # RL-STDMA
])

# 柱状图设置
bar_width = 0.15
index = np.arange(len(methods))

plt.figure(figsize=(12, 7))

# 绘制分组柱状图
for i, priority in enumerate(priorities):
    offset = (i - 1) * bar_width
    bars = plt.bar(index + offset, qos_data[:, i], bar_width,
                   label=priority,
                   color=sns.color_palette('viridis', 3)[i])

    # 在柱状图上添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

plt.title('不同优先级业务的QoS满足率', fontsize=14)
plt.xlabel('调度方法', fontsize=12)
plt.ylabel('QoS满足率 (%)', fontsize=12)
plt.xticks(index, methods)
plt.legend(loc='upper left', fontsize=10)
plt.ylim(0, 100)
plt.grid(True, linestyle='--', alpha=0.7, axis='y')

# 添加水平参考线表示目标QoS水平
plt.axhline(y=90, color='r', linestyle='--', alpha=0.5, label='高优先级目标')
plt.axhline(y=75, color='g', linestyle='--', alpha=0.5, label='中优先级目标')
plt.axhline(y=60, color='b', linestyle='--', alpha=0.5, label='低优先级目标')

plt.tight_layout()
plt.savefig('图4-5_不同优先级业务的QoS满足率.png', dpi=300, bbox_inches='tight')
plt.show()