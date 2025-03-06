import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['STFangsong']
plt.rcParams['axes.unicode_minus'] = False

# 创建雷达图投影
def radar_factory(num_vars, frame='circle'):
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = 'radar'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')
            self.set_theta_direction(-1)  # 顺时针方向
            self.set_thetagrids(np.degrees(theta))  # 设置角度刻度

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(np.concatenate((args[0], [args[0][0]])),
                                np.concatenate((args[1], [args[1][0]])),
                                closed=closed, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            if x.size != 0:
                line.set_data(np.concatenate((x, [x[0]])),
                              np.concatenate((y, [y[0]])))

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            spine = Spine(self, 'circle', Path.unit_circle())
            spine.set_transform(Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta

# 数据定义
protocols = ['OLSR', 'MP-OLSR', 'MP-DSR', 'AMLBR']
metrics = ['吞吐量', '延迟', 'PDR', '路由开销', '负载均衡', '能量效率']

# 归一化数据 (0-1)，值越高越好
data = np.array([
    [0.48, 0.42, 0.71, 0.68, 0.35, 0.38],  # OLSR
    [0.65, 0.52, 0.78, 0.52, 0.51, 0.53],  # MP-OLSR
    [0.69, 0.56, 0.82, 0.55, 0.59, 0.61],  # MP-DSR
    [0.88, 0.78, 0.91, 0.62, 0.77, 0.82]   # AMLBR
])

# 反转“延迟”和“路由开销”数据，使数值越高越好
data[:, 1] = 1 - data[:, 1]  # 反转延迟
data[:, 3] = 1 - data[:, 3]  # 反转路由开销

# 创建雷达图
fig = plt.figure(figsize=(10, 8))
theta = radar_factory(len(metrics), frame='circle')  # 这里可以选择 'circle' 或 'polygon'

ax = fig.add_subplot(111, projection='radar')

# 颜色列表
colors = ['#d62728', '#9467bd', '#8c564b', '#e377c2']

# 绘制数据
for i, protocol in enumerate(protocols):
    ax.plot(theta, data[i], color=colors[i], linewidth=2, label=protocol)
    ax.fill(theta, data[i], alpha=0.1, color=colors[i])

# 设置图表属性
ax.set_varlabels(metrics)
plt.legend(protocols, loc=(1.05, 0.95), labelspacing=0.1, fontsize=10)
plt.title('不同协议的综合性能比较', fontsize=14, y=1.1)

# 显示图表
plt.tight_layout()
plt.savefig('不同协议的综合性能比较.png', dpi=300, bbox_inches='tight')
plt.show()