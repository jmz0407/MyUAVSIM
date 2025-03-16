import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import logging

matplotlib.font_manager._log.setLevel(logging.WARNING)
plt.rcParams['font.sans-serif'] = ['STFangsong']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 设置全局字体大小
plt.rcParams.update({'font.size': 28})

# 创建一个无向图
G = nx.Graph()

# 添加节点
nodes = range(0, 10)
G.add_nodes_from(nodes)

# 添加边和权重
edges_with_weights = [
    (0, 1, 0.17),
    (0, 2, 0.65),
    (0, 9, 0.36),
    (1, 3, 0.01),
    (1, 4, 0.06),
    (1, 5, 0.19),
    (1, 6, 0.26),
    (1, 7, 0.8),
    (1, 8, 0.03),
    (2, 3, 0.13),
    (2, 4, 0.42),
    (2, 6, 0.53),
    (3, 7, 0.17),
    (3, 8, 0.28),
    (4, 9, 0.35),
    (5, 7, 0.32),
    (6, 7, 0.42),
]

for u, v, w in edges_with_weights:
    G.add_edge(u, v, weight=w)

# 设置节点位置 - 使用spring_layout可以自动调整，或者手动指定位置匹配原图
pos = {
    0: (0, 0.5),
    1: (1, 0.5),
    2: (-1, -0.5),
    3: (-0.5, -1.5),
    4: (-1, -2.5),
    5: (2, 0),
    6: (0, -1.5),
    7: (1.5, -1.5),
    8: (3, -1),
    9: (0, 1.5)
}

plt.figure(figsize=(12, 10))

# 绘制节点
nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue', alpha=0.8)

# 根据权重绘制边 - 粗线表示高权重链接，细线表示低权重链接
# 将权重分为两类
threshold = 0.3  # 可以调整阈值以区分高权重和低权重
high_weight_edges = [(u, v) for u, v, w in edges_with_weights if w >= threshold]
low_weight_edges = [(u, v) for u, v, w in edges_with_weights if w < threshold]

# 绘制高权重边 - 粗线
nx.draw_networkx_edges(G, pos, edgelist=high_weight_edges, width=2.5, alpha=0.7, edge_color='black')

# 绘制低权重边 - 细线
nx.draw_networkx_edges(G, pos, edgelist=low_weight_edges, width=1.0, alpha=0.5, edge_color='gray')

# 添加节点标签
nx.draw_networkx_labels(G, pos, font_size=14, font_family='sans-serif')

# 添加边标签（权重）
edge_labels = {(u, v): f"{w:.2f}" for u, v, w in edges_with_weights}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=18)

# 添加图例
high_weight_patch = mpatches.Patch(color='black', label='高权重链接 (≥0.3)')
low_weight_patch = mpatches.Patch(color='gray', label='低权重链接 (<0.3)')
node_patch = mpatches.Patch(color='lightblue', label='节点')

plt.legend(handles=[high_weight_patch, low_weight_patch, node_patch],
           loc='upper right', fontsize=12)

# 调整图的布局
plt.axis('off')
plt.tight_layout()

# 显示图
plt.savefig('network_graph.png', dpi=300, bbox_inches='tight')
plt.show()