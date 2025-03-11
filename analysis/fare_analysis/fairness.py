import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import random


# 定义Jain公平性指数计算函数
def calculate_fairness_index(allocations, demands=None):
    """
    计算Jain公平性指数

    参数:
        allocations: 各节点获得的时隙数列表
        demands: 各节点的需求时隙数列表(可选)

    返回:
        fairness_index: Jain公平性指数 [1/n, 1]
    """
    if demands is None:
        # 不考虑需求差异时的公平性指数
        x = np.array(allocations)
    else:
        # 考虑需求差异时的归一化公平性指数
        x = np.array(allocations) / np.array(demands)
        # 处理需求为0的情况
        x = np.where(np.array(demands) > 0, x, 1.0)

    n = len(x)
    return (np.sum(x) ** 2) / (n * np.sum(x ** 2))


# 创建随机无人机网络拓扑
def create_network_topology(n_drones, comm_range, width=1000, height=1000, altitude=100):
    """
    创建随机无人机网络拓扑

    参数:
        n_drones: 无人机数量
        comm_range: 通信范围
        width, height, altitude: 区域尺寸参数

    返回:
        G: NetworkX图对象
        positions: 节点位置字典
    """
    # 创建图
    G = nx.Graph()

    # 随机生成无人机位置
    positions = {}
    for i in range(n_drones):
        x = random.uniform(0, width)
        y = random.uniform(0, height)
        z = random.uniform(0, altitude)
        positions[i] = (x, y, z)
        G.add_node(i, pos=(x, y, z))

    # 基于通信范围添加边
    for i in range(n_drones):
        for j in range(i + 1, n_drones):
            pos_i = positions[i]
            pos_j = positions[j]
            # 计算3D欧氏距离
            distance = np.sqrt((pos_i[0] - pos_j[0]) ** 2 + (pos_i[1] - pos_j[1]) ** 2 + (pos_i[2] - pos_j[2]) ** 2)
            if distance <= comm_range:
                G.add_edge(i, j, weight=distance)

    return G, positions


# 构建干扰图
def build_interference_graph(G, interference_range):
    """
    构建干扰图

    参数:
        G: 原始网络拓扑图
        interference_range: 干扰范围

    返回:
        IG: 干扰图
    """
    IG = nx.Graph()
    IG.add_nodes_from(G.nodes())

    positions = nx.get_node_attributes(G, 'pos')

    # 添加干扰边
    for i in G.nodes():
        for j in G.nodes():
            if i != j:
                pos_i = positions[i]
                pos_j = positions[j]
                # 计算3D欧氏距离
                distance = np.sqrt((pos_i[0] - pos_j[0]) ** 2 + (pos_i[1] - pos_j[1]) ** 2 + (pos_i[2] - pos_j[2]) ** 2)
                if distance <= interference_range:
                    IG.add_edge(i, j)

    return IG


# 传统STDMA算法实现
def traditional_stdma(interference_graph, num_slots, demands=None):
    """
    传统STDMA时隙分配算法

    参数:
        interference_graph: 干扰图
        num_slots: 总时隙数
        demands: 节点需求列表(可选)

    返回:
        slots_allocation: 时隙分配结果 {slot_id: [node_list]}
        node_allocations: 每个节点获得的时隙数
    """
    slots_allocation = {i: [] for i in range(num_slots)}
    node_allocations = {node: 0 for node in interference_graph.nodes()}

    # 按节点度排序 (度越大的节点干扰越多，优先分配)
    nodes_by_degree = sorted(interference_graph.nodes(),
                             key=lambda n: interference_graph.degree(n),
                             reverse=True)

    for node in nodes_by_degree:
        # 为节点找到可用的时隙
        for slot in range(num_slots):
            slot_nodes = slots_allocation[slot]
            # 检查该时隙是否与当前节点冲突
            conflict = False
            for slot_node in slot_nodes:
                if slot_node in interference_graph.neighbors(node) or slot_node == node:
                    conflict = True
                    break

            if not conflict:
                slots_allocation[slot].append(node)
                node_allocations[node] += 1

                # 如果节点的需求已满足，则停止分配
                if demands and node_allocations[node] >= demands[node]:
                    break

    return slots_allocation, node_allocations


# 公平性优化的STDMA算法实现
def fair_stdma(interference_graph, num_slots, demands=None):
    """
    基于公平性优化的STDMA时隙分配算法

    参数:
        interference_graph: 干扰图
        num_slots: 总时隙数
        demands: 节点需求列表(可选)

    返回:
        slots_allocation: 时隙分配结果 {slot_id: [node_list]}
        node_allocations: 每个节点获得的时隙数
    """
    slots_allocation = {i: [] for i in range(num_slots)}
    node_allocations = {node: 0 for node in interference_graph.nodes()}

    # 初始化节点优先级
    if demands:
        node_priority = {node: demands[node] for node in interference_graph.nodes()}
    else:
        node_priority = {node: 1 for node in interference_graph.nodes()}

    # 循环分配，直到所有时隙分配完毕或者无法继续分配
    can_allocate = True
    while can_allocate:
        can_allocate = False
        # 按当前优先级排序
        nodes_by_priority = sorted(node_priority.keys(),
                                   key=lambda n: node_priority[n] / (node_allocations[n] + 0.1),
                                   reverse=True)

        for node in nodes_by_priority:
            if demands and node_allocations[node] >= demands[node]:
                continue  # 已满足需求

            # 为节点找到可用的时隙
            for slot in range(num_slots):
                slot_nodes = slots_allocation[slot]
                # 检查该时隙是否与当前节点冲突
                conflict = False
                for slot_node in slot_nodes:
                    if slot_node in interference_graph.neighbors(node) or slot_node == node:
                        conflict = True
                        break

                if not conflict:
                    slots_allocation[slot].append(node)
                    node_allocations[node] += 1
                    can_allocate = True
                    break  # 找到一个时隙后退出，给其他节点机会

        # 更新优先级，考虑已分配情况
        for node in node_priority:
            if demands:
                normalized_allocation = node_allocations[node] / max(demands[node], 1)
                node_priority[node] = demands[node] / (normalized_allocation + 0.1)
            else:
                node_priority[node] = 1 / (node_allocations[node] + 0.1)

    return slots_allocation, node_allocations


# 模拟GAT-PPO-STDMA的简化实现
def gat_ppo_stdma_simulation(interference_graph, num_slots, demands=None):
    """
    模拟GAT-PPO-STDMA的时隙分配结果 (简化版)

    注: 完整的GAT-PPO-STDMA实现需要训练好的GAT和PPO模型,
        这里使用简化版本模拟其特性
    """
    slots_allocation = {i: [] for i in range(num_slots)}
    node_allocations = {node: 0 for node in interference_graph.nodes()}

    # 计算节点中心度，模拟GAT学到的节点重要性
    centrality = nx.betweenness_centrality(interference_graph)

    # 模拟需求感知和拓扑感知的特性
    if demands:
        node_scores = {node: centrality[node] * demands[node] for node in interference_graph.nodes()}
    else:
        node_scores = centrality

    for iteration in range(3):  # 多次迭代优化，模拟PPO的策略改进
        # 按得分排序
        nodes_by_score = sorted(node_scores.keys(), key=lambda n: node_scores[n] / (node_allocations[node] + 0.1),
                                reverse=True)

        for node in nodes_by_score:
            # 对于每个时隙，计算分配该节点的"收益"
            slot_benefits = []
            for slot in range(num_slots):
                slot_nodes = slots_allocation[slot]
                # 检查冲突
                conflict = False
                for slot_node in slot_nodes:
                    if slot_node in interference_graph.neighbors(node) or slot_node == node:
                        conflict = True
                        break

                if conflict:
                    slot_benefits.append(-100)  # 冲突，大幅惩罚
                else:
                    # 计算该时隙的收益，考虑空间复用效率和公平性
                    spatial_reuse = len(slot_nodes) + 1  # 模拟空间复用奖励
                    fairness_reward = 1.0 / (node_allocations[node] + 1)  # 模拟公平性奖励
                    slot_benefits.append(spatial_reuse + 2 * fairness_reward)

            # 选择收益最高的时隙
            if max(slot_benefits) > 0:
                best_slot = np.argmax(slot_benefits)
                slots_allocation[best_slot].append(node)
                node_allocations[node] += 1

        # 更新节点得分，类似PPO的策略更新
        for node in node_scores:
            fairness_factor = 1.0
            if demands and demands[node] > 0:
                fairness_factor = demands[node] / (node_allocations[node] + 0.5)
            node_scores[node] = centrality[node] * fairness_factor

    return slots_allocation, node_allocations


# 随机生成业务需求
def generate_demands(n_drones, scenario='uniform'):
    """
    生成节点业务需求

    参数:
        n_drones: 节点数
        scenario: 场景类型 ('uniform', 'heterogeneous', 'dynamic')

    返回:
        demands: 各节点的需求时隙数列表
    """
    demands = np.zeros(n_drones)

    if scenario == 'uniform':
        # 均匀需求
        demands = np.ones(n_drones) * 3

    elif scenario == 'heterogeneous':
        # 异构需求
        high_priority = int(n_drones * 0.2)  # 20%高优先级节点
        medium_priority = int(n_drones * 0.3)  # 30%中优先级节点

        # 高优先级节点
        demands[:high_priority] = 5
        # 中优先级节点
        demands[high_priority:high_priority + medium_priority] = 3
        # 低优先级节点
        demands[high_priority + medium_priority:] = 1

        # 随机打乱
        np.random.shuffle(demands)

    elif scenario == 'dynamic':
        # 基本需求
        demands = np.ones(n_drones) * 2

        # 随机选择30%的节点需求突增
        surge_nodes = np.random.choice(n_drones, size=int(n_drones * 0.3), replace=False)
        demands[surge_nodes] = 6

    return demands


# 评估算法公平性
def evaluate_fairness(n_drones=15, comm_range=200, interference_range=300, num_slots=10, scenario='heterogeneous'):
    """
    评估不同算法的时隙分配公平性

    参数:
        n_drones: 无人机数量
        comm_range: 通信范围
        interference_range: 干扰范围
        num_slots: 总时隙数
        scenario: 业务场景 ('uniform', 'heterogeneous', 'dynamic')

    返回:
        results: 包含各算法公平性指标的字典
    """
    # 创建网络拓扑
    G, positions = create_network_topology(n_drones, comm_range)

    # 构建干扰图
    IG = build_interference_graph(G, interference_range)

    # 生成业务需求
    demands = generate_demands(n_drones, scenario)

    # 运行不同的时隙分配算法
    tradstdma_slots, tradstdma_alloc = traditional_stdma(IG, num_slots, demands)
    fairstdma_slots, fairstdma_alloc = fair_stdma(IG, num_slots, demands)
    gatppo_slots, gatppo_alloc = gat_ppo_stdma_simulation(IG, num_slots, demands)

    # 轮询式TDMA (基准)
    tdma_alloc = {node: min(demands[node], 1) for node in range(n_drones)}

    # 计算公平性指数
    tradstdma_fairness = calculate_fairness_index(list(tradstdma_alloc.values()), demands)
    fairstdma_fairness = calculate_fairness_index(list(fairstdma_alloc.values()), demands)
    gatppo_fairness = calculate_fairness_index(list(gatppo_alloc.values()), demands)
    tdma_fairness = calculate_fairness_index(list(tdma_alloc.values()), demands)

    # 计算空间复用率
    tradstdma_sru = sum(len(nodes) for nodes in tradstdma_slots.values()) / num_slots
    fairstdma_sru = sum(len(nodes) for nodes in fairstdma_slots.values()) / num_slots
    gatppo_sru = sum(len(nodes) for nodes in gatppo_slots.values()) / num_slots
    tdma_sru = 1.0  # TDMA无空间复用

    # 归一化资源计算
    tradstdma_norm = [tradstdma_alloc[i] / demands[i] if demands[i] > 0 else 1.0 for i in range(n_drones)]
    fairstdma_norm = [fairstdma_alloc[i] / demands[i] if demands[i] > 0 else 1.0 for i in range(n_drones)]
    gatppo_norm = [gatppo_alloc[i] / demands[i] if demands[i] > 0 else 1.0 for i in range(n_drones)]
    tdma_norm = [tdma_alloc[i] / demands[i] if demands[i] > 0 else 1.0 for i in range(n_drones)]

    results = {
        'fairness': {
            'Traditional STDMA': tradstdma_fairness,
            'Fair STDMA': fairstdma_fairness,
            'GAT-PPO-STDMA': gatppo_fairness,
            'Round-Robin TDMA': tdma_fairness
        },
        'spatial_reuse': {
            'Traditional STDMA': tradstdma_sru,
            'Fair STDMA': fairstdma_sru,
            'GAT-PPO-STDMA': gatppo_sru,
            'Round-Robin TDMA': tdma_sru
        },
        'normalized_resources': {
            'Traditional STDMA': tradstdma_norm,
            'Fair STDMA': fairstdma_norm,
            'GAT-PPO-STDMA': gatppo_norm,
            'Round-Robin TDMA': tdma_norm
        },
        'allocations': {
            'Traditional STDMA': tradstdma_alloc,
            'Fair STDMA': fairstdma_alloc,
            'GAT-PPO-STDMA': gatppo_alloc,
            'Round-Robin TDMA': tdma_alloc
        },
        'demands': demands,
        'slots': {
            'Traditional STDMA': tradstdma_slots,
            'Fair STDMA': fairstdma_slots,
            'GAT-PPO-STDMA': gatppo_slots
        }
    }

    return results, G, IG


# 可视化时隙分配结果
def visualize_slot_allocation(results, G, IG, scenario='heterogeneous'):
    """可视化不同算法的时隙分配和公平性结果"""

    # 1. 绘制公平性指数对比条形图
    plt.figure(figsize=(10, 6))
    algorithms = list(results['fairness'].keys())
    fairness_values = list(results['fairness'].values())

    bars = plt.bar(algorithms, fairness_values, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])

    # 在条形上方标注具体数值
    for bar, value in zip(bars, fairness_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom')

    plt.title(f'Fairness Index Comparison ({scenario} scenario)')
    plt.ylabel('Jain Fairness Index')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f'fairness_comparison_{scenario}.png', dpi=300, bbox_inches='tight')

    # 2. 绘制归一化资源分配箱线图
    plt.figure(figsize=(12, 6))
    norm_data = [results['normalized_resources'][alg] for alg in algorithms]

    plt.boxplot(norm_data, labels=algorithms)
    plt.title(f'Normalized Resource Distribution ({scenario} scenario)')
    plt.ylabel('Normalized Resource (Allocated/Demanded)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f'resource_distribution_{scenario}.png', dpi=300, bbox_inches='tight')

    # 3. 绘制空间复用率与公平性的散点图
    plt.figure(figsize=(8, 6))
    sru_values = list(results['spatial_reuse'].values())

    plt.scatter(sru_values, fairness_values, c=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'], s=100)

    for i, alg in enumerate(algorithms):
        plt.annotate(alg, (sru_values[i], fairness_values[i]),
                     xytext=(10, 5), textcoords='offset points')

    plt.title(f'Fairness vs. Spatial Reuse ({scenario} scenario)')
    plt.xlabel('Spatial Reuse Utilization')
    plt.ylabel('Fairness Index')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'fairness_vs_sru_{scenario}.png', dpi=300, bbox_inches='tight')

    # 4. 可视化时隙分配结果 (以GAT-PPO为例)
    plt.figure(figsize=(14, 8))

    # 获取节点的原始需求，用于颜色映射
    demands = results['demands']
    max_demand = max(demands)
    norm_demands = demands / max_demand

    # 获取GAT-PPO的分配结果
    gatppo_slots = results['slots']['GAT-PPO-STDMA']

    # 创建时隙分配网格
    n_slots = len(gatppo_slots)
    n_drones = len(demands)
    allocation_grid = np.zeros((n_drones, n_slots))

    for slot, nodes in gatppo_slots.items():
        for node in nodes:
            allocation_grid[node, slot] = 1

    plt.imshow(allocation_grid, cmap='Blues', aspect='auto')
    plt.colorbar(label='Allocated (1) / Not Allocated (0)')

    # 添加节点需求指示器
    demand_indicator = np.array(demands).reshape(-1, 1) / max_demand
    plt.imshow(demand_indicator, cmap='Reds', aspect='auto',
               extent=(-1, 0, -0.5, n_drones - 0.5))
    plt.colorbar(label='Normalized Demand')

    plt.title(f'GAT-PPO-STDMA Slot Allocation ({scenario} scenario)')
    plt.xlabel('Time Slot')
    plt.ylabel('Node ID')
    plt.tight_layout()
    plt.savefig(f'slot_allocation_{scenario}.png', dpi=300, bbox_inches='tight')

    plt.show()


# 主函数
def main():
    # 设置随机种子以保证可重复性
    np.random.seed(42)
    random.seed(42)

    # 评估不同场景下的公平性
    scenarios = ['uniform', 'heterogeneous', 'dynamic']

    for scenario in scenarios:
        print(f"\nEvaluating {scenario} scenario...")
        results, G, IG = evaluate_fairness(n_drones=15, scenario=scenario)

        # 打印公平性结果
        print("\nFairness Index:")
        for alg, fairness in results['fairness'].items():
            print(f"  {alg}: {fairness:.4f}")

        print("\nSpatial Reuse Utilization:")
        for alg, sru in results['spatial_reuse'].items():
            print(f"  {alg}: {sru:.4f}")

        # 可视化结果
        visualize_slot_allocation(results, G, IG, scenario)


if __name__ == "__main__":
    main()