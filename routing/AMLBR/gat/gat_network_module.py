import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import logging
from utils import config
from phy.large_scale_fading import euclidean_distance, maximum_communication_range
class GATLayer(nn.Module):
    """
    图注意力网络层，用于路由决策
    """

    def __init__(self, in_features, out_features, heads=1, dropout=0.1, concat=True):
        super(GATLayer, self).__init__()
        self.gat = GATConv(
            in_features,
            out_features,
            heads=heads,
            dropout=dropout,
            concat=concat
        )
        self.attention_weights = None

        # 注册钩子来捕获注意力权重
        def hook_fn(module, input, output):
            if hasattr(module, '_alpha'):
                self.attention_weights = module._alpha.detach()

        self.gat.register_forward_hook(hook_fn)

    def forward(self, x, edge_index):
        return self.gat(x, edge_index)

    def get_attention_weights(self):
        return self.attention_weights


class GATRoutingModel(nn.Module):
    """
    基于GAT的路由决策模型
    """

    def __init__(self, in_features, hidden_dim=32, heads=2):
        super(GATRoutingModel, self).__init__()
        self.gat1 = GATLayer(in_features, hidden_dim, heads=heads, concat=True)
        # 如果concat=True，第二层的输入维度需要是hidden_dim*heads
        self.gat2 = GATLayer(hidden_dim * heads, hidden_dim, heads=1, concat=False)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        # 第一层GAT
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        # 第二层GAT
        x = self.gat2(x, edge_index)
        x = F.elu(x)

        # 输出层
        scores = self.fc(x)
        return scores, x  # 返回节点评分和节点嵌入

    def get_attention_weights(self):
        # 返回第一层的注意力权重
        return self.gat1.get_attention_weights()


class GATNetworkAnalyzer:
    """
    使用GAT分析网络拓扑和计算路径质量
    """

    def __init__(self, simulator, num_drones, node_feature_dim=5):
        self.num_drones = num_drones
        self.node_feature_dim = node_feature_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.simulator = simulator
        # 初始化GAT模型
        self.model = GATRoutingModel(
            in_features=node_feature_dim,
            hidden_dim=32,
            heads=2
        ).to(self.device)

        # 初始化优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # 缓存
        self.node_embeddings = None
        self.path_quality_cache = {}
        self.last_update_time = 0

        logging.info(f"GAT网络分析器初始化完成，使用设备: {self.device}")

    def update_network_state(self, simulator, update_interval=1e6):
        """
        更新网络状态和模型
        """
        current_time = simulator.env.now

        # 控制更新频率
        if current_time - self.last_update_time < update_interval:
            return False

        # 构建图数据
        node_features, edge_index = self._build_graph_data(simulator)

        # 转换为PyTorch张量
        node_features = torch.FloatTensor(node_features).to(self.device)
        edge_index = torch.LongTensor(edge_index).to(self.device)

        # 前向传播
        self.model.eval()
        with torch.no_grad():
            scores, node_embeddings = self.model(node_features, edge_index)

        # 缓存节点嵌入
        self.node_embeddings = node_embeddings.cpu().numpy()

        # 清空路径质量缓存
        self.path_quality_cache = {}

        # 更新时间戳
        self.last_update_time = current_time

        logging.info(f"网络状态已更新，时间: {current_time}")
        return True

    def _build_graph_data(self, simulator):
        """
        从模拟器构建图数据
        """
        # 节点特征
        node_features = np.zeros((self.num_drones, self.node_feature_dim))

        # 边列表
        edges = []

        for i in range(self.num_drones):
            drone = simulator.drones[i]

            # 节点特征: [x坐标, y坐标, z坐标, 剩余能量比例, 队列占用比例]
            node_features[i, 0:3] = drone.coords
            node_features[i, 3] = drone.residual_energy / config.INITIAL_ENERGY
            node_features[i, 4] = drone.transmitting_queue.qsize() / drone.max_queue_size

            # 构建边
            for j in range(self.num_drones):
                if i != j:
                    drone2 = simulator.drones[j]
                    # 计算距离
                    dist = euclidean_distance(drone.coords, drone2.coords)
                    # 如果在通信范围内，添加一条边
                    if dist <= maximum_communication_range():
                        edges.append([i, j])

        # 转换为edge_index格式 [2, num_edges]
        edge_index = np.array(edges).T if edges else np.zeros((2, 0), dtype=int)

        return node_features, edge_index

    def evaluate_path_quality(self, path):
        """
        评估路径质量
        """
        if not path or len(path) < 2:
            return 0.0

        # 检查缓存
        path_key = tuple(path)
        if path_key in self.path_quality_cache:
            return self.path_quality_cache[path_key]

        # 如果没有嵌入信息，返回默认值
        if self.node_embeddings is None:
            return 0.5

        # 计算路径质量
        path_quality = 0.0

        # 方法1: 基于节点嵌入的相似性
        for i in range(len(path) - 1):
            node1 = path[i]
            node2 = path[i + 1]

            if 0 <= node1 < self.num_drones and 0 <= node2 < self.num_drones:
                # 计算两个节点嵌入的余弦相似度
                emb1 = self.node_embeddings[node1]
                emb2 = self.node_embeddings[node2]

                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                path_quality += max(0, similarity)  # 确保是正值

        # 平均每跳质量
        path_quality = path_quality / (len(path) - 1)

        # 缓存结果
        self.path_quality_cache[path_key] = path_quality

        return path_quality

    def get_node_importance(self):
        """
        获取节点重要性
        """
        if self.node_embeddings is None:
            return None

        # 基于节点嵌入的范数计算节点重要性
        node_importance = np.linalg.norm(self.node_embeddings, axis=1)

        # 归一化
        if np.max(node_importance) > 0:
            node_importance = node_importance / np.max(node_importance)

        return node_importance

    def train_model(self, simulator, epochs=10):
        """
        使用当前网络状态训练模型，增加详细日志和返回值
        """
        try:
            # 构建图数据
            node_features, edge_index = self._build_graph_data(simulator)

            # 转换为PyTorch张量
            node_features = torch.FloatTensor(node_features).to(self.device)
            edge_index = torch.LongTensor(edge_index).to(self.device)

            # 检查是否有足够的边
            if edge_index.shape[1] < 2:
                logging.warning(f"图中边数({edge_index.shape[1]})不足，无法进行有效训练")
                return False

            # 构建训练标签 (这里使用一个简单的启发式规则)
            labels = torch.zeros(self.num_drones, device=self.device)
            for i in range(self.num_drones):
                # 示例：基于剩余能量和邻居数量的启发式
                drone = simulator.drones[i]
                energy_factor = drone.residual_energy / config.INITIAL_ENERGY

                # 计算邻居数
                neighbor_count = 0
                for j in range(self.num_drones):
                    if i != j:
                        dist = euclidean_distance(drone.coords, simulator.drones[j].coords)
                        if dist <= maximum_communication_range():
                            neighbor_count += 1

                # 综合评分
                labels[i] = energy_factor * (0.5 + 0.5 * min(1.0, neighbor_count / 5))

            # 打印训练信息
            logging.info(f"GAT训练开始，节点数: {node_features.shape[0]}, 边数: {edge_index.shape[1]}, 轮次: {epochs}")

            # 训练循环
            self.model.train()
            for epoch in range(epochs):
                self.optimizer.zero_grad()

                # 前向传播
                scores, _ = self.model(node_features, edge_index)
                scores = scores.squeeze()

                # 计算损失 (MSE)
                loss = F.mse_loss(scores, labels)

                # 反向传播
                loss.backward()
                self.optimizer.step()

                logging.info(f"训练轮次 {epoch + 1}/{epochs}, 损失: {loss.item():.4f}")

            # 更新嵌入
            self.model.eval()
            with torch.no_grad():
                _, node_embeddings = self.model(node_features, edge_index)
                self.node_embeddings = node_embeddings.cpu().numpy()

            logging.info("GAT训练完成，已更新节点嵌入")
            return True

        except Exception as e:
            logging.error(f"GAT训练失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def debug_model_state(self):
        """打印模型当前状态，用于调试"""
        if self.node_embeddings is None:
            logging.info("GAT模型尚未生成节点嵌入")
            return

        # 打印节点嵌入统计
        emb_mean = np.mean(self.node_embeddings)
        emb_std = np.std(self.node_embeddings)
        emb_min = np.min(self.node_embeddings)
        emb_max = np.max(self.node_embeddings)

        logging.info(
            f"节点嵌入统计 - 均值: {emb_mean:.4f}, 标准差: {emb_std:.4f}, 最小值: {emb_min:.4f}, 最大值: {emb_max:.4f}")

        # 打印模型参数统计
        total_params = sum(p.numel() for p in self.model.parameters())
        logging.info(f"模型参数数量: {total_params}")

        # 检查注意力权重
        attention_weights = self.get_attention_matrix()
        if attention_weights is not None:
            att_mean = np.mean(attention_weights)
            att_std = np.std(attention_weights)
            logging.info(f"注意力权重统计 - 均值: {att_mean:.4f}, 标准差: {att_std:.4f}")
        else:
            logging.info("注意力权重尚未生成")

    # 在周期性监控中添加GAT模型状态检查

    def periodic_gat_monitoring(self):
        """定期监控GAT模型状态"""
        # 等待初始化完成
        yield self.simulator.env.timeout(1 * 1e6)  # 10秒后开始监控

        interval = 1 * 1e6  # 每15秒检查一次

        while True:
            yield self.simulator.env.timeout(interval)

            current_time = self.simulator.env.now / 1e6  # 转换为秒

            # 找到使用GAT的无人机
            gat_drones = []
            for i, drone in enumerate(self.simulator.drones):
                if hasattr(drone.routing_protocol, 'gat_analyzer'):
                    gat_drones.append((i, drone.routing_protocol))

            if gat_drones:
                logging.info(f"[{current_time:.1f}s] GAT模型状态监控:")

                for drone_id, router in gat_drones:
                    # 获取GAT使用率
                    stats = router.get_routing_stats()
                    gat_usage = stats.get('gat_usage_ratio', 0)

                    # 获取节点重要性
                    node_importance = router.gat_analyzer.get_node_importance()
                    if node_importance is not None:
                        my_importance = node_importance[drone_id]
                        logging.info(
                            f"  无人机 {drone_id} - GAT使用率: {gat_usage * 100:.1f}%, 重要性: {my_importance:.4f}")

                        # 调试GAT模型状态
                        router.gat_analyzer.debug_model_state()
                    else:
                        logging.info(f"  无人机 {drone_id} - GAT使用率: {gat_usage * 100:.1f}%, 节点重要性未计算")

    def debug_model_state(self):
        """打印模型当前状态，用于调试"""
        if self.node_embeddings is None:
            logging.info("GAT模型尚未生成节点嵌入")
            return

        # 打印节点嵌入统计
        emb_mean = np.mean(self.node_embeddings)
        emb_std = np.std(self.node_embeddings)
        emb_min = np.min(self.node_embeddings)
        emb_max = np.max(self.node_embeddings)

        logging.info(
            f"节点嵌入统计 - 均值: {emb_mean:.4f}, 标准差: {emb_std:.4f}, 最小值: {emb_min:.4f}, 最大值: {emb_max:.4f}")

        # 打印模型参数统计
        total_params = sum(p.numel() for p in self.model.parameters())
        logging.info(f"模型参数数量: {total_params}")

        # 检查注意力权重
        attention_weights = self.get_attention_matrix()
        if attention_weights is not None:
            att_mean = np.mean(attention_weights)
            att_std = np.std(attention_weights)
            logging.info(f"注意力权重统计 - 均值: {att_mean:.4f}, 标准差: {att_std:.4f}")
        else:
            logging.info("注意力权重尚未生成")

    def get_attention_matrix(self):
        """
        获取注意力矩阵
        """
        attention_weights = self.model.get_attention_weights()
        if attention_weights is None:
            return None

        # 转换为numpy数组
        return attention_weights.cpu().numpy()

    def recommend_paths(self, src_id, dst_id, all_paths, top_k=3):
        """
        根据GAT模型推荐最佳路径
        """
        if not all_paths:
            return []

        # 评估每条路径
        path_scores = []
        for path in all_paths:
            quality = self.evaluate_path_quality(path)
            path_scores.append((path, quality))

        # 排序并返回前K条
        path_scores.sort(key=lambda x: x[1], reverse=True)
        return [path for path, _ in path_scores[:top_k]]


# 使用示例
if __name__ == "__main__":
    # 仅作示例，实际使用时需要导入正确的模块
    from utils import config
    from utils.util_function import euclidean_distance
    from phy.large_scale_fading import maximum_communication_range


    class DummySimulator:
        def __init__(self):
            self.env = type('obj', (object,), {'now': 0})
            self.drones = []


    class DummyDrone:
        def __init__(self, id):
            self.identifier = id
            self.coords = np.random.rand(3)
            self.residual_energy = np.random.rand() * 10000
            self.transmitting_queue = type('obj', (object,), {'qsize': lambda: np.random.randint(0, 10)})
            self.max_queue_size = 20


    # 创建模拟数据
    simulator = DummySimulator()
    for i in range(10):
        simulator.drones.append(DummyDrone(i))

    # 初始化分析器
    analyzer = GATNetworkAnalyzer(10)

    # 更新网络状态
    analyzer.update_network_state(simulator)

    # 评估路径
    path = [0, 2, 5, 8]
    quality = analyzer.evaluate_path_quality(path)
    print(f"路径 {path} 的质量: {quality:.4f}")

    # 获取节点重要性
    importance = analyzer.get_node_importance()
    print("节点重要性:", importance)