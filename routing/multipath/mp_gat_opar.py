import copy
import logging
import math
import numpy as np
import random
from entities.packet import DataPacket
from topology.virtual_force.vf_packet import VfPacket
from utils import config
from utils.util_function import euclidean_distance
from phy.large_scale_fading import maximum_communication_range
from simulator.TrafficGenerator import TrafficRequirement
from routing.multipath.mp_amlb_opar import MP_AMLB_OPAR
from collections import defaultdict

# 尝试导入PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
    logging.info("PyTorch导入成功")
except ImportError:
    logging.warning("无法导入PyTorch，将使用基本AMLB-OPAR算法")
    HAS_TORCH = False

# 纯PyTorch实现的图注意力层
if HAS_TORCH:
    class PureGATLayer(nn.Module):
        def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2, concat=True):
            super(PureGATLayer, self).__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.dropout = dropout
            self.alpha = alpha
            self.concat = concat

            # 线性变换权重
            self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
            nn.init.xavier_uniform_(self.W.data)

            # 注意力机制参数
            self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
            nn.init.xavier_uniform_(self.a.data)

            # LeakyReLU激活函数
            self.leakyrelu = nn.LeakyReLU(self.alpha)

        def forward(self, x, adj):
            # x: 节点特征矩阵 [N, in_features]
            # adj: 邻接矩阵 [N, N]

            # 线性变换
            h = torch.mm(x, self.W)  # [N, out_features]
            N = h.size(0)

            # 计算注意力系数
            a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, N,
                                                                                              2 * self.out_features)
            e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

            # 掩码无连接的节点
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            attention = F.dropout(attention, self.dropout, training=self.training)

            # 聚合邻居特征
            h_prime = torch.matmul(attention, h)

            # 非线性激活
            if self.concat:
                return F.elu(h_prime)
            else:
                return h_prime


    class PureGAT(nn.Module):
        def __init__(self, in_dim, hidden_dim=64, out_dim=16, dropout=0.1, alpha=0.2, num_heads=2):
            super(PureGAT, self).__init__()
            self.dropout = dropout

            # 多头注意力第一层
            self.attentions = [PureGATLayer(in_dim, hidden_dim, dropout=dropout, alpha=alpha, concat=True)
                               for _ in range(num_heads)]
            for i, attention in enumerate(self.attentions):
                self.add_module(f'attention_{i}', attention)

            # 输出层
            self.out_att = PureGATLayer(hidden_dim * num_heads, out_dim, dropout=dropout, alpha=alpha, concat=False)

        def forward(self, x, adj):
            # x: 节点特征矩阵 [N, in_dim]
            # adj: 邻接矩阵 [N, N]

            # 多头注意力
            x = F.dropout(x, self.dropout, training=self.training)
            x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
            x = F.dropout(x, self.dropout, training=self.training)

            # 输出层
            x = self.out_att(x, adj)
            return x


class MP_GAT_OPAR(MP_AMLB_OPAR):
    def __init__(self, simulator, my_drone):
        super().__init__(simulator, my_drone)

        # GAT相关属性
        self.use_gat = HAS_TORCH and getattr(config, 'USE_GAT', True)
        self.feature_dim = 8  # 节点特征维度
        self.hidden_dim = 64  # 隐藏层维度
        self.embedding_dim = 16  # 输出嵌入维度
        self.num_heads = 2  # 注意力头数
        self.gat_model = None
        self.node_features = {}  # 节点特征字典
        self.edge_weights = {}  # 边权重
        self.adj_matrix = None  # 邻接矩阵
        self.node_map = {}  # 节点ID到索引的映射

        # 训练相关
        self.optimizer = None
        self.loss_fn = None
        self.learning_rate = 0.001
        self.batch_size = 32
        self.training_data = []  # 存储训练样本
        self.device = None

        # 初始化GAT模型
        if self.use_gat:
            self._initialize_gat_model()
            # 启动模型定期训练进程
            self.simulator.env.process(self.train_model_periodically())
            logging.info("MP_GAT_OPAR 初始化完成，已启用GAT功能")
        else:
            logging.info("MP_GAT_OPAR 初始化完成，使用基本AMLB-OPAR算法")

    def _initialize_gat_model(self):
        """初始化GAT模型"""
        if not self.use_gat:
            return

        try:
            # 设置设备
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # 定义损失函数
            self.loss_fn = nn.MSELoss()

            # 初始化模型
            self.gat_model = PureGAT(
                in_dim=self.feature_dim,
                hidden_dim=self.hidden_dim,
                out_dim=self.embedding_dim,
                num_heads=self.num_heads,
                dropout=0.1,
                alpha=0.2
            ).to(self.device)

            # 初始化优化器
            self.optimizer = torch.optim.Adam(self.gat_model.parameters(), lr=self.learning_rate)
            logging.info(f"GAT模型初始化完成，使用设备: {self.device}")
        except Exception as e:
            logging.error(f"初始化GAT模型时出错: {str(e)}")
            self.use_gat = False

    def _extract_node_features(self, drone_id):
        """提取节点特征"""
        if not self.use_gat:
            return None

        try:
            if drone_id not in self.simulator.drones:
                return torch.zeros(self.feature_dim, device=self.device)

            drone = self.simulator.drones[drone_id]

            # 提取节点特征：位置、速度、队列长度、能量等
            x, y, z = drone.coords
            queue_len = drone.transmitting_queue.qsize() / drone.max_queue_size if hasattr(drone,
                                                                                           'transmitting_queue') else 0
            energy = drone.energy / drone.initial_energy if hasattr(drone, 'energy') else 1.0
            neighbors_count = len(drone.motion_controller.neighbor_table.neighbors) if hasattr(drone.motion_controller,
                                                                                               'neighbor_table') else 0
            max_neighbors = 10  # 假设的最大邻居数

            # 归一化特征
            normalized_x = x / 500  # 假设场景尺寸为1000
            normalized_y = y / 500
            normalized_z = z / 500
            normalized_neighbors = neighbors_count / max_neighbors

            # 构造特征向量 [x, y, z, queue_length, energy, neighbors, 0, 0]
            features = [normalized_x, normalized_y, normalized_z, queue_len, energy, normalized_neighbors, 0, 0]
            return torch.tensor(features, dtype=torch.float32, device=self.device)
        except Exception as e:
            logging.error(f"提取节点特征时出错: {str(e)}")
            return torch.zeros(self.feature_dim, device=self.device) if self.use_gat else None

    def _build_network_graph(self):
        """构建网络图（邻接矩阵）"""
        if not self.use_gat:
            return None

        try:
            # 创建节点列表
            nodes = list(self.simulator.drones.keys())
            n = len(nodes)

            # 创建节点映射字典
            self.node_map = {node_id: idx for idx, node_id in enumerate(nodes)}

            # 创建邻接矩阵
            adj_matrix = torch.zeros((n, n), device=self.device)

            # 提取节点特征
            node_features = []
            for node_id in nodes:
                features = self._extract_node_features(node_id)
                node_features.append(features)

            # 构建邻接矩阵
            for i, src_id in enumerate(nodes):
                src_drone = self.simulator.drones[src_id]
                for j, dst_id in enumerate(nodes):
                    if src_id == dst_id:
                        # 自环设为1
                        adj_matrix[i, i] = 1
                        continue

                    dst_drone = self.simulator.drones[dst_id]

                    # 计算节点间距离
                    dist = euclidean_distance(src_drone.coords, dst_drone.coords)

                    # 如果距离在通信范围内
                    if dist <= self.max_comm_range:
                        # 设置边权重（归一化到0-1）
                        edge_weight = 1.0 / max(dist, 1.0)
                        adj_matrix[i, j] = edge_weight
                        self.edge_weights[(src_id, dst_id)] = edge_weight

            # 储存邻接矩阵
            self.adj_matrix = adj_matrix

            # 返回节点特征矩阵
            return torch.stack(node_features)
        except Exception as e:
            logging.error(f"构建网络图时出错: {str(e)}")
            return None

    def _collect_training_data(self, packet):
        """收集训练数据"""
        if not self.use_gat:
            return

        try:
            if not hasattr(packet, 'flow_id') or not hasattr(packet, 'path_index'):
                return

            # 如果数据包成功送达，收集训练数据
            src_id = packet.src_drone.identifier
            dst_id = packet.dst_drone.identifier
            path_index = packet.path_index
            path = self.path_cache.get(dst_id, [])[path_index] if dst_id in self.path_cache else []

            if not path:
                return

            # 计算实际性能指标
            latency = self.simulator.env.now - packet.creation_time
            hop_count = packet.get_current_ttl()

            # 创建训练样本
            sample = {
                'src_id': src_id,
                'dst_id': dst_id,
                'path': path,
                'latency': latency,
                'hop_count': hop_count,
                'creation_time': self.simulator.env.now
            }

            self.training_data.append(sample)

            # 限制训练数据大小
            if len(self.training_data) > 1000:
                self.training_data = self.training_data[-1000:]
        except Exception as e:
            logging.error(f"收集训练数据时出错: {str(e)}")

    def train_model(self):
        """训练GAT模型"""
        if not self.use_gat or len(self.training_data) < self.batch_size:
            return

        try:
            # 构建网络图
            node_features = self._build_network_graph()
            if node_features is None or self.adj_matrix is None:
                return

            # 准备训练数据
            paths = []
            targets = []

            for sample in random.sample(self.training_data, min(self.batch_size, len(self.training_data))):
                path = sample['path']
                latency = sample['latency']
                hop_count = sample['hop_count']

                # 归一化延迟 (假设最大延迟为1s)
                normalized_latency = min(latency / 1e6, 1.0)

                # 目标值: 路径质量的倒数 (延迟越小，质量越高)
                target = 1.0 - normalized_latency

                paths.append(path)
                targets.append(target)

            # 训练模型
            self.gat_model.train()

            # 获取节点嵌入
            with torch.no_grad():
                node_embeddings = self.gat_model(node_features, self.adj_matrix)

            # 计算路径嵌入和预测质量
            path_embeddings = []
            for path in paths:
                path_embedding = torch.zeros(self.embedding_dim, device=self.device)
                if len(path) > 1:
                    # 路径中每个链路的嵌入平均
                    for i in range(len(path) - 1):
                        if path[i] in self.node_map and path[i + 1] in self.node_map:
                            src_idx = self.node_map[path[i]]
                            dst_idx = self.node_map[path[i + 1]]
                            link_embedding = (node_embeddings[src_idx] + node_embeddings[dst_idx]) / 2
                            path_embedding += link_embedding
                    path_embedding /= (len(path) - 1)
                path_embeddings.append(path_embedding)

            # 计算预测值
            path_embeddings = torch.stack(path_embeddings)
            predictions = torch.sum(path_embeddings, dim=1)
            targets_tensor = torch.tensor(targets, dtype=torch.float32, device=self.device)

            # 训练步骤
            self.optimizer.zero_grad()
            loss = self.loss_fn(predictions, targets_tensor)
            loss.backward()
            self.optimizer.step()

            logging.info(f"GAT模型训练完成，损失: {loss.item()}")
        except Exception as e:
            logging.error(f"训练模型时出错: {str(e)}")

    def train_model_periodically(self):
        """定期训练模型的进程"""
        while True:
            yield self.simulator.env.timeout(5 * 1e6)  # 每5秒训练一次
            if self.use_gat and len(self.training_data) >= self.batch_size:
                self.train_model()

    def _update_path_stats(self, packet):
        """更新路径统计数据"""
        super()._update_path_stats(packet)

        # 额外收集训练数据
        if self.use_gat:
            self._collect_training_data(packet)

    def gat_path_ranking(self, src_id, dst_id, paths):
        """使用GAT模型对路径进行排序"""
        if not self.use_gat or not paths:
            return paths

        try:
            # 构建或更新图
            node_features = self._build_network_graph()
            if node_features is None or self.adj_matrix is None:
                return paths

            # 获取节点嵌入
            self.gat_model.eval()
            with torch.no_grad():
                node_embeddings = self.gat_model(node_features, self.adj_matrix)

            # 计算路径得分
            path_scores = []

            for path in paths:
                path_score = 0
                for i in range(len(path) - 1):
                    if path[i] in self.node_map and path[i + 1] in self.node_map:
                        src_idx = self.node_map[path[i]]
                        dst_idx = self.node_map[path[i + 1]]
                        # 计算链路嵌入的得分
                        link_embedding = torch.cat([node_embeddings[src_idx], node_embeddings[dst_idx]])
                        link_score = torch.sum(link_embedding).item()
                        path_score += link_score

                # 归一化得分（按路径长度）
                if len(path) > 1:
                    path_score /= (len(path) - 1)
                path_scores.append(path_score)

            # 按得分排序路径
            ranked_paths = [p for _, p in sorted(zip(path_scores, paths), key=lambda x: x[0], reverse=True)]

            return ranked_paths
        except Exception as e:
            logging.error(f"使用GAT排序路径时出错: {str(e)}")
            return paths

    def discover_multiple_paths(self, src_id, dst_id):
        """使用GAT增强版的多路径发现"""
        # 先使用原始方法找到路径
        paths = super().discover_multiple_paths(src_id, dst_id)

        # 使用GAT排序路径
        if self.use_gat and paths:
            ranked_paths = self.gat_path_ranking(src_id, dst_id, paths)
            logging.info(f"GAT-OPAR: 从 {src_id} 到 {dst_id} 的路径已重新排序")
            return ranked_paths

        return paths

    def _assign_path_for_flow(self, flow_id, dst_id):
        """基于GAT为流量分配路径"""
        if flow_id in self.flow_to_path:
            return self.flow_to_path[flow_id]

        # 检查是否有可用路径
        if dst_id not in self.path_cache or not self.path_cache[dst_id]:
            return 0

        # 使用GAT来评估每条路径的质量
        if self.use_gat:
            paths = self.path_cache[dst_id]
            ranked_paths = self.gat_path_ranking(self.my_drone.identifier, dst_id, paths)
            # 选择最高排名的路径
            path_index = self.path_cache[dst_id].index(ranked_paths[0])

            # 考虑负载平衡
            # 如果最高排名路径的负载过高，考虑次优路径
            if dst_id in self.path_load and path_index in self.path_load[dst_id]:
                avg_load = sum(self.path_load[dst_id].values()) / len(self.path_load[dst_id])
                if self.path_load[dst_id][path_index] > avg_load * 1.5:
                    # 尝试次优路径
                    for ranked_path in ranked_paths[1:]:
                        alt_index = self.path_cache[dst_id].index(ranked_path)
                        if self.path_load[dst_id][alt_index] < avg_load:
                            path_index = alt_index
                            break
        else:
            # 如果不使用GAT，返回负载最小的路径
            path_index = 0
            min_load = float('inf')

            if dst_id in self.path_load:
                for idx, load in self.path_load[dst_id].items():
                    if load < min_load:
                        min_load = load
                        path_index = idx

        # 记录分配给该流的路径
        self.flow_to_path[flow_id] = path_index
        return path_index

    def _rebalance_flows(self, dst_id):
        """使用GAT模型辅助的流量重平衡"""
        if not self.use_gat:
            super()._rebalance_flows(dst_id)
            return

        # 获取所有使用此目的地路径的流
        target_flows = {}
        for flow_id, path_idx in self.flow_to_path.items():
            if flow_id.split('_')[2] == str(dst_id):  # flow_src_dst_priority
                target_flows[flow_id] = path_idx

        if not target_flows:
            return

        # 找出负载最高的路径
        max_load = -1
        max_path = None

        for idx, load in self.path_load[dst_id].items():
            if load > max_load:
                max_load = load
                max_path = idx

        if max_path is None:
            return

        # 使用GAT评估其他路径
        paths = self.path_cache[dst_id]
        ranked_paths = self.gat_path_ranking(self.my_drone.identifier, dst_id, paths)

        # 找出使用负载最高路径的流
        flows_on_max_path = [f for f, p in target_flows.items() if p == max_path]

        if flows_on_max_path:
            # 选择一个流和最佳替代路径
            flow_to_move = random.choice(flows_on_max_path)

            # 寻找负载适中且排名较高的替代路径
            best_alt_path = None
            for ranked_path in ranked_paths:
                alt_index = self.path_cache[dst_id].index(ranked_path)

                # 不考虑当前路径
                if alt_index == max_path:
                    continue

                # 检查负载是否适中
                if dst_id in self.path_load and alt_index in self.path_load[dst_id]:
                    if self.path_load[dst_id][alt_index] < max_load * 0.7:
                        best_alt_path = alt_index
                        break

            # 如果找到合适的替代路径，执行流量迁移
            if best_alt_path is not None:
                self.flow_to_path[flow_to_move] = best_alt_path

                # 更新负载计数
                self.path_load[dst_id][max_path] -= 1
                self.path_load[dst_id][best_alt_path] += 1

                logging.info(f"GAT-OPAR: 负载平衡 - 将流 {flow_to_move} 从路径 {max_path} 转移到路径 {best_alt_path}")