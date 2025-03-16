import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
import math


class DynamicGATBlock(nn.Module):
    """支持可变节点数量的GAT块"""

    def __init__(self, in_channels, out_channels, heads=4, dropout=0.1):
        super().__init__()
        self.gat1 = GATConv(
            in_channels,
            out_channels,
            heads=heads,
            dropout=dropout,
            concat=True
        )
        self.gat2 = GATConv(
            out_channels * heads,
            out_channels,
            heads=1,
            dropout=dropout,
            concat=False
        )
        self.norm1 = nn.LayerNorm(out_channels * heads)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x, edge_index):
        """
        前向传播
        x: 节点特征 [num_nodes, in_channels]
        edge_index: 边连接 [2, num_edges]
        """
        identity = self.proj(x)

        # 第一个GAT层
        x = self.gat1(x, edge_index)
        x = self.norm1(x)
        x = F.elu(x)
        x = self.dropout(x)

        # 第二个GAT层
        x = self.gat2(x, edge_index)
        x = self.norm2(x)

        # 残差连接
        return F.elu(x + identity)


class DynamicGNNFeatureExtractor(BaseFeaturesExtractor):
    """能处理可变规模网络的特征提取器"""

    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        # 基本网络参数
        self.node_feat_dim = 32
        self.hidden_dim = 64
        self.output_dim = features_dim

        # 节点特征编码器
        self.node_encoder = nn.Sequential(
            nn.Linear(4, self.node_feat_dim),  # 4 = [x,y,z,度]
            nn.ReLU(),
            nn.LayerNorm(self.node_feat_dim)
        )

        # GAT层
        self.gat_block = DynamicGATBlock(
            in_channels=self.node_feat_dim,
            out_channels=self.hidden_dim,
            heads=4,
            dropout=0.1
        )

        # 交通信息编码器
        self.traffic_encoder = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.LayerNorm(32)
        )

        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim + 32, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim)
        )

    def parse_observation(self, flat_obs):
        """将扁平化的观察解析为结构化格式，支持填充数据"""
        # 确保我们处理的是numpy数组
        if isinstance(flat_obs, torch.Tensor):
            flat_obs_np = flat_obs.detach().cpu().numpy()
        else:
            flat_obs_np = flat_obs

        # 从填充观察中提取有效数据
        # 尝试估计实际节点数量
        non_zero_count = np.sum(flat_obs_np != 0)

        # 保守估计节点数量
        estimated_nodes = int(np.sqrt(non_zero_count / 4))  # 4是因为拓扑和链路生命期矩阵
        estimated_nodes = max(5, min(estimated_nodes, 50))  # 限制在合理范围内

        # 计算各部分大小
        start_idx = 0
        topology_size = estimated_nodes * estimated_nodes

        # 安全提取拓扑矩阵数据
        safe_end = min(start_idx + topology_size, len(flat_obs_np))
        topology_data = flat_obs_np[start_idx:safe_end]
        topology = np.zeros((estimated_nodes, estimated_nodes), dtype=np.float32)

        # 填充可用数据
        if len(topology_data) > 0:
            flat_size = min(topology_size, len(topology_data))
            topology.flat[:flat_size] = topology_data[:flat_size]

        # 节点位置数据
        start_idx = safe_end
        position_size = estimated_nodes * 3
        safe_end = min(start_idx + position_size, len(flat_obs_np))
        position_data = flat_obs_np[start_idx:safe_end]
        positions = np.zeros((estimated_nodes, 3), dtype=np.float32)

        if len(position_data) > 0:
            flat_size = min(position_size, len(position_data))
            positions_flat = np.zeros(position_size, dtype=np.float32)
            positions_flat[:flat_size] = position_data[:flat_size]
            positions = positions_flat.reshape(estimated_nodes, 3)

        # 路由数据
        start_idx = safe_end
        routing_size = estimated_nodes
        safe_end = min(start_idx + routing_size, len(flat_obs_np))
        routing_data = flat_obs_np[start_idx:safe_end]
        routing = np.full(estimated_nodes, -1, dtype=np.float32)

        if len(routing_data) > 0:
            flat_size = min(routing_size, len(routing_data))
            routing[:flat_size] = routing_data[:flat_size]

        # 链路生命期数据
        start_idx = safe_end
        lifetime_size = estimated_nodes * estimated_nodes
        safe_end = min(start_idx + lifetime_size, len(flat_obs_np))
        lifetime_data = flat_obs_np[start_idx:safe_end]
        link_lifetime = np.zeros((estimated_nodes, estimated_nodes), dtype=np.float32)

        if len(lifetime_data) > 0:
            flat_size = min(lifetime_size, len(lifetime_data))
            lifetime_flat = np.zeros(lifetime_size, dtype=np.float32)
            lifetime_flat[:flat_size] = lifetime_data[:flat_size]
            link_lifetime = lifetime_flat.reshape(estimated_nodes, estimated_nodes)

        # 交通信息数据
        start_idx = safe_end
        traffic_size = 5
        safe_end = min(start_idx + traffic_size, len(flat_obs_np))
        traffic_data = flat_obs_np[start_idx:safe_end]
        traffic_info = np.zeros(traffic_size, dtype=np.float32)

        if len(traffic_data) > 0:
            flat_size = min(traffic_size, len(traffic_data))
            traffic_info[:flat_size] = traffic_data[:flat_size]

        # 节点度数据
        start_idx = safe_end
        degrees_size = estimated_nodes
        safe_end = min(start_idx + degrees_size, len(flat_obs_np))
        degrees_data = flat_obs_np[start_idx:safe_end]
        node_degrees = np.zeros(estimated_nodes, dtype=np.float32)

        if len(degrees_data) > 0:
            flat_size = min(degrees_size, len(degrees_data))
            node_degrees[:flat_size] = degrees_data[:flat_size]

        # 获取当前设备
        device = next(self.parameters()).device

        # 转换为张量并移到正确的设备上
        return {
            'topology': torch.tensor(topology, dtype=torch.float32, device=device),
            'node_positions': torch.tensor(positions, dtype=torch.float32, device=device),
            'routing': torch.tensor(routing, dtype=torch.float32, device=device),
            'link_lifetime': torch.tensor(link_lifetime, dtype=torch.float32, device=device),
            'traffic_info': torch.tensor(traffic_info, dtype=torch.float32, device=device),
            'node_degrees': torch.tensor(node_degrees, dtype=torch.float32, device=device),
            'num_nodes': estimated_nodes
        }

    def forward(self, observations):
        """前向传播"""
        if not isinstance(observations, torch.Tensor):
            observations = torch.tensor(observations, dtype=torch.float32)

        if observations.device != next(self.parameters()).device:
            observations = observations.to(next(self.parameters()).device)

        batch_size = observations.shape[0]
        results = []

        for i in range(batch_size):
            obs = observations[i]
            # 解析观察数据
            obs_dict = self.parse_observation(obs)

            # 提取节点特征
            node_features = torch.cat([
                obs_dict['node_positions'],
                obs_dict['node_degrees'].unsqueeze(-1)
            ], dim=-1)
            node_features = self.node_encoder(node_features)

            # 提取边索引 - 确保边索引在正确的设备上
            edge_index = torch.nonzero(obs_dict['topology']).t().contiguous()

            # 处理没有边的情况
            if edge_index.shape[1] == 0:
                num_nodes = obs_dict['num_nodes']
                edge_index = torch.tensor([[j, j] for j in range(int(num_nodes))],
                                          dtype=torch.long,
                                          device=observations.device).t().contiguous()

            # GAT处理
            node_features = self.gat_block(node_features, edge_index)

            # 全局池化 - 使用与节点特征相同设备上的索引
            batch_index = torch.zeros(node_features.size(0), dtype=torch.long, device=node_features.device)
            pooled = global_mean_pool(node_features, batch_index)

            # 处理交通信息
            traffic_features = self.traffic_encoder(obs_dict['traffic_info'])

            # 确保维度匹配 - 修复关键错误
            # pooled 形状是 [1, hidden_dim]，traffic_features 形状是 [32]
            # 统一维度，确保都是 [batch, features]
            if pooled.dim() > 1 and traffic_features.dim() == 1:
                # 将traffic_features扩展为2D
                traffic_features = traffic_features.unsqueeze(0)  # [features] -> [1, features]

            # 合并特征 - 沿着特征维度（dim=1）连接
            combined = torch.cat([pooled, traffic_features], dim=1)
            results.append(combined)

        # 堆叠批次结果
        stacked = torch.cat(results, dim=0)  # 使用cat而不是stack，因为结果已经有批次维度

        # 特征融合
        return self.fusion(stacked)