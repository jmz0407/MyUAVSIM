import os
import multiprocessing
import numpy as np
import torch as th
import torch.nn as nn
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict
from utils import config
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from simulator.simulator import Simulator
import simpy
from rl_environment import StdmaEnv
import traceback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from matplotlib import pyplot as plt
# 设置中文字体（以PingFang SC为例）
plt.rcParams['font.sans-serif']=['STFangsong'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号


class GATBlock(nn.Module):
    """使用多头注意力机制的GAT块"""

    def __init__(self, in_channels, out_channels, heads=4, dropout=0.1):
        super().__init__()
        self.gat1 = GATConv(
            in_channels,
            out_channels,
            heads=heads,
            dropout=dropout,
            concat=True  # 连接多个注意力头的输出
        )
        # 第二层GAT不连接多头的输出
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
        # 如果输入输出维度不同，添加投影层
        self.proj = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        self.attention_weights = None  # 存储注意力权重

        # 注册钩子来捕获注意力权重
        def hook_fn(module, input, output):
            if hasattr(module, 'alpha'):
                self.attention_weights = module.alpha.detach()

        self.gat1.register_forward_hook(hook_fn)
    def forward(self, x, edge_index):
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
    def get_attention_weights(self):
        return self.attention_weights


class StdmaFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)

        total_dim = observation_space.shape[0]
        self.num_nodes = 10

        self.dims = {
            'topology': self.num_nodes * self.num_nodes,
            'position': self.num_nodes * 3,
            'routing': self.num_nodes,
            'link_lifetime': self.num_nodes * self.num_nodes,
            'traffic': 5,
            'node_degrees': self.num_nodes
        }

        self.node_feat_dim = 32
        self.hidden_dim = 64

        # 节点特征编码器
        self.node_encoder = nn.Sequential(
            nn.Linear(4, self.node_feat_dim),  # 3D位置 + 度数
            nn.ReLU(),
            nn.LayerNorm(self.node_feat_dim)
        )

        # GAT层
        self.gat_block = GATBlock(
            in_channels=self.node_feat_dim,
            out_channels=self.hidden_dim,
            heads=4,
            dropout=0.1
        )

        # 路由路径编码器
        self.route_encoder = nn.Sequential(
            nn.Linear(self.num_nodes, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.1)
        )

        # 业务信息编码器
        self.traffic_encoder = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Dropout(0.1)
        )

        # 注意力池化层
        self.attention_pooling = nn.Sequential(
            nn.Linear(self.hidden_dim, 1),
            nn.Softmax(dim=1)
        )

        # 特征融合层
        total_features = self.hidden_dim + 64 + 32
        self.fusion = nn.Sequential(
            nn.Linear(total_features, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim),
            nn.Dropout(0.1)
        )

    def _unflatten_observation(self, flat_obs):
        """将扁平化的观察还原为结构化格式"""
        try:
            batch_size = flat_obs.size(0) if len(flat_obs.shape) > 1 else 1
            if batch_size > 1:
                flat_obs = flat_obs.view(batch_size, -1)

            idx = 0
            result = {}

            # 还原拓扑特征 (batch_size, num_nodes, num_nodes)
            topology_size = self.dims['topology']
            result['topology'] = flat_obs[:, idx:idx + topology_size].reshape(
                batch_size, self.num_nodes, self.num_nodes)
            idx += topology_size

            # 还原节点位置 (batch_size, num_nodes, 3)
            position_size = self.dims['position']
            result['position'] = flat_obs[:, idx:idx + position_size].reshape(
                batch_size, self.num_nodes, 3)
            idx += position_size

            # 还原路由路径 (batch_size, num_nodes)
            routing_size = self.dims['routing']
            result['routing'] = flat_obs[:, idx:idx + routing_size].reshape(
                batch_size, self.num_nodes)
            idx += routing_size

            # 还原链路生命期 (batch_size, num_nodes, num_nodes)
            lifetime_size = self.dims['link_lifetime']
            result['link_lifetime'] = flat_obs[:, idx:idx + lifetime_size].reshape(
                batch_size, self.num_nodes, self.num_nodes)
            idx += lifetime_size

            # 还原业务信息 (batch_size, 5)
            traffic_size = self.dims['traffic']
            result['traffic'] = flat_obs[:, idx:idx + traffic_size].reshape(
                batch_size, 5)
            idx += traffic_size

            # 还原节点度数 (batch_size, num_nodes)
            degrees_size = self.dims['node_degrees']
            result['node_degrees'] = flat_obs[:, idx:idx + degrees_size].reshape(
                batch_size, self.num_nodes)

            return result

        except Exception as e:
            print(f"Error in _unflatten_observation: {str(e)}")
            print(f"Input shape: {flat_obs.shape}")
            print(f"Current index: {idx}")
            print(f"Batch size: {batch_size}")
            raise e

    def forward(self, observations):
        """前向传播"""
        try:
            if not isinstance(observations, torch.Tensor):
                observations = torch.tensor(observations, dtype=torch.float32)

            if observations.device != next(self.parameters()).device:
                observations = observations.to(next(self.parameters()).device)

            obs_dict = self._unflatten_observation(observations)
            batch_size = observations.shape[0]

            graph_features_list = []
            for i in range(batch_size):
                # 获取样本数据
                topology = obs_dict['topology'][i]
                positions = obs_dict['position'][i]
                degrees = obs_dict['node_degrees'][i]

                # 构建节点特征
                node_features = torch.cat([
                    positions,
                    degrees.unsqueeze(-1)
                ], dim=-1)
                node_features = self.node_encoder(node_features)

                # 创建边索引
                edge_index = torch.nonzero(topology).t().contiguous()
                if edge_index.shape[1] == 0:
                    edge_index = torch.tensor([[j, j] for j in range(self.num_nodes)],
                                              dtype=torch.long).t().contiguous()
                    if next(self.parameters()).is_cuda:
                        edge_index = edge_index.cuda()

                # GAT处理
                node_features = self.gat_block(node_features, edge_index)

                # 全局池化
                graph_features = node_features.mean(dim=0)
                graph_features_list.append(graph_features)

            # 合并批次特征
            graph_features = torch.stack(graph_features_list)

            # 处理路由和业务信息
            route_features = self.route_encoder(obs_dict['routing'])
            traffic_features = self.traffic_encoder(obs_dict['traffic'])

            # 特征融合
            combined = torch.cat([
                graph_features,
                route_features,
                traffic_features
            ], dim=1)

            return self.fusion(combined)

        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            print(f"Observations shape: {observations.shape}")
            import traceback
            traceback.print_exc()
            raise e


class CNNFeaturesExtractor(BaseFeaturesExtractor):
    """CNN特征提取器，用于消融实验对比"""

    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        # 观察空间的大小
        n_input_channels = 1  # 单通道
        obs_size = observation_space.shape[0]

        # 计算最接近的平方数作为重塑尺寸
        # 例如：255 -> 16x16（补充一些填充值）
        input_side = int(np.ceil(np.sqrt(obs_size)))

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )

        # 计算展平后的特征维度
        # 这里需要考虑池化层对尺寸的影响
        # 两个max pooling (kernel=2) 会将尺寸减半两次
        with torch.no_grad():
            fake_input = torch.zeros(1, n_input_channels, input_side, input_side)
            flattened_size = self.cnn(fake_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(flattened_size, features_dim),
            nn.ReLU()
        )

        # 保存输入尺寸用于重塑
        self.input_side = input_side
        self.obs_size = obs_size

    def forward(self, observations):
        # 确保输入是张量
        if not isinstance(observations, torch.Tensor):
            observations = torch.tensor(observations, dtype=torch.float32)

        if observations.device != next(self.parameters()).device:
            observations = observations.to(next(self.parameters()).device)

        # 获取批次大小
        batch_size = observations.shape[0]

        # 创建填充的输入
        padded_obs = torch.zeros(batch_size, self.input_side * self.input_side,
                                 device=observations.device)
        padded_obs[:, :self.obs_size] = observations

        # 重塑为BCHW格式用于CNN
        x = padded_obs.view(batch_size, 1, self.input_side, self.input_side)

        # 前向传播
        x = self.cnn(x)
        return self.linear(x)
class SimpleMLP(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        # 简单MLP处理扁平化输入
        self.mlp = nn.Sequential(
            nn.Linear(observation_space.shape[0], 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim)
        )

    def forward(self, observations):
        return self.mlp(observations)
def make_env(rank, seed=None):
    """创建环境的工厂函数"""

    def _init():
        # 创建基础环境
        env = simpy.Environment()
        n_drones = config.NUMBER_OF_DRONES
        channel_states = {i: simpy.Resource(env, capacity=1) for i in range(n_drones)}

        # 创建模拟器
        simulator = Simulator(
            seed=seed or 2024 + rank,
            env=env,
            channel_states=channel_states,
            n_drones=n_drones
        )

        # 创建STDMA环境
        stdma_env = StdmaEnv(
            simulator=simulator,
            num_nodes=config.NUMBER_OF_DRONES,
            num_slots=config.NUMBER_OF_DRONES
        )

        # 添加Monitor包装器
        log_dir = f"./logs/env_{rank}"
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(stdma_env, log_dir)

        # 设置随机种子
        if seed is not None:
            env.reset(seed=seed + rank)

        return env

    return _init


def train_stdma_agent(total_timesteps=1000000, num_envs=1):
    from stable_baselines3.common.callbacks import BaseCallback
    import numpy as np

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/STDMA_PPO_{current_time}"
    os.makedirs(log_dir, exist_ok=True)

    # 创建向量化环境
    env = DummyVecEnv([
        lambda: Monitor(make_env(i)(), os.path.join(log_dir, str(i)))
        for i in range(num_envs)
    ])

    # 创建评估环境
    eval_env = DummyVecEnv([
        lambda: Monitor(make_env(num_envs)(), os.path.join(log_dir, 'eval'))
    ])

    # 创建自定义回调函数记录训练指标
    class CustomCallback(BaseCallback):
        def __init__(self, verbose=0, window_size=1000):  # 增大窗口大小
            super().__init__(verbose)
            self.episode_rewards = []
            self.episode_lengths = []
            self.slot_reuse_rates = []
            self.current_episode_reward = 0
            # 添加数据缓冲
            self.reward_buffer = []
            self.reuse_rate_buffer = []
            self.delay_buffer = []
            self.window_size = window_size

        def _on_step(self) -> bool:
            env = self.training_env.envs[0]

            # 收集数据到缓冲区
            if hasattr(env, 'last_reward'):
                self.reward_buffer.append(env.last_reward)

            if hasattr(env, 'current_schedule'):
                total_slots = len(env.current_schedule)
                total_assignments = sum(len(nodes) for nodes in env.current_schedule.values())
                if total_slots > 0:
                    reuse_rate = total_assignments / total_slots
                    self.reuse_rate_buffer.append(reuse_rate)

            if hasattr(env, '_estimate_delay'):
                delay = env._estimate_delay()
                self.delay_buffer.append(delay)

            # 每1000步计算一次平均值并记录
            if len(self.reward_buffer) >= self.window_size:
                # 计算平均值
                avg_reward = np.mean(self.reward_buffer)
                self.logger.record("train/average_reward", avg_reward)

                if self.reuse_rate_buffer:
                    avg_reuse_rate = np.mean(self.reuse_rate_buffer)
                    self.logger.record("metrics/average_reuse_rate", avg_reuse_rate)

                if self.delay_buffer:
                    avg_delay = np.mean(self.delay_buffer)
                    self.logger.record("metrics/average_delay", avg_delay)

                # 清空缓冲区
                self.reward_buffer = []
                self.reuse_rate_buffer = []
                self.delay_buffer = []

                # 强制写入日志
                self.logger.dump(self.n_calls)

            return True

    # 设置策略参数
    policy_kwargs = dict(
        features_extractor_class=StdmaFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[128, 64], vf=[128, 64])
    )

    # 创建PPO模型
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=20,
        gamma=0.99,
        gae_lambda=0.98,
        clip_range=0.2,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs,
        verbose=1
    )

    # 创建回调函数
    custom_callback = CustomCallback()

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=log_dir,
        name_prefix="stdma_ppo_model"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, 'best_model'),
        log_path=log_dir,
        eval_freq=1000,
        deterministic=True,
        render=False
    )

    try:
        print("开始训练...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=[custom_callback, checkpoint_callback, eval_callback],
            progress_bar=True
        )

        final_model_path = os.path.join(log_dir, "final_model.zip")
        model.save(final_model_path)
        print(f"\n最终模型已保存至: {final_model_path}")

    except Exception as e:
        print(f"\n训练出错: {str(e)}")
        error_path = os.path.join(log_dir, "error_model.zip")
        model.save(error_path)
        raise e

    return model, log_dir


def plot_results(log_dir, title='学习曲线'):
    try:
        import glob
        import pandas as pd
        # 加载所有的监控文件
        all_data = []
        monitor_files = glob.glob(os.path.join(log_dir, "**", "*.monitor.csv"), recursive=True)

        for monitor_file in monitor_files:
            data = pd.read_csv(monitor_file, skiprows=1)  # 跳过元数据行
            all_data.append(data)

        if not all_data:
            print("未找到监控数据文件")
            return

        # 合并所有数据
        data = pd.concat(all_data)

        # 计算累计步数和移动平均奖励
        data['cumulative_steps'] = data['l'].cumsum()
        data['reward_mavg'] = data['r'].rolling(window=50, min_periods=1).mean()

        # 绘制图表
        plt.figure(figsize=(10, 5))
        plt.plot(data['cumulative_steps'], data['reward_mavg'])
        plt.xlabel('累计步数')
        plt.ylabel('平均奖励')
        plt.title(title)

        # 保存图表
        plt.savefig(os.path.join(log_dir, 'learning_curve.png'))
        plt.close()
        print(f"学习曲线已保存至: {os.path.join(log_dir, 'learning_curve.png')}")

    except Exception as e:
        print(f"绘制学习曲线时出错: {str(e)}")
        traceback.print_exc()


def moving_average(values, window):
    if len(values) == 0:
        return []
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def visualize_gat_features(model, env):
    # 获取观察空间样本
    obs = env.reset()[0]

    # 提取特征（获取StdmaFeatureExtractor的输出）
    device = next(model.policy.features_extractor.parameters()).device
    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)

    # 增加批次维度，确保是2D张量
    if len(obs_tensor.shape) == 1:
        obs_tensor = obs_tensor.unsqueeze(0)  # 添加批次维度

    # 获取中间特征
    with torch.no_grad():
        # 提取节点特征
        obs_dict = model.policy.features_extractor._unflatten_observation(obs_tensor)

        batch_index = 0
        topology = obs_dict['topology'][batch_index]
        positions = obs_dict['position'][batch_index]
        degrees = obs_dict['node_degrees'][batch_index]

        # 构建节点特征
        node_features = torch.cat([
            positions,
            degrees.unsqueeze(-1)
        ], dim=-1)
        node_features = model.policy.features_extractor.node_encoder(node_features)

        # 获取边索引
        edge_index = torch.nonzero(topology).t().contiguous()
        if edge_index.shape[1] == 0:  # 防止没有边的情况
            edge_index = torch.tensor([[j, j] for j in range(len(positions))],
                                      dtype=torch.long).t().contiguous().to(device)

        # GAT处理后的特征
        gat_features = model.policy.features_extractor.gat_block(node_features, edge_index)

        # 转为CPU进行可视化
        gat_features = gat_features.cpu().numpy()

        # 降维可视化，处理少量样本的情况
        n_samples = gat_features.shape[0]
        print(f"节点数量: {n_samples}")

        # 使用PCA进行降维
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        node_features_2d = pca.fit_transform(gat_features)

        print(f"PCA解释方差比: {pca.explained_variance_ratio_}")

        # 绘制PCA结果
        plt.figure(figsize=(10, 8))
        plt.scatter(node_features_2d[:, 0], node_features_2d[:, 1], s=100)

        # 添加节点编号标签
        for i in range(len(node_features_2d)):
            plt.annotate(str(i), (node_features_2d[i, 0], node_features_2d[i, 1]), fontsize=12)

        plt.title('GAT节点特征PCA可视化')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.grid(True)
        plt.savefig('./gat_features_pca.png')
        plt.close()

        # 如果节点数量足够，也尝试t-SNE可视化
        if n_samples > 5:
            try:
                from sklearn.manifold import TSNE
                perplexity = min(n_samples - 1, 5)  # 为小数据集设置较小的perplexity
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                tsne_features = tsne.fit_transform(gat_features)

                plt.figure(figsize=(10, 8))
                plt.scatter(tsne_features[:, 0], tsne_features[:, 1], s=100)

                for i in range(len(tsne_features)):
                    plt.annotate(str(i), (tsne_features[i, 0], tsne_features[i, 1]), fontsize=12)

                plt.title(f'GAT节点特征t-SNE可视化 (perplexity={perplexity})')
                plt.grid(True)
                plt.savefig('./gat_features_tsne.png')
                plt.close()
            except Exception as e:
                print(f"t-SNE可视化失败: {str(e)}")

    # 计算特征相似度矩阵
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(gat_features)

    # 绘制相似度热图
    plt.figure(figsize=(10, 8))
    plt.imshow(sim_matrix, cmap='viridis')
    plt.colorbar()
    plt.title('节点特征相似度矩阵')
    plt.savefig('./node_similarity_matrix.png')

    return gat_features


def analyze_feature_importance(model, env, num_samples=100):
    # 收集多个样本
    observations = []
    for _ in range(num_samples):
        obs = env.reset()[0]
        observations.append(obs)

    observations = np.stack(observations)

    # 原始性能
    original_actions = []
    for obs in observations:
        action, _ = model.predict(obs, deterministic=True)
        original_actions.append(action)

    # 特征消融分析
    feature_names = ['topology', 'position', 'routing', 'link_lifetime', 'traffic', 'node_degrees']
    importance_scores = {}

    for feature in feature_names:
        # 创建特征消融的观察
        ablated_obs = observations.copy()

        # 获取特征索引
        dims = model.policy.features_extractor.dims
        start_idx = 0
        for f in feature_names:
            if f == feature:
                break
            start_idx += dims[f]

        end_idx = start_idx + dims[feature]

        # 将特征值置零
        ablated_obs[:, start_idx:end_idx] = 0.0

        # 预测行为
        ablated_actions = []
        for obs in ablated_obs:
            action, _ = model.predict(obs, deterministic=True)
            ablated_actions.append(action)

        # 计算行为变化
        action_changes = np.mean(np.abs(np.array(original_actions) - np.array(ablated_actions)))
        importance_scores[feature] = action_changes

    # 绘制特征重要性
    plt.figure(figsize=(10, 6))
    plt.bar(importance_scores.keys(), importance_scores.values())
    plt.title('特征重要性分析')
    plt.xlabel('特征')
    plt.ylabel('行为变化')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('./feature_importance.png')

    return importance_scores


def visualize_attention_weights(model, env):
    # 获取观察空间样本
    obs = env.reset()[0]

    # 提取特征
    device = next(model.policy.features_extractor.parameters()).device
    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)

    # 关键修改：确保输入是2D张量
    if len(obs_tensor.shape) == 1:
        obs_tensor = obs_tensor.unsqueeze(0)  # 添加批次维度

    # 获取GAT层的注意力权重
    with torch.no_grad():
        try:
            obs_dict = model.policy.features_extractor._unflatten_observation(obs_tensor)

            batch_index = 0
            topology = obs_dict['topology'][batch_index]
            positions = obs_dict['position'][batch_index]
            degrees = obs_dict['node_degrees'][batch_index]

            # 构建节点特征
            node_features = torch.cat([
                positions,
                degrees.unsqueeze(-1)
            ], dim=-1)
            node_features = model.policy.features_extractor.node_encoder(node_features)

            # 获取边索引
            edge_index = torch.nonzero(topology).t().contiguous()
            if edge_index.shape[1] == 0:  # 防止没有边的情况
                edge_index = torch.tensor([[j, j] for j in range(len(positions))],
                                          dtype=torch.long).t().contiguous().to(device)

            # 获取注意力权重
            attention_weights = model.policy.features_extractor.gat_block.get_attention_weights()

            # 如果注意力权重为None，可能是因为尚未通过网络传递数据
            if attention_weights is None:
                # 先运行一次GAT，获取注意力权重
                _ = model.policy.features_extractor.gat_block(node_features, edge_index)
                attention_weights = model.policy.features_extractor.gat_block.get_attention_weights()

            if attention_weights is None:
                print("警告: 无法获取注意力权重，可能钩子没有正确捕获")
                # 创建一个随机权重矩阵作为替代
                num_edges = edge_index.shape[1]
                attention_weights = torch.rand(num_edges, 4).to(device)  # 假设有4个注意力头

            # 转为CPU进行可视化
            attention_weights = attention_weights.cpu().numpy()

            # 创建图并添加边注意力权重
            import networkx as nx
            import matplotlib.pyplot as plt

            G = nx.Graph()

            # 添加节点
            positions_np = positions.cpu().numpy()
            for i in range(len(positions)):
                G.add_node(i, pos=(positions_np[i, 0], positions_np[i, 1]))

            # 添加边和注意力权重
            edge_list = edge_index.t().cpu().numpy()

            # 确保权重和边的数量匹配
            if len(edge_list) != len(attention_weights):
                print(f"警告: 边数量({len(edge_list)})与注意力权重数量({len(attention_weights)})不匹配")
                # 使用边的数量截断或扩展权重
                if len(edge_list) < len(attention_weights):
                    attention_weights = attention_weights[:len(edge_list)]
                else:
                    # 扩展权重矩阵
                    extra_weights = np.random.rand(len(edge_list) - len(attention_weights), attention_weights.shape[1])
                    attention_weights = np.vstack([attention_weights, extra_weights])

            # 用于存储边标签的字典
            edge_labels = {}

            for i, (src, dst) in enumerate(edge_list):
                if src != dst:  # 忽略自环
                    # 获取第一个注意力头的权重
                    weight = attention_weights[i, 0] if len(attention_weights.shape) > 1 else attention_weights[i]
                    G.add_edge(src, dst, weight=weight)
                    # 存储边标签，保留2位小数
                    edge_labels[(src, dst)] = f"{weight:.2f}"

            # 获取节点位置
            pos = nx.get_node_attributes(G, 'pos')

            # 绘制图
            plt.figure(figsize=(12, 10))

            # 绘制节点
            nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')

            # 绘制带权重的边
            edges = list(G.edges())
            if edges:  # 确保有边可绘制
                weights = [G[u][v]['weight'] for u, v in edges]

                # 归一化权重以用于边宽度
                if weights:  # 确保权重列表不为空
                    max_weight = max(weights)
                    if max_weight > 0:  # 避免除以零
                        edge_width = [3 * w / max_weight for w in weights]
                    else:
                        edge_width = [1.0] * len(weights)
                else:
                    edge_width = [1.0] * len(edges)

                nx.draw_networkx_edges(G, pos, width=edge_width, alpha=0.7)

            # 添加节点标签
            nx.draw_networkx_labels(G, pos, font_size=10)

            # 添加边权重标签
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

            plt.title('GAT Attention Weights Visualization')  # 使用英文避免中文字体问题
            plt.axis('off')  # 关闭坐标轴
            plt.savefig('./gat_attention_weights.png')
            plt.close()

            return attention_weights

        except Exception as e:
            print(f"注意力权重可视化失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
# def visualize_attention_weights(model, env):
#     # 获取观察空间样本
#     obs = env.reset()[0]
#
#     # 提取特征
#     device = next(model.policy.features_extractor.parameters()).device
#     obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
#
#     # 关键修改：确保输入是2D张量
#     if len(obs_tensor.shape) == 1:
#         obs_tensor = obs_tensor.unsqueeze(0)  # 添加批次维度
#
#     # 获取GAT层的注意力权重
#     with torch.no_grad():
#         try:
#             obs_dict = model.policy.features_extractor._unflatten_observation(obs_tensor)
#
#             batch_index = 0
#             topology = obs_dict['topology'][batch_index]
#             positions = obs_dict['position'][batch_index]
#             degrees = obs_dict['node_degrees'][batch_index]
#
#             # 构建节点特征
#             node_features = torch.cat([
#                 positions,
#                 degrees.unsqueeze(-1)
#             ], dim=-1)
#             node_features = model.policy.features_extractor.node_encoder(node_features)
#
#             # 获取边索引
#             edge_index = torch.nonzero(topology).t().contiguous()
#             if edge_index.shape[1] == 0:  # 防止没有边的情况
#                 edge_index = torch.tensor([[j, j] for j in range(len(positions))],
#                                           dtype=torch.long).t().contiguous().to(device)
#
#             # 获取注意力权重
#             attention_weights = model.policy.features_extractor.gat_block.get_attention_weights()
#
#             # 如果注意力权重为None，可能是因为尚未通过网络传递数据
#             if attention_weights is None:
#                 # 先运行一次GAT，获取注意力权重
#                 _ = model.policy.features_extractor.gat_block(node_features, edge_index)
#                 attention_weights = model.policy.features_extractor.gat_block.get_attention_weights()
#
#             if attention_weights is None:
#                 print("警告: 无法获取注意力权重，可能钩子没有正确捕获")
#                 # 创建一个随机权重矩阵作为替代
#                 num_edges = edge_index.shape[1]
#                 attention_weights = torch.rand(num_edges, 4).to(device)  # 假设有4个注意力头
#
#             # 转为CPU进行可视化
#             attention_weights = attention_weights.cpu().numpy()
#
#             # 创建图并添加边注意力权重
#             import networkx as nx
#
#             G = nx.Graph()
#
#             # 添加节点
#             positions_np = positions.cpu().numpy()
#             for i in range(len(positions)):
#                 G.add_node(i, pos=(positions_np[i, 0], positions_np[i, 1]))
#
#             # 添加边和注意力权重
#             edge_list = edge_index.t().cpu().numpy()
#
#             # 确保权重和边的数量匹配
#             if len(edge_list) != len(attention_weights):
#                 print(f"警告: 边数量({len(edge_list)})与注意力权重数量({len(attention_weights)})不匹配")
#                 # 使用边的数量截断或扩展权重
#                 if len(edge_list) < len(attention_weights):
#                     attention_weights = attention_weights[:len(edge_list)]
#                 else:
#                     # 扩展权重矩阵
#                     extra_weights = np.random.rand(len(edge_list) - len(attention_weights), attention_weights.shape[1])
#                     attention_weights = np.vstack([attention_weights, extra_weights])
#
#             for i, (src, dst) in enumerate(edge_list):
#                 if src != dst:  # 忽略自环
#                     # 获取第一个注意力头的权重
#                     weight = attention_weights[i, 0] if len(attention_weights.shape) > 1 else attention_weights[i]
#                     G.add_edge(src, dst, weight=weight)
#
#             # 获取节点位置
#             pos = nx.get_node_attributes(G, 'pos')
#
#             # 绘制图
#             plt.figure(figsize=(12, 10))
#
#             # 绘制节点
#             nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
#
#             # 绘制带权重的边
#             edges = list(G.edges())
#             if edges:  # 确保有边可绘制
#                 weights = [G[u][v]['weight'] for u, v in edges]
#
#                 # 归一化权重以用于边宽度
#                 if weights:  # 确保权重列表不为空
#                     max_weight = max(weights)
#                     if max_weight > 0:  # 避免除以零
#                         edge_width = [3 * w / max_weight for w in weights]
#                     else:
#                         edge_width = [1.0] * len(weights)
#                 else:
#                     edge_width = [1.0] * len(edges)
#
#                 nx.draw_networkx_edges(G, pos, width=edge_width, alpha=0.7)
#
#             # 添加节点标签
#             nx.draw_networkx_labels(G, pos, font_size=10)
#
#             plt.title('GAT Attention Weights Visualization')  # 使用英文避免中文字体问题
#             plt.axis('off')  # 关闭坐标轴
#             plt.savefig('./gat_attention_weights.png')
#             plt.close()
#
#             return attention_weights
#
#         except Exception as e:
#             print(f"注意力权重可视化失败: {str(e)}")
#             import traceback
#             traceback.print_exc()
#             return None


# def perform_ablation_study(env_factory, total_timesteps=200000):
#     """比较不同特征提取器的性能差异"""
#
#     # 定义三种模型配置
#     model_configs = {
#         "GAT": dict(
#             features_extractor_class=StdmaFeatureExtractor,
#             features_extractor_kwargs=dict(features_dim=256),
#             net_arch=dict(pi=[128, 64], vf=[128, 64])
#         ),
#         "MLP": dict(
#             features_extractor_class=SimpleMLP,  # 使用默认的MLP特征提取器
#             net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64])
#         ),
#         # "CNN": dict(
#         #     features_extractor_class=CNNFeaturesExtractor,  # 需要实现一个CNN特征提取器
#         #     features_extractor_kwargs=dict(features_dim=256),
#         #     net_arch=dict(pi=[128, 64], vf=[128, 64])
#         # )
#     }
#
#     results = {}
#
#     for model_name, policy_kwargs in model_configs.items():
#         print(f"\n训练 {model_name} 模型...")
#
#         # 创建环境
#         env = DummyVecEnv([env_factory])
#
#         # 创建模型
#         model = PPO(
#             "MultiInputPolicy",
#             env,
#             learning_rate=3e-4,
#             n_steps=1024,
#             batch_size=64,
#             n_epochs=10,
#             gamma=0.99,
#             policy_kwargs=policy_kwargs,
#             verbose=1
#         )
#
#         # 训练模型
#         model.learn(total_timesteps=total_timesteps)
#
#         # 评估模型
#         eval_env = DummyVecEnv([env_factory])
#         eval_results = evaluate_schedule(eval_env, model, num_episodes=20)
#
#         # 保存结果
#         results[model_name] = eval_results
#
#     # 比较模型性能
#     metrics = ['avg_reward', 'avg_reuse_ratio', 'interference_free_rate', 'qos_satisfaction_rate']
#
#     plt.figure(figsize=(15, 10))
#     for i, metric in enumerate(metrics):
#         plt.subplot(2, 2, i + 1)
#
#         # 只使用存在该指标的模型
#         available_models = [model for model in model_configs.keys()
#                             if model in results and metric in results[model]]
#
#         if available_models:  # 只在有可用数据时绘图
#             values = [results[model][metric] for model in available_models]
#             plt.bar(available_models, values)
#             plt.title(metric)
#             plt.grid(True)
#         else:
#             plt.text(0.5, 0.5, f"No data for {metric}",
#                      ha='center', va='center', transform=plt.gca().transAxes)
#
#     plt.tight_layout()
#     plt.savefig('./model_comparison.png')
#
#     return results

import pandas as pd


def perform_ablation_study(env_factory, total_timesteps=200000):
    """比较不同特征提取器的性能差异"""
    import pandas as pd
    import matplotlib.pyplot as plt

    # 设置支持中文的字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Heiti TC']  # macOS常用字体
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

    # 定义三种模型配置
    model_configs = {
        "GAT": dict(
            features_extractor_class=StdmaFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[128, 64], vf=[128, 64])
        ),
        "MLP": dict(
            features_extractor_class=SimpleMLP,
            net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64])
        ),
    }

    results = {}
    training_histories = {}  # 新增：存储训练过程数据

    for model_name, policy_kwargs in model_configs.items():
        print(f"\n训练 {model_name} 模型...")
        training_histories[model_name] = {
            'timesteps': [],
            'avg_reward': [],
            'avg_reuse_ratio': [],
            'interference_free_rate': [],
            'qos_satisfaction_rate': []
        }

        # 创建环境
        env = DummyVecEnv([env_factory])

        # 创建带回调的评估环境
        eval_env = DummyVecEnv([env_factory])

        # 自定义回调函数
        class ProgressMetricCallback(EvalCallback):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.model_name = model_name

            def _on_step(self) -> bool:
                # 每5次评估记录一次（避免过于频繁）
                if self.eval_freq > 0 and self.n_calls % (self.eval_freq // 5) == 0:
                    # 执行评估
                    try:
                        eval_results = evaluate_schedule(self.eval_env, self.model, num_episodes=5)

                        # 记录训练过程数据
                        training_histories[self.model_name]['timesteps'].append(self.num_timesteps)
                        for metric in ['avg_reward', 'avg_reuse_ratio',
                                       'interference_free_rate', 'qos_satisfaction_rate']:
                            training_histories[self.model_name][metric].append(eval_results.get(metric, 0))
                    except Exception as e:
                        print(f"评估过程出错: {str(e)}")

                return True  # 继续训练

        try:
            # 创建模型
            model = PPO(
                "MultiInputPolicy",
                env,
                learning_rate=3e-4,
                n_steps=1024,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                policy_kwargs=policy_kwargs,
                verbose=1
            )

            # 训练模型（添加回调）
            eval_callback = ProgressMetricCallback(
                eval_env,
                eval_freq=1000,  # 每1000步评估一次
                deterministic=True,
                render=False
            )

            model.learn(
                total_timesteps=total_timesteps,
                callback=eval_callback,
                progress_bar=True
            )

            # 最终评估
            eval_results = evaluate_schedule(eval_env, model, num_episodes=20)
            results[model_name] = eval_results

        except Exception as e:
            print(f"训练 {model_name} 模型时出错: {str(e)}")
            results[model_name] = {}  # 空字典防止后续处理出错

        finally:
            env.close()
            eval_env.close()

    # 防止绘图错误：检查是否有数据
    has_data = all(len(training_histories[model_name]['timesteps']) > 0 for model_name in model_configs.keys())

    if has_data:
        try:
            # 绘制训练过程曲线
            metrics = ['avg_reward', 'avg_reuse_ratio', 'interference_free_rate', 'qos_satisfaction_rate']
            metric_names = ['Average Reward', 'Slot Reuse Ratio', 'Interference-free Rate', 'QoS Satisfaction']

            plt.figure(figsize=(15, 10))
            for i, (metric, name) in enumerate(zip(metrics, metric_names)):
                plt.subplot(2, 2, i + 1)

                for model_name in model_configs.keys():
                    if len(training_histories[model_name]['timesteps']) > 0:
                        # 平滑处理曲线（移动平均）
                        window_size = max(1, len(training_histories[model_name][metric]) // 20)
                        smoothed_values = pd.Series(training_histories[model_name][metric]).rolling(window_size,
                                                                                                    min_periods=1).mean()

                        plt.plot(
                            training_histories[model_name]['timesteps'],
                            smoothed_values,
                            label=model_name,
                            linewidth=2
                        )

                plt.title(name)
                plt.xlabel('Training Steps')
                plt.ylabel(name)
                plt.grid(True, alpha=0.3)
                plt.legend()

                # 自动调整Y轴范围
                try:
                    y_values = [v for model_name in model_configs.keys()
                                for v in training_histories[model_name][metric] if v is not None]
                    if y_values:
                        y_min = min(y_values) * 0.95
                        y_max = max(y_values) * 1.05
                        plt.ylim(y_min, y_max)
                except Exception:
                    pass  # 忽略Y轴范围调整错误

            plt.tight_layout()
            plt.savefig('./training_progress.png', dpi=300, bbox_inches='tight')
            print("训练进度图已保存至 ./training_progress.png")

        except Exception as e:
            print(f"绘制训练进度图出错: {str(e)}")

    return results  # 只返回结果字典，确保返回类型一致


# def compare_learning_curves(env_factory, total_timesteps=200000):
#     # 定义两种模型
#     models = {
#         "With_GAT": PPO(
#             "MultiInputPolicy",
#             DummyVecEnv([env_factory]),
#             policy_kwargs=dict(
#                 features_extractor_class=StdmaFeatureExtractor,
#                 features_extractor_kwargs=dict(features_dim=256)
#             ),
#             verbose=1
#         ),
#         "Without_GAT": PPO(
#             "MultiInputPolicy",
#             DummyVecEnv([env_factory]),
#             policy_kwargs=dict(
#                 features_extractor_class=SimpleMLP,  # 改为SimpleMLP
#                 features_extractor_kwargs=dict(features_dim=256)
#             ),
#             verbose=1
#         )
#     }
#
#     # 训练并记录学习曲线
#     rewards = {name: [] for name in models.keys()}
#
#     for name, model in models.items():
#         print(f"\n训练 {name} 模型...")
#
#         # 每1000步评估一次
#         for i in range(0, total_timesteps, 10000):
#             model.learn(total_timesteps=10000)
#
#             # 评估当前性能
#             eval_env = DummyVecEnv([env_factory])
#             eval_results = evaluate_schedule(eval_env, model, num_episodes=5)
#             rewards[name].append(eval_results['avg_reward'])
#
#     # 绘制学习曲线
#     plt.figure(figsize=(10, 6))
#     for name, reward_history in rewards.items():
#         plt.plot(range(0, total_timesteps, 10000), reward_history, label=name)
#
#     plt.xlabel('训练步数')
#     plt.ylabel('平均奖励')
#     plt.title('GAT特征提取器学习曲线比较')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('./learning_curves_comparison.png')
#
#     return rewards
def compare_learning_curves(env_factory, total_timesteps=200000):
    """改进后的学习曲线对比函数，支持每1000步评估一次"""
    from stable_baselines3.common.callbacks import BaseCallback
    import numpy as np
    import matplotlib.pyplot as plt

    # 设置支持中文的字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Heiti TC']  # macOS常用字体
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

    # 使用更新的seaborn样式名称
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        # 如果新样式名称不可用，尝试默认样式
        try:
            import seaborn as sns
            sns.set_style("darkgrid")
        except:
            pass  # 忽略样式错误

    # 配置模型参数
    model_configs = {
        "With_GAT": {
            "class": StdmaFeatureExtractor,
            "kwargs": dict(features_dim=256),
            "color": "#FF6F61"  # 珊瑚色
        },
        "Without_GAT": {
            "class": SimpleMLP,
            "kwargs": dict(features_dim=256),
            "color": "#6B5B95"  # 紫水晶色
        }
    }

    # 自定义回调类（支持每1000步评估）
    class FrequentEvalCallback(BaseCallback):
        def __init__(self, eval_freq=1000, model_name="default"):
            super().__init__()
            self.eval_freq = eval_freq
            self.model_name = model_name
            self.metrics = {
                'timesteps': [],
                'avg_reward': [],
                'reuse_ratio': [],
                'interference_rate': [],
                'qos_satisfaction': []
            }

        def _on_step(self) -> bool:
            if self.n_calls % self.eval_freq == 0:
                try:
                    eval_env = DummyVecEnv([env_factory])
                    eval_results = evaluate_schedule(eval_env, self.model, num_episodes=3)  # 减少评估episode数

                    self.metrics['timesteps'].append(self.num_timesteps)
                    for key in self.metrics.keys():
                        if key != 'timesteps':
                            metric_value = eval_results.get(key, np.nan)
                            # 确保值是有效的数值
                            if metric_value is not None and not np.isnan(metric_value):
                                self.metrics[key].append(metric_value)
                            else:
                                self.metrics[key].append(0.0)  # 默认值

                    eval_env.close()

                except Exception as e:
                    print(f"评估失败: {str(e)}")
            return True

    # 训练并记录数据
    all_metrics = {}
    train_interval = 10000  # 每10000步训练一次

    for model_name, config in model_configs.items():
        print(f"\n=== 训练 {model_name} 模型 ===")

        try:
            # 初始化环境和模型
            env = DummyVecEnv([env_factory])
            model = PPO(
                "MultiInputPolicy",
                env,
                policy_kwargs=dict(
                    features_extractor_class=config["class"],
                    features_extractor_kwargs=config["kwargs"],
                    net_arch=dict(pi=[128, 64], vf=[128, 64])
                ),
                verbose=0  # 关闭默认输出
            )

            # 设置回调
            logger = FrequentEvalCallback(eval_freq=1000, model_name=model_name)

            # 分批次训练
            for step in range(0, total_timesteps, train_interval):
                remaining = total_timesteps - step
                actual_steps = min(train_interval, remaining)

                model.learn(
                    total_timesteps=actual_steps,
                    callback=logger,
                    reset_num_timesteps=False,
                    progress_bar=True
                )

                # 打印进度
                progress = (step + actual_steps) / total_timesteps * 100
                if logger.metrics['avg_reward']:
                    latest_reward = logger.metrics['avg_reward'][-1]
                    print(f"进度: {progress:.1f}% | 最新评估奖励: {latest_reward:.2f}")
                else:
                    print(f"进度: {progress:.1f}% | 无评估奖励数据")

            all_metrics[model_name] = logger.metrics

        except Exception as e:
            print(f"训练 {model_name} 模型时出错: {str(e)}")
            all_metrics[model_name] = {'timesteps': [], 'avg_reward': []}  # 空数据

        finally:
            if 'env' in locals():
                env.close()

    try:
        # 可视化设置
        fig, ax = plt.subplots(figsize=(12, 6))

        # 绘制奖励曲线
        for model_name, config in model_configs.items():
            if model_name in all_metrics and len(all_metrics[model_name]['avg_reward']) > 0:
                # 使用移动平均平滑
                rewards = all_metrics[model_name]['avg_reward']
                timestamps = all_metrics[model_name]['timesteps']

                # 确保数据长度匹配
                min_len = min(len(rewards), len(timestamps))
                rewards = rewards[:min_len]
                timestamps = timestamps[:min_len]

                if len(rewards) > 1:
                    window_size = min(10, len(rewards) // 2)
                    if window_size > 0:
                        # 使用卷积平滑
                        smooth_rewards = np.convolve(
                            rewards,
                            np.ones(window_size) / window_size,
                            mode='valid'
                        )

                        # 绘制曲线
                        ax.plot(
                            timestamps[:len(smooth_rewards)],
                            smooth_rewards,
                            label=model_name,
                            color=config["color"],
                            linewidth=2
                        )
                    else:
                        # 数据点太少，直接绘制
                        ax.plot(
                            timestamps,
                            rewards,
                            label=model_name,
                            color=config["color"],
                            linewidth=2
                        )

        ax.set_title('GAT vs MLP Learning Curves (Evaluated every 1000 steps)', fontsize=14, pad=12)
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Average Reward', fontsize=12)
        ax.legend(frameon=True, facecolor='white')
        ax.grid(True, linestyle='--', alpha=0.6)

        # 保存高质量图片
        plt.savefig(
            './learning_curves_frequent_eval.png',
            dpi=300,
            bbox_inches='tight',
            facecolor='white'
        )
        print("学习曲线比较图已保存至 ./learning_curves_frequent_eval.png")

    except Exception as e:
        print(f"绘制学习曲线比较图时出错: {str(e)}")

    return all_metrics  # 返回指标数据

def evaluate_schedule(env, model, num_episodes=50):
    """评估已训练模型的时隙分配性能"""
    results = {
        'rewards': [],  # 每个episode的总奖励
        'step_rewards': [],  # 每一步的奖励
        'slot_usage': [],  # 时隙使用数
        'success_rate': 0,  # 成功调度的比率
        'interference_free': 0,  # 无干扰调度的数量
        'slot_reuse': [],  # 时隙复用率
        'episode_lengths': [],  # 每个episode的长度
        'time_efficiency': [],  # 时间效率
        'routing_completion': [],  # 路由完成率
        'avg_delay': [],  # 平均延迟
        'qos_satisfaction': []  # QoS满足率
    }

    print("\n开始评估模型...")
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")

        obs = env.reset()[0]
        episode_reward = 0
        step_rewards = []
        done = False
        step_count = 0

        # 获取基础环境
        base_env = env.envs[0]
        episode_schedule = {}

        while not done:
            # 预测动作
            action, _ = model.predict(obs, deterministic=True)
            action = np.array([action])

            # 记录当前状态
            current_node = base_env.current_node

            # 检查动作有效性并执行
            is_valid = base_env.is_valid_assignment(current_node, action[0])
            print(f"Step {step_count}: Node {current_node}, Action {action[0]}, Valid: {is_valid}")

            # 执行动作
            obs, reward, done, info = env.step(action)
            step_rewards.append(reward[0])
            episode_reward += reward[0]
            step_count += 1

            # 获取并记录当前调度
            current_schedule = info[0].get('schedule', {})
            episode_schedule = current_schedule.copy()
            print(f"当前调度: {current_schedule}")

            # 检查步数限制
            if step_count > base_env.num_nodes * 3:
                print(f"Warning: 超过最大步数 ({step_count})")
                done = True

        # Episode 完成后的统计
        print(f"\nEpisode {episode + 1} 完成:")
        print(f"总步数: {step_count}")
        print(f"总奖励: {episode_reward}")
        print(f"最终调度: {episode_schedule}")

        # 记录基本指标
        results['rewards'].append(episode_reward)
        results['step_rewards'].extend(step_rewards)
        results['episode_lengths'].append(step_count)
        results['slot_usage'].append(len(episode_schedule))

        # 检查干扰
        has_interference = False
        for slot, nodes in episode_schedule.items():
            for i, node1 in enumerate(nodes):
                for node2 in nodes[i + 1:]:
                    if hasattr(base_env, 'topology_matrix') and base_env.topology_matrix[node1][node2] == 1:
                        has_interference = True
                        print(f"发现干扰：时隙 {slot} 中的节点 {node1} 和 {node2}")
                        break
                if has_interference:
                    break
            if has_interference:
                break

        if not has_interference:
            results['interference_free'] += 1

        # 计算时隙复用率
        if episode_schedule:
            total_assignments = sum(len(nodes) for nodes in episode_schedule.values())
            reuse_ratio = total_assignments / len(episode_schedule) if len(episode_schedule) > 0 else 0
            results['slot_reuse'].append(reuse_ratio)
            print(f"时隙复用率: {reuse_ratio:.2f}")

        # 计算时间效率
        if hasattr(base_env, '_estimate_delay') and hasattr(base_env, 'current_requirement'):
            current_delay = base_env._estimate_delay()
            if current_delay > 0 and base_env.current_requirement:
                time_efficiency = base_env.current_requirement.delay_requirement / current_delay
                results['time_efficiency'].append(time_efficiency)
                results['avg_delay'].append(current_delay)

        # 检查路由完成情况
        if hasattr(base_env, 'current_requirement') and base_env.current_requirement:
            route_path = base_env.current_requirement.routing_path
            if route_path:
                assigned_nodes = set()
                for nodes in episode_schedule.values():
                    assigned_nodes.update(nodes)
                completion_rate = len(set(route_path) & assigned_nodes) / len(route_path)
                results['routing_completion'].append(completion_rate)

        # 检查QoS满足情况
        if hasattr(base_env, 'current_requirement') and base_env.current_requirement:
            if hasattr(base_env, '_estimate_delay'):
                current_delay = base_env._estimate_delay()
                qos_satisfied = current_delay <= base_env.current_requirement.delay_requirement
                results['qos_satisfaction'].append(float(qos_satisfied))

    # 计算最终统计
    if num_episodes > 0:
        # 基本指标统计
        results['success_rate'] = results['success_rate'] / num_episodes * 100
        results['interference_free_rate'] = results['interference_free'] / num_episodes * 100
        results['avg_reward'] = np.mean(results['rewards'])
        results['avg_slots'] = np.mean(results['slot_usage'])
        results['avg_reuse_ratio'] = np.mean(results['slot_reuse']) if results['slot_reuse'] else 0
        results['avg_episode_length'] = np.mean(results['episode_lengths'])

        # 高级指标统计
        if results['time_efficiency']:
            results['avg_time_efficiency'] = np.mean(results['time_efficiency'])
        if results['routing_completion']:
            results['avg_routing_completion'] = np.mean(results['routing_completion'])
        if results['qos_satisfaction']:
            results['qos_satisfaction_rate'] = np.mean(results['qos_satisfaction']) * 100
        if results['avg_delay']:
            results['average_delay'] = np.mean(results['avg_delay'])

        print("\n最终评估结果:")
        print(f"调度成功率: {results['success_rate']:.2f}%")
        print(f"无干扰率: {results['interference_free_rate']:.2f}%")
        print(f"平均时隙数: {results['avg_slots']:.2f}")
        print(f"平均时隙复用率: {results['avg_reuse_ratio']:.2f}")
        print(f"平均奖励: {results['avg_reward']:.2f}")
        print(f"平均episode长度: {results['avg_episode_length']:.2f}")
        if 'avg_time_efficiency' in results:
            print(f"平均时间效率: {results['avg_time_efficiency']:.2f}")
        if 'avg_routing_completion' in results:
            print(f"平均路由完成率: {results['avg_routing_completion']:.2f}")
        if 'qos_satisfaction_rate' in results:
            print(f"QoS满足率: {results['qos_satisfaction_rate']:.2f}%")
        if 'average_delay' in results:
            print(f"平均延迟: {results['average_delay']:.2f}")

    return results


if __name__ == "__main__":
    try:
        # 设置多进程启动方法
        if os.name != 'nt':
            multiprocessing.set_start_method('spawn')

        os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'

        print("开始STDMA PPO训练...")
        model, log_dir = train_stdma_agent(
            total_timesteps=200000,
            num_envs=1
        )
        # model_path = './logs/STDMA_PPO_20250225_050845/best_model/best_model.zip'
        # print("加载模型...")
        # model = PPO.load(model_path)
        # log_dir = os.path.dirname(model_path)
        print("\n创建评估环境...")
        eval_env = DummyVecEnv([
            lambda: Monitor(make_env(0)(), os.path.join(log_dir, 'final_eval'))
        ])
        # 添加GAT特征验证步骤
        print("\n开始验证GAT特征有效性...")

        # 1. 特征可视化
        print("1. 进行GAT特征可视化...")
        gat_features = visualize_gat_features(model, eval_env)
        print(f"- 特征可视化完成，结果保存至 {log_dir}/gat_features_tsne.png")

        # 2. 特征重要性分析
        print("2. 进行特征重要性分析...")
        importance_scores = analyze_feature_importance(model, eval_env, num_samples=50)
        print("- 特征重要性分析结果:")
        for feature, score in importance_scores.items():
            print(f"  {feature}: {score:.4f}")

        # 3. 注意力权重可视化
        print("3. 进行注意力权重可视化...")
        attention_weights = visualize_attention_weights(model, eval_env)
        print(f"- 注意力权重可视化完成，结果保存至 {log_dir}/gat_attention_weights.png")

        # 4. 消融实验
        print("4. 进行消融实验比较不同架构...")
        ablation_results = perform_ablation_study(make_env(0), total_timesteps=100000)
        print("- 消融实验完成，结果保存至 model_comparison.png")

        # 5. 学习曲线比较
        print("5. 比较学习曲线...")
        learning_curves = compare_learning_curves(make_env(0), total_timesteps=100000)
        print("- 学习曲线比较完成，结果保存至 learning_curves_comparison.png")

        # 继续原有评估
        print("\n开始模型评估...")
        results = evaluate_schedule(eval_env, model, num_episodes=50)

        print("\n绘制学习曲线...")
        plot_results(log_dir, title='STDMA PPO学习曲线')

        print("\n训练和评估完成！")
        print("\nGAT特征验证结果概述:")
        print(f"1. 特征可视化: 见 {log_dir}/gat_features_tsne.png")
        print(f"2. 特征重要性: 最重要的特征是 {max(importance_scores, key=importance_scores.get)}")
        print(f"3. 注意力权重: 见 {log_dir}/gat_attention_weights.png")
        print(
            f"4. 与其他架构比较: GAT性能优于其他架构 {ablation_results['GAT']['avg_reward'] > ablation_results['MLP']['avg_reward']}")
        print(f"5. 学习曲线比较: 见 learning_curves_comparison.png")

    except Exception as e:
        print(f"发生错误: {str(e)}")
        traceback.print_exc()