import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from datetime import datetime
import io
from PIL import Image
from collections import deque
from torch_geometric.nn import GATConv, global_mean_pool

# 导入自定义环境
from dynamic_env import DynamicStdmaEnv


class GATBlock(nn.Module):
    """GAT块，类似于GNN模型中的DynamicGATBlock"""

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
        """前向传播"""
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


class GNNFeatureExtractor(nn.Module):
    """基于图神经网络的特征提取器"""

    def __init__(self, max_nodes, output_dim=256):
        super().__init__()
        self.max_nodes = max_nodes
        self.node_feat_dim = 32
        self.hidden_dim = 64
        self.output_dim = output_dim

        # 节点特征编码器
        self.node_encoder = nn.Sequential(
            nn.Linear(4, self.node_feat_dim),  # 4 = [x,y,z,度]
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

        # 交通信息编码器
        self.traffic_encoder = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.LayerNorm(32)
        )

        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim + 32, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )

    def parse_observation(self, flat_obs):
        """解析扁平化的观察，转换为结构化数据"""
        # 确保我们处理的是numpy数组
        if isinstance(flat_obs, torch.Tensor):
            flat_obs_np = flat_obs.detach().cpu().numpy()
        else:
            flat_obs_np = flat_obs

        # 从填充观察中提取有效数据，估计实际节点数量
        non_zero_count = np.sum(flat_obs_np != 0)
        estimated_nodes = int(np.sqrt(non_zero_count / 4))  # 4是因为拓扑和链路生命期矩阵
        estimated_nodes = max(5, min(estimated_nodes, self.max_nodes))  # 限制在合理范围内

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

            # 全局池化
            batch_index = torch.zeros(node_features.size(0), dtype=torch.long, device=node_features.device)
            pooled = global_mean_pool(node_features, batch_index)

            # 处理交通信息
            traffic_features = self.traffic_encoder(obs_dict['traffic_info'])

            # 确保维度匹配
            if pooled.dim() > 1 and traffic_features.dim() == 1:
                traffic_features = traffic_features.unsqueeze(0)

            # 合并特征
            combined = torch.cat([pooled, traffic_features], dim=1)
            results.append(combined)

        # 堆叠批次结果
        stacked = torch.cat(results, dim=0)

        # 特征融合
        return self.fusion(stacked)


class DQNNetwork(nn.Module):
    """DQN网络结构"""

    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.feature_extractor = None  # 将在DQNAgent中设置

        # Q-网络层
        self.fc1 = nn.Linear(feature_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        """前向传播"""
        if self.feature_extractor is not None:
            x = self.feature_extractor(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    """DQN智能体实现"""

    def __init__(
            self,
            env,
            feature_dim=256,
            learning_rate=1e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.995,
            buffer_size=100000,
            batch_size=32,
            target_update_freq=1000,
            max_nodes=50,
            device=None
    ):
        self.env = env
        self.action_dim = env.action_space.n
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq

        # 设置设备
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 创建特征提取器
        self.feature_extractor = GNNFeatureExtractor(max_nodes=max_nodes, output_dim=feature_dim).to(self.device)

        # 创建Q网络和目标网络
        self.q_network = DQNNetwork(feature_dim, self.action_dim).to(self.device)
        self.target_network = DQNNetwork(feature_dim, self.action_dim).to(self.device)
        self.q_network.feature_extractor = self.feature_extractor
        self.target_network.feature_extractor = self.feature_extractor

        # 复制参数到目标网络
        self.update_target_network()

        # 设置优化器
        self.optimizer = optim.Adam(list(self.q_network.parameters()) + list(self.feature_extractor.parameters()),
                                    lr=learning_rate)

        # 创建经验回放缓冲区
        self.replay_buffer = deque(maxlen=buffer_size)

        # 设置探索参数
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # 训练计数器
        self.train_step_counter = 0
        self.episode_counter = 0

        # 性能跟踪
        self.episode_rewards = []
        self.reuse_ratios = []
        self.slot_counts = []

    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state, evaluate=False):
        """选择动作，使用epsilon-greedy策略"""
        if (not evaluate) and random.random() < self.epsilon:
            return self.env.action_space.sample()

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()

    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def add_to_buffer(self, state, action, reward, next_state, done):
        """添加经验到缓冲区"""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train_step(self):
        """执行一步训练"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # 从缓冲区采样
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # 转换为张量
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # 计算损失
        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        torch.nn.utils.clip_grad_norm_(self.feature_extractor.parameters(), 10.0)
        self.optimizer.step()

        # 定期更新目标网络
        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def train(self, total_timesteps, callback=None):
        """训练智能体"""
        timestep = 0
        losses = []

        while timestep < total_timesteps:
            state, _ = self.env.reset()
            episode_reward = 0
            done = False

            while not done and timestep < total_timesteps:
                # 选择动作
                action = self.select_action(state)

                # 执行动作
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # 保存经验
                self.add_to_buffer(state, action, reward, next_state, done)

                # 训练网络
                loss = self.train_step()
                if loss > 0:
                    losses.append(loss)

                # 更新状态
                state = next_state
                episode_reward += reward
                timestep += 1

                # 更新进度条
                if timestep % 1000 == 0:
                    avg_loss = np.mean(losses[-100:]) if losses else 0
                    print(f"步数: {timestep}/{total_timesteps}, Epsilon: {self.epsilon:.4f}, Loss: {avg_loss:.4f}")

                # 执行回调函数
                if callback:
                    callback.update(self, self.env, timestep, info, done)

            # 回合结束
            self.episode_counter += 1
            self.episode_rewards.append(episode_reward)

            # 计算复用率
            schedule = info.get('schedule', {})
            if schedule:
                slots_used = len(schedule)
                total_assignments = sum(len(nodes) for nodes in schedule.values())
                reuse_ratio = total_assignments / slots_used if slots_used > 0 else 0
                self.reuse_ratios.append(reuse_ratio)
                self.slot_counts.append(slots_used)

                # 打印回合信息
                if self.episode_counter % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    avg_reuse = np.mean(self.reuse_ratios[-10:]) if self.reuse_ratios else 0
                    print(
                        f"Episode {self.episode_counter}: Avg Reward = {avg_reward:.2f}, Avg Reuse = {avg_reuse:.2f}, Epsilon = {self.epsilon:.4f}")

            # 衰减探索率
            self.decay_epsilon()

        return self.episode_rewards, self.reuse_ratios, self.slot_counts

    def save(self, path):
        """保存模型"""
        torch.save({
            'feature_extractor': self.feature_extractor.state_dict(),
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        print(f"模型已保存到 {path}")

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        print(f"从 {path} 加载模型")

    def evaluate(self, env, episodes=10):
        """评估模型性能"""
        rewards = []
        reuse_ratios = []
        slot_counts = []

        for ep in range(episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = self.select_action(state, evaluate=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                state = next_state
                episode_reward += reward

            rewards.append(episode_reward)

            # 计算复用率
            schedule = info.get('schedule', {})
            slots_used = len(schedule)
            total_assignments = sum(len(nodes) for nodes in schedule.values())
            reuse_ratio = total_assignments / slots_used if slots_used > 0 else 0

            reuse_ratios.append(reuse_ratio)
            slot_counts.append(slots_used)

            print(f"评估 Episode {ep + 1}: 奖励={episode_reward:.2f}, 复用率={reuse_ratio:.2f}, 使用时隙={slots_used}")

        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_reuse_ratio': np.mean(reuse_ratios),
            'std_reuse_ratio': np.std(reuse_ratios),
            'mean_slots': np.mean(slot_counts),
            'std_slots': np.std(slot_counts)
        }


class DQNCallback:
    """DQN训练回调函数"""

    def __init__(self, node_size_range, max_nodes, log_dir=None, verbose=0):
        self.node_size_range = node_size_range
        self.max_nodes = max_nodes
        self.log_dir = log_dir
        self.verbose = verbose

        self.episode_count = 0
        self.rewards = []
        self.reuse_ratios = []
        self.slot_counts = []
        self.episode_lengths = []
        self.schedules = []
        self.network_sizes = []
        self.losses = []

        # 创建日志目录
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def update(self, agent, env, timestep, info, done):
        """更新回调状态"""
        # 当回合结束时
        if done:
            # 获取环境信息
            schedule = info.get('schedule', {})
            if schedule:
                self.schedules.append(schedule)

                # 计算复用率
                slot_count = len(schedule)
                total_assignments = sum(len(nodes) for nodes in schedule.values())
                reuse_ratio = total_assignments / slot_count if slot_count > 0 else 0

                # 记录统计
                self.slot_counts.append(slot_count)
                self.reuse_ratios.append(reuse_ratio)
                self.network_sizes.append(env.num_nodes)

                # 每10个回合记录日志
                self.episode_count += 1
                if self.verbose > 0 and self.episode_count % 10 == 0:
                    if len(agent.episode_rewards) >= 10:
                        avg_reward = np.mean(agent.episode_rewards[-10:])
                        avg_reuse = np.mean(self.reuse_ratios[-10:]) if self.reuse_ratios else 0
                        print(
                            f"Episode {self.episode_count}: Avg Reward = {avg_reward:.2f}, Avg Reuse = {avg_reuse:.2f}, Next Nodes = {env.num_nodes}")

                # 每50个回合生成可视化图表
                if self.episode_count % 50 == 0 and self.log_dir:
                    self._save_visualizations(agent, env)

                # 随机改变下一个回合的节点数量
                new_nodes = np.random.randint(self.node_size_range[0], self.node_size_range[1] + 1)
                env.num_nodes = new_nodes

    def _save_visualizations(self, agent, env):
        """保存训练可视化图表"""
        if len(agent.episode_rewards) < 10:
            return

        # 创建图表目录
        charts_dir = os.path.join(self.log_dir, "charts")
        os.makedirs(charts_dir, exist_ok=True)

        # 1. 奖励曲线
        plt.figure(figsize=(10, 6))
        plt.plot(agent.episode_rewards)
        plt.title('训练奖励')
        plt.xlabel('回合')
        plt.ylabel('回合奖励')
        plt.grid(True)
        plt.savefig(os.path.join(charts_dir, f"rewards_{self.episode_count}.png"))
        plt.close()

        # 2. 复用率曲线
        if self.reuse_ratios:
            plt.figure(figsize=(10, 6))
            plt.plot(self.reuse_ratios)
            plt.title('时隙复用率')
            plt.xlabel('回合')
            plt.ylabel('复用率')
            plt.grid(True)
            plt.savefig(os.path.join(charts_dir, f"reuse_ratio_{self.episode_count}.png"))
            plt.close()

        # 3. 时隙分配可视化 (最新的调度)
        if self.schedules:
            schedule = self.schedules[-1]
            fig, ax = plt.subplots(figsize=(10, 6))

            # 为每个时隙创建一个条形，高度是节点数量
            slots = []
            nodes_count = []

            for slot, nodes in sorted(schedule.items()):
                slots.append(slot)
                nodes_count.append(len(nodes))

            # 绘制条形图
            bars = ax.bar(slots, nodes_count, alpha=0.7)

            # 在每个条形顶部标记节点ID
            for i, (slot, nodes) in enumerate(sorted(schedule.items())):
                node_labels = ', '.join(str(n) for n in nodes)
                ax.text(i, nodes_count[i], node_labels, ha='center', va='bottom')

            # 设置标题和标签
            ax.set_title(f'Slot Assignment (Network Size: {env.num_nodes})')
            ax.set_xlabel('Slot Index')
            ax.set_ylabel('Number of Nodes')
            ax.set_xticks(slots)
            ax.set_ylim(0, max(nodes_count) + 1.5 if nodes_count else 1)

            # 绘制平均高度线
            avg_nodes = sum(nodes_count) / len(slots) if slots else 0
            ax.axhline(y=avg_nodes, color='r', linestyle='--', label=f'Avg: {avg_nodes:.2f}')
            ax.legend()

            # 保存图表
            plt.savefig(os.path.join(charts_dir, f"slot_assignment_{self.episode_count}.png"))
            plt.close(fig)

    def save_final_report(self, agent, save_path):
        """保存最终训练报告"""
        # 创建图表
        plt.figure(figsize=(12, 10))

        # 1. 奖励曲线
        plt.subplot(2, 2, 1)
        plt.plot(agent.episode_rewards)
        plt.title('训练奖励')
        plt.xlabel('回合')
        plt.ylabel('回合奖励')
        plt.grid(True)

        # 2. 复用率曲线
        if self.reuse_ratios:
            plt.subplot(2, 2, 2)
            plt.plot(self.reuse_ratios)
            plt.title('时隙复用率')
            plt.xlabel('回合')
            plt.ylabel('复用率')
            plt.grid(True)

        # 3. Epsilon曲线
        plt.subplot(2, 2, 3)
        epsilon_values = [agent.epsilon_start * (agent.epsilon_decay ** i) for i in range(self.episode_count)]
        epsilon_values = [max(epsilon, agent.epsilon_end) for epsilon in epsilon_values]
        plt.plot(epsilon_values)
        plt.title('探索率 (Epsilon) 衰减')
        plt.xlabel('回合')
        plt.ylabel('Epsilon')
        plt.grid(True)

        # 4. 时隙数曲线
        plt.subplot(2, 2, 4)
        if self.slot_counts:
            plt.plot(self.slot_counts)
            plt.title('时隙数量')
            plt.xlabel('回合')
            plt.ylabel('使用时隙数')
            plt.grid(True)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"最终报告已保存到 {save_path}")


# 训练DQN模型的主函数
def train_dqn(config):
    """训练DQN模型"""
    # 提取配置
    total_timesteps = config.get("total_timesteps", 500_000)
    node_size_range = config.get("node_size_range", (5, 30))
    save_dir = config.get("save_dir", "./models")
    experiment_name = config.get("experiment_name", f"dqn_stdma_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # 设置随机种子
    set_random_seed(42)

    # 创建保存目录
    model_save_path = os.path.join(save_dir, experiment_name)
    log_dir = os.path.join("./logs", experiment_name)
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"训练配置:")
    print(f"  总步数: {total_timesteps}")
    print(f"  节点范围: {node_size_range}")
    print(f"  日志目录: {log_dir}")
    print(f"  模型保存路径: {model_save_path}")

    # 计算最大节点数
    max_nodes = max(node_size_range)

    # 创建环境
    start_nodes = random.randint(*node_size_range)
    env = DynamicStdmaEnv(num_nodes=start_nodes, max_nodes=max_nodes)
    env = Monitor(env, os.path.join(log_dir, "train"))

    # 创建DQN智能体
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(
        env=env,
        feature_dim=256,
        learning_rate=3e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        buffer_size=100000,
        batch_size=32,
        target_update_freq=1000,
        max_nodes=max_nodes,
        device=device
    )

    # 创建回调
    callback = DQNCallback(
        node_size_range=node_size_range,
        max_nodes=max_nodes,
        log_dir=log_dir,
        verbose=1
    )

    # 训练模型
    try:
        print("开始训练...")
        rewards, reuse_ratios, slot_counts = agent.train(
            total_timesteps=total_timesteps,
            callback=callback
        )

        # 保存最终模型
        final_model_path = os.path.join(model_save_path, "final_model.pt")
        agent.save(final_model_path)
        print(f"模型训练完成，已保存到 {final_model_path}")

        # 保存训练报告
        report_path = os.path.join(model_save_path, "training_report.png")
        callback.save_final_report(agent, report_path)

        return agent, model_save_path

    except Exception as e:
        print(f"训练出错: {str(e)}")
        # 尝试保存中断时的模型
        try:
            error_model_path = os.path.join(model_save_path, "error_model.pt")
            agent.save(error_model_path)
            print(f"保存了错误时的模型: {error_model_path}")
        except:
            pass
        raise e

    finally:
        # 关闭环境
        env.close()


# 评估DQN模型的函数
def evaluate_dqn(model_path, node_sizes=[10, 20, 30, 40, 50], episodes=10):
    """评估DQN模型在不同规模网络上的性能"""
    results = {}

    # 设置随机种子
    set_random_seed(42)

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_nodes = max(node_sizes)

    try:
        print(f"\n开始评估模型: {model_path}")

        # 创建一个临时环境以初始化智能体
        temp_env = DynamicStdmaEnv(num_nodes=10, max_nodes=max_nodes)
        agent = DQNAgent(
            env=temp_env,
            max_nodes=max_nodes,
            device=device
        )
        agent.load(model_path)

    except Exception as e:
        print(f"加载模型失败: {e}")
        return {}

    for nodes in node_sizes:
        print(f"\n测试 {nodes} 节点网络...")
        env = DynamicStdmaEnv(num_nodes=nodes, max_nodes=max_nodes)

        # 评估模型
        result = agent.evaluate(env, episodes=episodes)
        results[nodes] = result

        print(f"  平均奖励: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
        print(f"  平均复用率: {result['mean_reuse_ratio']:.2f} ± {result['std_reuse_ratio']:.2f}")
        print(f"  平均时隙数: {result['mean_slots']:.2f} ± {result['std_slots']:.2f}")

    # 绘制结果
    if results:
        plt.figure(figsize=(15, 10))

        # 绘制平均奖励
        plt.subplot(2, 2, 1)
        plt.errorbar(
            list(results.keys()),
            [r['mean_reward'] for r in results.values()],
            yerr=[r['std_reward'] for r in results.values()],
            marker='o'
        )
        plt.title('平均奖励 vs 网络规模')
        plt.xlabel('节点数量')
        plt.ylabel('平均奖励')
        plt.grid(True)

        # 绘制平均复用率
        plt.subplot(2, 2, 2)
        plt.errorbar(
            list(results.keys()),
            [r['mean_reuse_ratio'] for r in results.values()],
            yerr=[r['std_reuse_ratio'] for r in results.values()],
            marker='o'
        )
        plt.title('平均复用率 vs 网络规模')
        plt.xlabel('节点数量')
        plt.ylabel('平均复用率')
        plt.grid(True)

        # 绘制时隙效率
        plt.subplot(2, 2, 3)
        slot_efficiency = [nodes / r['mean_slots'] for nodes, r in results.items()]
        plt.plot(list(results.keys()), slot_efficiency, marker='o')
        plt.title('时隙效率 vs 网络规模')
        plt.xlabel('节点数量')
        plt.ylabel('节点数/时隙数')
        plt.grid(True)

        # 绘制时隙数量
        plt.subplot(2, 2, 4)
        plt.errorbar(
            list(results.keys()),
            [r['mean_slots'] for r in results.values()],
            yerr=[r['std_slots'] for r in results.values()],
            marker='o'
        )
        plt.title('平均时隙数 vs 网络规模')
        plt.xlabel('节点数量')
        plt.ylabel('平均时隙数')
        plt.grid(True)

        plt.tight_layout()

        # 保存图表
        result_dir = os.path.dirname(model_path)
        plt.savefig(os.path.join(result_dir, "dqn_evaluation_results.png"))
        print(f"\n评估结果已保存至 {os.path.join(result_dir, 'dqn_evaluation_results.png')}")

    return results


# 比较两种模型的性能
def compare_models(ppo_results, dqn_results, save_path="model_comparison.png"):
    """比较PPO和DQN模型的性能"""
    if not ppo_results or not dqn_results:
        print("缺少评估结果，无法进行比较")
        return

    # 找到两个结果中共有的节点数量
    common_nodes = sorted(set(ppo_results.keys()).intersection(set(dqn_results.keys())))

    if not common_nodes:
        print("没有共同的节点数量，无法进行比较")
        return

    plt.figure(figsize=(16, 12))

    # 1. 比较平均奖励
    plt.subplot(2, 2, 1)
    plt.plot(common_nodes, [ppo_results[n]['mean_reward'] for n in common_nodes], 'o-', label='PPO')
    plt.plot(common_nodes, [dqn_results[n]['mean_reward'] for n in common_nodes], 's-', label='DQN')
    plt.title('平均奖励比较')
    plt.xlabel('节点数量')
    plt.ylabel('平均奖励')
    plt.legend()
    plt.grid(True)

    # 2. 比较平均复用率
    plt.subplot(2, 2, 2)
    plt.plot(common_nodes, [ppo_results[n]['mean_reuse_ratio'] for n in common_nodes], 'o-', label='PPO')
    plt.plot(common_nodes, [dqn_results[n]['mean_reuse_ratio'] for n in common_nodes], 's-', label='DQN')
    plt.title('平均复用率比较')
    plt.xlabel('节点数量')
    plt.ylabel('平均复用率')
    plt.legend()
    plt.grid(True)

    # 3. 比较时隙效率
    plt.subplot(2, 2, 3)
    ppo_efficiency = [n / ppo_results[n]['mean_slots'] for n in common_nodes]
    dqn_efficiency = [n / dqn_results[n]['mean_slots'] for n in common_nodes]
    plt.plot(common_nodes, ppo_efficiency, 'o-', label='PPO')
    plt.plot(common_nodes, dqn_efficiency, 's-', label='DQN')
    plt.title('时隙效率比较 (节点数/时隙数)')
    plt.xlabel('节点数量')
    plt.ylabel('效率')
    plt.legend()
    plt.grid(True)

    # 4. 比较时隙数量
    plt.subplot(2, 2, 4)
    plt.plot(common_nodes, [ppo_results[n]['mean_slots'] for n in common_nodes], 'o-', label='PPO')
    plt.plot(common_nodes, [dqn_results[n]['mean_slots'] for n in common_nodes], 's-', label='DQN')
    plt.title('平均时隙数比较')
    plt.xlabel('节点数量')
    plt.ylabel('平均时隙数')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"模型比较结果已保存至 {save_path}")

    # 打印数值对比
    print("\n数值对比:")
    print("=" * 80)
    print(
        f"{'节点数':<10}{'PPO奖励':<15}{'DQN奖励':<15}{'PPO复用率':<15}{'DQN复用率':<15}{'PPO时隙数':<15}{'DQN时隙数':<15}")
    print("-" * 80)

    for n in common_nodes:
        print(f"{n:<10}{ppo_results[n]['mean_reward']:<15.2f}{dqn_results[n]['mean_reward']:<15.2f}"
              f"{ppo_results[n]['mean_reuse_ratio']:<15.2f}{dqn_results[n]['mean_reuse_ratio']:<15.2f}"
              f"{ppo_results[n]['mean_slots']:<15.2f}{dqn_results[n]['mean_slots']:<15.2f}")

    print("=" * 80)


# 主函数
if __name__ == "__main__":
    # 设置随机种子
    set_random_seed(42)

    # 训练配置
    training_config = {
        "total_timesteps": 500_000,  # 总训练步数
        "node_size_range": (5, 30),  # 节点数量范围
        "experiment_name": f"dqn_stdma_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }

    # 训练DQN模型
    agent, model_path = train_dqn(training_config)

    # 评估DQN模型
    if model_path and os.path.exists(os.path.join(model_path, "final_model.pt")):
        dqn_results = evaluate_dqn(os.path.join(model_path, "final_model.pt"))

        # 如果存在PPO模型结果，进行比较
        ppo_model_path = input("输入PPO模型评估结果路径（如果存在）或按Enter跳过比较: ")
        # if ppo_model_path and os.path.exists(ppo_model_path):
        #     # 假设PPO评估结果可以从某个文件加载
        #     # 这里简化处理，实际使用时可能需要更复杂的加载逻辑
        #     compare_models(ppo_results, dqn_results,
        #                    save_path=os.path.join(os.path.dirname(model_path), "ppo_dqn_comparison.png"))