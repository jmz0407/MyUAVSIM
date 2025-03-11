import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from datetime import datetime
import matplotlib.pyplot as plt
import copy
from collections import deque
import random
from torch.utils.tensorboard import SummaryWriter

# 导入环境和工具
from rl_environment import StdmaEnv
from simulator.simulator import Simulator
import simpy
from utils import config
from torch_geometric.nn import GATConv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# 设置随机种子以便复现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 超参数
MEMORY_SIZE = 100000  # 经验回放缓冲区大小
BATCH_SIZE = 64  # 批量大小
GAMMA = 0.99  # 折扣因子
TAU = 0.005  # 目标网络软更新参数
POLICY_NOISE = 0.2  # 目标策略平滑的噪声
NOISE_CLIP = 0.5  # 噪声裁剪范围
POLICY_FREQ = 2  # 策略和目标网络延迟更新频率
LR_ACTOR = 0.0003  # Actor学习率
LR_CRITIC = 0.0003  # Critic学习率
EPSILON_START = 1.0  # 初始探索率
EPSILON_END = 0.1  # 最终探索率
EPSILON_DECAY = 5000  # 探索率衰减步数
HIDDEN_DIM = 256  # 隐藏层维度


class GATBlock(nn.Module):
    """带有多头注意力机制的GAT块"""

    def __init__(self, in_channels, out_channels, heads=4, dropout=0.1):
        super().__init__()
        self.gat1 = GATConv(
            in_channels,
            out_channels,
            heads=heads,
            dropout=dropout,
            concat=True  # 连接多头注意力输出
        )
        # 第二层GAT不连接多头输出
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

        # 注册钩子捕获注意力权重
        self.gat1.register_forward_hook(self.gat_attention_hook)

    def gat_attention_hook(self, module, input, output):
        """钩子函数，用于捕获GAT层的注意力权重"""
        if hasattr(module, 'alpha'):
            self.attention_weights = module.alpha.detach()

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
    """使用GAT架构的特征提取器"""

    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        total_dim = observation_space.shape[0]
        self.num_nodes = 10  # 假设有10个节点，与原始代码一致

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
            print(f"_unflatten_observation错误: {str(e)}")
            print(f"输入形状: {flat_obs.shape}")
            print(f"当前索引: {idx}")
            print(f"批次大小: {batch_size}")
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
            print(f"前向传播错误: {str(e)}")
            print(f"观察形状: {observations.shape}")
            import traceback
            traceback.print_exc()
            raise e


class Actor(nn.Module):
    """TD3 Actor网络 - 为离散动作空间调整"""

    def __init__(self, input_space, n_actions, features_dim=256, hidden_dim=256):
        super(Actor, self).__init__()

        # 特征提取器
        self.features_extractor = StdmaFeatureExtractor(
            observation_space=input_space,
            features_dim=features_dim
        )

        # Actor网络 - 输出离散动作的logits
        self.actor = nn.Sequential(
            nn.Linear(features_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions)
        )

    def forward(self, x):
        features = self.features_extractor(x)
        action_logits = self.actor(features)
        return action_logits

    def get_action(self, x, deterministic=False):
        """获取离散动作"""
        action_logits = self.forward(x)

        if deterministic:
            # 确定性策略 - 选择logits最高的动作
            action = torch.argmax(action_logits, dim=1).cpu().numpy()
        else:
            # 随机策略 - 基于softmax概率采样
            probs = F.softmax(action_logits, dim=1)
            dist = Categorical(probs)
            action = dist.sample().cpu().numpy()

        return action

    def get_action_with_logits(self, x):
        """返回logits和对应的动作"""
        action_logits = self.forward(x)
        return action_logits


class Critic(nn.Module):
    """TD3 Critic网络 - 双Q网络"""

    def __init__(self, input_space, n_actions, features_dim=256, hidden_dim=256):
        super(Critic, self).__init__()

        # 特征提取器 - 共享提取状态特征
        self.features_extractor = StdmaFeatureExtractor(
            observation_space=input_space,
            features_dim=features_dim
        )

        # 第一个Q网络
        self.q1 = nn.Sequential(
            nn.Linear(features_dim + n_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 第二个Q网络
        self.q2 = nn.Sequential(
            nn.Linear(features_dim + n_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 动作的one-hot编码维度
        self.n_actions = n_actions

    def forward(self, x, action):
        """前向传播计算Q值

        Args:
            x: 状态输入
            action: 整数动作索引 [batch_size]

        Returns:
            q1, q2: 两个Q网络的值
        """
        features = self.features_extractor(x)

        # 将动作转换为one-hot编码
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.long, device=features.device)

        # 如果action是单一维度，增加batch维度
        if action.dim() == 0:
            action = action.unsqueeze(0)

        one_hot_actions = F.one_hot(action, num_classes=self.n_actions).float()

        # 确保one_hot_actions和features有相同的batch size
        if features.dim() > one_hot_actions.dim():
            one_hot_actions = one_hot_actions.unsqueeze(0)

        # 拼接状态特征和动作
        x = torch.cat([features, one_hot_actions], dim=1)

        # 计算两个Q值
        q1 = self.q1(x)
        q2 = self.q2(x)

        return q1, q2

    def q1_forward(self, x, action):
        """只使用第一个Q网络计算值"""
        features = self.features_extractor(x)

        # 将动作转换为one-hot编码
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.long, device=features.device)

        # 如果action是单一维度，增加batch维度
        if action.dim() == 0:
            action = action.unsqueeze(0)

        one_hot_actions = F.one_hot(action, num_classes=self.n_actions).float()

        # 确保one_hot_actions和features有相同的batch size
        if features.dim() > one_hot_actions.dim():
            one_hot_actions = one_hot_actions.unsqueeze(0)

        # 拼接状态特征和动作
        x = torch.cat([features, one_hot_actions], dim=1)

        # 只计算第一个Q值
        q1 = self.q1(x)

        return q1


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)


class TD3Agent:
    """TD3代理类"""

    def __init__(self, env, hidden_dim=HIDDEN_DIM, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC,
                 buffer_size=MEMORY_SIZE, batch_size=BATCH_SIZE, gamma=GAMMA, tau=TAU,
                 policy_noise=POLICY_NOISE, noise_clip=NOISE_CLIP, policy_freq=POLICY_FREQ):

        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        # 确定设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 获取环境信息
        self.n_actions = env.action_space.n
        self.input_space = env.observation_space

        # 初始化Actor和Critic网络
        self.actor = Actor(self.input_space, self.n_actions, hidden_dim=hidden_dim).to(self.device)
        self.actor_target = Actor(self.input_space, self.n_actions, hidden_dim=hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(self.input_space, self.n_actions, hidden_dim=hidden_dim).to(self.device)
        self.critic_target = Critic(self.input_space, self.n_actions, hidden_dim=hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # 初始化经验回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_size)

        # 探索参数
        self.epsilon = EPSILON_START
        self.epsilon_end = EPSILON_END
        self.epsilon_decay = EPSILON_DECAY

        # 训练步计数
        self.total_steps = 0
        self.policy_update_counter = 0

    def select_action(self, state, evaluate=False):
        """选择动作，带有epsilon-greedy探索"""
        # 在评估模式下使用确定性策略
        if evaluate:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                return self.actor.get_action(state_tensor, deterministic=True)[0]

        # 使用epsilon-greedy探索
        if np.random.random() < self.epsilon:
            # 随机动作
            return np.random.randint(0, self.n_actions)
        else:
            # 基于策略的动作
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                return self.actor.get_action(state_tensor, deterministic=True)[0]

    def update_epsilon(self):
        """更新探索率"""
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon - (EPSILON_START - EPSILON_END) / self.epsilon_decay
        )

    def train_step(self):
        """执行一步TD3训练"""
        self.total_steps += 1

        # 记录训练指标
        metrics = {}

        # 如果没有足够的样本，跳过更新
        if len(self.replay_buffer) < self.batch_size:
            return metrics

        # 从回放缓冲区采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        with torch.no_grad():
            # 对目标策略添加噪声
            next_action_logits = self.actor_target(next_states)
            noise = torch.normal(0, self.policy_noise, size=next_action_logits.shape).to(self.device)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            noisy_next_action_logits = next_action_logits + noise

            # 基于添加噪声的logits选择下一个动作
            next_actions = torch.argmax(noisy_next_action_logits, dim=1)

            # 计算目标Q值
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        # 计算当前Q值
        current_q1, current_q2 = self.critic(states, actions)

        # Critic损失
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # 记录Critic损失和Q值
        metrics['critic_loss'] = critic_loss.item()
        metrics['q_value'] = current_q1.mean().item()

        # 优化Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 延迟策略更新
        if self.total_steps % self.policy_freq == 0:

            # 计算Actor损失 - 最大化Q值
            actor_logits = self.actor(states)
            actor_actions = torch.argmax(actor_logits, dim=1)
            actor_loss = -self.critic.q1_forward(states, actor_actions).mean()

            # 记录Actor损失
            metrics['actor_loss'] = actor_loss.item()

            # 优化Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 软更新目标网络
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.policy_update_counter += 1

        return metrics

    def save(self, directory):
        """保存模型"""
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.actor.state_dict(), os.path.join(directory, "td3_actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(directory, "td3_critic.pth"))

        print(f"模型已保存至: {directory}")

    def load(self, directory):
        """加载模型"""
        self.actor.load_state_dict(torch.load(os.path.join(directory, "td3_actor.pth")))
        self.critic.load_state_dict(torch.load(os.path.join(directory, "td3_critic.pth")))

        # 同步目标网络
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        print(f"模型已从{directory}加载")


def train_td3(env, max_episodes=10000, max_steps=None, save_interval=100):
    """训练TD3代理"""

    # 设置目录
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/STDMA_TD3_{current_time}"
    os.makedirs(log_dir, exist_ok=True)

    # 创建TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))

    # 创建并初始化代理
    agent = TD3Agent(env)

    # 训练数据记录
    episode_rewards = []
    episode_lengths = []
    avg_rewards = []

    print("开始TD3训练...")

    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        step = 0
        done = False

        # 收集每个回合的指标
        episode_q_values = []
        episode_actor_losses = []
        episode_critic_losses = []
        slot_reuse_rates = []

        while not done:
            # 选择动作
            action = agent.select_action(state)

            # 执行动作
            try:
                result = env.step(action)
                next_state, reward, done, _, info = result
                schedule = info['schedule']
                # 记录时隙复用率
                if schedule:
                    total_slots = len(schedule)
                    total_assignments = sum(len(nodes) for nodes in schedule.values())
                    reuse_rate = total_assignments / total_slots if total_slots > 0 else 0
                    slot_reuse_rates.append(reuse_rate)

                    # 尝试从info中获取调度信息
                    if 'schedule' in info:
                        schedule = info['schedule']
                        if schedule:
                            total_slots = len(schedule)
                            total_assignments = sum(len(nodes) for nodes in schedule.values())
                            reuse_rate = total_assignments / total_slots if total_slots > 0 else 0
                            slot_reuse_rates.append(reuse_rate)
            except Exception as e:
                print(f"环境执行动作时出错: {e}")
                continue

            # 保存经验
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # 更新代理并获取训练指标
            metrics = agent.train_step()
            if metrics:
                if 'critic_loss' in metrics:
                    episode_critic_losses.append(metrics['critic_loss'])
                if 'actor_loss' in metrics:
                    episode_actor_losses.append(metrics['actor_loss'])
                if 'q_value' in metrics:
                    episode_q_values.append(metrics['q_value'])

            # 更新状态
            state = next_state
            episode_reward += reward
            step += 1

            # 更新探索率
            agent.update_epsilon()

            # 检查是否达到最大步数
            if max_steps and step >= max_steps:
                done = True

        # 记录回合结果
        episode_rewards.append(episode_reward)
        episode_lengths.append(step)

        # 计算近期平均奖励
        avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        avg_rewards.append(avg_reward)

        # 记录到TensorBoard
        writer.add_scalar('Reward/Episode', episode_reward, episode)
        writer.add_scalar('Reward/Average', avg_reward, episode)
        writer.add_scalar('Steps/Episode', step, episode)
        writer.add_scalar('Exploration/Epsilon', agent.epsilon, episode)

        # 记录其他训练指标
        if episode_critic_losses:
            writer.add_scalar('Values/Q', np.mean(episode_q_values), episode)
        if slot_reuse_rates:
            writer.add_scalar('Performance/SlotReuseRate', np.mean(slot_reuse_rates), episode)

        # 每10个回合记录网络的参数直方图
        if episode % 10 == 0:
            for name, param in agent.actor.named_parameters():
                writer.add_histogram(f'Actor/{name}', param.clone().cpu().data.numpy(), episode)
            for name, param in agent.critic.named_parameters():
                writer.add_histogram(f'Critic/{name}', param.clone().cpu().data.numpy(), episode)

        # 打印训练进度
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{max_episodes} | Reward: {episode_reward:.2f} | " +
                  f"Avg Reward: {avg_reward:.2f} | Steps: {step} | Epsilon: {agent.epsilon:.4f}")

        # 定期保存模型和绘制学习曲线
        if (episode + 1) % save_interval == 0:
            agent.save(os.path.join(log_dir, f"checkpoint_{episode + 1}"))

            # 绘制学习曲线
            plt.figure(figsize=(16, 8))

            # 绘制奖励曲线
            plt.subplot(2, 2, 1)
            plt.plot(episode_rewards, label='Episode Reward')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('TD3 Learning Curve - Episode Rewards')
            plt.grid(True)

            # 绘制平均奖励曲线
            plt.subplot(2, 2, 2)
            plt.plot(avg_rewards, label='Average Reward', color='orange')
            plt.xlabel('Episode')
            plt.ylabel('Avg Reward')
            plt.title('TD3 Learning Curve - Average Rewards')
            plt.grid(True)

            # 绘制回合长度曲线
            plt.subplot(2, 2, 3)
            plt.plot(episode_lengths, label='Episode Length', color='green')
            plt.xlabel('Episode')
            plt.ylabel('Steps')
            plt.title('TD3 Learning Curve - Episode Lengths')
            plt.grid(True)

            # 绘制探索率衰减曲线
            plt.subplot(2, 2, 4)
            plt.plot(
                range(len(episode_rewards)),
                [max(EPSILON_END, EPSILON_START - (EPSILON_START - EPSILON_END) * min(i * 10 / EPSILON_DECAY, 1))
                 for i in range(len(episode_rewards))],
                label='Epsilon', color='purple'
            )
            plt.xlabel('Episode')
            plt.ylabel('Epsilon')
            plt.title('TD3 Learning Curve - Exploration Rate')
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(log_dir, f"learning_curve_{episode + 1}.png"))
            plt.close()

    # 保存最终模型
    agent.save(os.path.join(log_dir, "final_model"))

    # 绘制最终学习曲线
    plt.figure(figsize=(16, 8))

    # 绘制奖励曲线
    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards, label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('TD3 Learning Curve - Episode Rewards')
    plt.grid(True)

    # 绘制平均奖励曲线
    plt.subplot(2, 2, 2)
    plt.plot(avg_rewards, label='Average Reward', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Avg Reward')
    plt.title('TD3 Learning Curve - Average Rewards')
    plt.grid(True)

    # 绘制回合长度曲线
    plt.subplot(2, 2, 3)
    plt.plot(episode_lengths, label='Episode Length', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('TD3 Learning Curve - Episode Lengths')
    plt.grid(True)

    # 保存最终图表
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "final_learning_curve.png"))
    plt.close()

    # 关闭TensorBoard writer
    writer.close()

    print(f"训练完成，模型和日志保存在 {log_dir}")
    print(f"可以使用以下命令启动TensorBoard查看训练进度: tensorboard --logdir={log_dir}/tensorboard")

    return agent, episode_rewards, log_dir


def evaluate_td3(agent, env, num_episodes=50):
    """评估训练好的TD3代理"""

    # 创建TensorBoard SummaryWriter进行评估
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_log_dir = f"./logs/STDMA_TD3_eval_{current_time}"
    os.makedirs(eval_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(eval_log_dir, "tensorboard"))

    # 评估指标
    results = {
        'rewards': [],
        'step_rewards': [],
        'slot_usage': [],
        'interference_free': 0,
        'slot_reuse': [],
        'episode_lengths': [],
        'qos_satisfaction': []
    }

    print("\n开始评估TD3模型...")

    for episode in range(num_episodes):
        # 重置环境
        state, _ = env.reset()
        done = False
        episode_reward = 0
        step_rewards = []
        step_count = 0

        # 当前回合的调度
        episode_schedule = {}

        while not done:
            # 使用确定性策略选择动作
            action = agent.select_action(state, evaluate=True)

            # 检查动作有效性
            current_node = env.current_node
            is_valid = env.is_valid_assignment(current_node, action)
            print(f"步骤 {step_count}: 节点 {current_node}, 动作 {action}, 有效性: {is_valid}")

            # 执行动作
            try:
                result = env.step(action)
                next_state, reward, done, _, info = result
                schedule = info['schedule']
                episode_schedule = schedule.copy() if schedule else {}
            except Exception as e:
                print(f"环境执行动作时出错: {e}")
                continue

            step_rewards.append(reward)
            episode_reward += reward
            step_count += 1

            # 更新状态
            state = next_state

            # 防止无限循环
            if step_count > env.num_nodes * 3:
                print(f"警告: 超过最大步数 ({step_count})")
                done = True

        # 回合完成
        print(f"\n回合 {episode + 1} 完成:")
        print(f"总步数: {step_count}")
        print(f"总奖励: {episode_reward}")
        print(f"最终调度: {episode_schedule}")

        # 记录基本指标
        results['rewards'].append(episode_reward)
        results['step_rewards'].extend(step_rewards)
        results['episode_lengths'].append(step_count)
        results['slot_usage'].append(len(episode_schedule))

        # 记录到TensorBoard
        writer.add_scalar('Eval/Reward', episode_reward, episode)
        writer.add_scalar('Eval/Steps', step_count, episode)
        writer.add_scalar('Eval/SlotUsage', len(episode_schedule), episode)

        # 检查干扰
        has_interference = False
        for slot, nodes in episode_schedule.items():
            for i, node1 in enumerate(nodes):
                for node2 in nodes[i + 1:]:
                    if hasattr(env, 'topology_matrix') and env.topology_matrix[node1][node2] == 1:
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

            # 记录到TensorBoard
            writer.add_scalar('Eval/SlotReuseRatio', reuse_ratio, episode)

        # 检查QoS满足情况
        if hasattr(env, 'current_requirement') and env.current_requirement:
            if hasattr(env, '_estimate_delay'):
                current_delay = env._estimate_delay()
                qos_satisfied = current_delay <= env.current_requirement.delay_requirement
                results['qos_satisfaction'].append(float(qos_satisfied))

                # 记录到TensorBoard
                writer.add_scalar('Eval/QoSSatisfied', float(qos_satisfied), episode)
                writer.add_scalar('Eval/Delay', current_delay, episode)

    # 计算最终统计
    if num_episodes > 0:
        results['avg_reward'] = np.mean(results['rewards'])
        results['avg_slots'] = np.mean(results['slot_usage'])
        results['avg_reuse_ratio'] = np.mean(results['slot_reuse']) if results['slot_reuse'] else 0
        results['avg_episode_length'] = np.mean(results['episode_lengths'])
        results['interference_free_rate'] = results['interference_free'] / num_episodes * 100

        if results['qos_satisfaction']:
            results['qos_satisfaction_rate'] = np.mean(results['qos_satisfaction']) * 100

        # 记录最终统计到TensorBoard
        writer.add_scalar('EvalSummary/AvgReward', results['avg_reward'], 0)
        writer.add_scalar('EvalSummary/AvgSlots', results['avg_slots'], 0)
        writer.add_scalar('EvalSummary/AvgReuseRatio', results['avg_reuse_ratio'], 0)
        writer.add_scalar('EvalSummary/InterferenceFreeRate', results['interference_free_rate'], 0)
        if 'qos_satisfaction_rate' in results:
            writer.add_scalar('EvalSummary/QoSSatisfactionRate', results['qos_satisfaction_rate'], 0)

        # 创建汇总图表并保存到TensorBoard
        fig = plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(results['rewards'])
        plt.title('评估奖励')
        plt.xlabel('Episode')
        plt.ylabel('Reward')

        plt.subplot(2, 2, 2)
        plt.plot(results['slot_reuse'])
        plt.title('时隙复用率')
        plt.xlabel('Episode')
        plt.ylabel('Reuse Ratio')

        plt.tight_layout()

        # 将图表添加到TensorBoard
        writer.add_figure('EvalSummary/Plots', fig, 0)

        print("\n评估结果:")
        print(f"平均奖励: {results['avg_reward']:.2f}")
        print(f"无干扰率: {results['interference_free_rate']:.2f}%")
        print(f"平均时隙数: {results['avg_slots']:.2f}")
        print(f"平均时隙复用率: {results['avg_reuse_ratio']:.2f}")
        print(f"平均回合长度: {results['avg_episode_length']:.2f}")
        if 'qos_satisfaction_rate' in results:
            print(f"QoS满足率: {results['qos_satisfaction_rate']:.2f}%")

    # 关闭TensorBoard Writer
    writer.close()
    print(f"评估日志保存在 {eval_log_dir}")
    print(f"可以使用以下命令查看评估结果: tensorboard --logdir={eval_log_dir}/tensorboard")

    return results


def compare_td3_with_ppo(td3_agent, env, log_dir):
    """比较TD3和PPO的性能"""
    from stable_baselines3 import PPO

    # 创建TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join(log_dir, "comparison_tensorboard"))

    # 设置策略参数
    policy_kwargs = dict(
        features_extractor_class=StdmaFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[128, 64], vf=[128, 64])
    )

    # 创建并训练PPO模型
    print("\n训练PPO模型进行比较...")
    ppo_model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=os.path.join(log_dir, "ppo_tensorboard")
    )

    # 训练PPO
    ppo_model.learn(total_timesteps=200000)

    # 保存PPO模型
    ppo_model_path = os.path.join(log_dir, "ppo_model.zip")
    ppo_model.save(ppo_model_path)

    # 评估TD3
    print("\n评估TD3模型...")
    td3_results = evaluate_td3(td3_agent, env, num_episodes=20)

    # 评估PPO
    print("\n评估PPO模型...")

    # 定义一个包装器，使PPO具有与TD3相同的接口
    class PPOWrapper:
        def __init__(self, model):
            self.model = model

        def select_action(self, state, evaluate=True):
            state_tensor = torch.FloatTensor(state).reshape(1, -1)
            action, _ = self.model.predict(state_tensor, deterministic=evaluate)
            return action

    ppo_agent = PPOWrapper(ppo_model)
    ppo_results = evaluate_td3(ppo_agent, env, num_episodes=20)

    # 记录比较结果到TensorBoard
    metrics = ['avg_reward', 'avg_reuse_ratio', 'interference_free_rate', 'qos_satisfaction_rate']
    for metric in metrics:
        if metric in td3_results and metric in ppo_results:
            writer.add_scalars(f'Comparison/{metric}', {
                'TD3': td3_results[metric],
                'PPO': ppo_results[metric]
            }, 0)

    # 创建比较图表
    print("\n创建比较图表...")
    plt.figure(figsize=(15, 10))

    # 比较平均奖励
    plt.subplot(2, 2, 1)
    algorithms = ['TD3', 'PPO']
    rewards = [td3_results.get('avg_reward', 0), ppo_results.get('avg_reward', 0)]
    plt.bar(algorithms, rewards, color=['blue', 'green'])
    plt.title('平均奖励')
    plt.ylabel('奖励值')
    plt.grid(True, alpha=0.3)

    # 比较无干扰率
    plt.subplot(2, 2, 2)
    interference_rates = [td3_results.get('interference_free_rate', 0), ppo_results.get('interference_free_rate', 0)]
    plt.bar(algorithms, interference_rates, color=['blue', 'green'])
    plt.title('无干扰率 (%)')
    plt.ylabel('百分比')
    plt.grid(True, alpha=0.3)

    # 比较时隙复用率
    plt.subplot(2, 2, 3)
    reuse_ratios = [td3_results.get('avg_reuse_ratio', 0), ppo_results.get('avg_reuse_ratio', 0)]
    plt.bar(algorithms, reuse_ratios, color=['blue', 'green'])
    plt.title('平均时隙复用率')
    plt.ylabel('复用率')
    plt.grid(True, alpha=0.3)

    # 比较QoS满足率
    plt.subplot(2, 2, 4)
    qos_rates = [
        td3_results.get('qos_satisfaction_rate', 0),
        ppo_results.get('qos_satisfaction_rate', 0)
    ]
    plt.bar(algorithms, qos_rates, color=['blue', 'green'])
    plt.title('QoS满足率 (%)')
    plt.ylabel('百分比')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'td3_vs_ppo_comparison.png'))

    # 将图表添加到TensorBoard
    writer.add_figure('Comparison/Summary', plt.gcf(), 0)

    # 关闭TensorBoard writer
    writer.close()

    print(f"比较结果保存在 {log_dir}")
    print(f"可以使用以下命令查看比较结果: tensorboard --logdir={log_dir}")

    return td3_results, ppo_results
if __name__ == "__main__":
    try:
        # 设置显示中文
        plt.rcParams['font.sans-serif'] = ['STFangSong']
        plt.rcParams['axes.unicode_minus'] = False

        print("创建STDMA环境...")

        # 创建基础环境
        env = simpy.Environment()
        n_drones = config.NUMBER_OF_DRONES
        channel_states = {i: simpy.Resource(env, capacity=1) for i in range(n_drones)}

        # 创建模拟器
        simulator = Simulator(
            seed=42,
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

        # 检查命令行参数
        import sys

        mode = "compare"  # 默认为训练模式
        model_path = None

        if len(sys.argv) > 1:
            mode = sys.argv[1]
            if mode not in ["train", "eval", "compare", "visualize"]:
                print(f"未知模式: {mode}. 使用: train, eval, compare, visualize")
                mode = "eval"

            if mode in ["eval", "visualize"] and len(sys.argv) > 2:
                model_path = sys.argv[2]

        if mode == "train":
            # 训练TD3
            print("\n开始TD3训练...")
            agent, rewards, log_dir = train_td3(stdma_env, max_episodes=1000, save_interval=50)

            # 评估训练好的模型
            print("\n评估训练好的TD3模型...")
            eval_results = evaluate_td3(agent, stdma_env, num_episodes=20)

        elif mode == "eval":
            if model_path:
                print(f"\n从 {model_path} 加载TD3模型进行评估...")

                # 创建代理
                agent = TD3Agent(stdma_env)

                # 加载模型
                agent.load(model_path)

                # 评估模型
                eval_results = evaluate_td3(agent, stdma_env, num_episodes=20)
            else:
                print("评估模式需要指定模型路径，例如: python td3_agent.py eval ./logs/path_to_model")

        elif mode == "compare":
            # 训练并比较TD3和PPO
            print("\n开始TD3训练（用于比较）...")
            agent, rewards, log_dir = train_td3(stdma_env, max_episodes=300, save_interval=100)

            print("\n与PPO进行比较...")
            td3_results, ppo_results = compare_td3_with_ppo(agent, stdma_env, log_dir)

        elif mode == "visualize":
            if model_path:
                print(f"\n从 {model_path} 加载TD3模型进行可视化...")

                # 创建代理
                agent = TD3Agent(stdma_env)

                # 加载模型
                agent.load(model_path)

                # 可视化模型
                print("\n创建网络参数分布图...")

                # 创建TensorBoard SummaryWriter
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                viz_log_dir = f"./logs/STDMA_TD3_viz_{current_time}"
                os.makedirs(viz_log_dir, exist_ok=True)
                writer = SummaryWriter(log_dir=viz_log_dir)

                # 记录网络参数
                for name, param in agent.actor.named_parameters():
                    writer.add_histogram(f'Actor/{name}', param.clone().cpu().data.numpy(), 0)

                for name, param in agent.critic.named_parameters():
                    writer.add_histogram(f'Critic/{name}', param.clone().cpu().data.numpy(), 0)

                # 生成动作分布热图
                print("\n生成动作分布热图...")

                # 收集一系列状态
                states = []
                for _ in range(100):
                    state, _ = stdma_env.reset()
                    states.append(state)

                # 分析动作分布
                if states:
                    actions = []
                    logits_list = []

                    with torch.no_grad():
                        for state in states:
                            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                            logits = agent.actor(state_tensor)
                            action = torch.argmax(logits, dim=1).item()
                            actions.append(action)
                            logits_list.append(logits.cpu().numpy())

                    # 创建动作分布图
                    plt.figure(figsize=(10, 6))
                    plt.hist(actions, bins=range(stdma_env.action_space.n + 1), alpha=0.7)
                    plt.xlabel('动作')
                    plt.ylabel('频率')
                    plt.title('TD3模型动作分布')
                    plt.grid(True, alpha=0.3)
                    plt.savefig(os.path.join(viz_log_dir, 'action_distribution.png'))

                    # 添加到TensorBoard
                    writer.add_figure('Visualization/ActionDistribution', plt.gcf(), 0)

                    # 创建logits热图
                    if logits_list:
                        avg_logits = np.mean(logits_list, axis=0)[0]
                        plt.figure(figsize=(12, 4))
                        plt.bar(range(len(avg_logits)), avg_logits)
                        plt.xlabel('动作')
                        plt.ylabel('平均Logit值')
                        plt.title('TD3模型动作Logits分布')
                        plt.grid(True, alpha=0.3)
                        plt.savefig(os.path.join(viz_log_dir, 'logits_distribution.png'))

                        # 添加到TensorBoard
                        writer.add_figure('Visualization/LogitsDistribution', plt.gcf(), 0)

                # 关闭TensorBoard Writer
                writer.close()
                print(f"可视化结果保存在 {viz_log_dir}")
                print(f"可以使用以下命令查看可视化结果: tensorboard --logdir={viz_log_dir}")
            else:
                print("可视化模式需要指定模型路径，例如: python td3_agent.py visualize ./logs/path_to_model")

        print("\nTD3任务完成!")
        if 'log_dir' in locals():
            print(f"结果和模型保存至: {log_dir}")

    except Exception as e:
        import traceback

        print(f"发生错误: {str(e)}")
        traceback.print_exc()