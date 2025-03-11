#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UAV网络STDMA多算法对比评估脚本

此脚本对比评估在UAV网络环境中不同强化学习算法的性能:
- PPO (Proximal Policy Optimization)
- GAT-PPO (Graph Attention Network enhanced PPO)
- DQN (Deep Q-Network)
- SAC (Soft Actor-Critic)
- TD3 (Twin Delayed DDPG)

使用统一的环境配置和评估指标，确保公平比较。
"""

import os
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from stable_baselines3 import PPO, DQN, SAC, TD3
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
import gymnasium as gym
import simpy

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
try:
    plt.style.use('seaborn-v0_8-deep')
except:
    try:
        plt.style.use('seaborn-deep')
    except:
        pass  # 忽略样式错误

# 创建日志目录
log_dir = f"./logs/algorithm_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(log_dir, exist_ok=True)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"{log_dir}/comparison.log")
    ]
)

logger = logging.getLogger("AlgorithmComparison")

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {device}")

# ---------- 环境导入部分 ----------#
# 导入您的环境模块
try:
    # 尝试直接导入
    from rl_environment import StdmaEnv
    from simulator.simulator import Simulator
    from utils import config

    logger.info("成功导入环境模块")
except ImportError:
    logger.warning("无法直接导入环境模块，请确保环境模块路径正确")
    # 可以在这里添加模块路径
    import sys

    sys.path.append('.')  # 添加当前目录
    try:
        from rl_environment import StdmaEnv
        from simulator.simulator import Simulator
        from utils import config

        logger.info("成功导入环境模块(使用添加路径)")
    except ImportError as e:
        logger.error(f"导入环境模块失败: {e}")
        logger.error("请修改脚本中的导入路径")
        raise ImportError("请确保环境模块可导入，或修改脚本中的导入路径")


# ---------- 模型实现部分 ----------#

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


class StdmaGATFeatureExtractor(BaseFeaturesExtractor):
    """为STDMA调度设计的GAT特征提取器"""

    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        total_dim = observation_space.shape[0]
        try:
            self.num_nodes = config.NUMBER_OF_DRONES  # 从配置获取无人机数量
        except:
            self.num_nodes = 10  # 如果无法获取配置，使用默认值
            logger.warning("无法从config获取NUMBER_OF_DRONES，使用默认值10")

        # 假设环境观察的维度分布如下
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
            logger.error(f"Error in _unflatten_observation: {str(e)}")
            logger.error(f"Input shape: {flat_obs.shape}")
            logger.error(f"Current index: {idx}")
            logger.error(f"Batch size: {batch_size}")
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
            logger.error(f"Error in forward pass: {str(e)}")
            logger.error(f"Observations shape: {observations.shape}")
            import traceback
            traceback.print_exc()
            raise e


class SimpleMLP(BaseFeaturesExtractor):
    """简单MLP特征提取器，用于除GAT-PPO外的其他算法"""

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


# ---------- 环境创建部分 ----------#

def make_env(rank, n_drones=10, num_slots=10, seed=None):
    """创建环境的工厂函数"""

    def _init():
        try:
            # 创建基础环境
            env = simpy.Environment()
            channel_states = {i: simpy.Resource(env, capacity=1) for i in range(n_drones)}

            # 创建模拟器
            simulator = Simulator(
                seed=seed or 42 + rank,
                env=env,
                channel_states=channel_states,
                n_drones=n_drones
            )

            # 创建STDMA环境
            stdma_env = StdmaEnv(
                simulator=simulator,
                num_nodes=n_drones,
                num_slots=num_slots
            )

            # 添加Monitor包装器
            env_log_dir = os.path.join(log_dir, f"env_{rank}")
            os.makedirs(env_log_dir, exist_ok=True)
            env = Monitor(stdma_env, env_log_dir)

            # 设置随机种子
            if seed is not None:
                env.reset(seed=seed + rank)

            return env

        except Exception as e:
            logger.error(f"创建环境失败: {e}")
            import traceback
            traceback.print_exc()
            raise e

    return _init


# 连续动作空间包装器（用于SAC/TD3）
class DiscreteToContinuousSAC(gym.ActionWrapper):
    """SAC的离散到连续动作空间转换包装器"""

    def __init__(self, env):
        super().__init__(env)
        n_actions = env.action_space.n
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,))
        self.n_actions = n_actions

    def action(self, action):
        discrete_action = int(action[0] * self.n_actions)
        discrete_action = min(discrete_action, self.n_actions - 1)
        return discrete_action


class DiscreteToContinuousTD3(gym.ActionWrapper):
    """TD3的离散到连续动作空间转换包装器"""

    def __init__(self, env):
        super().__init__(env)
        n_actions = env.action_space.n
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.n_actions = n_actions

    def action(self, action):
        discrete_action = int((action[0] + 1) / 2 * self.n_actions)
        discrete_action = min(discrete_action, self.n_actions - 1)
        return discrete_action


# ---------- 算法训练和评估部分 ----------#

class MetricsCallback:
    """指标收集回调"""

    def __init__(self, trainer, eval_freq=5000):
        self.trainer = trainer
        self.eval_freq = eval_freq
        self.last_timestep = 0

    def __call__(self, locals, globals):
        try:
            if locals['self'].num_timesteps - self.last_timestep >= self.eval_freq:
                self.last_timestep = locals['self'].num_timesteps
                self.trainer._evaluate_and_record(locals['self'], locals['self'].num_timesteps)
        except Exception as e:
            logger.error(f"MetricsCallback错误: {e}")
        return True


class AlgorithmTrainer:
    """算法训练和评估类"""

    def __init__(self, algorithm_name, n_drones=10, num_slots=10, seed=42):
        self.algorithm_name = algorithm_name
        self.n_drones = n_drones
        self.num_slots = num_slots
        self.seed = seed

        # 创建算法专用日志目录
        self.algo_log_dir = os.path.join(log_dir, algorithm_name)
        os.makedirs(self.algo_log_dir, exist_ok=True)

        # 创建环境
        try:
            if algorithm_name in ["SAC", "TD3"]:
                # 连续动作空间算法需要特殊处理
                wrapper_class = DiscreteToContinuousSAC if algorithm_name == "SAC" else DiscreteToContinuousTD3

                # 创建基础环境
                base_env = make_env(0, n_drones, num_slots, seed)()
                # 包装为连续动作空间
                wrapped_env = wrapper_class(base_env)
                # 再包装为向量环境
                self.env = VecMonitor(
                    DummyVecEnv([lambda: wrapped_env]),
                    filename=os.path.join(self.algo_log_dir, "env/monitor.csv")
                )

                # 同样处理评估环境
                eval_base_env = make_env(1, n_drones, num_slots, seed)()
                eval_wrapped_env = wrapper_class(eval_base_env)
                self.eval_env = VecMonitor(
                    DummyVecEnv([lambda: eval_wrapped_env]),
                    filename=os.path.join(self.algo_log_dir, "eval_env/monitor.csv")
                )
            else:
                # 离散动作空间算法直接使用向量化环境
                self.env = make_vec_env(
                    lambda: make_env(0, n_drones, num_slots, seed)(),
                    n_envs=1,
                    monitor_dir=os.path.join(self.algo_log_dir, "env")
                )

                # 创建评估环境
                self.eval_env = make_vec_env(
                    lambda: make_env(1, n_drones, num_slots, seed)(),
                    n_envs=1,
                    monitor_dir=os.path.join(self.algo_log_dir, "eval_env")
                )
        except Exception as e:
            logger.error(f"创建环境失败: {e}")
            import traceback
            traceback.print_exc()
            raise e

        # 统计数据
        self.train_stats = {
            'timesteps': [],
            'rewards': [],
            'reuse_ratio': [],
            'qos_satisfaction': []
        }

        logger.info(f"初始化 {algorithm_name} 训练器完成")

    def _create_model(self):
        """创建对应算法的模型"""
        try:
            if self.algorithm_name == "GAT-PPO":
                # GAT-PPO使用图注意力网络
                policy_kwargs = dict(
                    features_extractor_class=StdmaGATFeatureExtractor,
                    features_extractor_kwargs=dict(features_dim=256),
                    net_arch=dict(pi=[128, 64], vf=[128, 64])
                )

                model = PPO(
                    "MlpPolicy",
                    self.env,
                    learning_rate=3e-4,
                    n_steps=1024,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=0.01,
                    policy_kwargs=policy_kwargs,
                    verbose=1,
                    tensorboard_log=os.path.join(self.algo_log_dir, "tb")
                )

            elif self.algorithm_name == "PPO":
                # 标准PPO使用MLP
                policy_kwargs = dict(
                    features_extractor_class=SimpleMLP,
                    features_extractor_kwargs=dict(features_dim=256),
                    net_arch=dict(pi=[128, 64], vf=[128, 64])
                )

                model = PPO(
                    "MlpPolicy",
                    self.env,
                    learning_rate=3e-4,
                    n_steps=1024,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=0.01,
                    policy_kwargs=policy_kwargs,
                    verbose=1,
                    tensorboard_log=os.path.join(self.algo_log_dir, "tb")
                )

            elif self.algorithm_name == "DQN":
                policy_kwargs = dict(
                    features_extractor_class=SimpleMLP,
                    features_extractor_kwargs=dict(features_dim=256)
                )

                model = DQN(
                    "MlpPolicy",
                    self.env,
                    learning_rate=1e-4,
                    buffer_size=10000,
                    learning_starts=1000,
                    batch_size=64,
                    tau=0.1,
                    gamma=0.99,
                    train_freq=4,
                    gradient_steps=1,
                    target_update_interval=1000,
                    policy_kwargs=policy_kwargs,
                    verbose=1,
                    tensorboard_log=os.path.join(self.algo_log_dir, "tb")
                )

            elif self.algorithm_name == "SAC":
                policy_kwargs = dict(
                    features_extractor_class=SimpleMLP,
                    features_extractor_kwargs=dict(features_dim=256),
                    net_arch=dict(pi=[128, 64], qf=[128, 64])
                )

                model = SAC(
                    "MlpPolicy",
                    self.env,
                    learning_rate=3e-4,
                    buffer_size=10000,
                    learning_starts=1000,
                    batch_size=64,
                    tau=0.005,
                    gamma=0.99,
                    train_freq=1,
                    gradient_steps=1,
                    policy_kwargs=policy_kwargs,
                    verbose=1,
                    tensorboard_log=os.path.join(self.algo_log_dir, "tb")
                )

            elif self.algorithm_name == "TD3":
                policy_kwargs = dict(
                    features_extractor_class=SimpleMLP,
                    features_extractor_kwargs=dict(features_dim=256),
                    net_arch=dict(pi=[128, 64], qf=[128, 64])
                )

                model = TD3(
                    "MlpPolicy",
                    self.env,
                    learning_rate=3e-4,
                    buffer_size=10000,
                    learning_starts=1000,
                    batch_size=64,
                    tau=0.005,
                    gamma=0.99,
                    train_freq=1,
                    policy_delay=2,
                    policy_kwargs=policy_kwargs,
                    verbose=1,
                    tensorboard_log=os.path.join(self.algo_log_dir, "tb")
                )

            else:
                raise ValueError(f"不支持的算法: {self.algorithm_name}")

            return model

        except Exception as e:
            logger.error(f"创建{self.algorithm_name}模型失败: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def train(self, total_timesteps=200000, eval_freq=5000):
        """训练模型并定期评估"""
        logger.info(f"开始训练 {self.algorithm_name} 算法, 总步数: {total_timesteps}")

        model = self._create_model()
        start_time = time.time()

        # 创建回调函数
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=os.path.join(self.algo_log_dir, "best_model"),
            log_path=self.algo_log_dir,
            eval_freq=eval_freq,
            deterministic=True,
            render=False
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=os.path.join(self.algo_log_dir, "checkpoints"),
            name_prefix=f"{self.algorithm_name}_model"
        )

        metrics_callback = MetricsCallback(self, eval_freq)

        # 训练模型
        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=[eval_callback, checkpoint_callback, metrics_callback],
                progress_bar=True
            )
        except Exception as e:
            logger.error(f"{self.algorithm_name} 训练中断: {str(e)}")
            import traceback
            traceback.print_exc()

        # 保存最终模型
        final_model_path = os.path.join(self.algo_log_dir, "final_model.zip")
        model.save(final_model_path)

        training_time = time.time() - start_time
        logger.info(f"{self.algorithm_name} 训练完成，耗时: {training_time:.2f}秒")

        # 记录训练时间
        self.train_stats['training_time'] = training_time

        # 保存训练统计数据
        self._save_stats()

        return model, final_model_path

    def _evaluate_and_record(self, model, timesteps, n_episodes=5):
        """评估模型并记录指标"""
        env = self.eval_env

        total_reward = 0
        total_reuse_ratio = 0
        total_qos_satisfaction = 0
        count_samples = 0

        for episode in range(n_episodes):
            obs = env.reset()[0]
            done = False
            episode_reward = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward[0]

                # 提取调度信息
                try:
                    base_env = env.envs[0]
                    if hasattr(base_env, 'unwrapped'):
                        base_env = base_env.unwrapped

                    if hasattr(base_env, 'current_schedule'):
                        schedule = base_env.current_schedule

                        # 计算复用率
                        if schedule:
                            total_assignments = sum(len(nodes) for nodes in schedule.values())
                            reuse_ratio = total_assignments / len(schedule) if len(schedule) > 0 else 0
                            total_reuse_ratio += reuse_ratio
                            count_samples += 1

                    # 提取QoS满足率
                    if hasattr(base_env, '_estimate_delay') and hasattr(base_env, 'current_requirement'):
                        delay = base_env._estimate_delay()
                        if base_env.current_requirement and base_env.current_requirement.delay_requirement:
                            qos_satisfied = float(delay <= base_env.current_requirement.delay_requirement)
                            total_qos_satisfaction += qos_satisfied
                except Exception as e:
                    logger.warning(f"获取环境状态时出错: {e}")

            total_reward += episode_reward

        # 计算平均值
        avg_reward = total_reward / n_episodes
        avg_reuse_ratio = total_reuse_ratio / max(1, count_samples)
        avg_qos_satisfaction = total_qos_satisfaction / max(1, count_samples)

        # 记录数据
        self.train_stats['timesteps'].append(timesteps)
        self.train_stats['rewards'].append(avg_reward)
        self.train_stats['reuse_ratio'].append(avg_reuse_ratio)
        self.train_stats['qos_satisfaction'].append(avg_qos_satisfaction)

        logger.info(
            f"{self.algorithm_name} 在 {timesteps} 步后评估: 平均奖励={avg_reward:.2f}, 复用率={avg_reuse_ratio:.2f}, QoS满足率={avg_qos_satisfaction:.2f}")

    def _save_stats(self):
        """保存训练统计数据"""
        stats_df = pd.DataFrame({
            'timesteps': self.train_stats['timesteps'],
            'rewards': self.train_stats['rewards'],
            'reuse_ratio': self.train_stats['reuse_ratio'],
            'qos_satisfaction': self.train_stats['qos_satisfaction']
        })

        stats_path = os.path.join(self.algo_log_dir, "training_stats.csv")
        stats_df.to_csv(stats_path, index=False)
        logger.info(f"{self.algorithm_name} 训练统计数据已保存到 {stats_path}")

    def evaluate(self, model_path=None, n_episodes=50):
        """详细评估模型性能"""
        logger.info(f"开始评估 {self.algorithm_name} 算法")

        # 加载模型
        if model_path is None:
            model_path = os.path.join(self.algo_log_dir, "best_model/best_model.zip")

        if not os.path.exists(model_path):
            logger.warning(f"未找到模型: {model_path}")
            return None

        try:
            # 根据算法类型加载模型
            if self.algorithm_name == "GAT-PPO" or self.algorithm_name == "PPO":
                model = PPO.load(model_path)
            elif self.algorithm_name == "DQN":
                model = DQN.load(model_path)
            elif self.algorithm_name == "SAC":
                model = SAC.load(model_path)
            elif self.algorithm_name == "TD3":
                model = TD3.load(model_path)

            # 评估指标
            metrics = {
                'rewards': [],
                'episode_lengths': [],
                'reuse_ratio': [],
                'interference_rate': [],
                'qos_satisfaction': [],
                'avg_delay': []
            }

            # 执行评估
            env = self.eval_env

            for episode in range(n_episodes):
                obs = env.reset()[0]
                done = [False]
                episode_reward = 0
                episode_length = 0

                # 评估指标临时存储
                episode_reuse_ratios = []
                episode_interference_rates = []
                episode_qos_satisfactions = []
                episode_delays = []

                while not done[0]:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = env.step(action)
                    episode_reward += reward[0]
                    episode_length += 1

                    # 收集调度指标
                    try:
                        base_env = env.envs[0]
                        if hasattr(base_env, 'unwrapped'):
                            base_env = base_env.unwrapped

                        if hasattr(base_env, 'current_schedule'):
                            schedule = base_env.current_schedule

                            if schedule:
                                # 计算复用率
                                total_assignments = sum(len(nodes) for nodes in schedule.values())
                                reuse_ratio = total_assignments / len(schedule) if len(schedule) > 0 else 0
                                episode_reuse_ratios.append(reuse_ratio)

                                # 计算干扰率
                                interference_count = 0
                                for slot, nodes in schedule.items():
                                    for i, node1 in enumerate(nodes):
                                        for node2 in nodes[i + 1:]:
                                            if hasattr(base_env, 'topology_matrix') and base_env.topology_matrix[node1][
                                                node2] == 1:
                                                interference_count += 1

                                interference_rate = interference_count / total_assignments if total_assignments > 0 else 0
                                episode_interference_rates.append(interference_rate)

                        # 收集QoS和延迟指标
                        if hasattr(base_env, '_estimate_delay') and hasattr(base_env, 'current_requirement'):
                            delay = base_env._estimate_delay()
                            episode_delays.append(delay)

                            if base_env.current_requirement and base_env.current_requirement.delay_requirement:
                                qos_satisfied = float(delay <= base_env.current_requirement.delay_requirement)
                                episode_qos_satisfactions.append(qos_satisfied)
                    except Exception as e:
                        logger.warning(f"获取环境状态时出错: {e}")

                # 记录本回合的指标
                metrics['rewards'].append(episode_reward)
                metrics['episode_lengths'].append(episode_length)

                if episode_reuse_ratios:
                    metrics['reuse_ratio'].append(np.mean(episode_reuse_ratios))

                if episode_interference_rates:
                    metrics['interference_rate'].append(np.mean(episode_interference_rates))

                if episode_qos_satisfactions:
                    metrics['qos_satisfaction'].append(np.mean(episode_qos_satisfactions))

                if episode_delays:
                    metrics['avg_delay'].append(np.mean(episode_delays))

                if episode % 10 == 0:
                    logger.info(
                        f"{self.algorithm_name} 评估: 完成第 {episode}/{n_episodes} 回合, 奖励: {episode_reward:.2f}")

            # 计算最终指标
            results = {
                'algorithm': self.algorithm_name,
                'avg_reward': np.mean(metrics['rewards']),
                'std_reward': np.std(metrics['rewards']),
                'avg_episode_length': np.mean(metrics['episode_lengths']),
                'avg_reuse_ratio': np.mean(metrics['reuse_ratio']) if metrics['reuse_ratio'] else 0,
                'avg_interference_rate': np.mean(metrics['interference_rate']) if metrics['interference_rate'] else 0,
                'qos_satisfaction_rate': np.mean(metrics['qos_satisfaction']) * 100 if metrics[
                    'qos_satisfaction'] else 0,
                'avg_delay': np.mean(metrics['avg_delay']) if metrics['avg_delay'] else 0
            }

            # 如果有训练时间，也加入结果
            if 'training_time' in self.train_stats:
                results['training_time'] = self.train_stats['training_time']

            # 保存评估结果
            results_df = pd.DataFrame([results])
            results_path = os.path.join(self.algo_log_dir, "evaluation_results.csv")
            results_df.to_csv(results_path, index=False)

            logger.info(f"{self.algorithm_name} 评估结果:")
            for key, value in results.items():
                if key != 'algorithm' and not np.isnan(value):
                    logger.info(f"  {key}: {value:.4f}")

            return results

        except Exception as e:
            logger.error(f"评估 {self.algorithm_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            return None


# ---------- 结果可视化部分 ----------#

def generate_learning_curves(all_stats):
    """生成学习曲线比较图"""
    metrics = ['rewards', 'reuse_ratio', 'qos_satisfaction']
    titles = ['平均奖励', '时隙复用率', 'QoS满足率']

    plt.figure(figsize=(18, 6))
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        plt.subplot(1, 3, i + 1)

        for algo_name, stats in all_stats.items():
            if 'timesteps' in stats and metric in stats:
                # 确保数据长度一致
                min_len = min(len(stats['timesteps']), len(stats[metric]))
                timesteps = stats['timesteps'][:min_len]
                values = stats[metric][:min_len]

                # 平滑处理
                if len(values) > 10:
                    window_size = max(1, len(values) // 10)
                    weights = np.ones(window_size) / window_size
                    values_smooth = np.convolve(values, weights, mode='valid')
                    timesteps_smooth = timesteps[window_size - 1:]
                    plt.plot(timesteps_smooth, values_smooth, label=algo_name, linewidth=2)
                else:
                    plt.plot(timesteps, values, label=algo_name, linewidth=2)

        plt.title(title, fontsize=14)
        plt.xlabel('训练步数', fontsize=12)
        plt.ylabel(title, fontsize=12)
        plt.grid(True, alpha=0.3)

        if i == 0:
            plt.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'learning_curves.png'), dpi=300, bbox_inches='tight')
    logger.info(f"学习曲线已保存到 {os.path.join(log_dir, 'learning_curves.png')}")


def generate_comparison_plots(results):
    """生成算法性能对比图表"""
    if not results:
        logger.warning("没有结果可供绘图")
        return

    # 创建DataFrame
    df = pd.DataFrame(results)

    # 要对比的指标
    metrics = [
        ('avg_reward', '平均奖励'),
        ('avg_reuse_ratio', '时隙复用率'),
        ('qos_satisfaction_rate', 'QoS满足率(%)'),
        ('avg_interference_rate', '干扰率'),
        ('avg_delay', '平均延迟(μs)')
    ]

    if 'training_time' in df.columns:
        metrics.append(('training_time', '训练时间(秒)'))

    # 条形图比较
    plt.figure(figsize=(15, 12))
    for i, (metric, title) in enumerate(metrics, 1):
        if metric not in df.columns:
            continue

        plt.subplot(3, 2, i)
        colors = ['#FF6F61', '#6B5B95', '#88B04B', '#F7CAC9', '#92A8D1']  # 自定义颜色

        bars = plt.bar(df['algorithm'], df[metric], color=colors[:len(df)])

        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.,
                     height + 0.01 * max(df[metric]),
                     f'{height:.2f}',
                     ha='center', va='bottom', rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "metrics_comparison.png"), dpi=300, bbox_inches='tight')
    logger.info(f"性能指标对比图已保存到 {os.path.join(log_dir, 'metrics_comparison.png')}")

    # 生成雷达图
    metrics_for_radar = ['avg_reward', 'avg_reuse_ratio', 'qos_satisfaction_rate']

    # 检查是否有其他可用指标
    if 'avg_interference_rate' in df.columns:
        metrics_for_radar.append('avg_interference_rate')
    if 'avg_delay' in df.columns:
        metrics_for_radar.append('avg_delay')
    if 'training_time' in df.columns:
        metrics_for_radar.append('training_time')

    # 确保所有指标都存在
    metrics_for_radar = [m for m in metrics_for_radar if m in df.columns]

    if len(metrics_for_radar) >= 3:  # 雷达图至少需要3个指标
        plt.figure(figsize=(10, 10))

        # 标准化数据
        df_normalized = df.copy()
        for metric in metrics_for_radar:
            if df[metric].max() > df[metric].min():
                if metric in ['avg_interference_rate', 'avg_delay', 'training_time']:  # 对于这些指标，值越小越好
                    df_normalized[metric] = 1 - (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
                else:
                    df_normalized[metric] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
            else:
                if metric in ['avg_interference_rate', 'avg_delay', 'training_time']:
                    df_normalized[metric] = 1  # 如果所有值相同且越小越好，给最高分
                else:
                    df_normalized[metric] = df[metric] / df[metric].max() if df[metric].max() > 0 else df[metric]

        # 设置雷达图
        angles = np.linspace(0, 2 * np.pi, len(metrics_for_radar), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形

        # 指标名称映射
        metric_labels = {
            'avg_reward': '平均奖励',
            'avg_reuse_ratio': '时隙复用率',
            'qos_satisfaction_rate': 'QoS满足率',
            'avg_interference_rate': '抗干扰能力',
            'avg_delay': '低延迟能力',
            'training_time': '训练效率'
        }

        labels = [metric_labels.get(m, m) for m in metrics_for_radar]

        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))

        for _, row in df_normalized.iterrows():
            algorithm = row['algorithm']
            values = [row[metric] for metric in metrics_for_radar]
            values += values[:1]  # 闭合数据

            ax.plot(angles, values, linewidth=2, label=algorithm)
            ax.fill(angles, values, alpha=0.1)

        # 设置图表标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], fontsize=10)

        # 添加图例
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
        plt.title('算法性能雷达图', size=18, y=1.05)

        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "radar_comparison.png"), dpi=300, bbox_inches='tight')
        logger.info(f"雷达图已保存到 {os.path.join(log_dir, 'radar_comparison.png')}")


# ---------- 主函数部分 ----------#

def main(algorithms=None, total_timesteps=200000, n_drones=10, num_slots=10, eval_episodes=50, seed=42):
    """执行完整的算法对比流程"""
    if algorithms is None:
        algorithms = ["PPO", "GAT-PPO", "DQN", "SAC", "TD3"]

    logger.info(f"开始UAV网络STDMA算法对比: {', '.join(algorithms)}")
    logger.info(f"环境配置: {n_drones}架无人机, {num_slots}个时隙, 训练步数: {total_timesteps}")

    # 设置随机种子
    set_random_seed(seed)
    torch.manual_seed(seed)

    # 训练和评估每种算法
    all_trainers = {}
    all_results = []
    all_stats = {}

    for algo_name in algorithms:
        try:
            logger.info(f"\n{'=' * 50}\n开始处理 {algo_name} 算法\n{'=' * 50}")

            trainer = AlgorithmTrainer(
                algorithm_name=algo_name,
                n_drones=n_drones,
                num_slots=num_slots,
                seed=seed
            )
            all_trainers[algo_name] = trainer

            # 训练模型
            model, model_path = trainer.train(
                total_timesteps=total_timesteps,
                eval_freq=10000
            )

            # 评估模型
            results = trainer.evaluate(
                model_path=model_path,
                n_episodes=eval_episodes
            )

            if results:
                all_results.append(results)

            # 收集训练统计数据
            all_stats[algo_name] = trainer.train_stats

        except Exception as e:
            logger.error(f"{algo_name} 处理失败: {str(e)}")
            import traceback
            traceback.print_exc()

    # 生成比较图表
    if all_results:
        generate_comparison_plots(all_results)

    if all_stats:
        generate_learning_curves(all_stats)

    # 保存综合结果
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_path = os.path.join(log_dir, "all_results.csv")
        results_df.to_csv(results_path, index=False)
        logger.info(f"综合结果已保存到 {results_path}")

    logger.info(f"算法对比评估完成，结果保存在: {log_dir}")
    return all_results, all_stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='UAV网络STDMA多算法对比评估')
    parser.add_argument('--algorithms', nargs='+', default=["PPO", "GAT-PPO", "DQN", "SAC", "TD3"],
                        help='要对比的算法列表')
    parser.add_argument('--timesteps', type=int, default=200000,
                        help='训练步数')
    parser.add_argument('--n_drones', type=int, default=10,
                        help='无人机数量')
    parser.add_argument('--num_slots', type=int, default=10,
                        help='时隙数量')
    parser.add_argument('--eval_episodes', type=int, default=50,
                        help='评估回合数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    args = parser.parse_args()

    try:
        main(
            algorithms=args.algorithms,
            total_timesteps=args.timesteps,
            n_drones=args.n_drones,
            num_slots=args.num_slots,
            eval_episodes=args.eval_episodes,
            seed=args.seed
        )
    except Exception as e:
        logger.error(f"运行失败: {e}")
        import traceback

        traceback.print_exc()