#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UAV网络环境适配不同强化学习算法的代码

此脚本提供了一组辅助函数，用于将rl_environment.py和stdma.py中的环境适配到
不同的强化学习算法（PPO, GAT-PPO, DQN, SAC, TD3）。
"""

import os
import logging
import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import PPO, DQN, SAC, TD3
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

# 导入环境模块
try:
    from rl_environment import StdmaEnv
except ImportError:
    pass  # 在实际使用时需要确保能够导入

logger = logging.getLogger(__name__)


# ---------- 环境包装器 ----------#

class DiscreteToContinuousSAC(gym.ActionWrapper):
    """SAC的离散到连续动作空间转换包装器"""

    def __init__(self, env):
        super().__init__(env)
        n_actions = env.action_space.n
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,))
        self.n_actions = n_actions
        logger.info(f"创建SAC连续动作空间: {self.action_space} (原离散空间: {n_actions})")

    def action(self, action):
        try:
            # 连续到离散的转换
            discrete_action = int(action[0] * self.n_actions)
            discrete_action = min(discrete_action, self.n_actions - 1)
            return discrete_action
        except Exception as e:
            logger.error(f"SAC动作转换错误: {e}")
            return 0  # 默认动作


class DiscreteToContinuousTD3(gym.ActionWrapper):
    """TD3的离散到连续动作空间转换包装器"""

    def __init__(self, env):
        super().__init__(env)
        n_actions = env.action_space.n
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.n_actions = n_actions
        logger.info(f"创建TD3连续动作空间: {self.action_space} (原离散空间: {n_actions})")

    def action(self, action):
        try:
            # 连续到离散的转换（TD3使用tanh输出，范围为[-1,1]）
            discrete_action = int((action[0] + 1) / 2 * self.n_actions)
            discrete_action = min(discrete_action, self.n_actions - 1)
            return discrete_action
        except Exception as e:
            logger.error(f"TD3动作转换错误: {e}")
            return 0  # 默认动作


# ---------- STDMA类修改 ----------#

def initialize_rl_controller(self, algorithm_type="PPO"):
    """初始化强化学习控制器，支持多种算法类型

    参数:
        algorithm_type: 算法类型，支持 "PPO", "GAT-PPO", "DQN", "SAC", "TD3"

    返回:
        (use_rl, rl_model, rl_env): 是否使用RL、模型实例和环境实例的元组
    """
    if algorithm_type == "PPO":
        return self._initialize_ppo_rl_controller()
    elif algorithm_type == "GAT-PPO":
        return self._initialize_gat_ppo_controller()
    elif algorithm_type == "DQN":
        return self._initialize_dqn_controller()
    elif algorithm_type == "SAC":
        return self._initialize_sac_controller()
    elif algorithm_type == "TD3":
        return self._initialize_td3_controller()
    else:
        logging.warning(f"未知的算法类型: {algorithm_type}，回退到PPO")
        return self._initialize_ppo_rl_controller()


def _initialize_gat_ppo_controller(self):
    """初始化GAT-PPO控制器"""
    try:
        from stable_baselines3 import PPO
        import os
        import torch

        # 导入GAT特征提取器
        from torch_geometric.nn import GATConv

        class GATBlock(nn.Module):
            """使用多头注意力机制的GAT块"""

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
                self.attention_weights = None

                def hook_fn(module, input, output):
                    if hasattr(module, 'alpha'):
                        self.attention_weights = module.alpha.detach()

                self.gat1.register_forward_hook(hook_fn)

            def forward(self, x, edge_index):
                identity = self.proj(x)
                x = self.gat1(x, edge_index)
                x = self.norm1(x)
                x = torch.nn.functional.elu(x)
                x = self.dropout(x)
                x = self.gat2(x, edge_index)
                x = self.norm2(x)
                return torch.nn.functional.elu(x + identity)

            def get_attention_weights(self):
                return self.attention_weights

        class StdmaGATFeatureExtractor(BaseFeaturesExtractor):
            """为STDMA调度设计的GAT特征提取器"""

            def __init__(self, observation_space, features_dim=256):
                super().__init__(observation_space, features_dim)

                total_dim = observation_space.shape[0]
                self.num_nodes = self.my_drone.simulator.n_drones

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
                    nn.Linear(4, self.node_feat_dim),
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

                # 特征融合层
                total_features = self.hidden_dim + 64 + 32
                self.fusion = nn.Sequential(
                    nn.Linear(total_features, features_dim),
                    nn.ReLU(),
                    nn.LayerNorm(features_dim),
                    nn.Dropout(0.1)
                )

                # 将my_drone引用保存为属性，以便在forward中使用
                self.my_drone = self

            def _unflatten_observation(self, flat_obs):
                """将扁平化的观察还原为结构化格式"""
                batch_size = flat_obs.size(0) if len(flat_obs.shape) > 1 else 1
                if batch_size > 1:
                    flat_obs = flat_obs.view(batch_size, -1)

                idx = 0
                result = {}

                # 还原拓扑特征
                topology_size = self.dims['topology']
                result['topology'] = flat_obs[:, idx:idx + topology_size].reshape(
                    batch_size, self.num_nodes, self.num_nodes)
                idx += topology_size

                # 还原节点位置
                position_size = self.dims['position']
                result['position'] = flat_obs[:, idx:idx + position_size].reshape(
                    batch_size, self.num_nodes, 3)
                idx += position_size

                # 还原路由路径
                routing_size = self.dims['routing']
                result['routing'] = flat_obs[:, idx:idx + routing_size].reshape(
                    batch_size, self.num_nodes)
                idx += routing_size

                # 还原链路生命期
                lifetime_size = self.dims['link_lifetime']
                result['link_lifetime'] = flat_obs[:, idx:idx + lifetime_size].reshape(
                    batch_size, self.num_nodes, self.num_nodes)
                idx += lifetime_size

                # 还原业务信息
                traffic_size = self.dims['traffic']
                result['traffic'] = flat_obs[:, idx:idx + traffic_size].reshape(
                    batch_size, 5)
                idx += traffic_size

                # 还原节点度数
                degrees_size = self.dims['node_degrees']
                result['node_degrees'] = flat_obs[:, idx:idx + degrees_size].reshape(
                    batch_size, self.num_nodes)

                return result

            def forward(self, observations):
                """前向传播"""
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

        # 构建模型路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(current_dir, "rl_controller/logs/GAT_PPO")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "best_model/best_model.zip")

        # 创建RL环境
        from rl_environment import StdmaEnv
        rl_env = StdmaEnv(
            simulator=self.simulator,
            num_nodes=self.my_drone.simulator.n_drones,
            num_slots=self.num_slots
        )

        if os.path.exists(model_path):
            # 加载现有模型
            rl_model = PPO.load(model_path)
            use_rl = True
            logging.info(f"成功加载GAT-PPO模型: {model_path}")
        else:
            # 创建并训练新模型
            logging.info("未找到GAT-PPO模型，正在创建新模型...")

            # 设置策略参数
            policy_kwargs = dict(
                features_extractor_class=StdmaGATFeatureExtractor,
                features_extractor_kwargs=dict(features_dim=256),
                net_arch=dict(pi=[128, 64], vf=[128, 64])
            )

            # 包装环境
            vec_env = DummyVecEnv([lambda: rl_env])

            # 创建PPO模型
            rl_model = PPO(
                "MlpPolicy",
                vec_env,
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
                policy_kwargs=policy_kwargs,
                verbose=1
            )

            # 训练模型（简短训练）
            rl_model.learn(total_timesteps=10000)

            # 保存模型
            os.makedirs(os.path.join(model_dir, "best_model"), exist_ok=True)
            rl_model.save(model_path)
            use_rl = True
            logging.info(f"创建并保存了新GAT-PPO模型: {model_path}")

        return use_rl, rl_model, rl_env

    except Exception as e:
        logging.error(f"初始化GAT-PPO控制器失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None, None


def _initialize_dqn_controller(self):
    """初始化DQN控制器"""
    try:
        from stable_baselines3 import DQN
        import os

        # 构建模型路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(current_dir, "rl_controller/dqn_models")
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, "stdma_dqn_model.zip")

        # 创建RL环境
        from rl_environment import StdmaEnv
        rl_env = StdmaEnv(
            simulator=self.simulator,
            num_nodes=self.my_drone.simulator.n_drones,
            num_slots=self.num_slots
        )

        if os.path.exists(model_path):
            rl_model = DQN.load(model_path)
            use_rl = True
            logging.info(f"成功加载DQN模型: {model_path}")
        else:
            # 创建并训练新模型
            logging.info("未找到DQN模型，正在创建新模型...")

            # 包装环境
            vec_env = DummyVecEnv([lambda: rl_env])

            rl_model = DQN(
                "MlpPolicy",
                vec_env,
                learning_rate=1e-4,
                buffer_size=10000,
                learning_starts=1000,
                batch_size=64,
                tau=0.1,
                gamma=0.99,
                train_freq=4,
                gradient_steps=1,
                target_update_interval=1000,
                verbose=1
            )

            # 训练模型（简短训练）
            rl_model.learn(total_timesteps=10000)

            # 保存模型
            rl_model.save(model_path)
            use_rl = True
            logging.info(f"创建并保存了新DQN模型: {model_path}")

        return use_rl, rl_model, rl_env

    except Exception as e:
        logging.error(f"初始化DQN控制器失败: {str(e)}")
        return False, None, None


def _initialize_sac_controller(self):
    """初始化SAC控制器"""
    try:
        from stable_baselines3 import SAC
        import os
        import gymnasium as gym

        # 构建模型路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(current_dir, "rl_controller/sac_models")
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, "stdma_sac_model.zip")

        # 创建RL环境
        from rl_environment import StdmaEnv
        rl_env = StdmaEnv(
            simulator=self.simulator,
            num_nodes=self.my_drone.simulator.n_drones,
            num_slots=self.num_slots
        )

        # SAC需要连续动作空间，创建一个包装器
        wrapped_env = DiscreteToContinuousSAC(rl_env)

        if os.path.exists(model_path):
            rl_model = SAC.load(model_path)
            use_rl = True
            logging.info(f"成功加载SAC模型: {model_path}")
        else:
            # 创建并训练新模型
            logging.info("未找到SAC模型，正在创建新模型...")

            # 包装环境
            vec_env = DummyVecEnv([lambda: wrapped_env])

            rl_model = SAC(
                "MlpPolicy",
                vec_env,
                learning_rate=3e-4,
                buffer_size=10000,
                learning_starts=1000,
                batch_size=64,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                verbose=1
            )

            # 训练模型（简短训练）
            rl_model.learn(total_timesteps=10000)

            # 保存模型
            rl_model.save(model_path)
            use_rl = True
            logging.info(f"创建并保存了新SAC模型: {model_path}")

        return use_rl, rl_model, wrapped_env

    except Exception as e:
        logging.error(f"初始化SAC控制器失败: {str(e)}")
        return False, None, None


def _initialize_td3_controller(self):
    """初始化TD3控制器"""
    try:
        from stable_baselines3 import TD3
        import os
        import gymnasium as gym

        # 构建模型路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(current_dir, "rl_controller/td3_models")
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, "stdma_td3_model.zip")

        # 创建RL环境
        from rl_environment import StdmaEnv
        rl_env = StdmaEnv(
            simulator=self.simulator,
            num_nodes=self.my_drone.simulator.n_drones,
            num_slots=self.num_slots
        )

        # TD3也需要连续动作空间
        wrapped_env = DiscreteToContinuousTD3(rl_env)

        if os.path.exists(model_path):
            rl_model = TD3.load(model_path)
            use_rl = True
            logging.info(f"成功加载TD3模型: {model_path}")
        else:
            # 创建并训练新模型
            logging.info("未找到TD3模型，正在创建新模型...")

            # 包装环境
            vec_env = DummyVecEnv([lambda: wrapped_env])

            rl_model = TD3(
                "MlpPolicy",
                vec_env,
                learning_rate=3e-4,
                buffer_size=10000,
                learning_starts=1000,
                batch_size=64,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                policy_delay=2,
                verbose=1
            )

            # 训练模型（简短训练）
            rl_model.learn(total_timesteps=10000)

            # 保存模型
            rl_model.save(model_path)
            use_rl = True
            logging.info(f"创建并保存了新TD3模型: {model_path}")

        return use_rl, rl_model, wrapped_env

    except Exception as e:
        logging.error(f"初始化TD3控制器失败: {str(e)}")
        return False, None, None


# ---------- 使用方法示例 ----------#

def integrate_with_stdma():
    """在STDMA类中集成多算法支持的示例代码"""
    # 这段代码展示如何在stdma.py中集成多种算法

    '''
    # 在stdma.py的__init__方法中添加

    # 初始化强化学习组件（添加算法类型参数）
    # 支持的算法: "PPO", "GAT-PPO", "DQN", "SAC", "TD3"
    self.use_rl, self.rl_model, self.rl_env = self._initialize_rl_controller("GAT-PPO")

    # 然后复制本文件中的所有初始化方法到stdma.py
    '''

    # STDMA类中mac_send方法修改示例（支持不同算法）
    '''
    def mac_send(self, packet):
        """MAC层发送函数"""
        if not self.slot_schedule:
            logging.error("时隙表未创建")
            return
        logging.info(f"Time {self.env.now}:UAV{self.my_drone.identifier} MAC layer received {type(packet).__name__}")

        if isinstance(packet, TrafficRequirement):
            logging.info(f"当前时隙表: {self.slot_schedule}")

            # 业务信息处理
            # ...

            yield self.env.process(self._transmit_packet(packet))
            try:
                if hasattr(self, 'use_rl') and self.use_rl and self.rl_model is not None:
                    # 获取当前算法类型
                    algo_type = self.rl_model.__class__.__name__  # 'PPO', 'DQN', 'SAC', 'TD3'

                    # 使用RL重新分配时隙
                    obs = self.rl_env.reset(requirement_data=packet)[0]
                    logging.info(f"UAV{self.my_drone.identifier} 业务路径: {packet.routing_path}")
                    new_schedule = {}

                    # 针对不同算法的处理
                    if algo_type in ["SAC", "TD3"]:
                        # 连续动作空间的算法需要特殊处理
                        for node in range(self.simulator.n_drones):
                            action, _ = self.rl_model.predict(obs, deterministic=True)
                            # 连续动作空间包装器会自动转换为离散动作
                            obs, _, done, _, _, new_schedule = self.rl_env.step(action)
                            if done:
                                break
                    else:
                        # 离散动作空间的算法(PPO, GAT-PPO, DQN)
                        for node in range(self.simulator.n_drones):
                            action, _ = self.rl_model.predict(obs, deterministic=True)
                            slot = int(action)
                            if slot not in new_schedule:
                                new_schedule[slot] = []

                            obs, _, done, _, _, new_schedule = self.rl_env.step(action)
                            if done:
                                break

                    # 更新时隙分配
                    self.slot_schedule = new_schedule
                    for node in self.simulator.drones:
                        if node.identifier != self.my_drone.identifier:
                            node.mac_protocol.slot_schedule = self.slot_schedule
                            logging.info(f"{node.identifier}基于{algo_type}模型更新时隙分配: {new_schedule}")
                else:
                    # 使用传统方法重新分配时隙
                    self.slot_schedule = self._create_tra_slot_schedule()
                    logging.info("使用传统方法更新时隙分配")

            except Exception as e:
                logging.error(f"时隙调整失败: {e}")
    '''

    pass  # 实际使用时移除此行


# ---------- 多算法对比测试 ----------#

def run_algorithm_comparison(n_drones=10, num_slots=10, seed=42):
    """运行算法对比测试的示例代码"""
    # 这段代码展示如何使用算法对比脚本

    '''
    # 导入算法对比脚本
    import sys
    sys.path.append('/path/to/script/directory')
    from complete_comparison_script import main as run_comparison

    # 运行比较
    results, stats = run_comparison(
        algorithms=["PPO", "GAT-PPO", "DQN", "SAC", "TD3"],
        total_timesteps=50000,  # 减少步数用于快速测试
        n_drones=n_drones,
        num_slots=num_slots,
        eval_episodes=10,
        seed=seed
    )

    # 分析结果
    for result in results:
        algo = result['algorithm']
        avg_reward = result['avg_reward']
        reuse_ratio = result['avg_reuse_ratio']
        print(f"{algo}: 奖励={avg_reward:.2f}, 复用率={reuse_ratio:.2f}")
    '''

    pass  # 实际使用时移除此行


if __name__ == "__main__":
    print("这是一个辅助模块，请在主程序中导入并使用相关函数。")
    print("用法示例:")
    print("  1. 在stdma.py中集成_initialize_rl_controller和其他算法初始化方法")
    print("  2. 修改初始化代码为: self.use_rl, self.rl_model, self.rl_env = self._initialize_rl_controller('GAT-PPO')")
    print("  3. 运行对比脚本: python complete_comparison_script.py")