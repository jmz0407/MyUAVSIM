import os
import numpy as np
import torch as th
import torch.nn as nn
from datetime import datetime
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict
from utils import  config
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from simulator.simulator import Simulator
import simpy
from mac.rl_controller.rl_environment import StdmaEnv
import traceback
class GraphConvBlock(nn.Module):
    """图卷积块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, out_channels, heads=4, concat=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = GATConv(out_channels, out_channels, heads=4, concat=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        return x


class StdmaFeatureExtractor(BaseFeaturesExtractor):
    """基于GNN的特征提取器 - 修复维度问题"""

    def __init__(self, observation_space, features_dim=512,):
        super().__init__(observation_space, features_dim)

        # 从观察空间中获取正确的维度
        total_dim = observation_space.shape[0]
        num_nodes = 10  # 确定节点数量为10

        # 计算每个部分的维度
        self.dims = {
            'topology': num_nodes * num_nodes,  # 10x10
            'position': num_nodes * 3,  # 10x3
            'routing': num_nodes,  # 10
            'link_lifetime': num_nodes * num_nodes,  # 10x10
            'traffic': 5,  # 5
            'node_degrees': num_nodes  # 10
        }

        print(f"Input dimension: {total_dim}")
        print(f"Calculated dimensions: {self.dims}")

        self.num_nodes = num_nodes
        self.node_feat_dim = 32
        self.hidden_dim = 64

        # 节点特征编码
        self.node_encoder = nn.Sequential(
            nn.Linear(4, self.node_feat_dim),  # 3维位置 + 1维度数
            nn.ReLU(),
            nn.LayerNorm(self.node_feat_dim)
        )

        # 简化的图卷积层
        self.gnn1 = GCNConv(self.node_feat_dim, self.hidden_dim)
        self.gnn2 = GCNConv(self.hidden_dim, self.hidden_dim)

        # 特征融合
        total_features = (
                self.hidden_dim +  # 图特征
                64 +  # 路由特征
                32  # 业务特征
        )

        self.fusion = nn.Sequential(
            nn.Linear(total_features, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim)
        )

    def _unflatten_observation(self, flat_obs):
        """将扁平化的观察还原为结构化格式"""
        try:
            batch_size = flat_obs.size(0) if len(flat_obs.shape) > 1 else 1
            if batch_size > 1:
                flat_obs = flat_obs.view(batch_size, -1)

            idx = 0
            result = {}

            # 还原拓扑特征
            topology_size = self.dims['topology']
            result['topology_features'] = flat_obs[:, idx:idx + topology_size].reshape(
                batch_size, self.num_nodes, self.num_nodes)
            idx += topology_size

            # 还原节点位置
            position_size = self.dims['position']
            result['node_positions'] = flat_obs[:, idx:idx + position_size].reshape(
                batch_size, self.num_nodes, 3)
            idx += position_size

            # 还原路由路径
            routing_size = self.dims['routing']
            result['routing_path'] = flat_obs[:, idx:idx + routing_size].reshape(
                batch_size, self.num_nodes)
            idx += routing_size

            # 还原链路生命期
            lifetime_size = self.dims['link_lifetime']
            result['link_lifetime'] = flat_obs[:, idx:idx + lifetime_size].reshape(
                batch_size, self.num_nodes, self.num_nodes)
            idx += lifetime_size

            # 还原业务信息
            traffic_size = self.dims['traffic']
            result['traffic_info'] = flat_obs[:, idx:idx + traffic_size].reshape(
                batch_size, 5)
            idx += traffic_size

            # 还原节点度数
            degrees_size = self.dims['node_degrees']
            result['node_degrees'] = flat_obs[:, idx:idx + degrees_size].reshape(
                batch_size, self.num_nodes)

            return result

        except Exception as e:
            print(f"Error in _unflatten_observation: {str(e)}")
            print(f"Input shape: {flat_obs.shape}")
            print(f"Current index: {idx}")
            raise e

    def forward(self, observations):
        """前向传播"""
        try:
            # 将observations转换为tensor
            if not isinstance(observations, torch.Tensor):
                observations = torch.tensor(observations, dtype=torch.float32)

            # 还原结构化数据
            obs_dict = self._unflatten_observation(observations)

            batch_size = observations.shape[0]

            # 处理每个批次
            graph_features_list = []
            for i in range(batch_size):
                # 获取单个样本的数据
                topology = obs_dict['topology_features'][i]
                positions = obs_dict['node_positions'][i]
                degrees = obs_dict['node_degrees'][i]

                # 构建节点特征
                node_features = torch.cat([
                    positions,
                    degrees.unsqueeze(-1)
                ], dim=-1)
                node_features = self.node_encoder(node_features)

                # 创建边索引
                edge_index = torch.nonzero(topology).t().contiguous()

                # GNN处理
                x = F.relu(self.gnn1(node_features, edge_index))
                x = F.relu(self.gnn2(x, edge_index))

                # 图级别池化
                graph_features = x.mean(dim=0)
                graph_features_list.append(graph_features)

            # 合并批次的图特征
            graph_features = torch.stack(graph_features_list)

            # 处理路由信息
            route_features = obs_dict['routing_path']
            route_encoder = nn.Linear(self.num_nodes, 64).to(observations.device)
            route_features = route_encoder(route_features)

            # 处理业务信息
            traffic_features = obs_dict['traffic_info']
            traffic_encoder = nn.Linear(5, 32).to(observations.device)
            traffic_features = traffic_encoder(traffic_features)

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
            raise e

    def _process_graph_data(self, topology, positions):
        """处理图数据"""
        # 计算节点度
        degrees = topology.sum(dim=1, keepdim=True)

        # 组合节点特征: 位置和度信息
        node_features = torch.cat([positions, degrees], dim=1)
        node_features = self.node_embedding(node_features)

        # 创建边索引
        edge_index = torch.nonzero(topology).t().contiguous()

        return node_features, edge_index



    def _create_edge_index(self, adj_matrix):
        """从邻接矩阵创建边索引"""
        edges = torch.nonzero(adj_matrix).t()
        return edges

    def _get_node_features(self, node_positions, topology_features):
        """构建节点特征"""
        # 使用节点位置信息
        node_features = self.node_embedding(node_positions)
        print(f"Node features shape: {node_features.shape}")  # 调试信息

        # 添加节点度信息
        degrees = topology_features.sum(dim=1, keepdim=True)  # 形状: [1, 1, 10]
        print(f"Degrees shape after sum: {degrees.shape}")  # 调试信息

        # 调整 degrees 的形状为 [1, 10, 1]
        degrees = degrees.permute(0, 2, 1)  # 将第 1 维和第 2 维交换，形状变为 [1, 10, 1]
        print(f"Degrees shape after permute: {degrees.shape}")  # 调试信息

        # 将 degrees 扩展为与 node_features 相同的形状
        degrees = degrees.repeat(1, 1, node_features.size(-1))  # 形状: [1, 10, 64]
        print(f"Expanded degrees shape: {degrees.shape}")  # 调试信息

        # 拼接节点特征和度信息（在特征维度上拼接）
        node_features = torch.cat([node_features, degrees], dim=-1)  # 形状: [1, 10, 128]
        print(f"Concatenated node features shape: {node_features.shape}")  # 调试信息
        return node_features


class CustomGNN(nn.Module):
    """自定义GNN层，处理动态图结构"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=False)
        self.conv2 = GATConv(hidden_dim, output_dim, heads=4, concat=False)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def make_training_env(simulator=None):
    """改进的环境创建函数"""
    env = simpy.Environment()
    n_drones = 10
    channel_states = {i: simpy.Resource(env, capacity=1)
                      for i in range(n_drones)}

    simulator = Simulator(
        seed=2024,
        env=env,
        channel_states=channel_states,
        n_drones=n_drones
    )

    # 创建环境
    env = StdmaEnv(
        simulator=simulator,
        num_nodes=config.NUMBER_OF_DRONES,
        num_slots=config.NUMBER_OF_DRONES
    )

    # 包装环境以记录数据
    from stable_baselines3.common.monitor import Monitor
    env = Monitor(env, filename=None)

    return env


def train_stdma_agent(total_timesteps=1000000):
    """训练STDMA智能体"""
    import time
    from datetime import datetime, timedelta

    # 创建日志目录
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/STDMA_{current_time}"
    os.makedirs(log_dir, exist_ok=True)

    print("Creating environment...")
    env = make_training_env()

    # 设置策略参数
    policy_kwargs = dict(
        features_extractor_class=StdmaFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[128, 64]
    )

    print("Creating model...")
    model = DQN(
        "MultiInputPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        verbose=0
    )

    # 创建进度回调
    class ProgressCallback(CheckpointCallback):
        def __init__(self, save_freq, save_path, name_prefix="rl_model", total_timesteps=0):
            super().__init__(save_freq, save_path, name_prefix)
            self.start_time = time.time()
            self.total_timesteps = total_timesteps

        def _on_step(self):
            result = super()._on_step()

            # 计算进度
            progress = self.num_timesteps / self.total_timesteps
            elapsed_time = time.time() - self.start_time

            if progress > 0:
                estimated_total_time = elapsed_time / progress
                remaining_time = estimated_total_time - elapsed_time

                # 转换为可读格式
                remaining_time = timedelta(seconds=int(remaining_time))
                elapsed_time = timedelta(seconds=int(elapsed_time))

                # 清除当前行并打印进度
                print(f"\r{' ' * 100}", end="\r")  # 清除行
                print(f"Progress: {progress:.1%} | "
                      f"Elapsed: {elapsed_time} | "
                      f"Remaining: {remaining_time} | "
                      f"Episodes: {self.n_calls}", end="\r")

            return result

    # 创建回调
    checkpoint_callback = ProgressCallback(
        save_freq=50000,
        save_path=log_dir,
        name_prefix="stdma_model",
        total_timesteps=total_timesteps
    )

    print("\nStarting training...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            log_interval=100,
            progress_bar=True
        )
        print("\nTraining completed!")

    except Exception as e:
        print(f"\nTraining error: {str(e)}")
        traceback.print_exc()
        model.save(os.path.join(log_dir, "error_model.zip"))
        raise e

    # 保存最终模型
    final_model_path = os.path.join(log_dir, "final_model.zip")
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")

    return model, log_dir


def evaluate_schedule(env, model, num_episodes=50):
    """评估已训练模型的时隙分配性能"""
    results = {
        'rewards': [],
        'slot_usage': [],
        'success_rate': 0,
        'interference_free': 0,
        'slot_reuse': []
    }

    print("\n开始评估模型...")
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")

        obs, _ = env.reset()
        episode_reward = 0
        terminated = truncated = False
        step_count = 0

        while not (terminated or truncated):
            # 使用模型预测动作
            action, _ = model.predict(obs, deterministic=True)

            # 执行动作前检查其有效性
            if hasattr(env, 'current_node') and hasattr(env, '_is_valid_assignment'):
                is_valid = env._is_valid_assignment(env.current_node, action)
                print(f"Step {step_count}: Node {env.current_node}, Action {action}, Valid: {is_valid}")

            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

            # 显示当前调度状态
            schedule = info.get('schedule', {})
            print(f"当前调度: {schedule}")

            # 如果step太多，强制终止
            if step_count > env.num_nodes * 3:  # 允许每个节点平均2次尝试
                print(f"Warning: Exceeding maximum steps ({step_count}), forcing termination")
                truncated = True

        print(f"\nEpisode {episode + 1} 完成:")
        print(f"总步数: {step_count}")
        print(f"总奖励: {episode_reward}")
        print(f"最终调度: {schedule}")

        # 记录结果
        results['rewards'].append(episode_reward)
        results['slot_usage'].append(len(schedule))

        # 检查干扰
        has_interference = False
        for slot, nodes in schedule.items():
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

        # 计算复用率
        if schedule:
            total_assignments = sum(len(nodes) for nodes in schedule.values())
            reuse_ratio = total_assignments / len(schedule) if len(schedule) > 0 else 0
            results['slot_reuse'].append(reuse_ratio)
            print(f"时隙复用率: {reuse_ratio:.2f}")

        # 检查调度质量（使用公开方法）
        if hasattr(env, 'check_schedule_quality'):  # 假设我们添加了这个公开方法
            if env.check_schedule_quality(schedule):
                results['success_rate'] += 1
                print("调度方案合格")
            else:
                print("调度方案不合格")

    # 计算最终统计
    if num_episodes > 0:
        results['success_rate'] = results['success_rate'] / num_episodes * 100
        results['interference_free_rate'] = results['interference_free'] / num_episodes * 100
        results['avg_reward'] = np.mean(results['rewards'])
        results['avg_slots'] = np.mean(results['slot_usage'])
        results['avg_reuse_ratio'] = np.mean(results['slot_reuse']) if results['slot_reuse'] else 0

        print("\n最终评估结果:")
        print(f"调度成功率: {results['success_rate']:.2f}%")
        print(f"无干扰率: {results['interference_free_rate']:.2f}%")
        print(f"平均时隙数: {results['avg_slots']:.2f}")
        print(f"平均时隙复用率: {results['avg_reuse_ratio']:.2f}")
        print(f"平均奖励: {results['avg_reward']:.2f}")

    return results


if __name__ == "__main__":
    # # 训练模型
    model, log_dir = train_stdma_agent(200000)
    # 创建评估环境
    eval_env = make_training_env()

    # 评估模型
    results = evaluate_schedule(eval_env, model)
    print(results)


