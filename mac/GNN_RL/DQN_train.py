import os
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import gymnasium as gym
from datetime import datetime
import io
from PIL import Image
# 导入自定义模型和环境
from gnn_model import DynamicGNNFeatureExtractor
from dynamic_env import DynamicStdmaEnv
import matplotlib
import logging

matplotlib.font_manager._log.setLevel(logging.WARNING)
plt.rcParams['font.sans-serif'] = ['STFangsong']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class EnhancedTensorboardCallback(BaseCallback):
    """
    记录丰富指标到TensorBoard的自定义回调函数
    """

    def __init__(self, node_size_range, max_nodes, verbose=0):
        super().__init__(verbose)
        self.node_size_range = node_size_range
        self.max_nodes = max_nodes
        self.episode_count = 0
        self.rewards = []
        self.reuse_ratios = []
        self.slot_counts = []
        self.episode_lengths = []
        self.schedules = []
        self.network_sizes = []

        # 性能指标
        self.time_efficiencies = []  # 时间效率 (节点数/使用时隙数)
        self.qos_satisfactions = []  # QoS满足率

        # 训练指标
        self.value_losses = []
        self.exploration_rates = []

        # 图表计数
        self.graph_step = 0

    def _on_training_start(self):
        # 记录模型架构
        try:
            model_architecture = str(self.model.policy)
            self.logger.record("model/architecture", model_architecture)
        except:
            pass

    def _on_step(self):
        # 读取training_env
        try:
            env = self.training_env.envs[0]

            # 记录学习率
            if hasattr(self.model, 'learning_rate'):
                if callable(self.model.learning_rate):
                    current_lr = self.model.learning_rate(self._n_calls)
                else:
                    current_lr = self.model.learning_rate
                self.logger.record("train/learning_rate", current_lr)

            # 记录探索率（DQN特有）
            if hasattr(self.model, 'exploration_rate'):
                self.logger.record("train/exploration_rate", self.model.exploration_rate)
                self.exploration_rates.append(self.model.exploration_rate)

            # 记录训练损失 (如果可用)
            if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
                # 使用字典的副本进行迭代，避免迭代过程中字典大小变化导致的错误
                for k, v in list(self.model.logger.name_to_value.items()):
                    if 'loss' in k.lower():
                        # 使用完全不同的键名命名空间，避免嵌套导致的冲突
                        if 'value_loss' in k.lower() or 'q_loss' in k.lower():
                            self.logger.record(f"dqn_value_loss", v)
                            self.value_losses.append(v)
                        elif 'loss' in k.lower():
                            self.logger.record(f"dqn_training_loss", v)

            # 记录当前网络规模
            if hasattr(env, 'num_nodes'):
                self.logger.record("environment/network_size", env.num_nodes)

            # 记录环境信息 - 确保infos和dones是有效的
            infos = self.locals.get('infos', [{}]) if self.locals else [{}]
            dones = self.locals.get('dones', [False]) if self.locals else [False]

            # 当一个回合完成时
            if len(infos) > 0 and dones[0]:
                info = infos[0]

                # 记录回合指标
                if 'episode' in info:
                    episode_reward = info['episode']['r']
                    episode_length = info['episode']['l']
                    self.rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)

                    # 记录到TensorBoard
                    self.logger.record("rollout/ep_rew_mean", episode_reward)
                    self.logger.record("rollout/ep_len_mean", episode_length)

                # 获取时隙分配
                schedule = info.get('schedule', {})
                if schedule:
                    # 创建schedule的深拷贝，避免引用后期可能变化的字典
                    schedule_copy = {k: list(v) for k, v in schedule.items()}
                    self.schedules.append(schedule_copy)

                    # 计算复用率
                    slot_count = len(schedule)
                    total_assignments = sum(len(nodes) for nodes in schedule.values())
                    reuse_ratio = total_assignments / slot_count if slot_count > 0 else 0
                    time_efficiency = env.num_nodes / slot_count if slot_count > 0 else 0

                    # 记录统计
                    self.slot_counts.append(slot_count)
                    self.reuse_ratios.append(reuse_ratio)
                    self.time_efficiencies.append(time_efficiency)
                    self.network_sizes.append(env.num_nodes)

                    # 记录到TensorBoard
                    self.logger.record("metrics/slot_count", slot_count)
                    self.logger.record("metrics/reuse_ratio", reuse_ratio)
                    self.logger.record("metrics/time_efficiency", time_efficiency)
                    self.logger.record("metrics/assignments", total_assignments)

                    # 每10个回合计算均值
                    if len(self.reuse_ratios) >= 10:
                        self.logger.record("metrics/avg_reuse_ratio_10ep", np.mean(self.reuse_ratios[-10:]))
                        self.logger.record("metrics/avg_time_efficiency_10ep", np.mean(self.time_efficiencies[-10:]))

                    # 每50个回合可视化时隙分配
                    self.episode_count += 1
                    if self.episode_count % 50 == 0:
                        self._log_slot_visualization(schedule, env.num_nodes)

                # 随机改变下一个回合的节点数量
                new_nodes = np.random.randint(self.node_size_range[0], self.node_size_range[1] + 1)
                env.num_nodes = new_nodes

                # 记录日志
                if self.verbose > 0 and self.episode_count % 10 == 0:
                    if len(self.rewards) >= 10:
                        avg_reward = np.mean(self.rewards[-10:])
                        avg_reuse = np.mean(self.reuse_ratios[-10:])
                        print(
                            f"回合 {self.episode_count}: 平均奖励 = {avg_reward:.2f}, 平均复用率 = {avg_reuse:.2f}, 下一回合节点数 = {new_nodes}")

            # 记录模型性能信息
            if hasattr(torch.cuda, 'is_available') and torch.cuda.is_available():
                self.logger.record("system/gpu_memory_allocated", torch.cuda.memory_allocated() / (1024 ** 2))

        except Exception as e:
            if self.verbose > 0:
                print(f"回调函数错误: {e}")

        return True

    def _log_slot_visualization(self, schedule, num_nodes):
        """创建并记录时隙分配可视化"""
        try:
            # 创建图表
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
            ax.set_title(f'时隙分配 (网络规模: {num_nodes})')
            ax.set_xlabel('时隙索引')
            ax.set_ylabel('节点数量')
            ax.set_xticks(slots)
            ax.set_ylim(0, max(nodes_count) + 1.5 if nodes_count else 1.5)

            # 绘制平均高度线
            avg_nodes = sum(nodes_count) / len(slots) if slots else 0
            ax.axhline(y=avg_nodes, color='r', linestyle='--', label=f'平均: {avg_nodes:.2f}')
            ax.legend()

            # 保存为图像并记录到TensorBoard
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf)
            img_tensor = np.array(img.getdata()).reshape(img.size[1], img.size[0], -1)
            img_tensor = img_tensor[:, :, :3]  # 只保留RGB通道

            # 记录到TensorBoard
            self.logger.record(f"charts/slot_assignment", img_tensor, step=self.graph_step)
            self.graph_step += 1

            plt.close(fig)

        except Exception as e:
            if self.verbose > 0:
                print(f"创建时隙可视化错误: {e}")


# 设置随机种子以确保可重现性
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# DQN模型训练函数
def train_dqn_model(config):
    """训练DQN强化学习模型"""
    # 提取配置
    total_timesteps = config.get("total_timesteps", 1_000_000)
    node_size_range = config.get("node_size_range", (5, 30))
    save_dir = config.get("save_dir", "./models")
    experiment_name = config.get("experiment_name", f"dqn_stdma_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # 设置随机种子
    set_seeds(42)

    # 创建保存目录
    model_save_path = os.path.join(save_dir, experiment_name)
    log_dir = os.path.join("./logs", experiment_name)
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"训练配置:")
    print(f"  总步数: {total_timesteps}")
    print(f"  环境数: {1}")  # 强制使用单环境
    print(f"  节点范围: {node_size_range}")
    print(f"  日志目录: {log_dir}")
    print(f"  模型保存路径: {model_save_path}")

    print("使用单环境训练模式以避免观察维度不一致问题")

    # 计算最大节点数
    max_nodes = max(node_size_range)

    # 创建单环境
    start_nodes = random.randint(*node_size_range)
    env = DynamicStdmaEnv(num_nodes=start_nodes, max_nodes=max_nodes)
    env = Monitor(env, os.path.join(log_dir, "train"))
    train_env = DummyVecEnv([lambda: env])

    # 创建评估环境
    eval_env_10nodes = DynamicStdmaEnv(num_nodes=10, max_nodes=max_nodes)
    eval_env_10nodes = Monitor(eval_env_10nodes, os.path.join(log_dir, "eval_10nodes"))
    eval_env_10 = DummyVecEnv([lambda: eval_env_10nodes])

    eval_env_20nodes = DynamicStdmaEnv(num_nodes=20, max_nodes=max_nodes)
    eval_env_20nodes = Monitor(eval_env_20nodes, os.path.join(log_dir, "eval_20nodes"))
    eval_env_20 = DummyVecEnv([lambda: eval_env_20nodes])

    # 创建回调
    tensorboard_callback = EnhancedTensorboardCallback(node_size_range, max_nodes, verbose=1)

    # 创建保存检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_save_path,
        name_prefix="dqn_gnn_stdma"
    )

    # 创建评估回调
    eval_callback_10 = EvalCallback(
        eval_env_10,
        best_model_save_path=os.path.join(model_save_path, "10nodes"),
        log_path=os.path.join(log_dir, "10nodes"),
        eval_freq=20000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )

    eval_callback_20 = EvalCallback(
        eval_env_20,
        best_model_save_path=os.path.join(model_save_path, "20nodes"),
        log_path=os.path.join(log_dir, "20nodes"),
        eval_freq=2000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )

    # 合并所有回调
    callbacks = [tensorboard_callback, checkpoint_callback, eval_callback_10, eval_callback_20]

    # 设置模型参数
    policy_kwargs = dict(
        features_extractor_class=DynamicGNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[128, 64]  # DQN通常使用更简单的网络架构
    )

    # 创建DQN模型
    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=5e-4,
        buffer_size=100000,  # 经验回放缓冲区大小
        learning_starts=1000,  # 开始学习前的步数
        batch_size=64,  # 每次学习的批次大小
        tau=1.0,  # 目标网络更新系数 (1.0 = 硬更新)
        gamma=0.99,  # 折扣因子
        train_freq=4,  # 每4步训练一次
        gradient_steps=1,  # 每次更新的梯度步数
        target_update_interval=1000,  # 目标网络更新频率
        exploration_fraction=0.2,  # 探索率衰减时间比例
        exploration_initial_eps=1.0,  # 初始探索率
        exploration_final_eps=0.05,  # 最终探索率
        max_grad_norm=10,  # 梯度裁剪
        verbose=1,
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs
    )

    # 训练模型
    try:
        print("开始训练...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )

        # 保存最终模型
        final_model_path = os.path.join(model_save_path, "final_model.zip")
        model.save(final_model_path)
        print(f"模型训练完成，已保存到 {final_model_path}")

        # 保存训练曲线
        plt.figure(figsize=(12, 8))

        # 绘制奖励曲线
        plt.subplot(2, 2, 1)
        plt.plot(tensorboard_callback.rewards)
        plt.title('训练奖励')
        plt.xlabel('回合')
        plt.ylabel('平均奖励')
        plt.grid(True)

        # 绘制复用率曲线
        if tensorboard_callback.reuse_ratios:
            plt.subplot(2, 2, 2)
            plt.plot(tensorboard_callback.reuse_ratios)
            plt.title('时隙复用率')
            plt.xlabel('回合')
            plt.ylabel('平均复用率')
            plt.grid(True)

        # 绘制探索率曲线
        if tensorboard_callback.exploration_rates:
            plt.subplot(2, 2, 3)
            plt.plot(tensorboard_callback.exploration_rates)
            plt.title('探索率')
            plt.xlabel('步数')
            plt.ylabel('探索率')
            plt.grid(True)

        # 绘制价值损失曲线
        if tensorboard_callback.value_losses:
            plt.subplot(2, 2, 4)
            plt.plot(tensorboard_callback.value_losses)
            plt.title('Q值损失')
            plt.xlabel('更新次数')
            plt.ylabel('损失')
            plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(model_save_path, "training_curves.png"))

        return model, model_save_path

    except Exception as e:
        print(f"训练出错: {str(e)}")
        # 尝试保存中断时的模型
        try:
            error_model_path = os.path.join(model_save_path, "error_model.zip")
            model.save(error_model_path)
            print(f"保存了错误时的模型: {error_model_path}")
        except:
            pass
        raise e

    finally:
        # 关闭环境
        train_env.close()
        eval_env_10.close()
        eval_env_20.close()


# 模型评估函数
def evaluate_dqn_model(model_path, node_sizes=[5, 10, 15, 20, 30], episodes=10):
    """评估DQN模型在不同规模网络上的性能"""
    results = {}

    # 设置随机种子
    set_seeds(42)

    try:
        model = DQN.load(model_path)
        print(f"\n开始评估模型: {model_path}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return {}

    max_nodes = max(node_sizes)

    for nodes in node_sizes:
        print(f"\n测试 {nodes} 节点网络...")
        env = DynamicStdmaEnv(num_nodes=nodes, max_nodes=max_nodes)

        episode_rewards = []
        reuse_ratios = []
        slot_counts = []

        for ep in range(episodes):
            obs, _ = env.reset(num_nodes=nodes)
            done = False
            ep_reward = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_reward += reward

            # 计算统计信息
            schedule = info.get('schedule', {})
            slots_used = len(schedule)
            total_assignments = sum(len(nodes) for nodes in schedule.values())
            reuse_ratio = total_assignments / slots_used if slots_used > 0 else 0

            episode_rewards.append(ep_reward)
            reuse_ratios.append(reuse_ratio)
            slot_counts.append(slots_used)

            print(f"  回合 {ep + 1}: 奖励={ep_reward:.2f}, 复用率={reuse_ratio:.2f}, 使用时隙={slots_used}")

        # 记录结果
        results[nodes] = {
            'mean_reward': np.mean(episode_rewards),
            'mean_reuse_ratio': np.mean(reuse_ratios),
            'mean_slots': np.mean(slot_counts),
            'std_reward': np.std(episode_rewards),
            'std_reuse_ratio': np.std(reuse_ratios),
            'std_slots': np.std(slot_counts)
        }

        print(f"  平均奖励: {results[nodes]['mean_reward']:.2f} ± {results[nodes]['std_reward']:.2f}")
        print(f"  平均复用率: {results[nodes]['mean_reuse_ratio']:.2f} ± {results[nodes]['std_reuse_ratio']:.2f}")
        print(f"  平均时隙数: {results[nodes]['mean_slots']:.2f} ± {results[nodes]['std_slots']:.2f}")

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


# 主函数
if __name__ == "__main__":
    # 设置随机种子
    set_seeds(42)

    # 训练配置
    training_config = {
        "total_timesteps": 200_000,  # 总训练步数
        "n_envs": 1,  # 强制单环境
        "node_size_range": (5, 30),  # 节点数量范围
        "experiment_name": f"dqn_stdma_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }

    # 训练模型
    model, model_path = train_dqn_model(training_config)

    # 评估模型
    if model_path and os.path.exists(os.path.join(model_path, "final_model.zip")):
        evaluate_dqn_model(os.path.join(model_path, "final_model.zip"))