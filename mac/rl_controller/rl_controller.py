from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from mac.rl_controller.rl_environment import StdmaEnv
import simpy
from simulator.simulator import Simulator
from utils import config
def train_stdma_agent(env, total_timesteps=100000):
    """使用DQN训练STDMA调度代理"""

    # 创建日志目录
    log_dir = f"./logs/STDMA_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)

    # 使用DQN算法
    model = DQN(
        "MultiInputPolicy",
        env,
        learning_rate=5e-4,
        batch_size=32,
        buffer_size=50000,
        learning_starts=1000,
        target_update_interval=500,
        train_freq=4,
        gamma=0.99,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=dict(
            net_arch=[64, 32]
        ),
        verbose=1,
        tensorboard_log=log_dir
    )

    # 创建评估回调
    eval_callback = EvalCallback(
        env,
        best_model_save_path=f"{log_dir}/best_model",
        log_path=f"{log_dir}/eval_results",
        eval_freq=2000,
        deterministic=True,
        render=False
    )

    # 创建检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=f"{log_dir}/checkpoints",
        name_prefix="stdma_model"
    )

    try:
        # 开始训练
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True
        )

        # 保存最终模型
        model.save(f"{log_dir}/final_model")

    except Exception as e:
        print(f"Training error: {e}")
        model.save(f"{log_dir}/interrupted_model")
        raise e

    return model





def train_stdma_agent(env, total_timesteps=1000000):
    """
    使用SB3训练STDMA调度代理

    Args:
        env: StdmaEnv环境实例
        total_timesteps: 总训练步数
    """
    # 创建日志目录
    log_dir = f"./logs/STDMA_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)

    # 环境向量化
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # 创建模型 (这里使用PPO算法)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,  # 降低学习率
        n_steps=512,  # 减小batch size
        batch_size=32,  # 减小batch size
        n_epochs=5,  # 减少每次更新的epoch数
        gamma=0.99,  # 折扣因子
        gae_lambda=0.95,  # GAE参数
        clip_range=0.2,  # PPO裁剪参数
        ent_coef=0.01,  # 增加探索
        policy_kwargs=dict(
            net_arch=dict(
                pi=[64, 64],  # policy网络结构
                vf=[64, 64]  # value网络结构
            )
        ),
        verbose=1,
        tensorboard_log=log_dir
    )

    # 创建独立的评估环境
    eval_env = StdmaEnv(
        simulator=env.envs[0].simulator,
        num_nodes=env.envs[0].num_nodes,
        num_slots=env.envs[0].num_slots
    )
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{log_dir}/best_model",
        log_path=f"{log_dir}/eval_results",
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    # 创建检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=f"{log_dir}/checkpoints",
        name_prefix="stdma_model"
    )

    # 开始训练
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback]
    )

    # 保存最终模型和环境统计信息
    model.save(f"{log_dir}/final_model")
    env.save(f"{log_dir}/vec_normalize.pkl")

    return model, env


def evaluate_stdma_agent(model, env, n_eval_episodes=10):
    """
    评估训练好的代理

    Args:
        model: 训练好的模型
        env: 评估环境
        n_eval_episodes: 评估回合数
    """
    # 收集评估指标
    rewards = []
    delays = []
    interference_counts = []
    utilizations = []

    obs = env.reset()
    for episode in range(n_eval_episodes):
        episode_reward = 0
        episode_delay = []
        episode_interference = []
        episode_utilization = []
        done = False

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            episode_reward += reward
            episode_delay.append(info[0]['delay'] if info[0]['delay'] is not None else 0)
            episode_interference.append(info[0]['interference'])
            episode_utilization.append(info[0]['utilization'])

        rewards.append(episode_reward)
        delays.append(np.mean(episode_delay))
        interference_counts.append(np.mean(episode_interference))
        utilizations.append(np.mean(episode_utilization))

        obs = env.reset()

    eval_results = {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_delay': np.mean(delays),
        'mean_interference': np.mean(interference_counts),
        'mean_utilization': np.mean(utilizations)
    }

    return eval_results


def plot_training_results(log_dir):
    """
    绘制训练结果

    Args:
        log_dir: 训练日志目录
    """
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('STDMA Training Results')

    # 加载数据
    eval_log = np.load(f"{log_dir}/eval_results/evaluations.npz")
    rewards = eval_log['results']
    timesteps = eval_log['timesteps']

    # 绘制奖励曲线
    axes[0, 0].plot(timesteps, rewards.mean(axis=1))
    axes[0, 0].fill_between(
        timesteps,
        rewards.mean(axis=1) - rewards.std(axis=1),
        rewards.mean(axis=1) + rewards.std(axis=1),
        alpha=0.3
    )
    axes[0, 0].set_title('Average Reward')
    axes[0, 0].set_xlabel('Timesteps')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)

    # 加载并绘制其他指标
    metrics = ['delay', 'interference', 'utilization']
    positions = [(0, 1), (1, 0), (1, 1)]

    for metric, pos in zip(metrics, positions):
        data = np.load(f"{log_dir}/eval_results/{metric}.npy")
        axes[pos[0], pos[1]].plot(timesteps, data)
        axes[pos[0], pos[1]].set_title(f'Average {metric.capitalize()}')
        axes[pos[0], pos[1]].set_xlabel('Timesteps')
        axes[pos[0], pos[1]].set_ylabel(metric.capitalize())
        axes[pos[0], pos[1]].grid(True)

    plt.tight_layout()
    plt.savefig(f"{log_dir}/training_results.png")
    plt.close()


def main():
    """
    主训练流程
    """


    # 创建模拟环境
    env = simpy.Environment()
    channel_states = {i: simpy.Resource(env, capacity=1) for i in range(config.NUMBER_OF_DRONES)}
    simulator = Simulator(
        seed=2024,
        env=env,
        channel_states=channel_states,
        n_drones=config.NUMBER_OF_DRONES
    )

    # 创建STDMA环境
    stdma_env = StdmaEnv(
        simulator=simulator,
        num_nodes=config.NUMBER_OF_DRONES,
        num_slots=config.NUMBER_OF_DRONES
    )

    # 训练代理
    print("Starting training...")
    model, trained_env = train_stdma_agent(stdma_env, total_timesteps=1000000)

    # 评估代理
    print("\nEvaluating trained agent...")
    eval_results = evaluate_stdma_agent(model, trained_env)

    # 打印评估结果
    print("\nEvaluation Results:")
    for metric, value in eval_results.items():
        print(f"{metric}: {value:.4f}")

    # 绘制训练结果
    log_dir = "./logs/STDMA_" + datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_training_results(log_dir)

    print(f"\nTraining results saved to {log_dir}")


if __name__ == "__main__":
    main()