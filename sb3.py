import gym
from elegantrl.train import train_agent
from elegantrl.agents import AgentPPO  # 使用 PPO 算法
from elegantrl import Config

def train_cartpole():
    # 创建 Gym 环境
    env_name = "CartPole-v1"
    env = gym.make(env_name)       # 用于训练的环境
    env_eval = gym.make(env_name)  # 用于评估的环境

    # 配置训练参数
    args = Config(agent_class=AgentPPO, env=env)
    args.env_eval = env_eval
    args.env_name = env_name

    # 环境相关参数
    args.env_num = 1                # 环境实例数量
    args.if_discrete = True         # 动作空间是否为离散
    args.target_reward = 200.0      # 目标分数 (CartPole 的满分为 200)
    args.max_step = env._max_episode_steps  # 每个 episode 的最大步数

    # 智能体超参数
    args.net_dim = 2**7             # 神经网络隐藏层大小
    args.batch_size = 2**7          # 批量大小
    args.gamma = 0.99               # 折扣因子
    args.learning_rate = 1e-3       # 学习率
    args.eval_times = 2**2          # 评估次数
    args.break_step = int(1e5)      # 提前停止训练的总步数

    # 启动训练
    train_agent(args)

if __name__ == "__main__":
    train_cartpole()
