import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# 创建一个简单的gym环境（此处使用CartPole环境作为示例）
env = gym.make('CartPole-v1')

# 将环境包装为DummyVecEnv（向量化环境）
env = DummyVecEnv([lambda: env])

# 创建PPO模型
model = PPO("MlpPolicy", env, verbose=1)

# 训练模型
model.learn(total_timesteps=10000)

# 测试模型
obs = env.reset()  # 获取环境的初始观察值
# 如果返回的是元组，则只获取第一个元素
if isinstance(obs, tuple):
    obs = obs[0]

for _ in range(100):  # 进行1000步测试
    action, _states = model.predict(obs, deterministic=True)  # 获取模型的动作选择
    obs, reward, done, info = env.step(action)  # 执行动作并返回结果
    print("Reward:", reward)  # 打印每步的奖励
    if done:  # 如果回合结束，重置环境
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

env.close()  # 关闭环境