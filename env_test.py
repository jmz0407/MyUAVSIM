import gym
import numpy as np
import random


# 定义时隙分配环境
class UAVSlotAllocationEnv(gym.Env):
    def __init__(self, num_uavs, num_slots, max_steps):
        super(UAVSlotAllocationEnv, self).__init__()

        # 环境参数
        self.num_uavs = num_uavs  # UAV的数量
        self.num_slots = num_slots  # 时隙的数量
        self.max_steps = max_steps  # 最大步数

        # 动作空间和观察空间
        self.action_space = gym.spaces.Discrete(num_slots)  # 每个UAV可以选择一个时隙
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(num_uavs,), dtype=np.float32)

        # 初始化环境状态
        self.state = np.zeros(self.num_uavs)
        self.current_step = 0

    def reset(self):
        self.state = np.zeros(self.num_uavs)
        self.current_step = 0
        return self.state

    def step(self, action):
        self.current_step += 1

        # 计算吞吐量
        throughput = self.calculate_throughput(action)

        # 判断是否结束
        done = self.current_step >= self.max_steps

        # 返回状态，奖励，是否结束，附加信息
        return self.state, throughput, done, {}

    def calculate_throughput(self, action):
        """
        根据时隙分配情况计算吞吐量，简化为每个时隙的UAV成功发送的吞吐量
        """
        # 计算每个UAV选择的时隙
        slot_counts = np.zeros(self.num_slots)
        for a in action:
            slot_counts[a] += 1

        # 计算吞吐量：假设每个时隙最多允许1个UAV，超载时吞吐量为0
        throughput = 0
        for count in slot_counts:
            if count == 1:  # 唯一的时隙分配
                throughput += 1  # 每个成功的分配加1
        return throughput

    def render(self):
        print(f"Current step: {self.current_step}, State: {self.state}")

