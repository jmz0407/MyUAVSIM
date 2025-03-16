import os
import numpy as np
import torch
from stable_baselines3 import PPO
from dynamic_env import DynamicStdmaEnv
from gnn_model import DynamicGNNFeatureExtractor


def load_and_use_model(model_path, network_size=10):
    """加载模型并在指定规模的网络上使用

    参数:
        model_path: 模型文件路径
        network_size: 网络节点数量
    """
    print(f"加载模型: {model_path}")
    model = PPO.load(model_path)

    print(f"创建 {network_size} 节点的环境...")
    env = DynamicStdmaEnv(num_nodes=network_size)

    print("开始模拟...")
    obs, _ = env.reset(num_nodes=network_size)
    done = False
    total_reward = 0
    step = 0

    while not done:
        # 预测动作
        action, _ = model.predict(obs, deterministic=True)

        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        step += 1

        # 打印当前状态
        print(f"Step {step}: 节点 {info['current_node']} 分配到时隙 {action}, 奖励 = {reward:.2f}")

    print("\n模拟完成!")
    print(f"总步数: {step}")
    print(f"总奖励: {total_reward:.2f}")

    # 打印最终的时隙分配结果
    schedule = info.get('schedule', {})
    print("\n最终时隙分配:")
    for slot, nodes in schedule.items():
        print(f"  时隙 {slot}: {nodes}")

    # 计算统计信息
    if schedule:
        total_slots = len(schedule)
        total_assignments = sum(len(nodes) for nodes in schedule.values())
        reuse_ratio = total_assignments / total_slots

        print(f"\n统计分析:")
        print(f"  使用时隙数: {total_slots}/{network_size} ({total_slots / network_size * 100:.1f}%)")
        print(f"  分配节点数: {total_assignments}")
        print(f"  复用率: {reuse_ratio:.2f}")


def test_model_adaptation(model_path, size_sequence=[10, 20, 10, 30, 5]):
    """测试模型在网络规模动态变化时的适应能力

    参数:
        model_path: 模型文件路径
        size_sequence: 网络规模变化序列
    """
    print(f"加载模型: {model_path}")
    model = PPO.load(model_path)

    # 创建初始环境
    env = DynamicStdmaEnv(num_nodes=size_sequence[0])

    for i, size in enumerate(size_sequence):
        print(f"\n===== 测试网络规模: {size} 节点 =====")

        # 重置环境到新规模
        obs, _ = env.reset(num_nodes=size)
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            # 预测动作
            action, _ = model.predict(obs, deterministic=True)

            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            steps += 1

            # 打印关键步骤
            if steps % 5 == 0 or done:
                print(f"  Step {steps}: 奖励 = {reward:.2f}, 累计奖励 = {episode_reward:.2f}")
                if done:
                    schedule = info.get('schedule', {})
                    if schedule:
                        total_slots = len(schedule)
                        total_assignments = sum(len(nodes) for nodes in schedule.values())
                        reuse_ratio = total_assignments / total_slots

                        print(f"\n  最终结果:")
                        print(f"    总步数: {steps}")
                        print(f"    总奖励: {episode_reward:.2f}")
                        print(f"    使用时隙数: {total_slots}/{size} ({total_slots / size * 100:.1f}%)")
                        print(f"    复用率: {reuse_ratio:.2f}")


def compare_with_baseline(model_path, network_sizes=[10, 20, 30, 40, 50]):
    """将GNN模型与基线方法进行比较

    参数:
        model_path: 模型文件路径
        network_sizes: 要测试的网络规模
    """
    print(f"加载GNN模型: {model_path}")
    model = PPO.load(model_path)

    results = {
        'gnn': {},
        'baseline': {}
    }

    for size in network_sizes:
        print(f"\n===== 测试网络规模: {size} 节点 =====")
        env = DynamicStdmaEnv(num_nodes=size)

        # 测试GNN模型
        print("\n--- GNN模型 ---")
        obs, _ = env.reset(num_nodes=size)
        done = False
        episode_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        schedule = info.get('schedule', {})
        if schedule:
            total_slots = len(schedule)
            total_assignments = sum(len(nodes) for nodes in schedule.values())
            reuse_ratio = total_assignments / total_slots

            results['gnn'][size] = {
                'reward': episode_reward,
                'slots': total_slots,
                'reuse_ratio': reuse_ratio
            }

            print(f"  总奖励: {episode_reward:.2f}")
            print(f"  使用时隙数: {total_slots}/{size} ({total_slots / size * 100:.1f}%)")
            print(f"  复用率: {reuse_ratio:.2f}")

        # 测试基线方法 (贪婪分配)
        print("\n--- 基线方法 (贪婪分配) ---")
        obs, _ = env.reset(num_nodes=size)
        done = False
        episode_reward = 0

        while not done:
            # 贪婪策略: 总是选择当前可用的最小时隙
            action = greedy_slot_assignment(env)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        schedule = info.get('schedule', {})
        if schedule:
            total_slots = len(schedule)
            total_assignments = sum(len(nodes) for nodes in schedule.values())
            reuse_ratio = total_assignments / total_slots

            results['baseline'][size] = {
                'reward': episode_reward,
                'slots': total_slots,
                'reuse_ratio': reuse_ratio
            }

            print(f"  总奖励: {episode_reward:.2f}")
            print(f"  使用时隙数: {total_slots}/{size} ({total_slots / size * 100:.1f}%)")
            print(f"  复用率: {reuse_ratio:.2f}")

    # 打印比较结果
    print("\n===== 比较结果 =====")
    print(
        f"{'网络规模':<10} | {'GNN奖励':>10} | {'基线奖励':>10} | {'GNN时隙':>10} | {'基线时隙':>10} | {'GNN复用率':>10} | {'基线复用率':>10}")
    print("-" * 80)

    for size in network_sizes:
        if size in results['gnn'] and size in results['baseline']:
            gnn = results['gnn'][size]
            baseline = results['baseline'][size]

            print(f"{size:<10} | {gnn['reward']:>10.2f} | {baseline['reward']:>10.2f} | "
                  f"{gnn['slots']:>10} | {baseline['slots']:>10} | "
                  f"{gnn['reuse_ratio']:>10.2f} | {baseline['reuse_ratio']:>10.2f}")


def greedy_slot_assignment(env):
    """贪婪时隙分配策略 (用作基线比较)"""
    # 检查哪些时隙可用
    valid_slots = []
    for slot in range(env.max_slots):
        if env._is_valid_assignment(env.current_node, slot):
            valid_slots.append(slot)

    # 选择最小的有效时隙
    return min(valid_slots) if valid_slots else 0


if __name__ == "__main__":
    # 模型路径 (替换为您的实际模型路径)
    model_path = "models/gnn_stdma_20250316_224023/final_model.zip"

    # 1. 简单使用示例
    load_and_use_model(model_path, network_size=15)

    # 2. 测试网络规模动态变化
    test_model_adaptation(model_path, size_sequence=[10, 20, 30, 15, 25])

    # 3. 与基线方法比较
    compare_with_baseline(model_path, network_sizes=[10, 20, 30, 15, 25])