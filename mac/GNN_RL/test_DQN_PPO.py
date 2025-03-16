import os
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import argparse

# 导入自定义模块
from train_model import train_model, evaluate_model as evaluate_ppo
from DQN_train import train_dqn, evaluate_dqn


# 创建结果保存目录
def create_experiment_dir():
    base_dir = "./comparison_results"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(base_dir, f"ppo_dqn_comparison_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


# 保存评估结果到JSON文件
def save_results(results, filename):
    # 将字典的键转换为字符串，因为JSON不能有整数键
    serializable_results = {}
    for key, value in results.items():
        serializable_results[str(key)] = value

    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=4)


# 比较PPO和DQN模型
def compare_models(ppo_results, dqn_results, save_path):
    # 找到两个结果中共有的节点数量
    common_nodes = sorted([int(n) for n in set(ppo_results.keys()).intersection(set(dqn_results.keys()))])

    if not common_nodes:
        print("没有共同的节点数量，无法进行比较")
        return

    # 将字符串键转回整数用于绘图
    if isinstance(list(ppo_results.keys())[0], str):
        ppo_results = {int(k): v for k, v in ppo_results.items()}
    if isinstance(list(dqn_results.keys())[0], str):
        dqn_results = {int(k): v for k, v in dqn_results.items()}

    plt.figure(figsize=(16, 12))

    # 1. 比较平均奖励
    plt.subplot(2, 2, 1)
    plt.plot(common_nodes, [ppo_results[n]['mean_reward'] for n in common_nodes], 'o-', label='PPO')
    plt.plot(common_nodes, [dqn_results[n]['mean_reward'] for n in common_nodes], 's-', label='DQN')
    plt.title('平均奖励比较')
    plt.xlabel('节点数量')
    plt.ylabel('平均奖励')
    plt.legend()
    plt.grid(True)

    # 2. 比较平均复用率
    plt.subplot(2, 2, 2)
    plt.plot(common_nodes, [ppo_results[n]['mean_reuse_ratio'] for n in common_nodes], 'o-', label='PPO')
    plt.plot(common_nodes, [dqn_results[n]['mean_reuse_ratio'] for n in common_nodes], 's-', label='DQN')
    plt.title('平均复用率比较')
    plt.xlabel('节点数量')
    plt.ylabel('平均复用率')
    plt.legend()
    plt.grid(True)

    # 3. 比较时隙效率
    plt.subplot(2, 2, 3)
    ppo_efficiency = [n / ppo_results[n]['mean_slots'] for n in common_nodes]
    dqn_efficiency = [n / dqn_results[n]['mean_slots'] for n in common_nodes]
    plt.plot(common_nodes, ppo_efficiency, 'o-', label='PPO')
    plt.plot(common_nodes, dqn_efficiency, 's-', label='DQN')
    plt.title('时隙效率比较 (节点数/时隙数)')
    plt.xlabel('节点数量')
    plt.ylabel('效率')
    plt.legend()
    plt.grid(True)

    # 4. 比较时隙数量
    plt.subplot(2, 2, 4)
    plt.plot(common_nodes, [ppo_results[n]['mean_slots'] for n in common_nodes], 'o-', label='PPO')
    plt.plot(common_nodes, [dqn_results[n]['mean_slots'] for n in common_nodes], 's-', label='DQN')
    plt.title('平均时隙数比较')
    plt.xlabel('节点数量')
    plt.ylabel('平均时隙数')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"模型比较结果已保存至 {save_path}")

    # 生成比较表格
    comparison_table = []
    comparison_table.append("模型性能比较表\n")
    comparison_table.append("=" * 80)
    comparison_table.append(
        f"{'节点数':<10}{'PPO奖励':<15}{'DQN奖励':<15}{'PPO复用率':<15}{'DQN复用率':<15}{'PPO时隙数':<15}{'DQN时隙数':<15}")
    comparison_table.append("-" * 80)

    for n in common_nodes:
        comparison_table.append(f"{n:<10}{ppo_results[n]['mean_reward']:<15.2f}{dqn_results[n]['mean_reward']:<15.2f}"
                                f"{ppo_results[n]['mean_reuse_ratio']:<15.2f}{dqn_results[n]['mean_reuse_ratio']:<15.2f}"
                                f"{ppo_results[n]['mean_slots']:<15.2f}{dqn_results[n]['mean_slots']:<15.2f}")

    comparison_table.append("=" * 80)

    # 分析哪个模型性能更好
    ppo_better_reward = sum(1 for n in common_nodes if ppo_results[n]['mean_reward'] > dqn_results[n]['mean_reward'])
    dqn_better_reward = sum(1 for n in common_nodes if dqn_results[n]['mean_reward'] > ppo_results[n]['mean_reward'])

    ppo_better_reuse = sum(
        1 for n in common_nodes if ppo_results[n]['mean_reuse_ratio'] > dqn_results[n]['mean_reuse_ratio'])
    dqn_better_reuse = sum(
        1 for n in common_nodes if dqn_results[n]['mean_reuse_ratio'] > ppo_results[n]['mean_reuse_ratio'])

    ppo_better_slots = sum(1 for n in common_nodes if ppo_results[n]['mean_slots'] < dqn_results[n]['mean_slots'])
    dqn_better_slots = sum(1 for n in common_nodes if dqn_results[n]['mean_slots'] < ppo_results[n]['mean_slots'])

    comparison_table.append("\n性能优势分析:")
    comparison_table.append(
        f"奖励: PPO更好 {ppo_better_reward}/{len(common_nodes)} 种情况, DQN更好 {dqn_better_reward}/{len(common_nodes)} 种情况")
    comparison_table.append(
        f"复用率: PPO更好 {ppo_better_reuse}/{len(common_nodes)} 种情况, DQN更好 {dqn_better_reuse}/{len(common_nodes)} 种情况")
    comparison_table.append(
        f"时隙数: PPO更好 {ppo_better_slots}/{len(common_nodes)} 种情况, DQN更好 {dqn_better_slots}/{len(common_nodes)} 种情况")

    # 保存比较结果到文本文件
    comparison_path = os.path.splitext(save_path)[0] + "_analysis.txt"
    with open(comparison_path, 'w') as f:
        f.write('\n'.join(comparison_table))

    print(f"比较分析已保存至 {comparison_path}")

    # 打印比较结果
    for line in comparison_table:
        print(line)


def main():
    parser = argparse.ArgumentParser(description='训练和比较PPO与DQN模型')
    parser.add_argument('--train', action='store_true', help='是否训练新模型')
    parser.add_argument('--ppo-model', type=str, default=None, help='PPO模型路径')
    parser.add_argument('--dqn-model', type=str, default=None, help='DQN模型路径')
    parser.add_argument('--timesteps', type=int, default=500000, help='训练步数')
    parser.add_argument('--eval-only', action='store_true', help='只进行评估')
    args = parser.parse_args()

    # 创建实验目录
    experiment_dir = create_experiment_dir()
    print(f"实验结果将保存到: {experiment_dir}")

    # 训练配置
    node_size_range = (5, 30)
    max_nodes = max(node_size_range)
    eval_node_sizes = [10, 20, 30, 40, 50]

    # 设置模型保存路径
    ppo_model_path = args.ppo_model
    dqn_model_path = args.dqn_model

    # 如果需要训练新模型
    if args.train and not args.eval_only:
        print("开始训练新模型...")

        # 训练PPO模型
        if not args.ppo_model:
            print("\n训练PPO模型...")
            ppo_config = {
                "total_timesteps": args.timesteps,
                "node_size_range": node_size_range,
                "save_dir": os.path.join(experiment_dir, "ppo"),
                "experiment_name": f"ppo_stdma_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }
            _, ppo_save_dir = train_model(ppo_config)
            ppo_model_path = os.path.join(ppo_save_dir, "final_model.zip")

        # 训练DQN模型
        if not args.dqn_model:
            print("\n训练DQN模型...")
            dqn_config = {
                "total_timesteps": args.timesteps,
                "node_size_range": node_size_range,
                "save_dir": os.path.join(experiment_dir, "dqn"),
                "experiment_name": f"dqn_stdma_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }
            _, dqn_save_dir = train_dqn(dqn_config)
            dqn_model_path = os.path.join(dqn_save_dir, "final_model.pt")

    # 确保模型路径有效
    if not ppo_model_path or not os.path.exists(ppo_model_path):
        print("错误: 未提供有效的PPO模型路径")
        return

    if not dqn_model_path or not os.path.exists(dqn_model_path):
        print("错误: 未提供有效的DQN模型路径")
        return

    # 评估模型
    print("\n评估PPO模型...")
    ppo_results = evaluate_ppo(ppo_model_path, node_sizes=eval_node_sizes)
    ppo_results_file = os.path.join(experiment_dir, "ppo_results.json")
    save_results(ppo_results, ppo_results_file)

    print("\n评估DQN模型...")
    dqn_results = evaluate_dqn(dqn_model_path, node_sizes=eval_node_sizes)
    dqn_results_file = os.path.join(experiment_dir, "dqn_results.json")
    save_results(dqn_results, dqn_results_file)

    # 比较模型性能
    print("\n比较PPO和DQN模型性能...")
    comparison_path = os.path.join(experiment_dir, "ppo_dqn_comparison.png")
    compare_models(ppo_results, dqn_results, comparison_path)


if __name__ == "__main__":
    main()