import os
import sys
import torch
import logging
import numpy as np

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from mac.rl_controller.a3c import A3CTrainer
from mac.rl_controller.rl_environment import StdmaEnv
from simulator.simulator import Simulator
from utils import config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def create_env():
    """创建环境的工厂函数"""
    # 设置随机种子
    seed = np.random.randint(0, 10000)
    
    # 创建环境参数
    env_params = {
        'simulation_time': config.SIM_TIME,
        'slot_duration': config.SLOT_DURATION,
        'packet_size': config.DATA_PACKET_LENGTH,
        'bit_rate': config.BIT_RATE
    }
    
    # 初始化信道状态（示例：假设所有信道都是好的）
    n_drones = 10
    channel_states = np.ones((n_drones, n_drones))
    
    # 创建模拟器实例
    simulator = Simulator(
        seed=seed,
        env=env_params,
        channel_states=channel_states,
        n_drones=n_drones
    )
    
    # 初始化无人机
    simulator.init_drones(n_drones)
    
    # 创建STDMA环境
    return StdmaEnv(simulator, num_nodes=n_drones, num_slots=10)

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")
    
    # 创建保存模型的目录
    os.makedirs("models", exist_ok=True)
    
    # 初始化A3C训练器
    trainer = A3CTrainer(
        env_creator=create_env,
        n_workers=4,  # 使用4个worker进程
        gamma=0.99,
        n_steps=5,
        learning_rate=0.001,
        device=device
    )
    
    try:
        logging.info("开始训练...")
        trainer.train()
    except KeyboardInterrupt:
        logging.info("训练被手动中断")
    except Exception as e:
        logging.error(f"训练过程中出现错误: {str(e)}")
        logging.error(f"错误详情: ", exc_info=True)
    finally:
        # 保存模型
        model_path = "models/a3c_model.pth"
        trainer.save_model(model_path)
        logging.info(f"模型已保存到: {model_path}")

if __name__ == "__main__":
    main() 