# failure_recovery_test.py

import simpy
import matplotlib.pyplot as plt
import numpy as np
from utils import config
from simulator.simulator import Simulator


def schedule_node_failure(sim, node_id, failure_time, recovery_time=None):
    """安排节点故障和恢复"""

    # 故障事件
    def node_fails():
        yield sim.env.timeout(failure_time)
        sim.drones[node_id].sleep = True
        print(f"Time {sim.env.now / 1e6:.2f}s: Node {node_id} fails")

    sim.env.process(node_fails())

    # 恢复事件（如果指定）
    if recovery_time is not None:
        def node_recovers():
            yield sim.env.timeout(recovery_time)
            sim.drones[node_id].sleep = False
            print(f"Time {sim.env.now / 1e6:.2f}s: Node {node_id} recovers")

        sim.env.process(node_recovers())


def run_failure_recovery_test(protocol_name):
    """运行故障恢复测试"""
    # 设置协议
    config.ROUTING_PROTOCOL = protocol_name

    # 创建仿真环境
    env = simpy.Environment()
    channel_states = {i: simpy.Resource(env, capacity=1) for i in range(config.NUMBER_OF_DRONES)}

    sim = Simulator(
        seed=2024,
        env=env,
        channel_states=channel_states,
        n_drones=config.NUMBER_OF_DRONES
    )

    # 安排节点故障
    # 选择几个中间节点故障
    failure_nodes = [5, 8, 12]  # 假设这些是中间转发节点

    for node_id in failure_nodes:
        # 故障发生在仿真的30%处，恢复在60%处
        failure_time = config.SIM_TIME * 0.3
        recovery_time = config.SIM_TIME * 0.6

        schedule_node_failure(sim, node_id, failure_time, recovery_time)

    # 收集时序数据
    time_points = []
    pdr_values = []