import logging
import math
from phy.phy import Phy
from utils import config
from utils.util_function import euclidean_distance
from entities.packet import DataPacket
from mac.LinkQualityManager import LinkQualityManager
from mac.LoadBalancer import LoadBalancer
from simulator.improved_traffic_generator import TrafficRequirement
import traceback
from mac.GNN_RL.dynamic_env import DynamicStdmaEnv  # 使用动态环境
from mac.GNN_RL.gnn_model import DynamicGNNFeatureExtractor  # 导入GNN模型
# 在 stdma.py 开头添加
import sys
import os

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 确保 GNN_RL 目录可导入
gnn_rl_dir = os.path.join(current_dir, "GNN_RL")
if gnn_rl_dir not in sys.path:
    sys.path.insert(0, gnn_rl_dir)
class Stdma:
    def __init__(self, drone):
        self.my_drone = drone
        self.simulator = drone.simulator
        self.env = drone.env
        self.time_slot_duration = config.SLOT_DURATION
        self.num_slots = config.NUMBER_OF_DRONES
        self.current_slot = 0
        self.phy = Phy(self)
        self.current_transmission = None
        self.slot_schedule = self._create_tra_slot_schedule()
        # 链路质量管理
        self.link_quality_manager = LinkQualityManager()

        self.traffic_requirements = {}  # 存储收到的业务需求
        # 添加负载均衡器
        self.load_balancer = LoadBalancer()
        # from mac.rl_controller.rl_controller import RLSlotController
        # self.rl_controller = RLSlotController(
        #     simulator =self.simulator,
        #     num_nodes=drone.simulator.n_drones,
        #     num_slots=self.num_slots
        # )

        # 添加负载监控进程
        #self.env.process(self._monitor_load())

        # 添加RL控制器
        # from mac.rl_controller.rl_environment import StdmaEnv
        # self.rl_env = StdmaEnv(
        #     simulator=self.simulator,
        #     num_nodes=drone.simulator.n_drones,
        #     num_slots=self.num_slots
        # )


        # 初始化强化学习组件
        # self.use_rl, self.rl_model, self.rl_env = self._initialize_rl_controller()
        self.use_rl, self.rl_model, self.rl_env = self._initialize_ppo_rl_controller()


        # # 创建RL环境和控制器
        # try:
        #     from mac.rl_controller.rl_environment import StdmaEnv
        #     self.rl_env = StdmaEnv(
        #         simulator=self.simulator,
        #         num_nodes=drone.simulator.n_drones,
        #         num_slots=self.num_slots
        #     )
        #
        #     from stable_baselines3 import DQN
        #     import os

        #     # 获取当前文件的目录
        #     current_dir = os.path.dirname(os.path.abspath(__file__))
        #     # 构建模型路径
        #     model_path = os.path.join(current_dir, "rl_controller/logs/STDMA_20250124_084456/final_model.zip")
        #
        #     # 打印路径信息进行调试
        #     print(f"当前目录: {current_dir}")
        #     print(f"尝试加载模型: {model_path}")
        #     print(f"模型文件是否存在: {os.path.exists(model_path)}")
        #
        #     if os.path.exists(model_path):
        #         self.rl_model = DQN.load(model_path)
        #         self.use_rl = True
        #         print(f"成功加载RL模型")
        #         logging.info(f"成功加载RL模型: {model_path}")
        #     else:
        #         self.use_rl = False
        #         self.rl_model = None  # 明确设置为None
        #         print(f"未找到RL模型，使用传统方法")
        #         logging.warning(f"未找到RL模型: {model_path}")
        #
        # except Exception as e:
        #     print(f"加载RL模型出错: {str(e)}")
        #     logging.error(f"加载RL模型失败: {e}")
        #     self.use_rl = False
        #     self.rl_model = None  # 明确设置为None



        # 数据流管理
        self.flow_queue = {}  # 存储数据流
        self.flow_stats = {}  # 流量统计

        # 启动进程
        self.env.process(self._slot_synchronization())
        # self.env.process(self._delayed_schedule_creation())
        # self.env.process(self._monitor_flows())

    def _create_tra_slot_schedule(self):
        """创建时隙分配表"""
        schedule = {}
        for i in range(self.num_slots):
            schedule[i] = [i % self.num_slots]  # 将整数值改为列表
        logging.info(f"TDMA schedule created for drone {self.my_drone.identifier}: {schedule}")
        return schedule

    def update_slot_schedule(self, new_schedule):
        """提供给强化学习算法的时隙更新接口"""
        self.slot_schedule = new_schedule
        logging.info(f"Updated slot schedule: {self.slot_schedule}")

    def _initialize_ppo_rl_controller(self):
        """初始化PPO控制器"""
        try:
            # 尝试导入所需模块
            try:
                import gnn_model
            except ImportError:
                print("无法导入 gnn_model，尝试从文件动态导入")
                # 尝试动态导入
                import importlib.util
                gnn_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GNN_RL/gnn_model.py")
                spec = importlib.util.spec_from_file_location("gnn_model", gnn_path)
                gnn_model = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(gnn_model)
                sys.modules["gnn_model"] = gnn_model
                print(f"已动态加载 gnn_model 从 {gnn_path}")

            from stable_baselines3 import PPO

            # 构建模型路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(current_dir, "GNN_RL/models/gnn_stdma_20250316_224023/")
            model_path = os.path.join(model_dir, "final_model.zip")

            if os.path.exists(model_path):
                # 先加载模型
                rl_model = PPO.load(model_path)

                # 加载成功后再创建环境
                logging.info("模型加载成功，初始化环境...")

                # 创建虚拟业务需求以帮助初始化
                from dataclasses import dataclass

                @dataclass
                class DummyRequirement:
                    source_id: int = 0
                    dest_id: int = 0
                    routing_path: list = None
                    delay_requirement: float = 100.0
                    num_packets: int = 50
                    qos_requirement: int = 1

                # 创建环境但不立即重置
                from mac.GNN_RL.dynamic_env import DynamicStdmaEnv
                rl_env = DynamicStdmaEnv(
                    simulator=self.simulator,
                    num_nodes=self.my_drone.simulator.n_drones,
                    max_nodes=max(30, self.my_drone.simulator.n_drones)
                )

                # 手动初始化current_requirement
                rl_env.current_requirement = DummyRequirement()

                # 成功初始化
                use_rl = True
                logging.info(f"成功加载PPO模型和环境")

                return use_rl, rl_model, rl_env
            else:
                logging.warning(f"模型文件不存在: {model_path}")
                return False, None, None

        except Exception as e:
            logging.error(f"初始化PPO控制器失败: {str(e)}")
            traceback.print_exc()
            return False, None, None

    # def _initialize_ppo_rl_controller(self):
    #     """初始化PPO控制器"""
    #     try:
    #         from stable_baselines3 import PPO
    #         import os
    #
    #         # 构建模型路径
    #         current_dir = os.path.dirname(os.path.abspath(__file__))
    #         # model_dir = os.path.join(current_dir, "rl_controller/logs/")
    #         # # 找到最新的模型目录
    #         # model_dirs = [d for d in os.listdir(model_dir) if d.startswith("STDMA_PPO_")]
    #         # if not model_dirs:
    #         #     return False, None, None
    #         #
    #         # latest_model_dir = max(model_dirs)
    #         # model_path = os.path.join(model_dir, latest_model_dir, "best_model/best_model.zip")
    #         # 构建模型路径
    #         model_dir = os.path.join(current_dir, "rl_controller/logs/STDMA_PPO_20250207_112404/best_model")
    #         model_path = os.path.join(model_dir, "best_model.zip")
    #         if os.path.exists(model_path):
    #             rl_model = PPO.load(model_path)
    #             use_rl = True
    #             logging.info(f"成功加载PPO模型: {model_path}")
    #             # 创建RL环境
    #             from mac.rl_controller.rl_environment import StdmaEnv
    #             rl_env = StdmaEnv(
    #                 simulator=self.simulator,
    #                 num_nodes=self.my_drone.simulator.n_drones,
    #                 num_slots=self.num_slots
    #             )
    #
    #             return use_rl, rl_model, rl_env
    #         else:
    #             return False, None, None
    #
    #     except Exception as e:
    #         logging.error(f"初始化PPO控制器失败: {str(e)}")
    #         return False, None, None
    def _initialize_rl_controller(self):
            """初始化强化学习控制器"""
            try:
                from stable_baselines3 import DQN
                import os

                # 构建模型路径
                current_dir = os.path.dirname(os.path.abspath(__file__))
                model_dir = os.path.join(current_dir, "rl_controller/logs/STDMA_20250131_161759/")
                model_path = os.path.join(model_dir, "final_model.zip")

                # 调试信息
                logging.info(f"尝试加载RL模型路径: {model_path}")
                logging.info(f"模型文件是否存在: {os.path.exists(model_path)}")

                # 加载模型
                if os.path.exists(model_path):
                    rl_model = DQN.load(model_path)
                    use_rl = True
                    logging.info(f"成功加载RL模型: {model_path}")

                    # 创建RL环境
                    from mac.rl_controller.rl_environment import StdmaEnv
                    rl_env = StdmaEnv(
                        simulator=self.simulator,
                        num_nodes=self.my_drone.simulator.n_drones,
                        num_slots=self.num_slots
                    )

                    return use_rl, rl_model, rl_env
                else:
                    logging.warning("未找到RL模型文件")
                    return False, None, None

            except Exception as e:
                logging.error(f"初始化RL控制器失败: {str(e)}")
                traceback.print_exc()
                return False, None, None
    def get_network_state(self):
        """获取当前网络状态，供强化学习使用"""
        state = {
            'traffic_loads': self._get_traffic_loads(),
            'queue_lengths': self._get_queue_lengths(),
            'link_qualities': self._get_link_qualities(),
            'node_positions': self._get_node_positions(),
            'current_schedule': self.slot_schedule
        }
        return state

    def _is_valid_slot(self, node_id, slot, schedule):
        """检查时隙分配是否冲突"""
        if slot in schedule:
            for assigned_node in schedule[slot]:
                if self._nodes_interfere(node_id, assigned_node):
                    return False
        return True

    def _find_backup_slot(self, node_id, schedule):
        """备用时隙选择策略"""
        for slot in range(self.num_slots):
            if self._is_valid_slot(node_id, slot, schedule):
                return slot
        return len(schedule)  # 新增时隙
    def _get_traffic_loads(self):
        """获取各节点的业务负载"""
        traffic_loads = {}
        for drone_id in range(self.simulator.n_drones):
            drone = self.simulator.drones[drone_id]
            traffic_loads[drone_id] = {
                'queue_size': drone.transmitting_queue.qsize(),
                'waiting_packets': len(drone.waiting_list)
            }
        return traffic_loads

    def _get_queue_lengths(self):
        """获取队列长度信息"""
        return {
            drone_id: self.simulator.drones[drone_id].transmitting_queue.qsize()
            for drone_id in range(self.simulator.n_drones)
        }

    def _get_link_qualities(self):
        """获取链路质量信息"""
        link_qualities = {}
        for i in range(self.simulator.n_drones):
            for j in range(i+1, self.simulator.n_drones):
                link_qualities[(i,j)] = self.link_quality_manager.get_link_quality(i, j)
        return link_qualities

    def _get_node_positions(self):
        """获取节点位置信息"""
        return {
            drone_id: self.simulator.drones[drone_id].coords
            for drone_id in range(self.simulator.n_drones)
        }

    def _collect_state_info(self):
        """定期收集网络状态信息"""
        while True:
            self.network_state = self.get_network_state()
            yield self.env.timeout(config.STATE_COLLECTION_INTERVAL)


    def _slot_synchronization(self):
        while True:
            self.current_slot = (self.env.now // self.time_slot_duration) % self.num_slots
            yield self.env.timeout(self.time_slot_duration)

    def _monitor_load(self):
        """
        监控网络负载情况并触发必要的调整
        """
        while True:
            yield self.env.timeout(1000)  # 每1ms检查一次

            # 更新每个节点的统计信息
            for drone_id in range(config.NUMBER_OF_DRONES):
                queue_length = len(self.flow_queue.get(f"flow_{drone_id}", []))
                throughput = self.flow_stats.get(f"flow_{drone_id}", {}).get('throughput', 0)
                delay = self.flow_stats.get(f"flow_{drone_id}", {}).get('avg_delay', 0)

                self.load_balancer.update_node_stats(drone_id, queue_length, throughput, delay)

            # 检查是否需要重新分配时隙
            high_load_nodes = self.load_balancer.get_high_load_nodes()
            if high_load_nodes:
                self._adjust_slot_allocation_for_load(high_load_nodes)

    def _adjust_slot_allocation_for_load(self, high_load_nodes):
        """
        为高负载节点调整时隙分配
        """
        current_distribution = self.load_balancer.get_load_distribution()

        # 记录调整前的状态
        logging.info("\n负载均衡调整开始:")
        logging.info("-" * 50)
        logging.info("当前负载分布:")
        for node_id, stats in current_distribution.items():
            logging.info(f"节点 {node_id}: 负载分数={stats['load_score']:.2f}, "
                         f"队列长度={stats['queue_length']}")

        # 尝试为高负载节点分配额外时隙
        schedule = self.slot_schedule.copy()
        for node_id in high_load_nodes:
            current_slots = [slot for slot, nodes in schedule.items()
                             if node_id in nodes]

            # 如果节点当前时隙数低于平均值，考虑分配额外时隙
            avg_slots_per_node = sum(len(nodes) for nodes in schedule.values()) / config.NUMBER_OF_DRONES
            if len(current_slots) < avg_slots_per_node:
                self._allocate_extra_slot(node_id, schedule)

        # 更新时隙分配
        self.slot_schedule = schedule

        # 记录调整结果
        logging.info("\n调整后的时隙分配:")
        self._print_schedule_info(schedule)

    def _allocate_extra_slot(self, node_id, schedule):
        """
        为指定节点分配额外的时隙
        """
        from phy.large_scale_fading import maximum_communication_range
        interference_range = maximum_communication_range() * 1.5

        # 寻找可以容纳该节点的时隙
        for slot in range(self.num_slots):
            if slot not in schedule:
                schedule[slot] = [node_id]
                return True

            # 检查是否可以加入现有时隙
            can_add = True
            for existing_node in schedule[slot]:
                dist = euclidean_distance(
                    self.simulator.drones[node_id].coords,
                    self.simulator.drones[existing_node].coords
                )
                if dist < interference_range:
                    can_add = False
                    break

            if can_add and len(schedule[slot]) < 3:  # 保持每个时隙最多3个节点
                schedule[slot].append(node_id)
                return True

        # 如果没有找到合适的时隙，创建新的时隙
        new_slot = self.num_slots
        schedule[new_slot] = [node_id]
        self.num_slots += 1
        return True

    # def _create_slot_schedule(self):
    #     """改进的时隙分配算法，确保合理的时隙数量"""
    #     schedule = {}
    #     from phy.large_scale_fading import maximum_communication_range
    #     interference_range = maximum_communication_range() * 2
    #
    #     # 初始化时隙数为节点总数的2/3（可以根据实际情况调整）
    #     self.num_slots = math.ceil(config.NUMBER_OF_DRONES * 2 / 3)
    #     min_slots = math.ceil(config.NUMBER_OF_DRONES / 3)  # 最少需要的时隙数
    #     max_slots = config.NUMBER_OF_DRONES # 最大时隙数
    #
    #     # 计算节点兼容性矩阵
    #     compatibility_matrix = self._calculate_compatibility_matrix(interference_range)
    #
    #     best_schedule = None
    #     best_metric = float('inf')  # 用于评估分配方案的优劣
    #
    #     # 尝试不同的时隙数，找到最优解
    #     for num_slots in range(min_slots, max_slots + 1):
    #         current_schedule = {}
    #         unassigned = set(range(config.NUMBER_OF_DRONES))
    #
    #         # 为每个时隙分配节点
    #         for slot in range(num_slots):
    #             current_schedule[slot] = []
    #             candidates = list(unassigned)
    #
    #             # 根据干扰关系选择合适的节点组合
    #             candidates.sort(key=lambda x: sum(compatibility_matrix[x][y]
    #                                               for y in unassigned), reverse=True)
    #
    #             for drone_id in candidates[:]:
    #                 # 检查是否可以加入当前时隙
    #                 if self._can_add_to_slot(drone_id, current_schedule[slot],
    #                                          compatibility_matrix):
    #                     current_schedule[slot].append(drone_id)
    #                     unassigned.remove(drone_id)
    #
    #         # 如果所有节点都已分配且评估指标更好，更新最优解
    #         if not unassigned:
    #             metric = self._evaluate_schedule(current_schedule, compatibility_matrix)
    #             if metric < best_metric:
    #                 best_metric = metric
    #                 best_schedule = current_schedule
    #                 self.num_slots = num_slots
    #
    #     if best_schedule:
    #         schedule = best_schedule
    #         logging.info(f"找到最优时隙分配方案，使用 {self.num_slots} 个时隙")
    #     else:
    #         logging.warning("未找到有效的时隙分配方案，使用默认分配")
    #         schedule = self._create_tra_slot_schedule()
    #
    #     self._print_schedule_info(schedule)
    #     return schedule
    #
    # 在stdma.py中修改_create_slot_schedule方法：
    def _create_slot_schedule(self):
        """使用RL模型生成时隙分配"""
        if self.use_rl and self.rl_model:
            schedule = {}
            obs, _ = self.rl_env.reset()  # 重置环境状态

            # 为每个节点分配时隙
            for node_id in range(self.simulator.n_drones):
                action, _ = self.rl_model.predict(obs, deterministic=True)
                slot = int(action)

                # 更新时隙分配表
                if slot not in schedule:
                    schedule[slot] = []
                schedule[slot].append(node_id)

                # 更新环境状态
                obs, _, done, _, _ = self.rl_env.step(action)
                if done:
                    break

            logging.info(f"RL生成的时隙分配: {schedule}")
            return schedule
        else:
            # 回退到传统方法
            return self._create_tra_slot_schedule()
    def _calculate_compatibility_matrix(self, interference_range):
        """计算节点兼容性矩阵"""
        n_drones = config.NUMBER_OF_DRONES
        matrix = [[False] * n_drones for _ in range(n_drones)]

        for i in range(n_drones):
            for j in range(n_drones):
                if i != j:
                    drone1 = self.simulator.drones[i]
                    drone2 = self.simulator.drones[j]

                    # 结合距离和链路质量判断兼容性
                    dist = euclidean_distance(drone1.coords, drone2.coords)
                    link_quality = self.link_quality_manager.get_link_quality(i, j)

                    matrix[i][j] = (dist >= interference_range and
                                    (link_quality == -1 or link_quality >= 0.7))

        return matrix
    def _can_add_to_slot(self, drone_id, slot_nodes, compatibility_matrix):
        """检查节点是否可以加入当前时隙"""
        # 检查与时隙中所有已有节点的兼容性
        return all(compatibility_matrix[drone_id][assigned_id]
                   for assigned_id in slot_nodes)

    def _evaluate_schedule(self, schedule, compatibility_matrix):
        """评估时隙分配方案的质量"""
        # 评估指标：时隙数 + 平均每个时隙的节点干扰程度
        interference_score = 0
        total_pairs = 0

        for slot_nodes in schedule.values():
            if len(slot_nodes) > 1:
                for i in range(len(slot_nodes)):
                    for j in range(i + 1, len(slot_nodes)):
                        if not compatibility_matrix[slot_nodes[i]][slot_nodes[j]]:
                            interference_score += 1
                        total_pairs += 1

        avg_interference = interference_score / total_pairs if total_pairs > 0 else 0
        return len(schedule) + avg_interference * 5  # 权重可调
    def _print_schedule_info(self, schedule):
        """打印详细的时隙分配信息"""
        info = "\nSTDMA时隙分配详情:\n" + "-" * 50 + "\n"

        for slot, nodes in schedule.items():
            info += f"时隙 {slot}:\n"
            info += f"  节点: {', '.join(f'UAV-{n}' for n in nodes)}\n"

            # 打印该时隙内节点间的链路质量
            if len(nodes) > 1:
                info += "  节点间链路质量:\n"
                for i, node1 in enumerate(nodes):
                    for node2 in nodes[i + 1:]:
                        quality = self.link_quality_manager.get_link_quality(node1, node2)
                        info += f"    UAV-{node1} <-> UAV-{node2}: {quality:.2f}\n"

        info += "-" * 50
        logging.info(info)
    def _print_schedule_info(self, schedule):
        """打印详细的时隙分配信息"""
        info = "\nSTDMA时隙分配详情:\n" + "-" * 50 + "\n"

        for slot, nodes in schedule.items():
            info += f"时隙 {slot}:\n"
            info += f"  节点: {', '.join(f'UAV-{n}' for n in nodes)}\n"

            # 打印该时隙内节点间的链路质量
            if len(nodes) > 1:
                info += "  节点间链路质量:\n"
                for i, node1 in enumerate(nodes):
                    for node2 in nodes[i + 1:]:
                        quality = self.link_quality_manager.get_link_quality(node1, node2)
                        info += f"    UAV-{node1} <-> UAV-{node2}: {quality:.2f}\n"

        info += "-" * 50
        logging.info(info)


    def _delayed_schedule_creation(self):
        yield self.env.timeout(1000)
        self.slot_schedule = self._create_slot_schedule()

    def _monitor_flows(self):
        while True:
            yield self.env.timeout(5000)
            self._update_flow_stats()
            self._adjust_slots()

    def _update_flow_stats(self):
        for flow_id, stats in self.flow_stats.items():
            queue = self.flow_queue.get(flow_id, [])
            if queue:
                stats.update({
                    'queue_size': len(queue),
                    'avg_delay': sum(self.env.now - p.creation_time for p in queue) / len(queue),
                    'throughput': stats['sent_packets'] / (self.env.now / 1e6)
                })

    def _adjust_slots(self):
        # 检查是否需要重新分配
        needs_reallocation = any(
            stats['avg_delay'] > 2000 or stats['throughput'] < 1
            for stats in self.flow_stats.values()
        )
        if needs_reallocation:
            self._adjust_slots_with_rl()  #

    def _adjust_slots_with_rl(self):
        """使用RL模型动态调整时隙分配"""
        # 获取当前网络状态
        obs = self.rl_env._get_observation()

        # 让RL模型重新生成时隙分配
        new_schedule = {}
        for node in range(self.simulator.n_drones):
            action, _ = self.rl_model.predict(obs, deterministic=True)
            slot = int(action)

            if slot not in new_schedule:
                new_schedule[slot] = []
            new_schedule[slot].append(node)

            # 更新观察
            obs, _, _, _, _ = self.rl_env.step(action)

        # 更新时隙分配
        self.slot_schedule = new_schedule
        logging.info(f"RL adjusted schedule: {new_schedule}")

    # def _create_slot_schedule(self):
    #     schedule = {}
    #     from phy.large_scale_fading import maximum_communication_range
    #     interference_range = maximum_communication_range() * 1.2
    #
    #     unassigned = set(range(config.NUMBER_OF_DRONES))
    #     slot = 0
    #
    #     while unassigned:
    #         if slot >= self.num_slots:
    #             self.num_slots += 1
    #
    #         schedule[slot] = []
    #         remaining = list(unassigned)
    #
    #         for drone_id in remaining:
    #             if all(euclidean_distance(
    #                     self.simulator.drones[drone_id].coords,
    #                     self.simulator.drones[assigned_id].coords
    #             ) >= interference_range
    #                    for assigned_id in schedule[slot]):
    #                 schedule[slot].append(drone_id)
    #                 unassigned.remove(drone_id)
    #         slot += 1
    #
    #     logging.info(f"STDMA调度表: {schedule}")
    #     return schedule

    def mac_send(self, packet):
        """MAC层发送函数"""
        if not self.slot_schedule:
            logging.error("时隙表未创建")
            return
        logging.info(f"Time {self.env.now}:UAV{self.my_drone.identifier} MAC layer received {type(packet).__name__}")

        if isinstance(packet, TrafficRequirement):
            # logging.info(f"当前时隙表: {self.slot_schedule}")

            # 准备所需信息
            traffic_info = {
                'source_id': packet.source_id,
                'dest_id': packet.dest_id,
                'num_packets': packet.num_packets,
                'delay_requirement': packet.delay_requirement,
                'qos_requirement': packet.qos_requirement,
                'route_path' : packet.routing_path
            }
            #日志记录traffic
            logging.info(f"UAV{self.my_drone.identifier} received traffic requirement: {traffic_info}")
            # yield self.env.process(self._transmit_packet(packet))
            try:
                if hasattr(self, 'use_rl') and self.use_rl and self.rl_model is not None:
                    # 使用RL重新分配时隙
                    # copy_packet = copy.deepcopy(packet)
                    obs = self.rl_env.reset(requirement_data=packet)[0]
                    #记录当前的业务路径
                    logging.info(f"UAV{self.my_drone.identifier} 业务路径: {packet.routing_path}")
                    new_schedule = {}
                    done = False
                    while not done:
                        action, _ = self.rl_model.predict(obs, deterministic=True)
                        slot = int(action)
                        if slot not in new_schedule:
                            new_schedule[slot] = []

                        obs, _, done, _, info = self.rl_env.step(action)
                        new_schedule = info.get('schedule', {})


                    # 更新时隙分配
                    self.slot_schedule = new_schedule
                    for node in self.simulator.drones:
                        if node.identifier != self.my_drone.identifier:
                            node.mac_protocol.slot_schedule = self.slot_schedule
                            logging.info(f"{node.identifier}基于RL模型更新时隙分配: {new_schedule}")
                else:
                    # 使用传统方法重新分配时隙
                    self.slot_schedule = self._create_tra_slot_schedule()
                    logging.info("使用传统方法更新时隙分配")
            except Exception as e:
                import traceback
                logging.error(f"时隙调整失败: {e}")
                logging.error(f"错误详情: {traceback.format_exc()}")
            except Exception as e:
                logging.error(f"时隙调整失败: {e}")
                # 如果调整失败，保持当前时隙分配不变

            # 检查并修正时隙表格式
            for slot, nodes in self.slot_schedule.items():
                if not isinstance(nodes, list):
                    logging.error(f"Slot {slot} 的节点不是列表格式: {type(nodes)}")
                    if isinstance(nodes, int):
                        self.slot_schedule[slot] = [nodes]


        # 数据包流管理
        elif isinstance(packet, DataPacket):
            # flow_id = f"flow_{packet.src_drone.identifier}_{packet.dst_drone.identifier}"
            # self.flow_queue.setdefault(flow_id, []).append(packet)
            # self.flow_stats.setdefault(flow_id, {
            #     'sent_packets': 0,
            #     'avg_delay': 0,
            #     'throughput': 0,
            #     'queue_size': 0
            # })

            mac_start_time = self.env.now

            # # 确保时隙表中的值都是列表
            # schedule = {slot: (nodes if isinstance(nodes, list) else [nodes])
            #             for slot, nodes in self.slot_schedule.items()}

            # 使用修正后的时隙表查找分配
            assigned_slot = next((slot for slot, drones in self.slot_schedule.items()
                                  if self.my_drone.identifier in drones), None)
            # logging.info(f"UAV{self.my_drone.identifier} assigned to slot {assigned_slot}")
            # if assigned_slot is None:
            #     logging.error(f"无人机{self.my_drone.identifier}未分配时隙")
            #     return

            current_time = self.env.now
            slot_start = (
                                     current_time // self.time_slot_duration) * self.time_slot_duration + assigned_slot * self.time_slot_duration
            slot_end = slot_start + self.time_slot_duration

            if current_time < slot_start:
                yield self.env.timeout(slot_start - current_time)
            elif current_time >= slot_end:
                yield self.env.timeout(self.num_slots * self.time_slot_duration - (current_time - slot_start))

            yield self.env.process(self._transmit_packet(packet))
            #mac时延
            logging.info(f"UAV{self.my_drone.identifier} MAC 时延: {self.env.now - current_time}")
            self.simulator.metrics.mac_delay.append((self.env.now - current_time) / 1e3)
        else:
            yield self.env.process(self._transmit_packet(packet))


    def _transmit_packet(self, packet):
        self.current_transmission = packet

        if packet.transmission_mode == 0:
            packet.increase_ttl()
            self.phy.unicast(packet, packet.next_hop_id)
            logging.info(f"Time {self.env.now}:UAV{self.my_drone.identifier} transmitted {type(packet).__name__} to {packet.next_hop_id}")
            yield self.env.timeout(packet.packet_length / config.BIT_RATE * 1e6)

        elif packet.transmission_mode == 1:
            packet.increase_ttl()
            self.phy.broadcast(packet)
            yield self.env.timeout(packet.packet_length / config.BIT_RATE * 1e6)

        if isinstance(packet, DataPacket):
            flow_id = f"flow_{packet.src_drone.identifier}_{packet.dst_drone.identifier}"
            # self.flow_stats[flow_id]['sent_packets'] += 1
            # self.flow_queue[flow_id].remove(packet)

        self.current_transmission = None