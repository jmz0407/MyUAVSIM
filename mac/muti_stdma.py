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
from mac.GNN_RL.muti_dynamic_env import MUtiDynamicStdmaEnv  # 使用动态环境
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


class MutiStdma:
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
        self.has_processed_traffic = False  # 跟踪是否已处理过业务需求
        # 链路质量管理
        self.link_quality_manager = LinkQualityManager()

        # 多流支持 - 存储业务需求
        self.traffic_requirements = {}  # 存储收到的业务需求 {flow_id: requirement}
        self.active_flows = set()  # 活跃流ID集合
        self.flow_routes = {}  # 流的路由路径 {flow_id: routing_path}
        self.flow_stats = {}  # 流量统计 {flow_id: stats_dict}

        # 添加负载均衡器
        self.load_balancer = LoadBalancer()

        # 初始化强化学习组件
        self.use_rl, self.rl_model, self.rl_env = self._initialize_ppo_rl_controller()

        # 数据流管理
        self.flow_queue = {}  # 存储数据流

        # 时隙分配历史
        self.slot_assignment_history = []  # 记录历史时隙分配
        self.last_assignment_time = 0  # 上次分配时间

        # 启动进程
        self.env.process(self._slot_synchronization())
        self.env.process(self._monitor_flows())
        self.env.process(self._schedule_maintenance())
    def _create_tra_slot_schedule(self):
        """创建时隙分配表"""
        schedule = {}
        for i in range(self.num_slots):
            schedule[i] = i % self.num_slots
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
            model_dir = os.path.join(current_dir, "GNN_RL/models/gnn_stdma_20250317_022809/")
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
                rl_env = MUtiDynamicStdmaEnv(
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

    def _update_slot_schedule(self, old_schedule, new_schedule):
        """
        智能更新时隙分配表，确保没有节点干扰

        参数:
            old_schedule: 旧的时隙分配表 {slot: nodes}
            new_schedule: 新的时隙分配表 {slot: nodes}

        返回:
            合并后的时隙分配表
        """
        logging.info("更新时隙分配表...")

        # 规范化输入格式
        normalized_old = {}
        for slot, nodes in old_schedule.items():
            if isinstance(nodes, int):
                normalized_old[slot] = [nodes]
            else:
                normalized_old[slot] = list(nodes) if isinstance(nodes, list) else [nodes]

        normalized_new = {}
        for slot, nodes in new_schedule.items():
            if isinstance(nodes, int):
                normalized_new[slot] = [nodes]
            else:
                normalized_new[slot] = list(nodes) if isinstance(nodes, list) else [nodes]

        # 提取现有分配的所有节点
        old_nodes = set()
        for nodes in normalized_old.values():
            old_nodes.update(nodes)

        # 提取新分配的所有节点
        new_nodes = set()
        for nodes in normalized_new.values():
            new_nodes.update(nodes)

        # 获取在路径上的节点（应该保持其分配）
        path_nodes = set()
        for req in self.traffic_requirements.values():
            if hasattr(req, 'is_active') and req.is_active and hasattr(req, 'routing_path') and req.routing_path:
                path_nodes.update(req.routing_path)

        logging.info(f"路径节点: {path_nodes}")

        # 计算干扰关系
        interference_matrix = self._calculate_interference_matrix()

        # 合并策略：保留路径上节点的新分配，其他节点保持原有分配
        merged_schedule = {}

        # 1. 首先添加新分配中的路径节点
        for slot, nodes in normalized_new.items():
            if slot not in merged_schedule:
                merged_schedule[slot] = []

            for node in nodes:
                # 只添加路径上的节点，并且确保不重复添加
                if node in path_nodes and node not in merged_schedule[slot]:
                    # 检查是否与同一时隙中的现有节点冲突
                    if all(not interference_matrix[node][existing_node] for existing_node in merged_schedule[slot]):
                        merged_schedule[slot].append(node)
                    else:
                        # 如果有干扰，找一个新时隙
                        new_slot = 0
                        while True:
                            if new_slot not in merged_schedule:
                                merged_schedule[new_slot] = [node]
                                break

                            # 检查此时隙是否没有干扰
                            if all(not interference_matrix[node][existing_node] for existing_node in
                                   merged_schedule[new_slot]):
                                merged_schedule[new_slot].append(node)
                                break

                            new_slot += 1

        # 2. 然后添加旧分配中未被新分配的节点
        for slot, nodes in normalized_old.items():
            if slot not in merged_schedule:
                merged_schedule[slot] = []

            for node in nodes:
                # 检查节点是否已在任何时隙中
                already_assigned = False
                for s, n_list in merged_schedule.items():
                    if node in n_list:
                        already_assigned = True
                        break

                if not already_assigned:
                    # 检查是否与同一时隙中的现有节点冲突
                    if all(not interference_matrix[node][existing_node] for existing_node in merged_schedule[slot]):
                        merged_schedule[slot].append(node)
                    else:
                        # 如果有干扰，找一个新时隙
                        new_slot = 0
                        while True:
                            if new_slot not in merged_schedule:
                                merged_schedule[new_slot] = [node]
                                break

                            # 检查此时隙是否没有干扰
                            if all(not interference_matrix[node][existing_node] for existing_node in
                                   merged_schedule[new_slot]):
                                merged_schedule[new_slot].append(node)
                                break

                            new_slot += 1

        # 3. 检查并修复可能的遗漏节点
        assigned_nodes = set()
        for nodes in merged_schedule.values():
            assigned_nodes.update(nodes)

        missing_nodes = old_nodes.union(new_nodes) - assigned_nodes

        # 为遗漏节点找时隙
        for node in missing_nodes:
            # 尝试找一个没有干扰的时隙
            assigned = False
            for slot in range(max(merged_schedule.keys(), default=0) + 2):  # +2 确保有额外空间
                if slot not in merged_schedule:
                    merged_schedule[slot] = []

                # 检查此时隙是否没有干扰
                if all(not interference_matrix[node][existing_node] for existing_node in merged_schedule[slot]):
                    merged_schedule[slot].append(node)
                    assigned = True
                    break

            # 如果无法找到合适时隙，创建一个新时隙
            if not assigned:
                new_slot = max(merged_schedule.keys(), default=0) + 1
                merged_schedule[new_slot] = [node]

        # 4. 清理空时隙
        merged_schedule = {slot: nodes for slot, nodes in merged_schedule.items() if nodes}

        # 5. 重新映射时隙编号（可选）
        # 如果你希望时隙编号连续，可以取消下面的注释
        remapped_schedule = {}
        for i, (_, nodes) in enumerate(sorted(merged_schedule.items())):
            remapped_schedule[i] = nodes
        merged_schedule = remapped_schedule

        # 记录结果
        old_count = len(old_schedule)
        new_count = len(new_schedule)
        merged_count = len(merged_schedule)

        logging.info(f"合并时隙分配: 旧={old_count}个时隙, 新={new_count}个时隙, 合并后={merged_count}个时隙")

        # 更新时隙表
        self.slot_schedule = merged_schedule
        return merged_schedule

    def _calculate_interference_matrix(self):
        """
        计算节点间的干扰关系矩阵

        返回:
            干扰矩阵: n_nodes × n_nodes 的布尔矩阵，True表示两节点互相干扰
        """
        n_drones = self.simulator.n_drones
        interference_matrix = [[False for _ in range(n_drones)] for _ in range(n_drones)]

        # 获取干扰范围
        from phy.large_scale_fading import maximum_communication_range
        interference_range = maximum_communication_range() * 1.5  # 干扰范围通常大于通信范围

        # 计算所有节点对之间的干扰关系
        for i in range(n_drones):
            # 节点自身不干扰
            interference_matrix[i][i] = False

            for j in range(i + 1, n_drones):
                try:
                    # 计算节点间距离
                    dist = euclidean_distance(
                        self.simulator.drones[i].coords,
                        self.simulator.drones[j].coords
                    )

                    # 如果距离小于干扰范围，则存在干扰
                    has_interference = dist < interference_range

                    # 干扰是双向的
                    interference_matrix[i][j] = has_interference
                    interference_matrix[j][i] = has_interference

                except Exception as e:
                    logging.error(f"计算节点 {i} 和 {j} 之间的干扰关系时出错: {e}")
                    # 如果计算出错，保守地认为存在干扰
                    interference_matrix[i][j] = True
                    interference_matrix[j][i] = True

        return interference_matrix

    def _schedule_maintenance(self):
        """
        定期监控和维护时隙分配
        - 检查活跃流
        - 更新过期流
        - 确保时隙分配的连续性
        """
        while True:
            try:
                # 每100ms检查一次
                yield self.env.timeout(1e4)

                current_time = self.env.now

                # 1. 检查流的活跃状态
                expired_flows = []
                for flow_id, req in self.traffic_requirements.items():
                    # 如果请求已经超过其生命周期（假设需要添加一个生命周期属性）
                    if hasattr(req, 'lifetime') and (current_time - req.creation_timestamp) > req.lifetime:
                        expired_flows.append(flow_id)
                        logging.info(f"流 {flow_id} 已过期，将被移除")

                # 2. 移除过期流
                for flow_id in expired_flows:
                    if flow_id in self.traffic_requirements:
                        # 标记为非活跃但不删除，以保留历史记录
                        self.traffic_requirements[flow_id].is_active = False
                        if flow_id in self.active_flows:
                            self.active_flows.remove(flow_id)
                        logging.info(f"已将流 {flow_id} 标记为非活跃")

                # 3. 如果最近有新流添加并且上次分配时间较早，重新优化时隙分配
                time_since_last_assignment = current_time - self.last_assignment_time
                if time_since_last_assignment > 5e6:  # 如果超过5秒没有重新分配
                    # 检查是否有新的活跃流
                    new_active_flows = False
                    for flow_id, req in self.traffic_requirements.items():
                        if hasattr(req, 'is_active') and req.is_active and flow_id not in self.active_flows:
                            new_active_flows = True
                            self.active_flows.add(flow_id)

                    # 如果有新流或过期流，重新优化时隙分配
                    if new_active_flows or expired_flows:
                        yield self.env.process(self._reoptimize_slot_assignment())
                        self.last_assignment_time = current_time

                # 4. 检查时隙分配是否有问题（例如未分配的活跃节点）
                assigned_nodes = set()
                for nodes in self.slot_schedule.values():
                    if isinstance(nodes, list):
                        assigned_nodes.update(nodes)
                    else:
                        assigned_nodes.add(nodes)

                # 收集所有应该被分配的节点
                should_be_assigned = set()
                for flow_id in self.active_flows:
                    if flow_id in self.flow_routes:
                        should_be_assigned.update(self.flow_routes[flow_id])

                # 检查是否有节点应该被分配但未被分配
                unassigned_nodes = should_be_assigned - assigned_nodes
                if unassigned_nodes:
                    logging.warning(f"发现未分配时隙的节点: {unassigned_nodes}，将重新优化分配")
                    yield self.env.process(self._reoptimize_slot_assignment())
                    self.last_assignment_time = current_time
                # 定期验证时隙分配的完整性
                # 在 verify_schedule_completeness 之后
                if not self.verify_schedule_completeness():
                    logging.warning("时隙分配不完整，尝试直接补充缺失节点")
                    self.fix_missing_nodes()

                    # 将更新后的时隙分配传播到其他节点
                    for node in self.simulator.drones:
                        if node.identifier != self.my_drone.identifier:
                            node.mac_protocol.slot_schedule = self.slot_schedule

            except Exception as e:
                logging.error(f"时隙维护过程中出错: {e}")
                traceback.print_exc()

    def fix_missing_nodes(self):
        """
        直接在当前时隙表中补充缺失的节点
        """
        # 1. 获取所有已分配节点
        assigned_nodes = set()
        for slot, nodes in self.slot_schedule.items():
            if isinstance(nodes, list):
                assigned_nodes.update(nodes)
            else:
                assigned_nodes.add(nodes)

        # 2. 获取应该被分配的节点
        expected_nodes = set()
        for req in self.traffic_requirements.values():
            if hasattr(req, 'is_active') and req.is_active and hasattr(req, 'routing_path') and req.routing_path:
                expected_nodes.update(req.routing_path)

        # 3. 找出缺失的节点
        missing_nodes = expected_nodes - assigned_nodes

        if not missing_nodes:
            logging.info("没有缺失的节点，无需补充")
            return True

        logging.info(f"检测到缺失的节点: {missing_nodes}，正在补充...")

        # 4. 计算干扰关系用于分配决策
        interference_matrix = self._calculate_interference_matrix()

        # 5. 为每个缺失节点找到合适的时隙
        for node in missing_nodes:
            # 5.1 首先尝试找一个现有时隙添加，避免增加总时隙数
            best_slot = None
            min_interference = float('inf')

            for slot in sorted(self.slot_schedule.keys()):
                # 获取当前时隙中的节点
                slot_nodes = self.slot_schedule[slot]
                slot_nodes_list = slot_nodes if isinstance(slot_nodes, list) else [slot_nodes]

                # 计算与此时隙中节点的总干扰
                total_interference = 0
                has_direct_interference = False

                for existing_node in slot_nodes_list:
                    if interference_matrix[node][existing_node]:
                        has_direct_interference = True
                        break

                    # 计算节点距离作为干扰强度
                    try:
                        dist = euclidean_distance(
                            self.simulator.drones[node].coords,
                            self.simulator.drones[existing_node].coords
                        )
                        # 距离越近干扰越大
                        interference = max(0, 1 - dist / (self.max_comm_range * 2))
                        total_interference += interference
                    except Exception:
                        # 如果计算距离出错，假设有高干扰
                        total_interference += 1

                # 如果没有直接干扰且总干扰低于当前最小值
                if not has_direct_interference and total_interference < min_interference:
                    min_interference = total_interference
                    best_slot = slot

            # 5.2 如果找到了合适的现有时隙，添加节点
            if best_slot is not None and min_interference < 0.5:  # 设置干扰阈值
                if isinstance(self.slot_schedule[best_slot], list):
                    self.slot_schedule[best_slot].append(node)
                else:
                    self.slot_schedule[best_slot] = [self.slot_schedule[best_slot], node]
                logging.info(f"将节点 {node} 添加到现有时隙 {best_slot}")
            else:
                # 5.3 如果没有合适的现有时隙，创建一个新时隙
                new_slot = max(self.slot_schedule.keys(), default=-1) + 1
                self.slot_schedule[new_slot] = [node]
                logging.info(f"为节点 {node} 创建新时隙 {new_slot}")

        # 6. 再次验证是否所有节点都已分配
        missing_after_fix = expected_nodes - set(sum([
            slot_nodes if isinstance(slot_nodes, list) else [slot_nodes]
            for slot_nodes in self.slot_schedule.values()
        ], []))

        if missing_after_fix:
            logging.warning(f"修复后仍有缺失节点: {missing_after_fix}")
            return False
        else:
            logging.info(f"成功补充所有缺失节点，更新后的时隙表: {self.slot_schedule}")
            return True

    def _reoptimize_slot_assignment(self):
        """重新优化时隙分配，确保不中断活跃流"""
        try:
            yield self.env.timeout(0)
            # 1. 收集所有活跃流的需求
            active_requirements = []
            for flow_id in self.active_flows:
                if flow_id in self.traffic_requirements:
                    req = self.traffic_requirements[flow_id]
                    if hasattr(req, 'is_active') and req.is_active:
                        active_requirements.append(req)

            if not active_requirements:
                logging.info("没有活跃流，无需重新优化")
                return

            logging.info(f"重新优化 {len(active_requirements)} 个活跃流的时隙分配")

            # 2. 保存旧的时隙分配
            old_schedule = self.slot_schedule.copy() if self.slot_schedule else {}

            # 3. 使用RL模型重新生成时隙分配
            if self.use_rl and self.rl_model is not None:
                # 使用所有活跃流重置环境
                obs = self.rl_env.reset(active_requirements=active_requirements)[0]

                # 记录所有活跃流的路由路径
                for req in active_requirements:
                    logging.info(
                        f"活跃流 {req.flow_id if hasattr(req, 'flow_id') else f'{req.source_id}->{req.dest_id}'} 路径: {req.routing_path}")

                # 为每个节点生成时隙分配
                new_schedule = {}
                for node in range(self.simulator.n_drones):
                    action, _ = self.rl_model.predict(obs, deterministic=True)
                    slot = int(action)
                    if slot not in new_schedule:
                        new_schedule[slot] = []

                    obs, _, done, _, info = self.rl_env.step(action)
                    if info and 'schedule' in info:
                        new_schedule = info.get('schedule', {})
                    if done:
                        break

                # 4. 合并新旧时隙分配
                self._update_slot_schedule(old_schedule, new_schedule)



                # 6. 记录此次分配
                self.slot_assignment_history.append({
                    'time': self.env.now,
                    'schedule': self.slot_schedule.copy(),
                    'active_flows': list(self.active_flows),
                    'num_slots': len(self.slot_schedule)
                })

                # 验证优化后的时隙分配
                if not self.verify_schedule_completeness():
                    logging.warning("时隙分配不完整，尝试直接补充缺失节点")
                    self.fix_missing_nodes()

                # 将更新后的时隙分配传播到其他节点
                for node in self.simulator.drones:
                    if node.identifier != self.my_drone.identifier:
                        node.mac_protocol.slot_schedule = self.slot_schedule

                logging.info(f"时隙分配重新优化完成，当前有 {len(self.slot_schedule)} 个时隙")

        except Exception as e:
            logging.error(f"重新优化时隙分配时出错: {e}")
            traceback.print_exc()

    def _assign_missing_nodes(self):
        """为缺失的节点直接分配时隙（备用方法）"""
        # 获取所有已分配节点
        assigned_nodes = set()
        for nodes in self.slot_schedule.values():
            if isinstance(nodes, list):
                assigned_nodes.update(nodes)
            else:
                assigned_nodes.add(nodes)

        # 获取应该被分配的节点
        expected_nodes = set()
        for req in self.traffic_requirements.values():
            if hasattr(req, 'is_active') and req.is_active and hasattr(req, 'routing_path'):
                expected_nodes.update(req.routing_path)

        # 找到缺失的节点
        missing_nodes = expected_nodes - assigned_nodes

        if not missing_nodes:
            return

        logging.info(f"为缺失节点手动分配时隙: {missing_nodes}")

        # 计算干扰关系
        interference_matrix = self._calculate_interference_matrix()

        # 为每个缺失节点找到合适的时隙
        for node in missing_nodes:
            # 尝试找到可以加入的现有时隙
            assigned = False
            for slot in sorted(self.slot_schedule.keys()):
                slot_nodes = self.slot_schedule[slot]
                normalized_nodes = slot_nodes if isinstance(slot_nodes, list) else [slot_nodes]

                # 检查是否有干扰
                has_interference = False
                for existing_node in normalized_nodes:
                    if interference_matrix[node][existing_node]:
                        has_interference = True
                        break

                if not has_interference:
                    # 可以加入此时隙
                    if isinstance(slot_nodes, list):
                        self.slot_schedule[slot].append(node)
                    else:
                        self.slot_schedule[slot] = [self.slot_schedule[slot], node]
                    assigned = True
                    break

            # 如果无法加入现有时隙，创建新时隙
            if not assigned:
                new_slot = max(self.slot_schedule.keys(), default=-1) + 1
                self.slot_schedule[new_slot] = [node]

        logging.info(f"完成缺失节点的手动分配，更新后的时隙表: {self.slot_schedule}")
    def _monitor_flows(self):
        """监控流状态和性能"""
        while True:
            try:
                # 每500ms检查一次
                yield self.env.timeout(5e5)

                # 更新流统计
                for flow_id in self.active_flows:
                    if flow_id not in self.flow_stats:
                        self.flow_stats[flow_id] = {
                            'sent_packets': 0,
                            'received_packets': 0,
                            'start_time': self.env.now,
                            'last_activity': self.env.now,
                            'throughput': 0,
                            'avg_delay': 0,
                            'packet_ids': set()
                        }

                    # 查找属于此流的数据包统计
                    if flow_id in self.traffic_requirements:
                        req = self.traffic_requirements[flow_id]
                        source_id = req.source_id
                        dest_id = req.dest_id

                        # 更新统计信息
                        stats = self.flow_stats[flow_id]
                        stats['last_activity'] = self.env.now

                        # 计算吞吐量和延迟
                        if hasattr(self.simulator, 'metrics'):
                            # 提取属于此流的数据包ID
                            flow_packet_ids = set()
                            for packet_id in self.simulator.metrics.datapacket_arrived:
                                # 这里需要一种方法来识别属于此流的数据包
                                # 简化实现: 检查数据包的源和目标
                                if packet_id in self.simulator.metrics.deliver_time_dict:
                                    # 此处可能需要根据您的系统修改判断逻辑
                                    if flow_id.endswith(f"{source_id}_{dest_id}"):
                                        flow_packet_ids.add(packet_id)

                            # 更新接收的数据包数
                            new_packets = flow_packet_ids - stats['packet_ids']
                            stats['received_packets'] += len(new_packets)
                            stats['packet_ids'].update(new_packets)

                            # 计算平均延迟
                            if new_packets:
                                delay_sum = sum(self.simulator.metrics.deliver_time_dict[pid] for pid in new_packets)
                                avg_new_delay = delay_sum / len(new_packets)

                                # 更新加权平均延迟
                                old_weight = 0.7
                                new_weight = 0.3
                                if stats['avg_delay'] == 0:
                                    stats['avg_delay'] = avg_new_delay
                                else:
                                    stats['avg_delay'] = old_weight * stats['avg_delay'] + new_weight * avg_new_delay

                            # 计算吞吐量（Kbps）
                            duration = (self.env.now - stats['start_time']) / 1e6  # 秒
                            if duration > 0:
                                bits_received = stats['received_packets'] * config.DATA_PACKET_LENGTH
                                stats['throughput'] = bits_received / duration / 1000  # Kbps

                # 打印流统计
                if self.active_flows and self.env.now % 1e7 < 5e5:  # 每10秒左右打印一次
                    flow_stats_str = "\n当前活跃流统计:\n"
                    flow_stats_str += "-" * 60 + "\n"
                    flow_stats_str += f"{'流ID':<20} {'接收/发送':<15} {'吞吐量(Kbps)':<15} {'平均延迟(ms)':<15}\n"
                    flow_stats_str += "-" * 60 + "\n"

                    for flow_id in self.active_flows:
                        if flow_id in self.flow_stats:
                            stats = self.flow_stats[flow_id]
                            flow_stats_str += f"{flow_id:<20} {stats['received_packets']}/{stats.get('sent_packets', '?'):<15} {stats['throughput']:<15.2f} {stats['avg_delay'] / 1000:<15.2f}\n"

                    flow_stats_str += "-" * 60
                    logging.info(flow_stats_str)

            except Exception as e:
                logging.error(f"监控流状态时出错: {e}")
                traceback.print_exc()
    # 在stdma.py中需要修改的部分
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
    #         # model_dir = os.path.join(current_dir, "rl_controller/logs/STDMA_PPO_20250207_112404/best_model")
    #         # model_path = os.path.join(model_dir, "best_model.zip")
    #         model_dir = os.path.join(current_dir, "GNN_RL/models/gnn_stdma_20250316_224023/")
    #         model_path = os.path.join(model_dir, "final_model.zip")
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
    def verify_schedule_completeness(self):
        """验证时隙分配的完整性"""
        # 获取所有已分配节点
        assigned_nodes = set()
        for nodes in self.slot_schedule.values():
            if isinstance(nodes, list):
                assigned_nodes.update(nodes)
            else:
                assigned_nodes.add(nodes)

        # 获取应该被分配的节点
        expected_nodes = set()
        for req in self.traffic_requirements.values():
            if hasattr(req, 'is_active') and req.is_active and hasattr(req, 'routing_path'):
                expected_nodes.update(req.routing_path)

        # 检查是否有遗漏
        missing_nodes = expected_nodes - assigned_nodes
        if missing_nodes:
            logging.warning(f"时隙分配不完整，以下节点缺少分配: {missing_nodes}")
            return False

        return True

    def mac_send(self, packet):
        """改进的MAC层发送函数，支持批量业务处理"""
        if not self.slot_schedule:
            logging.error("时隙表未创建")
            return

        logging.info(f"Time {self.env.now}:UAV{self.my_drone.identifier} MAC layer received {type(packet).__name__}")

        # 处理业务需求
        if isinstance(packet, TrafficRequirement):
            # 生成流ID
            flow_id = f"flow_{packet.source_id}_{packet.dest_id}"
            logging.info(f"当前时隙表: {self.slot_schedule}")

            # 存储流量需求并标记为活跃
            packet.is_active = True
            if not hasattr(packet, 'flow_id'):
                packet.flow_id = flow_id
            self.traffic_requirements[packet.flow_id] = packet

            # 日志记录traffic
            logging.info(f"UAV{self.my_drone.identifier} received traffic requirement: {packet.flow_id}")

            # 检查该需求是否为批量处理的一部分
            is_batched = hasattr(packet, 'batch_id') and packet.batch_id

            # 如果已经有时隙分配的决策，则不需要重新分配
            has_existing_schedule = self.slot_schedule and len(self.slot_schedule) > 0

            try:
                # 只有以下情况才需要执行RL重新分配:
                # 1. 第一次收到业务需求（没有已有时隙分配）
                # 2. 收到的是单个业务需求（非批量处理的一部分）
                # 3. 收到批量处理的第一个业务需求
                needs_reallocation = (
                        not has_existing_schedule or
                        (not is_batched) or
                        (is_batched and not hasattr(self, 'processed_batch_ids') or
                         packet.batch_id not in getattr(self, 'processed_batch_ids', set()))
                )

                if needs_reallocation and hasattr(self, 'use_rl') and self.use_rl and self.rl_model is not None:
                    # 初始化已处理批次ID集合（如果不存在）
                    if not hasattr(self, 'processed_batch_ids'):
                        self.processed_batch_ids = set()

                    # 如果是批量处理的一部分，记录已处理的批次ID
                    if is_batched:
                        self.processed_batch_ids.add(packet.batch_id)

                    # 获取所有活跃的流
                    active_requirements = [req for req in self.traffic_requirements.values()
                                           if hasattr(req, 'is_active') and req.is_active]

                    logging.info(f"处理 {len(active_requirements)} 个活跃流")

                    # 智能检测环境是否支持多流
                    try:
                        if hasattr(self.rl_env,
                                   'supports_multiple_requirements') and self.rl_env.supports_multiple_requirements:
                            logging.info("环境支持多流参数，传入所有活跃流")
                            obs = self.rl_env.reset(active_requirements=active_requirements)[0]
                        else:
                            # 尝试传入所有活跃流的列表
                            try:
                                logging.info("尝试传入所有活跃流")
                                obs = self.rl_env.reset(requirement_data=active_requirements)[0]
                            except Exception:
                                # 回退到使用第一个流
                                logging.info("回退到使用第一个活跃流")
                                obs = self.rl_env.reset(requirement_data=active_requirements[0])[0]
                    except Exception as e:
                        logging.warning(f"重置环境时出错: {e}，使用第一个活跃流")
                        obs = self.rl_env.reset(requirement_data=active_requirements[0])[0]

                    # 记录所有活跃流的路由路径
                    for req in active_requirements:
                        if hasattr(req, 'routing_path') and req.routing_path:
                            logging.info(
                                f"活跃流 {req.flow_id if hasattr(req, 'flow_id') else f'{req.source_id}->{req.dest_id}'} 路径: {req.routing_path}")

                    # 保存旧的时隙分配
                    old_schedule = self.slot_schedule.copy() if self.slot_schedule else {}

                    # 初始化新的时隙表
                    new_schedule = {}

                    # 为每个节点生成时隙分配
                    for node in range(self.simulator.n_drones):
                        action, _ = self.rl_model.predict(obs, deterministic=True)
                        slot = int(action)

                        obs, _, done, _, info = self.rl_env.step(action)
                        if info and 'schedule' in info:
                            new_schedule = info.get('schedule', {})
                        if done:
                            break

                    # 检查首次业务需求处理
                    if not hasattr(self, 'has_processed_traffic') or not self.has_processed_traffic:
                        # 首次处理：完全替换时隙表
                        self.slot_schedule = new_schedule
                        self.has_processed_traffic = True
                        logging.info(f"首次业务需求处理，完全替换时隙表: {new_schedule}")
                    else:
                        # 后续处理：合并新旧时隙分配
                        self._update_slot_schedule(old_schedule, new_schedule)
                        logging.info(f"后续业务需求处理，合并时隙表")

                    # 验证和修复
                    if hasattr(self, 'verify_schedule_completeness'):
                        if not self.verify_schedule_completeness():
                            logging.warning("时隙分配不完整，尝试直接补充缺失节点")
                            self.fix_missing_nodes()

                    # 将更新后的时隙分配传播到其他节点
                    for node in self.simulator.drones:
                        if node.identifier != self.my_drone.identifier:
                            node.mac_protocol.slot_schedule = self.slot_schedule
                            node.mac_protocol.has_processed_traffic = True
                            logging.info(f"{node.identifier}基于RL模型更新时隙分配")
                else:
                    # 如果不需要重新分配，记录日志
                    if is_batched:
                        logging.info(f"批次 {packet.batch_id} 的一部分，已经处理过，跳过时隙重分配")
                    elif has_existing_schedule:
                        logging.info(f"已有时隙分配，跳过时隙重分配")
                    else:
                        # 使用传统方法重新分配时隙
                        self.slot_schedule = self._create_tra_slot_schedule()
                        logging.info("使用传统方法更新时隙分配")

            except Exception as e:
                logging.error(f"时隙调整失败: {e}")
                import traceback
                traceback.print_exc()
                # 如果调整失败，保持当前时隙分配不变

            # 检查并修正时隙表格式
            for slot, nodes in self.slot_schedule.items():
                if not isinstance(nodes, list):
                    logging.error(f"Slot {slot} 的节点不是列表格式: {type(nodes)}")
                    if isinstance(nodes, int):
                        self.slot_schedule[slot] = [nodes]
        # 数据包流管理
        elif isinstance(packet, DataPacket):
            mac_start_time = self.env.now

            # 使用修正后的时隙表查找分配
            assigned_slot = next((slot for slot, drones in self.slot_schedule.items()
                                  if self.my_drone.identifier in drones), None)

            if assigned_slot is None:
                logging.error(f"无人机{self.my_drone.identifier}未分配时隙")
                return

            current_time = self.env.now
            time_slot_duration = self.time_slot_duration
            num_slots = self.num_slots

            # 计算当前时隙的开始和结束时间
            slot_start = (current_time // time_slot_duration) * time_slot_duration + assigned_slot * time_slot_duration
            slot_end = slot_start + time_slot_duration

            # 等待合适的时隙
            if current_time < slot_start:
                yield self.env.timeout(slot_start - current_time)
            elif current_time >= slot_end:
                yield self.env.timeout(num_slots * time_slot_duration - (current_time - slot_start))

            # 发送数据包
            yield self.env.process(self._transmit_packet(packet))

            # 记录MAC层延迟
            mac_delay = self.env.now - mac_start_time
            logging.info(f"UAV{self.my_drone.identifier} MAC 时延: {mac_delay}")
            self.simulator.metrics.mac_delay.append(mac_delay / 1e3)
        else:
            # 其他类型的数据包直接传输
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

