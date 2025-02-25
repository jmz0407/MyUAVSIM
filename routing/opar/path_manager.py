import numpy as np
from collections import defaultdict
import logging
from phy.large_scale_fading import maximum_communication_range, euclidean_distance

class PathManager:
    """路径管理器"""

    def __init__(self, simulator, drone):
        self.simulator = simulator
        self.drone = drone
        self.path_cache = {}  # (src, dst) -> (path, timestamp)
        self.backup_paths = defaultdict(list)  # (src, dst) -> [path1, path2, ...]
        self.path_metrics = defaultdict(dict)  # (src, dst) -> {'stability': float, 'quality': float, ...}
        self.cache_timeout = 1e6  # 1秒缓存超时
        self.max_comm_range = maximum_communication_range()  # 获取最大通信范围

    def find_path(self, cost_matrix, src_id, dst_id, constraints=None):
        """寻找满足约束的最优路径"""
        current_time = self.simulator.env.now
        cache_key = (src_id, dst_id)

        # 检查缓存
        if cache_key in self.path_cache:
            path, timestamp = self.path_cache[cache_key]
            if current_time - timestamp < self.cache_timeout:
                if self._validate_path(path, constraints):
                    return path

        # 使用Dijkstra算法寻找路径
        path = self._dijkstra(cost_matrix, src_id, dst_id, constraints)
        if path:
            # 更新缓存
            self.path_cache[cache_key] = (path, current_time)
            # 查找备份路径
            self._update_backup_paths(cost_matrix, src_id, dst_id)

        return path

    def _dijkstra(self, cost_matrix, src_id, dst_id, constraints=None):
        """改进的Dijkstra算法实现"""
        n_nodes = len(cost_matrix)
        distance = np.full(n_nodes, np.inf)
        distance[src_id] = 0
        previous = np.full(n_nodes, -1)
        visited = np.zeros(n_nodes, dtype=bool)

        while True:
            # 找到最短距离的未访问节点
            unvisited_distances = np.where(~visited, distance, np.inf)
            current = np.argmin(unvisited_distances)

            if unvisited_distances[current] == np.inf:
                break

            if current == dst_id:
                break

            visited[current] = True

            # 更新邻居距离
            for neighbor in range(n_nodes):
                if not visited[neighbor] and cost_matrix[current, neighbor] != np.inf:
                    # 检查约束条件
                    if constraints and not self._check_constraints(current, neighbor, constraints):
                        continue

                    new_distance = distance[current] + cost_matrix[current, neighbor]
                    if new_distance < distance[neighbor]:
                        distance[neighbor] = new_distance
                        previous[neighbor] = current

        # 构建路径
        if distance[dst_id] == np.inf:
            return None

        path = []
        current = dst_id
        while current != -1:
            path.insert(0, current)
            current = previous[current]

        return path

    def _validate_path(self, path, constraints=None):
        """验证路径是否仍然有效"""
        if not path or len(path) < 2:
            return False

        for i in range(len(path) - 1):
            node1, node2 = path[i], path[i + 1]
            # 检查节点是否仍然可达
            if not self._nodes_reachable(node1, node2):
                return False
            # 检查约束条件
            if constraints and not self._check_constraints(node1, node2, constraints):
                return False

        return True

    def _nodes_reachable(self, node1_id, node2_id):
        """检查两个节点是否在通信范围内"""
        node1 = self.simulator.drones[node1_id]
        node2 = self.simulator.drones[node2_id]
        distance = euclidean_distance(node1.coords, node2.coords)
        return distance <= self.max_comm_range
    def _check_constraints(self, node1_id, node2_id, constraints):
        """检查是否满足约束条件"""
        for constraint in constraints:
            if not constraint(node1_id, node2_id):
                return False
        return True

    def _update_backup_paths(self, cost_matrix, src_id, dst_id):
        """更新备份路径"""
        backup_paths = []
        main_path = self.path_cache.get((src_id, dst_id), (None, None))[0]

        if not main_path:
            return

        # 修改代价矩阵以寻找备用路径
        temp_matrix = cost_matrix.copy()

        # 增加主路径上链路的代价
        for i in range(len(main_path) - 1):
            node1, node2 = main_path[i], main_path[i + 1]
            temp_matrix[node1, node2] *= 2
            temp_matrix[node2, node1] *= 2

        # 寻找首选备份路径
        backup1 = self._dijkstra(temp_matrix, src_id, dst_id)
        if backup1 and backup1 != main_path:
            backup_paths.append(backup1)

            # 继续增加已使用链路的代价来寻找第二备份路径
            for i in range(len(backup1) - 1):
                node1, node2 = backup1[i], backup1[i + 1]
                temp_matrix[node1, node2] *= 2
                temp_matrix[node2, node1] *= 2

            backup2 = self._dijkstra(temp_matrix, src_id, dst_id)
            if backup2 and backup2 not in [main_path, backup1]:
                backup_paths.append(backup2)

        self.backup_paths[(src_id, dst_id)] = backup_paths
        logging.info(f'Updated backup paths for {src_id}->{dst_id}: {len(backup_paths)} paths found')

    def handle_link_failure(self, failed_link):
        """处理链路故障"""
        src_node, dst_node = failed_link
        affected_paths = []

        # 找出受影响的路径
        for (s, d), (path, _) in self.path_cache.items():
            if self._path_contains_link(path, failed_link):
                affected_paths.append((s, d))

        # 重新计算受影响的路径
        for s, d in affected_paths:
            # 尝试使用备份路径
            if (s, d) in self.backup_paths:
                valid_backup = next(
                    (path for path in self.backup_paths[(s, d)]
                     if self._validate_path(path)), None)
                if valid_backup:
                    self.path_cache[(s, d)] = (valid_backup, self.simulator.env.now)
                    logging.info(f'Switched to backup path for {s}->{d}')
                    continue

            # 如果没有可用的备份路径，删除缓存
            if (s, d) in self.path_cache:
                del self.path_cache[(s, d)]

        logging.info(f'Handled link failure {failed_link}: {len(affected_paths)} paths affected')

    def _path_contains_link(self, path, link):
        """检查路径是否包含指定链路"""
        if not path:
            return False

        src, dst = link
        return any(
            (path[i] == src and path[i + 1] == dst) or
            (path[i] == dst and path[i + 1] == src)
            for i in range(len(path) - 1)
        )