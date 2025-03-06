import logging
import numpy as np
import math
from collections import defaultdict
from utils import config
from entities.packet import DataPacket


class MultiPathLoadBalancer:
    """
    Advanced load balancer for multi-path routing in UAV networks.
    Implements multiple load balancing strategies and adaptive path selection.

    Attributes:
        simulator: Reference to the simulator instance
        path_selection_strategy: Strategy for distributing traffic ('weighted', 'round_robin', 'adaptive')
        max_paths: Maximum number of paths to maintain per destination
        path_quality_threshold: Minimum quality threshold for path selection
        path_stats: Statistics for each path (delay, loss rate, throughput, etc.)
        traffic_stats: Statistics for different traffic types
        load_history: Historical load information for prediction
        active_flows: Currently active traffic flows
    """

    def __init__(self, simulator, max_paths=3, path_selection_strategy='weighted'):
        self.simulator = simulator
        self.max_paths = max_paths
        self.path_selection_strategy = path_selection_strategy
        self.path_quality_threshold = 0.3  # Minimum acceptable path quality (0-1)

        # Path statistics tracking
        self.path_stats = defaultdict(lambda: defaultdict(dict))  # src → dst → path_id → stats
        self.traffic_stats = defaultdict(dict)  # flow_id → stats
        self.load_history = defaultdict(list)  # node_id → [historical load values]
        self.active_flows = {}  # flow_id → flow info

        # Weights for path quality calculation
        self.weights = {
            'delay': 0.4,
            'loss_rate': 0.3,
            'capacity': 0.2,
            'stability': 0.1
        }

        # Path monitoring interval
        self.monitor_interval = config.MONITOR_INTERVAL

        logging.info(
            f"Initialized MultiPathLoadBalancer with {path_selection_strategy} strategy, max_paths={max_paths}")

    def update_path_stats(self, src_id, dst_id, path_id, metrics):
        """
        Update statistics for a specific path.

        Args:
            src_id: Source drone ID
            dst_id: Destination drone ID
            path_id: Path identifier
            metrics: Dictionary containing metrics (delay, loss_rate, throughput, etc.)
        """
        # Ensure path_id entry exists
        if path_id not in self.path_stats[src_id][dst_id]:
            self.path_stats[src_id][dst_id][path_id] = {
                'delay': float('inf'),
                'loss_rate': 1.0,
                'throughput': 0,
                'capacity': 0,
                'stability': 0,
                'last_updated': self.simulator.env.now,
                'samples': 0,
                'quality': 0
            }

        stats = self.path_stats[src_id][dst_id][path_id]
        stats['samples'] += 1

        # Exponential moving average for each metric
        alpha = 0.3  # Weight for new samples

        for metric, value in metrics.items():
            if metric in stats:
                stats[metric] = (1 - alpha) * stats[metric] + alpha * value

        # Update last updated time
        stats['last_updated'] = self.simulator.env.now

        # Recalculate path quality
        self._calculate_path_quality(src_id, dst_id, path_id)

        # Log significant changes
        if stats['samples'] % 10 == 0:  # Log every 10 samples
            logging.info(f"Path stats updated - src:{src_id}, dst:{dst_id}, path:{path_id}, "
                         f"quality:{stats['quality']:.2f}, delay:{stats['delay']:.2f}ms")

    def _calculate_path_quality(self, src_id, dst_id, path_id):
        """Calculate overall path quality based on multiple metrics (0-1 scale)"""
        stats = self.path_stats[src_id][dst_id][path_id]

        # Normalize metrics to 0-1 scale (1 is best)
        max_delay = 500  # Maximum acceptable delay in ms
        norm_delay = max(0, 1 - (stats['delay'] / max_delay))

        norm_loss = 1 - stats['loss_rate']
        norm_throughput = min(1, stats['throughput'] / 1000)  # Normalize to 1 Mbps

        # Combine metrics using weights
        quality = (
                self.weights['delay'] * norm_delay +
                self.weights['loss_rate'] * norm_loss +
                self.weights['capacity'] * norm_throughput +
                self.weights['stability'] * stats['stability']
        )

        # Update quality score
        stats['quality'] = quality
        return quality

    def select_paths(self, src_id, dst_id, packet=None, num_paths=None):
        """
        Select best paths for packet routing based on current statistics and strategy.

        Args:
            src_id: Source drone ID
            dst_id: Destination drone ID
            packet: Data packet to route (optional)
            num_paths: Number of paths to select (defaults to self.max_paths)

        Returns:
            List of selected path IDs ordered by preference
        """
        if num_paths is None:
            num_paths = self.max_paths

        # Check if we have statistics for this src-dst pair
        if dst_id not in self.path_stats[src_id] or not self.path_stats[src_id][dst_id]:
            logging.warning(f"No path statistics available for {src_id}->{dst_id}")
            return []

        # Get all available paths
        available_paths = list(self.path_stats[src_id][dst_id].keys())

        # Filter paths based on quality threshold
        qualified_paths = [
            path_id for path_id in available_paths
            if self.path_stats[src_id][dst_id][path_id]['quality'] >= self.path_quality_threshold
        ]

        if not qualified_paths:
            logging.warning(f"No qualified paths found for {src_id}->{dst_id}")
            # Fall back to best available path if none meet the threshold
            if available_paths:
                best_path = max(
                    available_paths,
                    key=lambda p: self.path_stats[src_id][dst_id][p]['quality']
                )
                return [best_path]
            return []

        # Apply path selection strategy
        if self.path_selection_strategy == 'weighted':
            return self._weighted_path_selection(src_id, dst_id, qualified_paths, num_paths)
        elif self.path_selection_strategy == 'round_robin':
            return self._round_robin_selection(src_id, dst_id, qualified_paths, num_paths)
        elif self.path_selection_strategy == 'adaptive':
            return self._adaptive_path_selection(src_id, dst_id, qualified_paths, packet, num_paths)
        else:
            # Default to selecting paths by quality
            sorted_paths = sorted(
                qualified_paths,
                key=lambda p: self.path_stats[src_id][dst_id][p]['quality'],
                reverse=True
            )
            return sorted_paths[:num_paths]

    def _weighted_path_selection(self, src_id, dst_id, paths, num_paths):
        """Select paths using weighted probability based on quality"""
        if not paths:
            return []

        # Get quality scores
        qualities = [self.path_stats[src_id][dst_id][p]['quality'] for p in paths]

        # Ensure all qualities are positive
        min_quality = min(qualities)
        if min_quality <= 0:
            adjusted_qualities = [q - min_quality + 0.01 for q in qualities]
        else:
            adjusted_qualities = qualities

        # Calculate weights (probabilities)
        total = sum(adjusted_qualities)
        weights = [q / total for q in adjusted_qualities]

        # Select paths based on weighted probabilities
        selected_indices = np.random.choice(
            len(paths),
            size=min(num_paths, len(paths)),
            replace=False,
            p=weights
        )

        return [paths[i] for i in selected_indices]

    def _round_robin_selection(self, src_id, dst_id, paths, num_paths):
        """Simple round-robin path selection"""
        if not paths:
            return []

        flow_id = f"{src_id}_{dst_id}"

        if flow_id not in self.active_flows:
            self.active_flows[flow_id] = {'last_path_index': 0}

        # Get last used index
        last_index = self.active_flows[flow_id]['last_path_index']

        # Sort paths by quality for initial ordering
        sorted_paths = sorted(
            paths,
            key=lambda p: self.path_stats[src_id][dst_id][p]['quality'],
            reverse=True
        )

        # Select paths in round-robin fashion
        selected = []
        for i in range(num_paths):
            index = (last_index + i) % len(sorted_paths)
            selected.append(sorted_paths[index])

        # Update last index
        self.active_flows[flow_id]['last_path_index'] = (last_index + num_paths) % len(sorted_paths)

        return selected

    def _adaptive_path_selection(self, src_id, dst_id, paths, packet, num_paths):
        """
        Adaptive path selection based on packet priority, flow type, and network conditions
        """
        if not paths:
            return []

        # Default to quality-based sorting
        sorted_paths = sorted(
            paths,
            key=lambda p: self.path_stats[src_id][dst_id][p]['quality'],
            reverse=True
        )

        # If no packet specifics are provided, return quality-sorted paths
        if packet is None:
            return sorted_paths[:num_paths]

        # Check if this is a high-priority packet
        if hasattr(packet, 'priority') and packet.priority > 0:
            # For high priority, select lowest delay paths
            delay_sorted = sorted(
                paths,
                key=lambda p: self.path_stats[src_id][dst_id][p]['delay']
            )
            return delay_sorted[:num_paths]

        # For regular traffic, consider current path loads
        path_loads = {}
        for path_id in paths:
            # Calculate path load based on recent traffic
            path_nodes = self._get_path_nodes(path_id)
            avg_load = 0
            if path_nodes:
                loads = [self._get_node_load(node_id) for node_id in path_nodes]
                avg_load = sum(loads) / len(loads)
            path_loads[path_id] = avg_load

        # Balance between quality and load
        balanced_score = {}
        for path_id in paths:
            quality = self.path_stats[src_id][dst_id][path_id]['quality']
            load = path_loads[path_id]
            # Higher score for higher quality and lower load
            balanced_score[path_id] = quality * (1 - min(1, load / 0.8))

        # Sort by balanced score
        balanced_paths = sorted(
            paths,
            key=lambda p: balanced_score[p],
            reverse=True
        )

        return balanced_paths[:num_paths]

    def distribute_packet(self, packet, selected_paths):
        """
        Determine which path to use for a specific packet based on current strategy

        Args:
            packet: The data packet to route
            selected_paths: List of available paths

        Returns:
            Selected path ID
        """
        if not selected_paths:
            return None

        if len(selected_paths) == 1:
            return selected_paths[0]

        # Get source and destination
        src_id = packet.src_drone.identifier
        dst_id = packet.dst_drone.identifier
        flow_id = f"{src_id}_{dst_id}"

        # Update packet multi-path flag
        packet.is_multipath = True

        # Check if we should use parallel transmission
        if packet.path_selection_strategy == 'parallel' and hasattr(packet, 'retries') and packet.retries == 0:
            # For initial transmission of high-priority packets, use multiple paths simultaneously
            if hasattr(packet, 'priority') and packet.priority > 0:
                packet.parallel_paths = selected_paths
                logging.info(f"Using parallel transmission for high-priority packet {packet.packet_id} "
                             f"on paths: {selected_paths}")
                return selected_paths

        # For regular packets or retransmissions, select a single path
        # Track per-flow path selection to maintain packet ordering
        if flow_id not in self.active_flows:
            self.active_flows[flow_id] = {
                'path_counters': defaultdict(int),
                'last_path': None,
                'packets_per_path': {}
            }

        flow_data = self.active_flows[flow_id]

        # Distribute packets across paths based on quality ratio
        total_quality = sum(self.path_stats[src_id][dst_id][p]['quality'] for p in selected_paths)
        if total_quality <= 0:
            # Equal distribution if all paths have zero quality
            path_weights = [1 / len(selected_paths)] * len(selected_paths)
        else:
            path_weights = [self.path_stats[src_id][dst_id][p]['quality'] / total_quality
                            for p in selected_paths]

        # Select path based on weighted distribution
        path_index = 0
        r = np.random.random()
        cumulative_prob = 0
        for i, weight in enumerate(path_weights):
            cumulative_prob += weight
            if r <= cumulative_prob:
                path_index = i
                break

        selected_path = selected_paths[path_index]

        # Update path counters
        flow_data['path_counters'][selected_path] += 1
        flow_data['last_path'] = selected_path

        logging.debug(f"Selected path {selected_path} for packet {packet.packet_id} "
                      f"from {src_id} to {dst_id}")

        return selected_path

    def update_node_load(self, node_id, queue_size, processing_rate=None):
        """Update load information for a specific node"""
        # Calculate normalized load (0-1)
        max_queue = config.MAX_QUEUE_SIZE
        load = min(1.0, queue_size / max_queue)

        # Store in history (limited to last 10 measurements)
        self.load_history[node_id].append(load)
        if len(self.load_history[node_id]) > 10:
            self.load_history[node_id].pop(0)

        return load

    def _get_node_load(self, node_id):
        """Get average load for a node based on history"""
        if node_id not in self.load_history or not self.load_history[node_id]:
            return 0

        return sum(self.load_history[node_id]) / len(self.load_history[node_id])

    def _get_path_nodes(self, path_id):
        """Extract nodes in a path from path_id"""
        # This is a placeholder - in a real implementation, you'd need to
        # maintain a mapping from path_id to actual path nodes
        # For now we'll return an empty list
        return []

    def handle_path_failure(self, src_id, dst_id, path_id):
        """
        Handle a path failure by marking the path as unavailable
        and redistributing active flows
        """
        if dst_id in self.path_stats[src_id] and path_id in self.path_stats[src_id][dst_id]:
            # Mark path as failed
            self.path_stats[src_id][dst_id][path_id]['quality'] = 0
            self.path_stats[src_id][dst_id][path_id]['stability'] = 0
            self.path_stats[src_id][dst_id][path_id]['loss_rate'] = 1.0

            logging.warning(f"Path failure detected: src:{src_id}, dst:{dst_id}, path:{path_id}")

            # Find active flows using this path
            flow_id = f"{src_id}_{dst_id}"
            if flow_id in self.active_flows:
                flow_data = self.active_flows[flow_id]

                # If this was the last used path, select a new one
                if flow_data.get('last_path') == path_id:
                    # Clear the last path so next packet will select a new one
                    flow_data['last_path'] = None

                    logging.info(f"Cleared last path for flow {flow_id} due to path failure")

    def get_path_metrics(self, src_id, dst_id):
        """Get metrics for all paths between src and dst"""
        if dst_id not in self.path_stats[src_id]:
            return {}

        return {
            path_id: {
                'quality': stats['quality'],
                'delay': stats['delay'],
                'loss_rate': stats['loss_rate'],
                'throughput': stats['throughput']
            }
            for path_id, stats in self.path_stats[src_id][dst_id].items()
        }

    def get_flow_distribution(self, src_id, dst_id):
        """Get packet distribution across paths for a specific flow"""
        flow_id = f"{src_id}_{dst_id}"
        if flow_id not in self.active_flows:
            return {}

        return dict(self.active_flows[flow_id]['path_counters'])

    def monitor_paths(self):
        """
        Process to periodically monitor path quality and update statistics
        This should be called as a SimPy process
        """
        while True:
            # Wait for monitor interval
            yield self.simulator.env.timeout(self.monitor_interval)

            # Check each active path
            for src_id in self.path_stats:
                for dst_id in self.path_stats[src_id]:
                    for path_id in list(self.path_stats[src_id][dst_id].keys()):
                        stats = self.path_stats[src_id][dst_id][path_id]

                        # Check if path has been updated recently
                        time_since_update = self.simulator.env.now - stats['last_updated']
                        if time_since_update > 2 * self.monitor_interval:
                            # Path hasn't been used recently, reduce stability score
                            decay_factor = 0.9  # 10% decay per check
                            stats['stability'] *= decay_factor

                            # Recalculate quality
                            self._calculate_path_quality(src_id, dst_id, path_id)

                        # If path quality is very low, consider it inactive
                        if stats['quality'] < 0.1 and time_since_update > 5 * self.monitor_interval:
                            logging.info(f"Removing inactive path: src:{src_id}, dst:{dst_id}, path:{path_id}")
                            del self.path_stats[src_id][dst_id][path_id]

            # Log status of active flows
            for flow_id, flow_data in self.active_flows.items():
                if 'path_counters' in flow_data and flow_data['path_counters']:
                    src_id, dst_id = flow_id.split('_')
                    logging.info(f"Flow {flow_id} distribution: {dict(flow_data['path_counters'])}")


class MultipathRouter:
    """
    Enhanced multi-path routing component that integrates with the load balancer
    and provides path discovery, maintenance, and selection functionality.
    """

    def __init__(self, drone, max_paths=3):
        self.drone = drone
        self.simulator = drone.simulator
        self.max_paths = max_paths
        self.paths_cache = defaultdict(list)  # dst_id -> list of paths
        self.path_expiry = {}  # path_id -> expiry time
        self.path_hop_count = {}  # path_id -> hop count

        # Create load balancer instance
        self.load_balancer = MultiPathLoadBalancer(
            simulator=self.simulator,
            max_paths=max_paths,
            path_selection_strategy='adaptive'
        )

        # Start monitoring process
        self.simulator.env.process(self.load_balancer.monitor_paths())
        self.simulator.env.process(self.monitor_paths())

    def discover_paths(self, dst_id):
        """
        Discover multiple paths to destination using various routing algorithms
        Returns a list of path IDs that were discovered
        """
        logging.info(f"UAV {self.drone.identifier} discovering paths to {dst_id}")
        discovered_paths = []

        # Get existing base routing protocol instance
        base_router = self.drone.routing_protocol

        # Method 1: Use existing routing protocol's path(s)
        # We'll use the same path calculation but with different parameters/weights
        try:
            # Get direct path from base routing
            for i in range(min(3, self.max_paths)):
                # Adjust cost matrix parameters slightly for each path
                if hasattr(base_router, 'calculate_cost_matrix'):
                    # Calculate slightly different cost matrices for path diversity
                    jitter = 0.05 * (i + 1)  # Small randomness for diversity
                    cost_matrix = base_router.calculate_cost_matrix(jitter=jitter)

                    # Use the router's path finding algorithm
                    if hasattr(base_router, 'dijkstra'):
                        path = base_router.dijkstra(
                            cost_matrix,
                            self.drone.identifier,
                            dst_id,
                            i  # Pass a parameter to ensure different paths
                        )

                        if path and len(path) > 1:
                            path_id = f"path_{self.drone.identifier}_{dst_id}_{i}"
                            self._store_path(path_id, path, dst_id)
                            discovered_paths.append(path_id)

                            # Calculate and store hop count
                            self.path_hop_count[path_id] = len(path) - 1
                            logging.info(f"Found path {path_id} with {len(path) - 1} hops: {path}")
        except Exception as e:
            logging.error(f"Error discovering paths using base routing: {str(e)}")

        # Method 2: Use geographic routing for an additional path
        # Simple geographic approach - get nodes that are roughly in the direction of destination
        if len(discovered_paths) < self.max_paths:
            try:
                geo_path = self._geographic_path(dst_id)
                if geo_path and len(geo_path) > 1:
                    path_id = f"geo_path_{self.drone.identifier}_{dst_id}"
                    self._store_path(path_id, geo_path, dst_id)
                    discovered_paths.append(path_id)
                    self.path_hop_count[path_id] = len(geo_path) - 1
                    logging.info(f"Found geographic path {path_id} with {len(geo_path) - 1} hops: {geo_path}")
            except Exception as e:
                logging.error(f"Error discovering geographic path: {str(e)}")

        # Method 3: Try to find a more stable path based on UAV mobility patterns
        if len(discovered_paths) < self.max_paths:
            try:
                stable_path = self._stable_mobility_path(dst_id)
                if stable_path and len(stable_path) > 1:
                    path_id = f"stable_path_{self.drone.identifier}_{dst_id}"
                    self._store_path(path_id, stable_path, dst_id)
                    discovered_paths.append(path_id)
                    self.path_hop_count[path_id] = len(stable_path) - 1
                    logging.info(f"Found stable path {path_id} with {len(stable_path) - 1} hops: {stable_path}")
            except Exception as e:
                logging.error(f"Error discovering stable path: {str(e)}")

        # Update path cache
        self.paths_cache[dst_id] = discovered_paths

        return discovered_paths

    def _geographic_path(self, dst_id):
        """Find a path based on geographic positions"""
        # Simple implementation - get nodes that are in the general direction of destination
        src_pos = self.drone.coords
        dst_pos = self.simulator.drones[dst_id].coords

        # Vector from source to destination
        direction_vector = [
            dst_pos[0] - src_pos[0],
            dst_pos[1] - src_pos[1],
            dst_pos[2] - src_pos[2]
        ]

        # Normalize vector
        magnitude = math.sqrt(sum(x * x for x in direction_vector))
        if magnitude > 0:
            direction_vector = [x / magnitude for x in direction_vector]

        # Find nodes that are roughly in this direction
        candidates = []
        for node_id, drone in enumerate(self.simulator.drones):
            if node_id == self.drone.identifier or node_id == dst_id:
                continue

            node_pos = drone.coords

            # Vector from source to this node
            node_vector = [
                node_pos[0] - src_pos[0],
                node_pos[1] - src_pos[1],
                node_pos[2] - src_pos[2]
            ]

            node_dist = math.sqrt(sum(x * x for x in node_vector))
            if node_dist > 0:
                node_vector = [x / node_dist for x in node_vector]

            # Calculate dot product to see if node is in similar direction
            dot_product = sum(a * b for a, b in zip(direction_vector, node_vector))

            # If dot product is positive and high, node is in similar direction
            if dot_product > 0.7:  # Cosine similarity threshold
                distance_to_dst = math.sqrt(sum((a - b) ** 2 for a, b in zip(node_pos, dst_pos)))
                candidates.append((node_id, distance_to_dst))

        # Sort by distance to destination
        candidates.sort(key=lambda x: x[1])

        # Construct a path through a few of these nodes
        path = [self.drone.identifier]
        visited = {self.drone.identifier}

        remaining_dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(src_pos, dst_pos)))

        # Add up to 2 intermediate nodes
        for i in range(min(2, len(candidates))):
            node_id = candidates[i][0]

            # Ensure we're making progress toward destination
            node_pos = self.simulator.drones[node_id].coords
            dist_to_dst = math.sqrt(sum((a - b) ** 2 for a, b in zip(node_pos, dst_pos)))

            if dist_to_dst < remaining_dist and node_id not in visited:
                path.append(node_id)
                visited.add(node_id)
                remaining_dist = dist_to_dst

        # Add destination
        path.append(dst_id)

        return path

    def _stable_mobility_path(self, dst_id):
        """Find a path with more stable links based on mobility patterns"""
        # Start with the source node
        path = [self.drone.identifier]
        visited = {self.drone.identifier}

        current_id = self.drone.identifier
        dst_pos = self.simulator.drones[dst_id].coords

        # Keep adding nodes until we reach the destination
        while current_id != dst_id:
            current_pos = self.simulator.drones[current_id].coords

            # Find all nodes within communication range
            neighbors = []
            for node_id, drone in enumerate(self.simulator.drones):
                if node_id in visited:
                    continue

                node_pos = drone.coords
                distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(current_pos, node_pos)))

                # Check if node is within communication range
                if distance <= config.SENSING_RANGE:
                    # Calculate stability based on velocity vectors
                    stability = self._calculate_link_stability(current_id, node_id)

                    # Calculate progress toward destination
                    dist_to_dst = math.sqrt(sum((a - b) ** 2 for a, b in zip(node_pos, dst_pos)))

                    # Add to neighbors list with metrics
                    neighbors.append((node_id, stability, dist_to_dst))

            if not neighbors:
                # No viable next hop, path is incomplete
                return None

            # Sort by combination of stability and distance progress
            neighbors.sort(key=lambda x: (0.7 * x[1] - 0.3 * x[2]), reverse=True)

            # Select best neighbor
            next_id = neighbors[0][0]
            path.append(next_id)
            visited.add(next_id)
            current_id = next_id

            # If the path is getting too long, abort
            if len(path) > 5:
                return None

            # If we added the destination, we're done
            if next_id == dst_id:
                break

        return path

    def _calculate_link_stability(self, node1_id, node2_id):
        """计算两个节点之间链路的稳定性（0-1范围，1表示最稳定）"""
        try:
            # 获取两个节点
            drone1 = self.simulator.drones[node1_id]
            drone2 = self.simulator.drones[node2_id]

            # 获取位置和速度向量
            pos1 = drone1.coords
            pos2 = drone2.coords

            # 尝试获取速度向量，如果可用
            vel1 = drone1.velocity if hasattr(drone1, 'velocity') else [0, 0, 0]
            vel2 = drone2.velocity if hasattr(drone2, 'velocity') else [0, 0, 0]

            # 计算相对速度向量
            rel_vel = [v1 - v2 for v1, v2 in zip(vel1, vel2)]
            rel_speed = math.sqrt(sum(v * v for v in rel_vel))

            # 计算当前距离
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))

            # 通信范围
            comm_range = config.SENSING_RANGE

            # 如果两个节点相对静止或移动缓慢，链路将更稳定
            if rel_speed < 1.0:
                return 0.9  # 非常稳定

            # 计算相对方向
            direction = [p2 - p1 for p1, p2 in zip(pos1, pos2)]
            direction_mag = math.sqrt(sum(d * d for d in direction))

            if direction_mag > 0:
                direction = [d / direction_mag for d in direction]

            # 计算相对速度与连接方向的点积
            # 负值表示节点正在靠近，正值表示正在远离
            dot_product = sum(a * b for a, b in zip(rel_vel, direction))

            # 预测这两个节点将保持在通信范围内的时间
            # 如果正在靠近，则为正，如果正在远离，则为负
            if abs(dot_product) < 0.001:  # 几乎垂直移动
                predicted_time = 100  # 假设很长时间
            else:
                # 如果dot_product为负，节点正在靠近，否则正在远离
                if dot_product < 0:
                    # 计算还需要多长时间才能达到最小距离
                    closing_time = abs((distance - 0.1 * comm_range) / dot_product)
                    predicted_time = 100  # 假设很长时间
                else:
                    # 计算还需要多长时间才能超出通信范围
                    separation_time = (comm_range - distance) / dot_product
                    predicted_time = separation_time

            # 标准化稳定性分数 (0-1)
            # 使用指数衰减函数将预测时间映射到稳定性分数
            max_time = 60  # 假设60秒是最大稳定期
            stability = 1.0 - math.exp(-predicted_time / max_time)

            return max(0.1, min(0.95, stability))  # 限制在0.1到0.95之间

        except Exception as e:
            logging.error(f"计算链路稳定性出错: {str(e)}")
            return 0.5  # 出错时返回中等稳定性值

    def _store_path(self, path_id, path, dst_id):
        """存储路径并设置过期时间"""
        # 存储路径
        self.paths_cache[dst_id].append(path_id)

        # 设置过期时间 (当前时间 + 30秒)
        self.path_expiry[path_id] = self.simulator.env.now + 30 * 1e6

        # 初始化或更新路径统计信息
        # 计算路径初始质量指标
        delay = self._estimate_path_delay(path)
        hop_count = len(path) - 1

        # 更新负载均衡器的路径统计信息
        self.load_balancer.update_path_stats(
            self.drone.identifier,
            dst_id,
            path_id,
            {
                'delay': delay,
                'loss_rate': 0.05 * hop_count,  # 根据跳数估计丢包率
                'throughput': 500 / hop_count,  # 根据跳数估计吞吐量
                'capacity': 1000,
                'stability': 0.9 - (0.1 * hop_count)  # 根据跳数估计稳定性
            }
        )

    def _estimate_path_delay(self, path):
        """估算路径的端到端延迟（毫秒）"""
        if not path or len(path) < 2:
            return float('inf')

        hop_count = len(path) - 1

        # 基本传输延迟 (毫秒)
        transmission_delay = (config.DATA_PACKET_LENGTH / config.BIT_RATE) * 1000

        # 排队延迟估计 (每跳10毫秒)
        queuing_delay = 10

        # MAC访问延迟估计
        mac_delay = 5

        # 总延迟
        total_delay = hop_count * (transmission_delay + queuing_delay + mac_delay)

        return total_delay

    def get_paths(self, dst_id):
        """获取到目的地的有效路径"""
        # 检查缓存中是否有有效路径
        current_time = self.simulator.env.now
        valid_paths = []

        for path_id in list(self.paths_cache.get(dst_id, [])):
            # 检查路径是否过期
            if path_id in self.path_expiry and current_time < self.path_expiry[path_id]:
                valid_paths.append(path_id)
            else:
                # 移除过期路径
                if path_id in self.path_expiry:
                    del self.path_expiry[path_id]
                if path_id in self.path_hop_count:
                    del self.path_hop_count[path_id]

                # 从缓存中移除
                if path_id in self.paths_cache[dst_id]:
                    self.paths_cache[dst_id].remove(path_id)

        # 如果没有有效路径，尝试发现新路径
        if not valid_paths:
            valid_paths = self.discover_paths(dst_id)

        return valid_paths

    def get_next_hop(self, packet):
        """为数据包选择下一跳"""
        if not isinstance(packet, DataPacket):
            # 非数据包使用默认路由
            return None

        dst_id = packet.dst_drone.identifier

        # 检查多路径路由开关
        if not config.MULTIPATH_ENABLED:
            # 如果多路径路由未启用，使用基础路由协议
            return None

        # 获取有效路径
        paths = self.get_paths(dst_id)
        if not paths:
            # 如果没有可用路径，返回None让基础路由协议处理
            logging.warning(f"UAV {self.drone.identifier}: 没有到达 {dst_id} 的可用路径")
            return None

        # 使用负载均衡器选择路径
        selected_paths = self.load_balancer.select_paths(
            self.drone.identifier,
            dst_id,
            packet,
            num_paths=min(3, len(paths))
        )

        if not selected_paths:
            logging.warning(f"UAV {self.drone.identifier}: 负载均衡器没有选择到 {dst_id} 的路径")
            return None

        # 为特定数据包分配路径
        path_id = self.load_balancer.distribute_packet(packet, selected_paths)

        if not path_id:
            logging.warning(f"UAV {self.drone.identifier}: 无法为数据包 {packet.packet_id} 分配路径")
            return None

        # 现在我们需要从path_id获取实际的路径和下一跳
        # 这需要一个从path_id到路径的映射
        # 由于_store_path只保存了path_id而没有实际存储路径内容，这里需要补充实现

        # 假设这里有一个_get_path函数可以获取path_id对应的实际路径
        path = self._get_path(path_id)
        if not path:
            logging.error(f"UAV {self.drone.identifier}: 无法获取路径 {path_id} 的内容")
            return None

        # 找到当前节点在路径中的位置
        if self.drone.identifier not in path:
            logging.error(f"UAV {self.drone.identifier}: 不在路径 {path_id} 中")
            return None

        current_index = path.index(self.drone.identifier)

        # 检查是否已经是最后一跳
        if current_index >= len(path) - 1:
            logging.info(f"UAV {self.drone.identifier}: 数据包 {packet.packet_id} 已到达最后一跳")
            return None

        # 获取下一跳
        next_hop = path[current_index + 1]

        # 更新数据包的路由路径信息
        packet.routing_path = path
        packet.next_hop_id = next_hop

        # 更新路径统计
        metrics = {
            'delay': self._get_link_delay(self.drone.identifier, next_hop),
            'loss_rate': self._get_link_loss_rate(self.drone.identifier, next_hop),
            'throughput': self._get_link_throughput(self.drone.identifier, next_hop),
            'stability': self._calculate_link_stability(self.drone.identifier, next_hop)
        }

        self.load_balancer.update_path_stats(
            self.drone.identifier,
            dst_id,
            path_id,
            metrics
        )

        logging.info(f"UAV {self.drone.identifier}: 为数据包 {packet.packet_id} 选择下一跳 {next_hop}")

        return next_hop

    def _get_path(self, path_id):
        """从path_id获取实际路径"""
        # 这是一个简化实现，实际上这可能需要从某个存储中获取
        # 在完整实现中，应该维护一个从path_id到实际路径的映射

        # 解析path_id获取相关信息
        if isinstance(path_id, list):
            # 如果path_id已经是列表，直接返回
            return path_id
        elif isinstance(path_id, str):
            # 处理字符串格式的path_id
            parts = path_id.split('_')
            # 原有处理逻辑...
        else:
            logging.error(f"不支持的path_id类型: {type(path_id)}")
            return None

        # 添加调试日志
        logging.debug(f"Parsing path_id: {path_id}, parts: {parts}")

        # 检查parts的长度和格式
        if len(parts) < 4:
            logging.error(f"无效的path_id格式: {path_id}")
            return None

        # 根据您的错误，parts[1]是'path'而不是数字
        # 修改解析逻辑以适应您的path_id格式
        try:
            if parts[0] == 'path':
                src_id = int(parts[2])
                dst_id = int(parts[3])
                path_index = int(parts[4]) if len(parts) > 4 else 0
            else:
                src_id = int(parts[1])
                dst_id = int(parts[2])
                path_index = int(parts[3]) if len(parts) > 3 else 0
        except ValueError as e:
            logging.error(f"解析path_id失败: {path_id}, 错误: {str(e)}")
            return None

        path_type = parts[0]
        src_id = int(parts[1])
        dst_id = int(parts[2])

        # 基于path_id类型重新计算路径
        if path_type == "path":
            # 使用基础路由重新计算
            base_router = self.drone.routing_protocol
            path_index = int(parts[3])

            if hasattr(base_router, 'calculate_cost_matrix') and hasattr(base_router, 'dijkstra'):
                jitter = 0.05 * (path_index + 1)
                cost_matrix = base_router.calculate_cost_matrix(jitter=jitter)
                return base_router.dijkstra(cost_matrix, src_id, dst_id, path_index)

        elif path_type == "geo":
            # 使用地理路由重新计算
            return self._geographic_path(dst_id)

        elif path_type == "stable":
            # 使用稳定路径重新计算
            return self._stable_mobility_path(dst_id)

        return None

    def _get_link_delay(self, src_id, dst_id):
        """估计两个节点之间的链路延迟（毫秒）"""
        # 基本传输延迟
        transmission_delay = (config.DATA_PACKET_LENGTH / config.BIT_RATE) * 1000

        # 根据队列长度估计排队延迟
        if hasattr(self.simulator.drones[dst_id], 'transmitting_queue'):
            queue_size = self.simulator.drones[dst_id].transmitting_queue.qsize()
            queuing_delay = queue_size * 5  # 每个包5毫秒
        else:
            queuing_delay = 10  # 默认值

        # 根据链路质量估计额外延迟
        # 获取两个节点之间的距离
        src_pos = self.simulator.drones[src_id].coords
        dst_pos = self.simulator.drones[dst_id].coords
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(src_pos, dst_pos)))

        # 距离越远，延迟越高
        distance_factor = distance / config.SENSING_RANGE
        distance_delay = 5 * distance_factor  # 最大5毫秒额外延迟

        total_delay = transmission_delay + queuing_delay + distance_delay

        return total_delay

    def _get_link_loss_rate(self, src_id, dst_id):
        """估计两个节点之间的链路丢包率（0-1）"""
        # 获取两个节点之间的距离
        src_pos = self.simulator.drones[src_id].coords
        dst_pos = self.simulator.drones[dst_id].coords
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(src_pos, dst_pos)))

        # 距离越远，丢包率越高
        distance_factor = distance / config.SENSING_RANGE

        # 基础丢包率
        base_loss = 0.01  # 1%的基础丢包率

        # 距离导致的额外丢包率
        distance_loss = 0.1 * (distance_factor ** 2)  # 最大10%的距离相关丢包

        # 网络拥塞导致的丢包
        congestion_loss = 0
        if hasattr(self.simulator.drones[dst_id], 'transmitting_queue'):
            queue_size = self.simulator.drones[dst_id].transmitting_queue.qsize()
            queue_factor = queue_size / config.MAX_QUEUE_SIZE
            congestion_loss = 0.2 * queue_factor  # 最大20%的拥塞相关丢包

        total_loss = base_loss + distance_loss + congestion_loss

        # 限制在0-1范围内
        return min(0.95, max(0.01, total_loss))

    def _get_link_throughput(self, src_id, dst_id):
        """估计两个节点之间的链路吞吐量（bps）"""
        # 基础吞吐量
        base_throughput = config.BIT_RATE  # 最大比特率

        # 获取两个节点之间的距离
        src_pos = self.simulator.drones[src_id].coords
        dst_pos = self.simulator.drones[dst_id].coords
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(src_pos, dst_pos)))

        # 距离越远，吞吐量越低
        distance_factor = distance / config.SENSING_RANGE
        distance_throughput = base_throughput * (1 - 0.3 * (distance_factor ** 2))  # 最大30%的距离相关降低

        # 网络拥塞导致的吞吐量降低
        congestion_throughput = distance_throughput
        if hasattr(self.simulator.drones[dst_id], 'transmitting_queue'):
            queue_size = self.simulator.drones[dst_id].transmitting_queue.qsize()
            queue_factor = queue_size / config.MAX_QUEUE_SIZE
            congestion_throughput = distance_throughput * (1 - 0.5 * queue_factor)  # 最大50%的拥塞相关降低

        return max(base_throughput * 0.1, congestion_throughput)  # 至少保留10%的基础吞吐量

    def monitor_paths(self):
        """定期监控路径状态并更新统计信息"""
        while True:
            # 每1秒监控一次
            yield self.simulator.env.timeout(1 * 1e6)

            # 查看所有缓存的路径
            for dst_id in list(self.paths_cache.keys()):
                valid_paths = self.get_paths(dst_id)

                if valid_paths:
                    for path_id in valid_paths:
                        # 刷新路径统计
                        self._refresh_path_stats(path_id, dst_id)

                        # 检查是否有路径质量显著下降
                        path_stats = self.load_balancer.path_stats[self.drone.identifier][dst_id].get(path_id, {})

                        if path_stats.get('quality', 0) < 0.2:
                            logging.warning(
                                f"UAV {self.drone.identifier}: 到 {dst_id} 的路径 {path_id} 质量较低，尝试寻找替代路径")

                            # 尝试发现新路径来替代
                            self.discover_paths(dst_id)

    def _refresh_path_stats(self, path_id, dst_id):
        """刷新路径统计信息"""
        path = self._get_path(path_id)
        if not path or len(path) < 2:
            return

        # 计算路径质量指标
        total_delay = 0
        total_loss = 0
        min_throughput = float('inf')
        total_stability = 0

        # 计算路径上每个链路的质量并累计
        for i in range(len(path) - 1):
            src_node = path[i]
            dst_node = path[i + 1]

            # 获取链路质量指标
            link_delay = self._get_link_delay(src_node, dst_node)
            link_loss = self._get_link_loss_rate(src_node, dst_node)
            link_throughput = self._get_link_throughput(src_node, dst_node)
            link_stability = self._calculate_link_stability(src_node, dst_node)

            # 累加延迟
            total_delay += link_delay

            # 累乘丢包率 (计算端到端成功率)
            if i == 0:
                total_loss = link_loss
            else:
                total_loss = 1 - (1 - total_loss) * (1 - link_loss)

            # 取最小吞吐量
            min_throughput = min(min_throughput, link_throughput)

            # 累加稳定性 (取平均值)
            total_stability += link_stability

        # 计算平均稳定性
        avg_stability = total_stability / (len(path) - 1) if len(path) > 1 else 0

        # 更新负载均衡器的路径统计
        self.load_balancer.update_path_stats(
            self.drone.identifier,
            dst_id,
            path_id,
            {
                'delay': total_delay,
                'loss_rate': total_loss,
                'throughput': min_throughput if min_throughput != float('inf') else 0,
                'stability': avg_stability
            }
        )

        # 延长路径过期时间
        self.path_expiry[path_id] = self.simulator.env.now + 30 * 1e6