import logging
import random
from utils import config
from utils.util_function import euclidean_distance
from entities.packet import DataPacket
from routing.mp_dsr.dsr import GlobalDSR


class GlobalMPDSR(GlobalDSR):
    """
    Multi-Path Dynamic Source Routing (MP-DSR) protocol implementation using global routing information

    This implementation extends GlobalDSR to support multiple paths for each destination.
    It uses the simulator's global routing information to find multiple paths and
    provides various strategies for path selection.

    Features:
    - Maintains multiple paths to each destination
    - Supports different path selection strategies
    - Uses global routing information instead of control packets
    - Distributes traffic across multiple paths
    """

    def __init__(self, simulator, my_drone):
        super().__init__(simulator, my_drone)

        # Path selection strategy
        self.path_selection_strategy = config.PATH_SELECTION_STRATEGY if hasattr(config,
                                                                                 'PATH_SELECTION_STRATEGY') else 'round_robin'
        self.path_stats = {}  # {dst_id: {path_key: {metrics}}}

        logging.info(
            f"UAV {self.my_drone.identifier} initialized GlobalMPDSR with strategy '{self.path_selection_strategy}'")

    def next_hop_selection(self, packet):
        """
        Select the next hop using multiple paths when available

        Args:
            packet: The packet to be forwarded

        Returns:
            has_route (bool): Whether a route is available
            packet (object): Either the original packet ready to send or a control packet
            enquire (bool): Whether a route discovery was initiated
        """
        has_route = False
        enquire = False

        # For non-data packets, forward as is
        if not isinstance(packet, DataPacket):
            return True, packet, False

        # For data packets, find a route to the destination
        dst_id = packet.dst_drone.identifier

        # If I am the destination, no need to route
        if dst_id == self.my_drone.identifier:
            return False, packet, False

        # If this is the source node, try to find multiple paths
        if packet.src_drone.identifier == self.my_drone.identifier:
            # Get routes from cache or discover new ones
            routes = self._get_routes(dst_id)

            if routes:
                # Select a route based on the strategy
                selected_route = self._select_route(dst_id, routes, packet)

                logging.info(f"UAV {self.my_drone.identifier} selected route to {dst_id}: {selected_route}")
                packet.routing_path = selected_route
                packet.next_hop_id = selected_route[1]  # First hop after source
                has_route = True
            else:
                # No route available
                logging.warning(f"UAV {self.my_drone.identifier} could not find route to {dst_id}")
                has_route = False
        else:
            # I am an intermediate node, check the source route in the packet
            if hasattr(packet, 'routing_path') and packet.routing_path:
                path = packet.routing_path
                # Find my position in the path
                try:
                    my_index = path.index(self.my_drone.identifier)
                    if my_index < len(path) - 1:
                        # I'm not the last node in the path
                        packet.next_hop_id = path[my_index + 1]
                        has_route = True
                    else:
                        # I'm the last node but not the destination - something is wrong
                        logging.warning(f"UAV {self.my_drone.identifier} is last in path but not destination")
                        has_route = False
                except ValueError:
                    # I'm not in the path - something is wrong
                    logging.warning(f"UAV {self.my_drone.identifier} not in routing path: {path}")
                    has_route = False
            else:
                # Packet without routing path - should not happen in DSR
                logging.warning(f"UAV {self.my_drone.identifier} received packet without routing path")
                has_route = False

        return has_route, packet, enquire

    def _get_routes(self, dst_id):
        """
        Get all routes to the destination

        Args:
            dst_id: Destination node ID

        Returns:
            list: List of routes to the destination or empty list if no route is available
        """
        # Check if we already have routes in cache
        if dst_id in self.route_cache and self.route_cache[dst_id]:
            return self.route_cache[dst_id]
        else:
            # No routes in cache, discover new ones
            routes = self._find_all_routes(self.my_drone.identifier, dst_id)
            if routes:
                self.route_cache[dst_id] = routes
            return routes

    def _select_route(self, dst_id, routes, packet=None):
        """
        Select a route from multiple available routes based on the strategy

        Args:
            dst_id: Destination node ID
            routes: List of available routes
            packet: Optional data packet being routed

        Returns:
            list: Selected route
        """
        if not routes:
            return None

        if len(routes) == 1:
            return routes[0]

        # Initialize path index for this destination if needed
        if dst_id not in self.current_path_index:
            self.current_path_index[dst_id] = 0

        # Different path selection strategies
        if self.path_selection_strategy == 'round_robin':
            # Simple round-robin selection
            index = self.current_path_index[dst_id]
            self.current_path_index[dst_id] = (index + 1) % len(routes)
            return routes[index]

        elif self.path_selection_strategy == 'random':
            # Random selection
            return random.choice(routes)

        elif self.path_selection_strategy == 'weighted':
            # Weighted selection based on path metrics
            if dst_id not in self.path_stats:
                self.path_stats[dst_id] = {}

            # Calculate weights for each path
            weights = []
            for route in routes:
                route_key = str(route)  # Use string representation as key

                # If we don't have stats for this path, initialize them
                if route_key not in self.path_stats[dst_id]:
                    self.path_stats[dst_id][route_key] = {
                        'success_count': 1,
                        'failure_count': 0,
                        'avg_delay': 0,
                        'last_used': 0
                    }

                # Calculate weight based on success rate and path length
                stats = self.path_stats[dst_id][route_key]
                success_rate = stats['success_count'] / (stats['success_count'] + stats['failure_count'])
                path_length = len(route)

                # Weight = success_rate / (path_length * (1 + recent_usage_penalty))
                time_since_last_used = max(1, (self.simulator.env.now - stats['last_used']) / 1e6)
                recent_usage_penalty = 0 if time_since_last_used > 1 else (1 - time_since_last_used)

                weight = success_rate / (path_length * (1 + recent_usage_penalty))
                weights.append(weight)

            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]

                # Select path based on weights
                selected_index = random.choices(range(len(routes)), weights=weights)[0]
                return routes[selected_index]
            else:
                # Fallback to round-robin
                index = self.current_path_index[dst_id]
                self.current_path_index[dst_id] = (index + 1) % len(routes)
                return routes[index]

        elif self.path_selection_strategy == 'adaptive':
            # Adaptive selection based on packet priority or congestion
            if packet and hasattr(packet, 'priority') and packet.priority > 0:
                # For high priority packets, use the shortest path
                return routes[0]
            else:
                # For regular packets, distribute across paths
                index = self.current_path_index[dst_id]
                self.current_path_index[dst_id] = (index + 1) % len(routes)
                return routes[index]

        else:
            # Default to first (shortest) path
            return routes[0]

    def update_path_stats(self, dst_id, route, success, delay=None):
        """
        Update statistics for a path

        Args:
            dst_id: Destination node ID
            route: Route used
            success: Whether the transmission was successful
            delay: End-to-end delay if available
        """
        if dst_id not in self.path_stats:
            self.path_stats[dst_id] = {}

        route_key = str(route)

        if route_key not in self.path_stats[dst_id]:
            self.path_stats[dst_id][route_key] = {
                'success_count': 0,
                'failure_count': 0,
                'avg_delay': 0,
                'last_used': self.simulator.env.now
            }

        stats = self.path_stats[dst_id][route_key]

        # Update usage time
        stats['last_used'] = self.simulator.env.now

        if success:
            stats['success_count'] += 1
            if delay is not None:
                # Update running average of delay
                old_avg = stats['avg_delay']
                total_count = stats['success_count']
                stats['avg_delay'] = ((old_avg * (total_count - 1)) + delay) / total_count
        else:
            stats['failure_count'] += 1

    def packet_reception(self, packet, src_drone_id):
        """
        Process a received packet with multipath enhancements

        Args:
            packet: Received packet
            src_drone_id: ID of the sender drone
        """
        current_time = self.simulator.env.now

        if isinstance(packet, DataPacket):
            # If I'm the destination, update path statistics
            if packet.dst_drone.identifier == self.my_drone.identifier:
                if hasattr(packet, 'routing_path') and packet.routing_path:
                    route = packet.routing_path
                    src_id = packet.src_drone.identifier
                    delay = current_time - packet.creation_time

                    # Update path statistics with successful delivery
                    self.update_path_stats(src_id, route, True, delay)

                    # Record metrics
                    hop_count = len(route) - 1 if route else 0
                    self.simulator.metrics.deliver_time_dict[packet.packet_id] = delay
                    self.simulator.metrics.throughput_dict[packet.packet_id] = config.DATA_PACKET_LENGTH / (delay / 1e6)
                    self.simulator.metrics.hop_cnt_dict[packet.packet_id] = hop_count
                    self.simulator.metrics.datapacket_arrived.add(packet.packet_id)

                    logging.info(
                        f"UAV {self.my_drone.identifier} received data packet {packet.packet_id} from {src_id}, "
                        f"latency: {delay} μs, hops: {hop_count}, path: {route}")
            else:
                # I'm an intermediate node, forward the packet according to the routing path
                if hasattr(packet, 'routing_path') and packet.routing_path:
                    try:
                        my_index = packet.routing_path.index(self.my_drone.identifier)

                        if my_index < len(packet.routing_path) - 1:
                            # I'm not the last node, forward the packet
                            next_hop_id = packet.routing_path[my_index + 1]
                            packet.next_hop_id = next_hop_id

                            # Check if next hop is reachable
                            next_hop_drone = self.simulator.drones[next_hop_id]
                            dist = euclidean_distance(self.my_drone.coords, next_hop_drone.coords)

                            if dist <= self.max_comm_range and not next_hop_drone.sleep:
                                # Next hop is reachable, forward the packet
                                self.my_drone.transmitting_queue.put(packet)
                                logging.info(
                                    f"UAV {self.my_drone.identifier} forwarding data packet {packet.packet_id} to {next_hop_id}")
                            else:
                                # Next hop is unreachable, mark this path as failed
                                src_id = packet.src_drone.identifier
                                route = packet.routing_path
                                self.update_path_stats(src_id, route, False)
                                logging.warning(
                                    f"UAV {self.my_drone.identifier} could not reach next hop {next_hop_id}, "
                                    f"marking path {route} as failed")
                        else:
                            logging.warning(
                                f"UAV {self.my_drone.identifier} is last in path but not destination: {packet.routing_path}")
                    except ValueError:
                        logging.warning(f"UAV {self.my_drone.identifier} not in routing path: {packet.routing_path}")
                else:
                    logging.warning(f"UAV {self.my_drone.identifier} received data packet without routing path")