import logging
import copy
from utils import config
from utils.util_function import euclidean_distance
from entities.packet import DataPacket


class GlobalDSR:
    """
    Dynamic Source Routing (DSR) protocol implementation using global routing information

    This implementation does not use control packets for route discovery and maintenance.
    Instead, it directly queries the simulator's global neighbor table to find routes.

    This approach is useful for:
    1. Simplifying the routing protocol for simulation purposes
    2. Focusing on other aspects of the simulation (e.g., MAC protocol, mobility patterns)
    3. Providing a baseline for comparison with fully distributed routing protocols
    """

    def __init__(self, simulator, my_drone):
        self.simulator = simulator
        self.my_drone = my_drone

        # Route cache for storing discovered paths
        self.route_cache = {}  # {dst_id: [path1, path2, ...]}
        self.max_paths = config.MAX_PATHS if hasattr(config, 'MAX_PATHS') else 3
        self.max_comm_range = self._calculate_max_comm_range()

        # Path selection strategy and state
        self.current_path_index = {}  # {dst_id: index}

        # Start periodic route cache update
        self.simulator.env.process(self._update_route_cache())

    def _calculate_max_comm_range(self):
        """Calculate the maximum communication range based on config parameters"""
        from phy.large_scale_fading import maximum_communication_range
        return maximum_communication_range()

    def next_hop_selection(self, packet):
        """
        Select the next hop for a packet based on DSR protocol

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

        # If this is the source node, try to find a route
        if packet.src_drone.identifier == self.my_drone.identifier:
            # Get a route from cache or discover a new one
            route = self._get_route(dst_id)

            if route:
                # Route available
                logging.info(f"UAV {self.my_drone.identifier} found route to {dst_id}: {route}")
                packet.routing_path = route
                packet.next_hop_id = route[1]  # First hop after source
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

    def _get_route(self, dst_id):
        """
        Get a route to the destination

        Args:
            dst_id: Destination node ID

        Returns:
            list: A route to the destination or None if no route is available
        """
        # Check if we already have a route in cache
        if dst_id in self.route_cache and self.route_cache[dst_id]:
            # Use round-robin to select from multiple paths
            if dst_id not in self.current_path_index:
                self.current_path_index[dst_id] = 0

            index = self.current_path_index[dst_id]
            paths = self.route_cache[dst_id]

            # Update index for next time
            self.current_path_index[dst_id] = (index + 1) % len(paths)

            return paths[index]
        else:
            # No route in cache, discover a new one
            return self._discover_route(dst_id)

    def _discover_route(self, dst_id):
        """
        Discover a route to the destination using global routing information

        Args:
            dst_id: Destination node ID

        Returns:
            list: A route to the destination or None if no route is available
        """
        # Get all routes using Dijkstra's algorithm
        routes = self._find_all_routes(self.my_drone.identifier, dst_id)

        if not routes:
            return None

        # Update route cache
        self.route_cache[dst_id] = routes

        # Return the first (shortest) route
        return routes[0] if routes else None

    def _find_all_routes(self, src_id, dst_id):
        """
        Find all routes from source to destination using global routing information

        Args:
            src_id: Source node ID
            dst_id: Destination node ID

        Returns:
            list: List of routes (each route is a list of node IDs)
        """
        # Use simulator's global neighbor table
        if not hasattr(self.simulator, 'global_neighbor_table') or not self.simulator.global_neighbor_table:
            # If global table not available, use a cost matrix based on node distances
            cost_matrix = self._calculate_cost_matrix()
            routes = []

            # Find the shortest path first
            shortest_path = self._dijkstra(cost_matrix, src_id, dst_id)
            if shortest_path:
                routes.append(shortest_path)

                # Find additional paths by increasing the cost of links in the shortest path
                for i in range(1, self.max_paths):
                    # Create a new cost matrix with increased costs for links in existing paths
                    mod_cost_matrix = cost_matrix.copy()

                    # Increase the cost of links in existing paths
                    for path in routes:
                        for j in range(len(path) - 1):
                            node1 = path[j]
                            node2 = path[j + 1]
                            mod_cost_matrix[node1][node2] *= (1 + i * 0.5)  # Increase cost by 50%, 100%, etc.
                            mod_cost_matrix[node2][node1] *= (1 + i * 0.5)

                    # Find another path with the modified cost matrix
                    alt_path = self._dijkstra(mod_cost_matrix, src_id, dst_id)
                    if alt_path and alt_path not in routes and len(alt_path) <= len(shortest_path) * 1.5:
                        # Only add if it's not too long and not already in routes
                        routes.append(alt_path)

            return routes
        else:
            # Use simulator's global neighbor table
            # Calculate routes using breadth-first search
            return self._find_routes_from_neighbor_table(src_id, dst_id)

    def _find_routes_from_neighbor_table(self, src_id, dst_id):
        """
        Find routes using the simulator's global neighbor table

        Args:
            src_id: Source node ID
            dst_id: Destination node ID

        Returns:
            list: List of routes (each route is a list of node IDs)
        """
        routes = []

        # Breadth-first search for paths
        visited = set([src_id])
        queue = [([src_id], visited)]  # (path, visited)

        while queue and len(routes) < self.max_paths:
            path, visited = queue.pop(0)
            current = path[-1]

            # If we reached the destination, add this path to routes
            if current == dst_id:
                routes.append(path)
                continue

            # Check all neighbors
            neighbor_table = self.simulator.global_neighbor_table
            if current in neighbor_table:
                for neighbor in neighbor_table[current]:
                    if neighbor not in visited:
                        # Create a new path and visited set
                        new_path = path + [neighbor]
                        new_visited = visited.copy()
                        new_visited.add(neighbor)

                        # Add to queue if not too long
                        if len(new_path) <= config.NUMBER_OF_DRONES:
                            queue.append((new_path, new_visited))

                # Sort queue by path length to prioritize shorter paths
                queue.sort(key=lambda x: len(x[0]))

        return routes

    def _calculate_cost_matrix(self):
        """
        Calculate cost matrix for all drones based on distance

        Returns:
            list: Cost matrix where cost_matrix[i][j] is the cost from drone i to drone j
        """
        n_drones = config.NUMBER_OF_DRONES
        cost_matrix = [[float('inf') for _ in range(n_drones)] for _ in range(n_drones)]

        # Set diagonal to 0
        for i in range(n_drones):
            cost_matrix[i][i] = 0

        # Calculate cost based on distance
        for i in range(n_drones):
            drone1 = self.simulator.drones[i]
            if drone1.sleep:
                continue  # Skip sleeping drones

            for j in range(i + 1, n_drones):
                drone2 = self.simulator.drones[j]
                if drone2.sleep:
                    continue  # Skip sleeping drones

                dist = euclidean_distance(drone1.coords, drone2.coords)

                # Only connect drones within communication range
                if dist <= self.max_comm_range:
                    # Cost is proportional to distance
                    cost_matrix[i][j] = dist
                    cost_matrix[j][i] = dist

        return cost_matrix

    def _dijkstra(self, cost_matrix, src, dst, option=0):
        """
        Find the shortest path using Dijkstra's algorithm

        Args:
            cost_matrix: Cost matrix
            src: Source node ID
            dst: Destination node ID
            option: Option flag (not used in basic algorithm)

        Returns:
            list: Shortest path from source to destination
        """
        n_drones = len(cost_matrix)

        # Initialize distances and paths
        distances = [float('inf')] * n_drones
        distances[src] = 0
        previous = [-1] * n_drones
        visited = [False] * n_drones

        # Main Dijkstra loop
        for _ in range(n_drones):
            # Find the unvisited node with the smallest distance
            min_dist = float('inf')
            min_node = -1
            for node in range(n_drones):
                if not visited[node] and distances[node] < min_dist:
                    min_dist = distances[node]
                    min_node = node

            if min_node == -1:
                break  # No more nodes to process

            visited[min_node] = True

            # If we reached the destination, we're done
            if min_node == dst:
                break

            # Update distances to neighbors
            for neighbor in range(n_drones):
                if not visited[neighbor] and cost_matrix[min_node][neighbor] < float('inf'):
                    new_dist = distances[min_node] + cost_matrix[min_node][neighbor]
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        previous[neighbor] = min_node

        # Build the path
        if distances[dst] < float('inf'):
            path = []
            current = dst
            while current != -1:
                path.insert(0, current)
                current = previous[current]
            return path
        else:
            return []  # No path found

    def _update_route_cache(self):
        """Periodically update the route cache using global routing information"""
        while True:
            yield self.simulator.env.timeout(5 * 1e6)  # Update every 5 seconds

            # Update routes for all destinations
            for dst_id in range(config.NUMBER_OF_DRONES):
                if dst_id != self.my_drone.identifier:
                    routes = self._find_all_routes(self.my_drone.identifier, dst_id)
                    if routes:
                        self.route_cache[dst_id] = routes
                    elif dst_id in self.route_cache:
                        del self.route_cache[dst_id]

            logging.info(
                f"UAV {self.my_drone.identifier} updated route cache, destinations: {list(self.route_cache.keys())}")

    def packet_reception(self, packet, src_drone_id):
        """
        Process a received packet

        Args:
            packet: Received packet
            src_drone_id: ID of the sender drone
        """
        current_time = self.simulator.env.now

        if isinstance(packet, DataPacket):
            # For data packets, check if I'm the destination
            if packet.dst_drone.identifier == self.my_drone.identifier:
                # I'm the destination, compute metrics
                latency = current_time - packet.creation_time
                hop_count = len(packet.routing_path) - 1 if hasattr(packet,
                                                                    'routing_path') and packet.routing_path else 0

                # Update metrics
                self.simulator.metrics.deliver_time_dict[packet.packet_id] = latency
                self.simulator.metrics.throughput_dict[packet.packet_id] = config.DATA_PACKET_LENGTH / (latency / 1e6)
                self.simulator.metrics.hop_cnt_dict[packet.packet_id] = hop_count
                self.simulator.metrics.datapacket_arrived.add(packet.packet_id)

                logging.info(
                    f"UAV {self.my_drone.identifier} received data packet {packet.packet_id} from {packet.src_drone.identifier}, "
                    f"latency: {latency} μs, hops: {hop_count}")
            else:
                # I'm an intermediate node, check if I'm in the routing path
                if hasattr(packet, 'routing_path') and packet.routing_path:
                    try:
                        my_index = packet.routing_path.index(self.my_drone.identifier)

                        if my_index < len(packet.routing_path) - 1:
                            # I'm not the last node, forward the packet
                            next_hop_id = packet.routing_path[my_index + 1]
                            packet.next_hop_id = next_hop_id

                            # Forward the packet
                            self.my_drone.transmitting_queue.put(packet)
                            logging.info(
                                f"UAV {self.my_drone.identifier} forwarding data packet {packet.packet_id} to {next_hop_id}")
                        else:
                            logging.warning(
                                f"UAV {self.my_drone.identifier} is last in path but not destination: {packet.routing_path}")
                    except ValueError:
                        logging.warning(f"UAV {self.my_drone.identifier} not in routing path: {packet.routing_path}")
                else:
                    logging.warning(f"UAV {self.my_drone.identifier} received data packet without routing path")