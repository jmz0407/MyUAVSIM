import copy
import random
import logging
from entities.packet import DataPacket, AckPacket
from topology.virtual_force.vf_packet import VfPacket
from utils import config
from phy.large_scale_fading import maximum_communication_range
from routing.dsdv.dsdv_packet import DsdvHelloPacket


class DsdvRoutingTable:
    """DSDV Routing Table implementation"""

    def __init__(self, env, my_drone):
        self.env = env
        self.my_drone = my_drone
        self.routing_table = {}

        # Format: [next_hop, metric, sequence_number, update_time]
        self.routing_table[my_drone.identifier] = [
            my_drone.identifier, 0, 0, self.env.now
        ]
        logging.info(f'UAV {my_drone.identifier} initialized routing table')

    def update_item(self, hello_packet, current_time):
        """Update routing table based on received hello packet"""
        src_id = hello_packet.src_drone.identifier
        received_table = hello_packet.routing_table
        updated = False

        # 忽略来自自己的路由更新
        if src_id == self.my_drone.identifier:
            return False

        logging.info(f'UAV {self.my_drone.identifier} received routing update from {src_id}')
        logging.info(f'Current routing table: {self.routing_table}')
        logging.info(f'Received routing info: {received_table}')

        # First update the direct route to sender
        if src_id not in self.routing_table:
            self.routing_table[src_id] = [src_id, 1, 0, current_time]
            updated = True
            logging.info(f'UAV {self.my_drone.identifier} added direct route to {src_id}')

        # Then process all routes in received table
        for dest_id, route_info in received_table.items():
            # 跳过关于自己的路由信息
            if dest_id == self.my_drone.identifier:
                continue

            new_metric = route_info[1] + 1
            if new_metric >= config.NUMBER_OF_DRONES:  # Prevent routing loops
                logging.debug(f'Route to {dest_id} via {src_id} exceeds max metric')
                continue

            if dest_id not in self.routing_table:
                self.routing_table[dest_id] = [
                    src_id, new_metric, route_info[2], current_time
                ]
                updated = True
                logging.info(f'UAV {self.my_drone.identifier} added route to {dest_id} via {src_id}')
            else:
                current_route = self.routing_table[dest_id]
                if (route_info[2] > current_route[2] or
                        (route_info[2] == current_route[2] and new_metric < current_route[1])):
                    current_route[0] = src_id
                    current_route[1] = new_metric
                    current_route[2] = route_info[2]
                    current_route[3] = current_time
                    updated = True
                    logging.info(f'UAV {self.my_drone.identifier} updated route to {dest_id} via {src_id}')

        if updated:
            logging.info(f'Updated routing table for UAV {self.my_drone.identifier}: {self.routing_table}')
        return updated

    def purge(self):
        """Remove stale routes"""
        current_time = self.env.now
        stale_timeout = 3 * 1e6  # Increased to 3s for more stability
        purged = False

        for dest_id in list(self.routing_table.keys()):
            if (dest_id != self.my_drone.identifier and
                    current_time - self.routing_table[dest_id][3] > stale_timeout):
                route_info = self.routing_table[dest_id]
                del self.routing_table[dest_id]
                purged = True
                logging.info(f'UAV {self.my_drone.identifier} removed stale route to {dest_id}, '
                             f'age: {(current_time - route_info[3]) / 1e6:.2f}s')

        return 1 if purged else 0

    def has_entry(self, dest_id):
        """Check if route exists and return next hop"""
        if dest_id in self.routing_table:
            next_hop = self.routing_table[dest_id][0]
            metric = self.routing_table[dest_id][1]
            logging.info(f'UAV {self.my_drone.identifier} found route to {dest_id} '
                         f'via {next_hop} with metric {metric}')
            return next_hop
        logging.info(f'UAV {self.my_drone.identifier} has no route to {dest_id}')
        return self.my_drone.identifier


class Dsdv:
    def __init__(self, simulator, my_drone):
        self.simulator = simulator
        self.my_drone = my_drone
        self.hello_interval = 0.25 * 1e6  # 0.25s interval
        self.routing_table = DsdvRoutingTable(self.simulator.env, my_drone)
        self.max_comm_range = maximum_communication_range()

        # Start periodic processes
        self.simulator.env.process(self.initial_update())
        self.simulator.env.process(self.broadcast_hello_packet_periodically())
        self.simulator.env.process(self.detect_broken_link_periodically())

    def broadcast_hello_packet(self):
        """Broadcast hello packet directly through PHY layer"""
        if not self.my_drone.sleep:
            config.GL_ID_HELLO_PACKET += 1

            # Update sequence number
            self.routing_table.routing_table[self.my_drone.identifier][2] += 2

            hello_packet = DsdvHelloPacket(
                src_drone=self.my_drone,
                creation_time=self.simulator.env.now,
                id_hello_packet=config.GL_ID_HELLO_PACKET,
                hello_packet_length=config.HELLO_PACKET_LENGTH,
                routing_table=self.routing_table.routing_table,
                simulator=self.simulator
            )

            # Directly broadcast through PHY layer
            if hasattr(self.my_drone.mac_protocol, 'phy'):
                self.my_drone.mac_protocol.phy.broadcast(hello_packet)
                self.simulator.metrics.control_packet_num += 1
                logging.info(
                    f'UAV {self.my_drone.identifier} directly broadcasting hello packet {hello_packet.packet_id}')
            else:
                logging.error(f'UAV {self.my_drone.identifier} has no PHY layer')

    def broadcast_hello_packet_periodically(self):
        """Periodically broadcast hello packets"""
        while True:
            if not self.my_drone.sleep:
                self.broadcast_hello_packet()
                jitter = random.randint(100, 1000)
                yield self.simulator.env.timeout(self.hello_interval + jitter)

    def check_waiting_list(self):
        """Periodically check packets in waiting list"""
        while True:
            if not self.my_drone.sleep:
                yield self.simulator.env.timeout(0.1 * 1e6)  # 每0.1秒检查一次

                # 获取当前等待列表的副本以避免迭代时修改
                current_waiting_list = self.my_drone.waiting_list.copy()

                for packet in current_waiting_list:
                    if self.simulator.env.now >= packet.creation_time + packet.deadline:
                        # 如果包已过期，从等待列表中移除
                        self.my_drone.waiting_list.remove(packet)
                        logging.info(f'Packet {packet.packet_id} removed from waiting list (expired)')
                        continue

                    if isinstance(packet, DataPacket):
                        dst_id = packet.dst_drone.identifier
                        next_hop_id = self.routing_table.has_entry(dst_id)

                        if next_hop_id != self.my_drone.identifier:
                            # 找到路由，将包放回传输队列
                            if self.my_drone.transmitting_queue.qsize() < self.my_drone.max_queue_size:
                                packet.next_hop_id = next_hop_id
                                self.my_drone.transmitting_queue.put(packet)
                                self.my_drone.waiting_list.remove(packet)
                                logging.info(
                                    f'Packet {packet.packet_id} moved from waiting list to transmission queue, '
                                    f'next hop: {next_hop_id}')
                        else:
                            # 仍然没有路由，触发一次更新
                            if random.random() < 0.2:  # 20%的概率触发更新，避免更新过于频繁
                                self.broadcast_hello_packet()

                if len(current_waiting_list) > 0:
                    logging.info(f'UAV {self.my_drone.identifier} waiting list size: {len(current_waiting_list)}')
            else:
                break
    def initial_update(self):
        """Send immediate initial update"""
        yield self.simulator.env.timeout(100)  # Small delay for initialization
        self.broadcast_hello_packet()
        logging.info(f'UAV {self.my_drone.identifier} sent initial routing update')

    def detect_broken_link_periodically(self):
        """Check for broken links periodically"""
        while True:
            yield self.simulator.env.timeout(1 * 1e6)  # Increased to 1s for stability
            if self.routing_table.purge():
                self.broadcast_hello_packet()
                logging.info(f'UAV {self.my_drone.identifier} detected broken links, broadcasting update')

    def next_hop_selection(self, packet):
        """Select next hop for packet routing"""
        has_route = True
        enquire = False

        if isinstance(packet, DataPacket):
            dst_id = packet.dst_drone.identifier
            next_hop_id = self.routing_table.has_entry(dst_id)

            if next_hop_id == self.my_drone.identifier:
                has_route = False
                logging.info(f'UAV {self.my_drone.identifier} no route to {dst_id}')
            else:
                packet.next_hop_id = next_hop_id
                logging.info(f'UAV {self.my_drone.identifier} found route to {dst_id} via {next_hop_id}')

        return has_route, packet, enquire

    def packet_reception(self, packet, src_drone_id):
        """Handle received packets"""
        current_time = self.simulator.env.now
        if isinstance(packet, DsdvHelloPacket):
            # 忽略自己发出的包
            if src_drone_id == self.my_drone.identifier:
                logging.debug(f'UAV {self.my_drone.identifier} ignoring self-originated hello packet')
                return

            logging.info(f'UAV {self.my_drone.identifier} received hello packet from {src_drone_id}')
            if self.routing_table.update_item(packet, current_time):
                self.broadcast_hello_packet()
        elif isinstance(packet, DataPacket):
            packet_copy = copy.copy(packet)
            logging.info('~~~ DataPacket: %s is received by UAV: %s at time: %s',
                         packet_copy.packet_id, self.my_drone.identifier, current_time)

            # 如果数据包的目标是当前 UAV，则进行处理
            if packet_copy.dst_drone.identifier == self.my_drone.identifier:
                latency = current_time - packet_copy.creation_time  # in us
                self.simulator.metrics.deliver_time_dict[packet_copy.packet_id] = latency
                self.simulator.metrics.throughput_dict[packet_copy.packet_id] = config.DATA_PACKET_LENGTH / (
                            latency / 1e6)
                self.simulator.metrics.hop_cnt_dict[packet_copy.packet_id] = packet_copy.get_current_ttl()
                self.simulator.metrics.datapacket_arrived.add(packet_copy.packet_id)

                logging.info('Latency: %s us, Throughput: %s', latency,
                             self.simulator.metrics.throughput_dict[packet_copy.packet_id])
                logging.info('DataPacket: %s delivered to UAV: %s', packet_copy.packet_id, self.my_drone.identifier)

                # 发送 ACK 如果需要
                if config.ACK_TIMEOUT > 0:
                    config.GL_ID_ACK_PACKET += 1
                    src_drone = self.simulator.drones[src_drone_id]
                    ack_packet = AckPacket(
                        src_drone=self.my_drone,
                        dst_drone=src_drone,
                        ack_packet_id=config.GL_ID_ACK_PACKET,
                        ack_packet_length=config.ACK_PACKET_LENGTH,
                        ack_packet=packet_copy,
                        simulator=self.simulator,
                        creation_time=current_time
                    )
                    # 添加 ACK 到发送队列
                    if self.my_drone.transmitting_queue.qsize() < self.my_drone.max_queue_size:
                        self.my_drone.transmitting_queue.put(ack_packet)
                        logging.info('ACK packet queued for packet: %s', packet_copy.packet_id)
            else:
                # 如果目标不是当前 UAV，将包放入队列中等待
                if self.my_drone.transmitting_queue.qsize() < self.my_drone.max_queue_size:
                    self.my_drone.transmitting_queue.put(packet_copy)
                    logging.info('DataPacket: %s queued for forwarding at UAV: %s',
                                 packet_copy.packet_id, self.my_drone.identifier)
                else:
                    logging.warning('Queue full at UAV: %s, DataPacket: %s dropped',
                                    self.my_drone.identifier, packet_copy.packet_id)

        elif isinstance(packet, VfPacket):
            logging.info('At time %s, UAV: %s receives a VF packet from UAV: %s, packet id: %s',
                         current_time, self.my_drone.identifier, src_drone_id, packet.packet_id)

            # 更新邻居表
            self.my_drone.motion_controller.neighbor_table.add_neighbor(packet, current_time)

            # 如果是 hello 消息，发送一个 ack
            if packet.msg_type == 'hello':
                config.GL_ID_VF_PACKET += 1
                ack_packet = VfPacket(
                    src_drone=self.my_drone,
                    creation_time=current_time,
                    id_hello_packet=config.GL_ID_VF_PACKET,
                    hello_packet_length=config.HELLO_PACKET_LENGTH,
                    simulator=self.simulator
                )
                ack_packet.msg_type = 'ack'
                self.my_drone.transmitting_queue.put(ack_packet)
                logging.info('VF ACK packet queued for packet: %s', packet.packet_id)

    def handle_packet_delivery(self, packet, sender_id):
        """Handle packet that reached its destination"""
        latency = self.simulator.env.now - packet.creation_time
        self.simulator.metrics.deliver_time_dict[packet.packet_id] = latency
        self.simulator.metrics.throughput_dict[packet.packet_id] = (
                config.DATA_PACKET_LENGTH / (latency / 1e6)
        )
        self.simulator.metrics.hop_cnt_dict[packet.packet_id] = packet.get_current_ttl()
        self.simulator.metrics.datapacket_arrived.add(packet.packet_id)

        # Send ACK
        if config.ACK_TIMEOUT > 0:
            self.send_ack_packet(packet, sender_id)