import logging
import simpy
import math
from phy.phy import Phy
from utils import config
from utils.util_function import euclidean_distance
from topology.virtual_force.vf_packet import VfPacket
from entities.packet import DataPacket

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
        self.slot_schedule = None

        # 数据流管理
        self.flow_queue = {}  # 存储数据流
        self.flow_stats = {}  # 流量统计

        # 启动进程
        self.env.process(self._slot_synchronization())
        self.env.process(self._delayed_schedule_creation())
        self.env.process(self._monitor_flows())

    def _slot_synchronization(self):
        while True:
            self.current_slot = (self.env.now // self.time_slot_duration) % self.num_slots
            yield self.env.timeout(self.time_slot_duration)

    def _delayed_schedule_creation(self):
        yield self.env.timeout(1)
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
            self.slot_schedule = self._create_slot_schedule()

    def _create_slot_schedule(self):
        schedule = {}
        from phy.large_scale_fading import maximum_communication_range
        interference_range = maximum_communication_range() * 1.2

        unassigned = set(range(config.NUMBER_OF_DRONES))
        slot = 0

        while unassigned:
            if slot >= self.num_slots:
                self.num_slots += 1

            schedule[slot] = []
            remaining = list(unassigned)

            for drone_id in remaining:
                if all(euclidean_distance(
                        self.simulator.drones[drone_id].coords,
                        self.simulator.drones[assigned_id].coords
                ) >= interference_range
                       for assigned_id in schedule[slot]):
                    schedule[slot].append(drone_id)
                    unassigned.remove(drone_id)
            slot += 1

        logging.info(f"STDMA调度表: {schedule}")
        return schedule

    def mac_send(self, packet):
        if not self.slot_schedule:
            logging.error("时隙表未创建")
            return

        # 控制包直接发送
        if isinstance(packet, VfPacket):
            yield self.env.process(self._transmit_packet(packet))
            return

        # 数据包流管理
        if isinstance(packet, DataPacket):
            flow_id = f"flow_{packet.src_drone.identifier}_{packet.dst_drone.identifier}"
            self.flow_queue.setdefault(flow_id, []).append(packet)
            self.flow_stats.setdefault(flow_id, {
                'sent_packets': 0,
                'avg_delay': 0,
                'throughput': 0,
                'queue_size': 0
            })

            mac_start_time = self.env.now
            assigned_slot = next((slot for slot, drones in self.slot_schedule.items()
                                  if self.my_drone.identifier in drones), None)

            if assigned_slot is None:
                logging.error(f"无人机{self.my_drone.identifier}未分配时隙")
                return

            current_time = self.env.now
            slot_start = (
                                     current_time // self.time_slot_duration) * self.time_slot_duration + assigned_slot * self.time_slot_duration
            slot_end = slot_start + self.time_slot_duration

            if current_time < slot_start:
                yield self.env.timeout(slot_start - current_time)
            elif current_time >= slot_end:
                yield self.env.timeout(self.num_slots * self.time_slot_duration - (current_time - slot_start))

            yield self.env.process(self._transmit_packet(packet))
            self.simulator.metrics.mac_delay.append(self.env.now - mac_start_time)

    def _transmit_packet(self, packet):
        self.current_transmission = packet

        if packet.transmission_mode == 0:
            packet.increase_ttl()
            self.phy.unicast(packet, packet.next_hop_id)
            yield self.env.timeout(packet.packet_length / config.BIT_RATE * 1e6)

        elif packet.transmission_mode == 1:
            packet.increase_ttl()
            self.phy.broadcast(packet)
            yield self.env.timeout(packet.packet_length / config.BIT_RATE * 1e6)

        if isinstance(packet, DataPacket):
            flow_id = f"flow_{packet.src_drone.identifier}_{packet.dst_drone.identifier}"
            self.flow_stats[flow_id]['sent_packets'] += 1
            self.flow_queue[flow_id].remove(packet)

        self.current_transmission = None