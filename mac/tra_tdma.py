import random
import simpy
import logging
from utils import config
import copy

# config logging
logging.basicConfig(filename='running_log.log',
                    filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=config.LOGGING_LEVEL)
from collections import defaultdict
import numpy as np

class Tdma:
    """
    Time Division Multiple Access (TDMA) MAC protocol.

    In TDMA, each drone is assigned a time slot during which it can send packets.
    There is no contention as each drone knows its allocated time.
    """

    def __init__(self, drone, num_slots=10):
        self.my_drone = drone
        self.simulator = drone.simulator
        self.env = drone.env
        self.num_slots = num_slots  # Number of time slots
        self.slot_duration = config.SLOT_DURATION  # Duration of each time slot
        self.slot_schedule = self.create_slot_schedule()  # Slot assignment for each drone
        self.current_slot = 0  # Current time slot being processed
        self.current_transmission = None  # Tracks if there's an ongoing transmission

    def create_slot_schedule(self):
        """
        Create a slot schedule for each drone.
        Each drone is assigned a unique slot in a round-robin fashion.
        """
        schedule = {}
        drone_count = len(self.my_drone.simulator.drones)
        for i in range(drone_count):
            assigned_slot = i % self.num_slots  # Assign time slots cyclically
            schedule[i] = assigned_slot
            logging.info(f"UAV {i} is assigned to time slot {assigned_slot}")
        return schedule

    def mac_send(self, pkd):
        """
        Control when drone can send packet using TDMA protocol.
        :param pkd: the packet that needs to send.
        """
        self.current_transmission = pkd
        transmission_attempt = pkd.number_retransmission_attempt[self.my_drone.identifier]

        # Determine the time slot based on the drone's identifier
        assigned_slot = self.slot_schedule[self.my_drone.identifier]
        current_time = self.env.now

        # Compute the start and end time of the current slot
        slot_start_time = (current_time // self.slot_duration) * self.slot_duration + assigned_slot * self.slot_duration
        slot_end_time = slot_start_time + self.slot_duration

        logging.info("UAV: %s is assigned to time slot %s from %s to %s.",
                     self.my_drone.identifier, assigned_slot, slot_start_time, slot_end_time)

        if current_time >= slot_start_time and current_time < slot_end_time:
            # It's this drone's turn to send
            logging.info('UAV: %s can send packet (pkd id: %s) at time: %s',
                         self.my_drone.identifier, pkd.packet_id, self.env.now)

            # Here we simulate the transmission by invoking the PHY layer (just like in CSMA/CA)
            yield self.env.process(self.send_packet(pkd))
        else:
            # It's not this drone's turn, wait for the next slot
            logging.info('UAV: %s has to wait for its turn at time: %s', self.my_drone.identifier, self.env.now)
            yield self.env.timeout(slot_end_time - current_time)  # Wait until the next slot

    def send_packet(self, pkd):
        """
        Simulate the process of sending a packet.
        """
        transmission_mode = pkd.transmission_mode
        logging.info('UAV: %s starting to send data packet: %s at time: %s',
                     self.my_drone.identifier, pkd.packet_id, self.env.now)

        if transmission_mode == 0:  # Unicast
            next_hop_id = pkd.next_hop_id
            pkd.increase_ttl()
            logging.info("Sending unicast packet from UAV: %s to UAV: %s", self.my_drone.identifier, next_hop_id)
            self.my_drone.phy.unicast(pkd, next_hop_id)
            yield self.env.timeout(pkd.packet_length / config.BIT_RATE * 1e6)  # Transmission delay
        elif transmission_mode == 1:  # Broadcast
            pkd.increase_ttl()
            logging.info("Broadcasting packet from UAV: %s", self.my_drone.identifier)
            self.my_drone.phy.broadcast(pkd)
            yield self.env.timeout(pkd.packet_length / config.BIT_RATE * 1e6)  # Transmission delay


class Metrics:
    """
    Tools for statistics of network performance.

    1. Packet Delivery Ratio (PDR): is the ratio of number of packets received at the destinations to the number
       of packets sent from the sources.
    2. Average end-to-end (E2E) delay: is the time a packet takes to route from a source to its destination.
    3. Routing Load: is calculated as the ratio between the number of control packets transmitted
       to the number of packets actually received.
    4. Throughput: it can be defined as a measure of how fast the data is sent from its source to its destination.
    5. Hop count: used to record the number of router output ports through which the packet should pass.
    """

    def __init__(self, simulator):
        self.simulator = simulator
        self.control_packet_num = 0
        self.datapacket_generated = set()  # All data packets generated
        self.datapacket_arrived = set()  # All data packets that arrive at the destination
        self.datapacket_generated_num = 0

        self.delivery_time = []
        self.deliver_time_dict = defaultdict()

        self.throughput = []
        self.throughput_dict = defaultdict()

        self.hop_cnt = []
        self.hop_cnt_dict = defaultdict()

        self.mac_delay = []

        self.collision_num = 0

    def print_metrics(self):
        # Calculate the average end-to-end delay
        for key in self.deliver_time_dict.keys():
            self.delivery_time.append(self.deliver_time_dict[key])

        for key2 in self.throughput_dict.keys():
            self.throughput.append(self.throughput_dict[key2])

        for key3 in self.hop_cnt_dict.keys():
            self.hop_cnt.append(self.hop_cnt_dict[key3])

        # Calculate metrics
        e2e_delay = np.mean(self.delivery_time) / 1e3  # unit: ms
        pdr = len(self.datapacket_arrived) / self.datapacket_generated_num * 100  # in %

        # Avoid division by zero if no data packets arrived
        rl = self.control_packet_num / len(self.datapacket_arrived) if len(self.datapacket_arrived) > 0 else 0

        throughput = np.mean(self.throughput) / 1e3  # in Kbps
        hop_cnt = np.mean(self.hop_cnt)
        average_mac_delay = np.mean(self.mac_delay) if self.mac_delay else 0  # Avoid nan if empty

        # Print metrics
        print('Totally sent: ', self.datapacket_generated_num, ' data packets')
        print('Packet delivery ratio is: ', pdr, '%')
        print('Average end-to-end delay is: ', e2e_delay, 'ms')
        print('Routing load is: ', rl)
        print('Average throughput is: ', throughput, 'Kbps')
        print('Average hop count is: ', hop_cnt)
        print('Collision num is: ', self.collision_num)
        print('Average mac delay is: ', average_mac_delay, 'ms')
