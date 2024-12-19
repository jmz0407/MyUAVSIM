import logging
from utils import config

# config logging
logging.basicConfig(filename='running_log.log',
                    filemode='w',  # there are two modes: 'a' and 'w'
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=config.LOGGING_LEVEL
                    )


class Phy:
    """
    Physical layer implementation

    Attributes:
        mac: mac protocol that installed
        env: simulation environment created by simpy
        my_drone: the drone that installed the physical protocol

    Future work: take co-channel interference into account, calculate the SINR before receiving the packet

    Author: Zihao Zhou, eezihaozhou@gmail.com
    Created at: 2024/1/11
    Updated at: 2024/4/25
    """

    def __init__(self, mac):
        self.mac = mac
        self.env = mac.env
        self.my_drone = mac.my_drone

    def unicast(self, packet, next_hop_id):
        """
        Unicast packet through the wireless channel
        :param packet: the data packet or ACK packet that needs to be transmitted
        :param next_hop_id: the identifier of the next hop drone
        :return: none
        """

        # energy consumption
        # Log the packet sending event
        logging.info(
            f"UAV: {self.my_drone.identifier} sending unicast packet {packet.packet_id} to UAV {next_hop_id} at time {self.env.now}")
        energy_consumption = (packet.packet_length / config.BIT_RATE) * config.TRANSMITTING_POWER
        self.my_drone.residual_energy -= energy_consumption
        logging.info(
            f"UAV: {self.my_drone.identifier} energy consumption for unicast: {energy_consumption:.3f} J, remaining energy: {self.my_drone.residual_energy:.3f} J")

        # transmit through the channel
        message = [packet, self.env.now, self.my_drone.identifier, 0]

        self.my_drone.simulator.channel.unicast_put(message, next_hop_id)
        logging.info(f"UAV: {self.my_drone.identifier} successfully sent unicast packet {packet.packet_id} to UAV {next_hop_id}.")

    def broadcast(self, packet):
        """
        Broadcast packet through the wireless channel
        :param packet: tha packet (hello packet, etc.) that needs to be broadcast
        :return: none
        """
        # Log the broadcast event
        logging.info(f"UAV: {self.my_drone.identifier} broadcasting packet {packet.packet_id} at time {self.env.now}")
        # energy consumption
        energy_consumption = (packet.packet_length / config.BIT_RATE) * config.TRANSMITTING_POWER
        self.my_drone.residual_energy -= energy_consumption
        logging.info(
            f"UAV: {self.my_drone.identifier} energy consumption for broadcast: {energy_consumption:.3f} J, remaining energy: {self.my_drone.residual_energy:.3f} J")
        # transmit through the channel
        message = [packet, self.env.now, self.my_drone.identifier, 0]

        self.my_drone.simulator.channel.broadcast_put(message)
        logging.info(f"UAV: {self.my_drone.identifier} successfully broadcasted packet {packet.packet_id}.")

    def multicast(self, packet, dst_id_list):
        """
        Multicast packet through the wireless channel
        :param packet: tha packet that needs to be multicasted
        :param dst_id_list: list of ids for multicast destinations
        :return: none
        """
        # Log the multicast event
        logging.info(
            f"UAV: {self.my_drone.identifier} multicasting packet {packet.packet_id} to {dst_id_list} at time {self.env.now}")

        # a transmission delay should be considered
        yield self.env.timeout(packet.packet_length / config.BIT_RATE * 1e6)

        # energy consumption
        energy_consumption = (packet.packet_length / config.BIT_RATE) * config.TRANSMITTING_POWER
        self.my_drone.residual_energy -= energy_consumption
        logging.info(
            f"UAV: {self.my_drone.identifier} energy consumption for multicast: {energy_consumption:.3f} J, remaining energy: {self.my_drone.residual_energy:.3f} J")

        # transmit through the channel
        message = [packet, self.env.now, self.my_drone.identifier]

        self.my_drone.simulator.channel.multicast_put(message, dst_id_list)
        logging.info(f"UAV: {self.my_drone.identifier} successfully multicasted packet {packet.packet_id} to {dst_id_list}.")