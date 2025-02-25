import math
import logging
from utils import config
from utils.util_function import euclidean_distance


# config logging
logging.basicConfig(filename='running_log.log',
                    filemode='w',  # there are two modes: 'a' and 'w'
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG
                    )


# def sinr_calculator(my_drone, main_drones_list, all_transmitting_drones_list):
#     """
#     calculate signal to signal-to-interference-plus-noise ratio
#     :param my_drone: receiver drone
#     :param main_drones_list: list of drones that wants to transmit packet to receiver
#     :param all_transmitting_drones_list: list of all drones currently transmitting packet
#     :return: list of sinr of each main drone
#     """
#
#     simulator = my_drone.simulator
#     transmit_power = config.TRANSMITTING_POWER
#     noise_power = config.NOISE_POWER
#
#     sinr_list = []  # record the sinr of all transmitter
#     receiver = my_drone
#
#     logging.info('Main node list: %s', main_drones_list)
#     for transmitter_id in main_drones_list:
#         transmitter = simulator.drones[transmitter_id]
#         interference_list = all_transmitting_drones_list[:]
#         interference_list.remove(transmitter_id)
#
#         main_link_path_loss = general_path_loss(receiver, transmitter)
#         receive_power = transmit_power * main_link_path_loss
#         interference_power = 0
#
#         if len(interference_list) != 0:
#             logging.info('Has interference')
#             # my_drone.simulator.metrics.collision_num += 1
#             for interference_id in interference_list:
#                 interference = simulator.drones[interference_id]
#
#                 logging.info('Main node is: %s, interference node is: %s, distance between them is: %s, main link'
#                              ' distance is: %s, interference link distance is: %s',
#                              transmitter_id, interference_id, euclidean_distance(transmitter.coords, interference.coords),
#                              euclidean_distance(transmitter.coords, receiver.coords),
#                              euclidean_distance(interference.coords, receiver.coords))
#
#                 interference_link_path_loss = general_path_loss(receiver, interference)
#                 interference_power += transmit_power * interference_link_path_loss
#         else:
#             logging.info('No interference, main link distance is: %s',
#                          euclidean_distance(transmitter.coords, receiver.coords))
#
#         sinr = 10 * math.log10(receive_power / (noise_power + interference_power))
#         logging.info('The SINR of main link is: %s', sinr)
#         sinr_list.append(sinr)
#
#     return sinr_list

def sinr_calculator(my_drone, main_drones_list, all_transmitting_drones_list):
    """
    Calculate SINR for TDMA protocol
    In TDMA, there should be no interference as each node transmits in its own slot
    """
    simulator = my_drone.simulator
    transmit_power = config.TRANSMITTING_POWER
    noise_power = config.NOISE_POWER
    sinr_list = []
    receiver = my_drone

    logging.info('Main node list: %s', main_drones_list)

    # 检查是否存在多个同时传输
    if len(all_transmitting_drones_list) > 1:
        pass
        # logging.warning(f"Multiple transmissions detected in TDMA at time {simulator.env.now}")
        # logging.warning(f"Transmitting nodes: {all_transmitting_drones_list}")

    for transmitter_id in main_drones_list:
        transmitter = simulator.drones[transmitter_id]

        # 计算路径损耗
        main_link_path_loss = general_path_loss(receiver, transmitter)
        receive_power = transmit_power * main_link_path_loss

        # TDMA中只考虑噪声，不应该有干扰
        sinr = 10 * math.log10(receive_power / noise_power)
        # logging.info('The SNR of main link is: %s', sinr)
        sinr_list.append(sinr)

        # 记录通信距离
        distance = euclidean_distance(transmitter.coords, receiver.coords)
        # logging.info('Communication distance: %s m', distance)

    return sinr_list


def stdma_sinr_calculator(my_drone, main_drones_list, all_transmitting_drones_list):
    simulator = my_drone.simulator
    transmit_power = config.TRANSMITTING_POWER
    noise_power = config.NOISE_POWER
    sinr_list = []
    receiver = my_drone

    logging.info('主传输节点: %s', main_drones_list)

    for transmitter_id in main_drones_list:
        transmitter = simulator.drones[transmitter_id]
        main_link_path_loss = general_path_loss(receiver, transmitter)
        receive_power = transmit_power * main_link_path_loss

        interference_power = 0
        # 只考虑同时传输节点的干扰
        interference_list = [node_id for node_id in all_transmitting_drones_list
                             if node_id != transmitter_id and node_id != receiver.identifier]

        for interferer_id in interference_list:
            interferer = simulator.drones[interferer_id]
            interference_path_loss = general_path_loss(receiver, interferer)
            interference_power += transmit_power * interference_path_loss

            distance = euclidean_distance(interferer.coords, receiver.coords)
            # logging.info('来自节点 %s 对接收器 %s 的干扰: 距离 = %.2f m, 功率 = %.6f',
            #              interferer_id, receiver.identifier, distance,
            #              transmit_power * interference_path_loss)

        sinr = 10 * math.log10(receive_power / (noise_power + interference_power))
        logging.info('节点 %s 到接收器 %s 的SINR: %.2f dB',
                      transmitter_id, receiver.identifier, sinr)

        if sinr != config.SNR_THRESHOLD:
            logging.info('节点 %s 的数据包成功被节点 %s 接收',
                         transmitter_id, receiver.identifier)
        else:
            logging.warning('节点 %s 的数据包未能被节点 %s 接收 (SINR: %.2f dB < 阈值: %.2f dB)',
                            transmitter_id, receiver.identifier, sinr, config.SNR_THRESHOLD)

        sinr_list.append(sinr)

    return sinr_list

def general_path_loss(receiver, transmitter):
    """
    general path loss model of line-of-sight (LoS) channels without system loss

    References:
        [1] J. Sabzehali, et al., "Optimizing number, placement, and backhaul connectivity of multi-UAV networks," in
            IEEE Internet of Things Journal, vol. 9, no. 21, pp. 21548-21560, 2022.

    :param receiver: the drone that receives the packet
    :param transmitter: the drone that sends the packet
    :return: path loss
    """

    c = config.LIGHT_SPEED
    fc = config.CARRIER_FREQUENCY
    alpha = 2  # path loss exponent

    distance = euclidean_distance(receiver.coords, transmitter.coords)

    if distance != 0:
        path_loss = (c / (4 * math.pi * fc * distance)) ** alpha
    else:
        path_loss = 1

    return path_loss


def maximum_communication_range():
    c = config.LIGHT_SPEED
    fc = config.CARRIER_FREQUENCY
    alpha = config.PATH_LOSS_EXPONENT  # path loss exponent
    transmit_power_db = 10 * math.log10(config.TRANSMITTING_POWER)
    noise_power_db = 10 * math.log10(config.NOISE_POWER)
    snr_threshold_db = config.SNR_THRESHOLD

    path_loss_db = transmit_power_db - noise_power_db - snr_threshold_db

    max_comm_range = (c * (10 ** (path_loss_db / (alpha * 10)))) / (4 * math.pi * fc)

    return max_comm_range
