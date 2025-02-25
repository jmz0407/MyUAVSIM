import random
import logging
from entities.packet import DataPacket
from phy.phy import Phy
from utils import config
from utils.util_function import euclidean_distance
import math

class FPRP:
    def __init__(self, drone):
        self.my_drone = drone
        self.simulator = drone.simulator
        self.env = drone.env
        self.phy = Phy(self)

        # 基本参数
        self.slot_duration = config.SLOT_DURATION
        self.num_slots = config.NUMBER_OF_DRONES
        self.frame_duration = self.slot_duration * self.num_slots
        self.phase_duration = self.slot_duration // 10

        # 预约和空间复用参数
        self.reservation_table = {}  # {frame_id: {slot_id: set(drone_ids)}}
        self.reuse_distance = config.REUSE_DISTANCE
        self.reservation_probability = 0.8

        # 邻居管理
        self.one_hop_neighbors = set()  # 一跳邻居
        self.two_hop_neighbors = set()  # 二跳邻居
        self.neighbor_slots = {}  # {drone_id: set(slot_ids)}
        self.neighbor_positions = {}  # {drone_id: coords}

        # 状态变量
        self.current_frame = 0
        self.reserved_slots = set()
        self.current_transmission = None
        self.reservation_attempts = {}  # {slot_id: attempt_count}
        self.max_reservation_attempts = 3
        self.reservation_responses = {}  # {slot_id: {drone_id: response}}

        # 添加动态时隙管理
        self.base_slot_duration = config.SLOT_DURATION
        self.current_slot_duration = self.base_slot_duration
        self.min_slot_duration = self.base_slot_duration * 0.5
        self.max_slot_duration = self.base_slot_duration * 1.5

        # 空间复用参数
        self.base_reuse_distance = config.REUSE_DISTANCE
        self.current_reuse_distance = self.base_reuse_distance
        self.min_reuse_distance = self.base_reuse_distance * 0.6

        # 负载监控
        self.queue_history = []
        self.load_check_interval = 1000  # 微秒

        # 启动负载监控进程
        self.env.process(self.monitor_network_load())



        # 启动关键进程
        self.env.process(self.frame_synchronization())
        self.env.process(self.update_neighbor_info())

    def print_frame_allocation(self):
        """打印当前时帧的分配情况"""
        logging.info(f"\n{'=' * 20} 时帧分配状态 {'=' * 20}")
        logging.info(f"当前时间: {self.env.now / 1e6:.3f}秒")
        logging.info(f"时帧编号: {self.current_frame}")
        logging.info(f"当前时隙长度: {self.current_slot_duration:.2f}微秒")
        logging.info(f"当前复用距离: {self.current_reuse_distance:.2f}米")

        # 创建时隙分配表
        slot_info = {}
        for slot_id in range(self.num_slots):
            slot_info[slot_id] = {
                'occupied_by': set(),
                'concurrent_transmissions': 0,
                'utilization': 0.0
            }

            if self.current_frame in self.reservation_table and \
                    slot_id in self.reservation_table[self.current_frame]:
                drones = self.reservation_table[self.current_frame][slot_id]
                slot_info[slot_id]['occupied_by'] = drones
                slot_info[slot_id]['concurrent_transmissions'] = len(drones)
                slot_info[slot_id]['utilization'] = len(drones) / self.num_slots * 100

        # 打印时隙分配详情
        logging.info("\n时隙分配详情:")
        logging.info(f"{'时隙ID':^8} | {'占用无人机':^20} | {'并发数':^8} | {'利用率':^8}")
        logging.info("-" * 50)

        total_concurrent = 0
        for slot_id in range(self.num_slots):
            info = slot_info[slot_id]
            drones_str = ','.join(map(str, info['occupied_by'])) if info['occupied_by'] else '空闲'
            logging.info(
                f"{slot_id:^8} | {drones_str:^20} | {info['concurrent_transmissions']:^8} | {info['utilization']:^.2f}%")
            total_concurrent += info['concurrent_transmissions']

        # 打印统计信息
        logging.info("\n统计信息:")
        avg_concurrent = total_concurrent / self.num_slots
        total_utilization = sum(info['utilization'] for info in slot_info.values())
        logging.info(f"平均并发传输数: {avg_concurrent:.2f}")
        logging.info(f"总体时帧利用率: {total_utilization:.2f}%")

        # 打印空间复用情况
        logging.info("\n空间复用分析:")
        for slot_id, info in slot_info.items():
            if info['concurrent_transmissions'] > 1:
                drones = info['occupied_by']
                logging.info(f"\n时隙 {slot_id} 的空间复用详情:")
                for drone1 in drones:
                    for drone2 in drones:
                        if drone1 < drone2:
                            distance = euclidean_distance(
                                self.simulator.drones[drone1].coords,
                                self.simulator.drones[drone2].coords
                            )
                            logging.info(f"无人机 {drone1} 和 {drone2} 之间距离: {distance:.2f}米")

        logging.info(f"{'=' * 50}\n")

    def monitor_network_load(self):
        """监控网络负载并调整参数"""
        while True:
            current_queue_size = self.my_drone.transmitting_queue.qsize()
            self.queue_history.append(current_queue_size)
            if len(self.queue_history) > 10:
                self.queue_history.pop(0)

            old_slot_duration = self.current_slot_duration
            old_reuse_distance = self.current_reuse_distance

            # 根据历史负载调整参数
            self._adjust_parameters()

            # 记录参数变化
            if old_slot_duration != self.current_slot_duration:
                logging.info(f"无人机 {self.my_drone.identifier} - 时间 {self.env.now / 1e6:.3f}秒: "
                             f"时隙长度从 {old_slot_duration:.2f}微秒 调整为 {self.current_slot_duration:.2f}微秒 "
                             f"(当前队列大小: {current_queue_size}/{self.my_drone.max_queue_size})")

            if old_reuse_distance != self.current_reuse_distance:
                logging.info(f"无人机 {self.my_drone.identifier} - 时间 {self.env.now / 1e6:.3f}秒: "
                             f"复用距离从 {old_reuse_distance:.2f}米 调整为 {self.current_reuse_distance:.2f}米")

            yield self.env.timeout(self.load_check_interval)

    def _adjust_parameters(self):
        """根据网络状况调整协议参数"""
        avg_queue_size = sum(self.queue_history) / len(self.queue_history)
        queue_trend = self._calculate_queue_trend()

        # 调整时隙长度
        # self.current_slot_duration = self._adjust_slot_duration(avg_queue_size)

        # 调整复用距离
        if queue_trend > 0:  # 队列增长趋势
            self.current_reuse_distance = max(
                self.min_reuse_distance,
                self.current_reuse_distance * 0.9
            )
        else:  # 队列下降趋势
            self.current_reuse_distance = min(
                self.base_reuse_distance,
                self.current_reuse_distance * 1.1
            )

    def _calculate_queue_trend(self):
        """计算队列变化趋势"""
        if len(self.queue_history) < 2:
            return 0
        return self.queue_history[-1] - self.queue_history[0]

    def _adjust_slot_duration(self, queue_size):
        """动态调整时隙长度"""
        old_duration = self.current_slot_duration
        new_duration = old_duration

        if queue_size > self.my_drone.max_queue_size * 0.8:
            new_duration = min(old_duration * 1.2, self.max_slot_duration)
            if new_duration != old_duration:
                logging.info(f"无人机 {self.my_drone.identifier} - 时间 {self.env.now / 1e6:.3f}秒: "
                             f"由于队列负载较高 ({queue_size}/{self.my_drone.max_queue_size}), "
                             f"增加时隙长度至 {new_duration:.2f}微秒")
        elif queue_size < self.my_drone.max_queue_size * 0.2:
            new_duration = max(old_duration * 0.8, self.min_slot_duration)
            if new_duration != old_duration:
                logging.info(f"无人机 {self.my_drone.identifier} - 时间 {self.env.now / 1e6:.3f}秒: "
                             f"由于队列负载较低 ({queue_size}/{self.my_drone.max_queue_size}), "
                             f"减少时隙长度至 {new_duration:.2f}微秒")

        return new_duration

    def _find_available_slot(self):
        """查找可用时隙"""
        current_slot = (self.env.now // self.current_slot_duration) % self.num_slots
        base_slot = self.my_drone.identifier % self.num_slots

        logging.info(f"无人机 {self.my_drone.identifier} - 时间 {self.env.now / 1e6:.3f}秒: "
                     f"开始查找可用时隙 (当前时隙: {current_slot}, 基础时隙: {base_slot})")

        # 首先检查基础时隙
        if self._check_aggressive_reuse(base_slot):
            wait_time = ((base_slot - current_slot) % self.num_slots) * self.current_slot_duration
            logging.info(f"无人机 {self.my_drone.identifier} - 时间 {self.env.now / 1e6:.3f}秒: "
                         f"使用基础时隙 {base_slot}, 等待时间 {wait_time:.2f}微秒")
            return base_slot, wait_time

        # 寻找可并发传输的时隙
        next_hop_id = getattr(self.current_transmission, 'next_hop_id', None)
        for offset in range(self.num_slots):
            slot_id = (base_slot + offset) % self.num_slots

            if next_hop_id and self._check_concurrent_transmission(slot_id, next_hop_id):
                wait_time = ((slot_id - current_slot) % self.num_slots) * self.current_slot_duration
                logging.info(f"无人机 {self.my_drone.identifier} - 时间 {self.env.now / 1e6:.3f}秒: "
                             f"找到可并发传输的时隙 {slot_id}, 目标无人机 {next_hop_id}")
                return slot_id, wait_time

        logging.warning(f"无人机 {self.my_drone.identifier} - 时间 {self.env.now / 1e6:.3f}秒: "
                        f"未找到可用时隙")
        return None, None

    def _check_aggressive_reuse(self, slot_id):
        """检查是否可以激进地复用时隙"""
        if slot_id not in self.reservation_table.get(self.current_frame, {}):
            return True

        for existing_id in self.reservation_table[self.current_frame][slot_id]:
            if existing_id != self.my_drone.identifier:
                distance = euclidean_distance(
                    self.my_drone.coords,
                    self.simulator.drones[existing_id].coords
                )
                # 使用当前的动态复用距离
                if distance < self.current_reuse_distance:
                    return False
        return True

    def _check_concurrent_transmission(self, slot_id, target_id):
        """检查是否可以并发传输"""
        if slot_id not in self.reservation_table.get(self.current_frame, {}):
            return True

        for existing_id in self.reservation_table[self.current_frame][slot_id]:
            if existing_id != self.my_drone.identifier:
                # 计算SINR
                distance_to_target = euclidean_distance(
                    self.my_drone.coords,
                    self.simulator.drones[target_id].coords
                )
                interference_distance = euclidean_distance(
                    self.simulator.drones[existing_id].coords,
                    self.simulator.drones[target_id].coords
                )

                signal_power = config.TRANSMITTING_POWER / (distance_to_target ** config.PATH_LOSS_EXPONENT)
                interference = config.TRANSMITTING_POWER / (interference_distance ** config.PATH_LOSS_EXPONENT)
                noise = config.NOISE_POWER

                sinr = 10 * math.log10(signal_power / (interference + noise))

                logging.info(f"无人机 {self.my_drone.identifier} - 时间 {self.env.now / 1e6:.3f}秒: "
                             f"时隙 {slot_id} 的SINR计算结果: {sinr:.2f}分贝 "
                             f"(目标无人机: {target_id}, "
                             f"与目标距离: {distance_to_target:.2f}米, "
                             f"干扰距离: {interference_distance:.2f}米)")

                if sinr < config.SNR_THRESHOLD:
                    return False
        return True



    def _reserve_slot(self, slot_id):
        """改进的时隙预约"""
        # 预约当前帧和下一帧
        for frame_id in [self.current_frame, self.current_frame + 1]:
            if frame_id not in self.reservation_table:
                self.reservation_table[frame_id] = {}
            if slot_id not in self.reservation_table[frame_id]:
                self.reservation_table[frame_id][slot_id] = set()

            self.reservation_table[frame_id][slot_id].add(self.my_drone.identifier)

        self.reserved_slots.add(slot_id)
        logging.info(f"Drone {self.my_drone.identifier} reserved slot {slot_id}")
    def _check_collision(self, slot_id):
        """增强的冲突检测"""
        # 检查当前帧的时隙占用
        if self.current_frame in self.reservation_table and \
                slot_id in self.reservation_table[self.current_frame]:
            for existing_id in self.reservation_table[self.current_frame][slot_id]:
                if existing_id != self.my_drone.identifier:
                    distance = euclidean_distance(
                        self.my_drone.coords,
                        self.simulator.drones[existing_id].coords
                    )
                    if distance < self.reuse_distance * 1.5:  # 增加安全距离
                        return True

        # 检查邻居节点的时隙使用
        for neighbor_id in self.one_hop_neighbors:
            if neighbor_id in self.neighbor_slots and \
                    slot_id in self.neighbor_slots[neighbor_id]:
                return True

        # 检查二跳邻居的时隙使用
        for neighbor_id in self.two_hop_neighbors:
            if neighbor_id in self.neighbor_slots and \
                    slot_id in self.neighbor_slots[neighbor_id]:
                return True

        return False

    def frame_synchronization(self):
        """修改帧同步进程，添加时帧分配打印"""
        while True:
            self.current_frame = (self.env.now // self.frame_duration)
            self._cleanup_old_reservations()

            # 打印时帧分配情况
            self.print_frame_allocation()

            yield self.env.timeout(self.frame_duration)

    def update_neighbor_info(self):
        """更新邻居信息进程"""
        while True:
            # 更新一跳邻居
            self.one_hop_neighbors = set(self.my_drone.get_neighbors())

            # 更新二跳邻居
            self.two_hop_neighbors.clear()
            for neighbor_id in self.one_hop_neighbors:
                neighbor = self.simulator.drones[neighbor_id]
                if hasattr(neighbor, 'get_neighbors'):
                    neighbor_neighbors = set(neighbor.get_neighbors())
                    self.two_hop_neighbors.update(neighbor_neighbors)

            # 移除一跳邻居和自身
            self.two_hop_neighbors -= self.one_hop_neighbors
            self.two_hop_neighbors.discard(self.my_drone.identifier)

            # 更新邻居位置
            for neighbor_id in self.one_hop_neighbors | self.two_hop_neighbors:
                self.neighbor_positions[neighbor_id] = self.simulator.drones[neighbor_id].coords

            yield self.env.timeout(self.frame_duration // 2)

    def _cleanup_old_reservations(self):
        """清理过期预约"""
        frames_to_remove = [frame_id for frame_id in self.reservation_table
                            if frame_id < self.current_frame]
        for frame_id in frames_to_remove:
            del self.reservation_table[frame_id]

    def _check_hidden_terminal_conflict(self, slot_id):
        """检查隐藏终端冲突"""
        for neighbor_id in self.two_hop_neighbors:
            if neighbor_id in self.neighbor_slots and slot_id in self.neighbor_slots[neighbor_id]:
                distance = euclidean_distance(
                    self.my_drone.coords,
                    self.neighbor_positions[neighbor_id]
                )
                if distance < self.reuse_distance * 2:
                    return True
        return False

    def _check_exposed_terminal(self, slot_id, next_hop_id):
        """检查暴露终端问题"""
        if next_hop_id is None:
            return False

        for neighbor_id in self.one_hop_neighbors:
            if neighbor_id in self.neighbor_slots and slot_id in self.neighbor_slots[neighbor_id]:
                # 计算发送者和接收者之间的距离
                sender_receiver_distance = euclidean_distance(
                    self.my_drone.coords,
                    self.simulator.drones[next_hop_id].coords
                )
                # 计算干扰节点和接收者之间的距离
                interferer_receiver_distance = euclidean_distance(
                    self.simulator.drones[neighbor_id].coords,
                    self.simulator.drones[next_hop_id].coords
                )

                # 如果干扰节点距离接收者足够远，允许并发传输
                if interferer_receiver_distance > self.reuse_distance * 1.5:
                    return False
        return True

    def _create_control_packet(self, msg_type, slot_id):
        """创建控制数据包"""
        packet = DataPacket(
            src_drone=self.my_drone,
            dst_drone=self.my_drone,
            creation_time=self.env.now,
            data_packet_id=config.GL_ID_HELLO_PACKET,
            data_packet_length=config.HELLO_PACKET_LENGTH,
            simulator=self.simulator
        )
        packet.transmission_mode = 1  # 广播模式
        packet.msg_type = msg_type
        packet.slot_id = slot_id
        return packet

    def reserve_slot(self, slot_id, next_hop_id):
        """改进的时隙预约过程"""
        success = False

        # 初始化该时隙的响应集合
        self.reservation_responses[slot_id] = {}

        # 发送预约请求
        request_packet = self._create_control_packet("request", slot_id)
        yield self.env.process(self._transmit_packet(request_packet))

        # 等待响应
        timeout = self.slot_duration // 2
        start_time = self.env.now
        while self.env.now - start_time < timeout:
            if len(self.reservation_responses[slot_id]) >= len(self.one_hop_neighbors):
                success = True
                break
            yield self.env.timeout(10)

        if success:
            self._add_reservation(slot_id)
            logging.info(f"Drone {self.my_drone.identifier} successfully reserved slot {slot_id}")
        else:
            logging.warning(f"Drone {self.my_drone.identifier} failed to reserve slot {slot_id}")

        # 清理响应记录
        if slot_id in self.reservation_responses:
            del self.reservation_responses[slot_id]

        return success

    def _verify_responses(self, slot_id):
        """验证预约响应"""
        if slot_id not in self.reservation_responses:
            return False

        responses = self.reservation_responses[slot_id]
        required_neighbors = self.one_hop_neighbors
        received_responses = set(responses.keys())

        return required_neighbors.issubset(received_responses)

    def _add_reservation(self, slot_id):
        """添加时隙预约"""
        if self.current_frame not in self.reservation_table:
            self.reservation_table[self.current_frame] = {}
        if slot_id not in self.reservation_table[self.current_frame]:
            self.reservation_table[self.current_frame][slot_id] = set()

        self.reservation_table[self.current_frame][slot_id].add(self.my_drone.identifier)
        self.reserved_slots.add(slot_id)

    def mac_send(self, packet):
        """改进的MAC发送函数"""
        if isinstance(packet, DataPacket):
            self.current_transmission = packet
            mac_start_time = self.env.now
            retry_count = 0
            max_retries = 3

            while retry_count < max_retries:
                slot_id, wait_time = self._find_available_slot()

                if slot_id is not None:
                    if wait_time > 0:
                        yield self.env.timeout(wait_time)

                    # 发送前再次检查冲突
                    if not self._check_collision(slot_id):
                        yield self.env.process(self._transmit_packet(packet))

                        mac_delay = self.env.now - mac_start_time
                        self.simulator.metrics.mac_delay.append(mac_delay / 1e3)
                        logging.info(f"Successfully sent packet {packet.packet_id} in slot {slot_id}")
                        break

                # 如果发送失败，等待随机时间后重试
                retry_count += 1
                if retry_count < max_retries:
                    backoff_time = random.randint(1, 2 ** retry_count) * self.slot_duration
                    yield self.env.timeout(backoff_time)

            if retry_count >= max_retries:
                logging.warning(f"Failed to send packet {packet.packet_id} after {max_retries} attempts")

            self.current_transmission = None

    def _transmit_packet(self, packet):
        """执行实际的数据包传输"""
        transmission_mode = packet.transmission_mode
        if transmission_mode == 0:  # 单播
            next_hop_id = packet.next_hop_id
            packet.increase_ttl()
            self.phy.unicast(packet, next_hop_id)
            yield self.env.timeout(packet.packet_length / config.BIT_RATE * 1e6)
        elif transmission_mode == 1:  # 广播
            packet.increase_ttl()
            self.phy.broadcast(packet)
            yield self.env.timeout(packet.packet_length / config.BIT_RATE * 1e6)

    def handle_control_packet(self, packet, sender_id):
        """
        处理接收到的控制包
        Args:
            packet: 控制包
            sender_id: 发送者ID
        """
        logging.info(
            f"Drone {self.my_drone.identifier} handling control packet from {sender_id}, type: {packet.msg_type}")

        if packet.msg_type == "request":
            slot_id = packet.slot_id
            # 检查该时隙是否可以被请求者使用
            if not self._check_local_conflict(slot_id, sender_id):
                response = self._create_control_packet("response", slot_id)
                logging.info(f"Drone {self.my_drone.identifier} sending response to {sender_id} for slot {slot_id}")
                self.env.process(self._transmit_packet(response))

                # 更新本地信息
                if sender_id not in self.neighbor_slots:
                    self.neighbor_slots[sender_id] = set()
                self.neighbor_slots[sender_id].add(slot_id)

        elif packet.msg_type == "response":
            slot_id = packet.slot_id
            if slot_id in self.reservation_responses:
                self.reservation_responses[slot_id][sender_id] = True
                logging.info(f"Drone {self.my_drone.identifier} received response from {sender_id} for slot {slot_id}")

    def _check_local_conflict(self, slot_id, requester_id):
        """
        检查本地是否存在冲突
        Args:
            slot_id: 请求的时隙
            requester_id: 请求者ID
        """
        # 检查该时隙是否已被自己使用
        if slot_id in self.reserved_slots:
            return True

        # 检查与邻居的距离
        requester_coords = self.simulator.drones[requester_id].coords
        for neighbor_id, slots in self.neighbor_slots.items():
            if slot_id in slots:
                distance = euclidean_distance(
                    requester_coords,
                    self.simulator.drones[neighbor_id].coords
                )
                if distance < self.reuse_distance:
                    return True

        return False