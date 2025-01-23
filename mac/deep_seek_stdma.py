import logging
import simpy
import random
from phy.phy import Phy
from utils import config
from entities.packet import DataPacket
from topology.virtual_force.vf_packet import VfPacket


class DeepSeekStdma:
    """
    Spatial Time Division Multiple Access (STDMA) MAC Protocol
    支持区分控制包和数据包的处理

    属性:
        my_drone: 当前无人机节点
        simulator: 仿真器实例
        env: SimPy仿真环境
        slot_duration: 时隙长度(μs)
        num_slots: 总时隙数
        slot_schedule: 时隙分配表
        current_transmission: 当前传输任务
        phy: 物理层接口
    """

    def __init__(self, drone):
        self.my_drone = drone
        self.simulator = drone.simulator
        self.env = drone.env

        # 基础参数配置
        self.slot_duration = config.SLOT_DURATION
        self.num_slots = config.NUMBER_OF_DRONES

        # 时隙分配表初始化
        self.slot_schedule = self._init_schedule()

        # 当前传输状态
        self.current_transmission = None

        # 物理层接口
        self.phy = Phy(self)

        # 启动时隙同步进程
        self.env.process(self._slot_synchronization())
        logging.info(f"STDMA initialized for drone {self.my_drone.identifier}")

    def _init_schedule(self):
        """初始化时隙分配表"""
        schedule = {}
        for i in range(self.num_slots):
            schedule[i] = {
                'owner': None,
                'state': 'idle'
            }
        return schedule

    def _slot_synchronization(self):
        """时隙同步进程"""
        while True:
            yield self.env.timeout(self.slot_duration)

    def mac_send(self, packet):
        """
        MAC层发送函数 - 区分控制包和数据包处理

        Args:
            packet: 需要发送的数据包
        """
        if not self.slot_schedule:
            logging.error("时隙表未创建")
            return

        self.current_transmission = packet
        mac_start_time = self.env.now

        # 控制包(VF packets)直接发送,不需要时隙预约
        if isinstance(packet, VfPacket):
            logging.info(f"UAV {self.my_drone.identifier} sending control packet {packet.packet_id}")
            yield self.env.process(self._transmit_control_packet(packet))

        # 数据包需要通过STDMA机制发送
        elif isinstance(packet, DataPacket):
            # 获取可用时隙
            assigned_slot = self._get_available_slot()
            if assigned_slot is None:
                logging.warning(f"No available slot for UAV {self.my_drone.identifier}")
                return

            # 计算等待时间
            current_time = self.env.now
            slot_start_time = (current_time // self.slot_duration + assigned_slot) * self.slot_duration
            wait_time = slot_start_time - current_time

            if wait_time > 0:
                yield self.env.timeout(wait_time)

            logging.info(
                f"UAV {self.my_drone.identifier} sending data packet {packet.packet_id} in slot {assigned_slot}")
            yield self.env.process(self._transmit_data_packet(packet))

        # 计算MAC延迟
        mac_delay = self.env.now - mac_start_time
        self.simulator.metrics.mac_delay.append(mac_delay)
        self.current_transmission = None

    # 添加这个新函数用于打印时隙状态
    def _print_slot_status(self):
        """打印当前时隙分配状态"""
        status = "\n当前时隙分配状态:\n"
        status += "-" * 50 + "\n"
        status += "时隙号 | 拥有者 | 状态\n"
        status += "-" * 50 + "\n"

        for slot_id in range(self.num_slots):
            info = self.slot_schedule[slot_id]
            owner = info['owner'] if info['owner'] is not None else "无"
            state = info['state']
            status += f"  {slot_id:2d}   |   {owner:3}  | {state}\n"

        status += "-" * 50
        logging.info(status)

    # 修改时隙选择函数
    def _get_available_slot(self):
        """基于空间复用的时隙选择"""
        from utils.util_function import euclidean_distance
        from phy.large_scale_fading import maximum_communication_range

        logging.info(f"\n[寻找时隙] UAV {self.my_drone.identifier} 开始寻找可用时隙")
        self._print_slot_status()  # 打印当前时隙状态

        available_slots = []
        max_range = maximum_communication_range()

        # 遍历所有时隙
        for slot_id, info in self.slot_schedule.items():
            if info['state'] == 'idle':
                logging.info(f"[空闲时隙] 时隙 {slot_id} 当前空闲")
                available_slots.append(slot_id)
                continue

            if info['owner'] == self.my_drone.identifier:
                logging.info(f"[己方时隙] 时隙 {slot_id} 属于本机")
                available_slots.append(slot_id)
                continue

            # 检查是否可以复用该时隙
            owner = self.simulator.drones[info['owner']]
            distance = euclidean_distance(self.my_drone.coords, owner.coords)

            if distance > 2 * max_range:
                logging.info(
                    f"[复用时隙] 时隙 {slot_id} 可以复用: 与UAV {info['owner']} 距离={distance:.2f}m > {2 * max_range:.2f}m")
                available_slots.append(slot_id)
            else:
                logging.info(
                    f"[占用时隙] 时隙 {slot_id} 不可用: 与UAV {info['owner']} 距离={distance:.2f}m <= {2 * max_range:.2f}m")

        if not available_slots:
            logging.warning("[无可用时隙] 没有找到可用的时隙")
            return None

        # 从可用时隙中随机选择
        selected_slot = random.choice(available_slots)
        prev_owner = self.slot_schedule[selected_slot]['owner']

        self.slot_schedule[selected_slot]['owner'] = self.my_drone.identifier
        self.slot_schedule[selected_slot]['state'] = 'busy'

        if prev_owner is not None:
            logging.info(f"[时隙复用] UAV {self.my_drone.identifier} 复用了UAV {prev_owner} 的时隙 {selected_slot}")
        else:
            logging.info(f"[时隙分配] UAV {self.my_drone.identifier} 获得空闲时隙 {selected_slot}")

        self._print_slot_status()  # 打印更新后的时隙状态
        return selected_slot

    def _transmit_control_packet(self, packet):
        """
        发送控制包(VF packets)

        Args:
            packet: VfPacket类型的控制包
        """
        packet.increase_ttl()

        # 控制包使用广播方式发送
        self.phy.broadcast(packet)

        # 等待传输完成
        transmission_time = packet.packet_length / config.BIT_RATE * 1e6
        yield self.env.timeout(transmission_time)

    def _transmit_data_packet(self, packet):
        """
        发送数据包(Data packets)

        Args:
            packet: DataPacket类型的数据包
        """
        packet.increase_ttl()

        # 数据包使用单播方式发送到下一跳
        self.phy.unicast(packet, packet.next_hop_id)

        # 等待传输完成
        transmission_time = packet.packet_length / config.BIT_RATE * 1e6
        yield self.env.timeout(transmission_time)