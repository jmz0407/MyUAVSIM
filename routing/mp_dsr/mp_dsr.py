import logging
import numpy as np
import math
import queue
from collections import defaultdict
from utils import config
from entities.packet import DataPacket, Packet


class RouteRequestPacket(Packet):
    """MP-DSR 路由请求包"""

    def __init__(self, src_drone, dst_drone, req_id, creation_time, simulator, route_record=None):
        # 路由请求包的ID格式: "RREQ_<源ID>_<目的ID>_<请求ID>"
        packet_id = f"RREQ_{src_drone.identifier}_{dst_drone.identifier}_{req_id}"
        packet_length = config.HELLO_PACKET_LENGTH  # 使用与Hello包相同的长度

        super().__init__(packet_id, packet_length, creation_time, simulator)

        self.src_drone = src_drone
        self.dst_drone = dst_drone
        self.req_id = req_id
        self.transmission_mode = 1  # 广播模式

        # 路由记录，存储已经访问的节点
        self.route_record = route_record or [src_drone.identifier]

        # 为辅助路由发现设置最大跳数限制
        self.max_hop_count = config.MAX_TTL


class RouteReplyPacket(Packet):
    """MP-DSR 路由回复包"""

    def __init__(self, src_drone, dst_drone, reply_id, route, creation_time, simulator):
        # 路由回复包的ID格式: "RREP_<源ID>_<目的ID>_<回复ID>"
        packet_id = f"RREP_{src_drone.identifier}_{dst_drone.identifier}_{reply_id}"
        packet_length = config.HELLO_PACKET_LENGTH  # 使用与Hello包相同的长度

        super().__init__(packet_id, packet_length, creation_time, simulator)

        self.src_drone = src_drone
        self.dst_drone = dst_drone
        self.reply_id = reply_id
        self.transmission_mode = 0  # 单播模式

        # 完整的路由路径
        self.route = route
        self.next_hop_id = self._get_next_hop()

    def _get_next_hop(self):
        """获取下一跳节点ID"""
        # 从路由路径中找出当前节点的位置
        try:
            current_index = self.route.index(self.src_drone.identifier)
            if current_index < len(self.route) - 1:
                return self.route[current_index + 1]
        except ValueError:
            # 当前节点不在路径中，这是一个错误情况
            logging.error(f"MP-DSR: 节点 {self.src_drone.identifier} 不在回复路由中")

        return None


class RouteErrorPacket(Packet):
    """MP-DSR 路由错误包"""

    def __init__(self, src_drone, dst_drone, broken_link, creation_time, simulator):
        # 路由错误包的ID格式: "RERR_<源ID>_<目的ID>_<断开链路>"
        packet_id = f"RERR_{src_drone.identifier}_{dst_drone.identifier}_{broken_link[0]}_{broken_link[1]}"
        packet_length = config.HELLO_PACKET_LENGTH  # 使用与Hello包相同的长度

        super().__init__(packet_id, packet_length, creation_time, simulator)

        self.src_drone = src_drone
        self.dst_drone = dst_drone
        self.broken_link = broken_link  # 格式: (from_node, to_node)
        self.transmission_mode = 0  # 单播模式
        self.next_hop_id = dst_drone.identifier


class MPDSR:
    """
    Multipath Dynamic Source Routing (MP-DSR) 协议实现

    MP-DSR是经典DSR协议的多路径扩展，旨在发现和维护多条路径，
    提高网络的容错能力和负载平衡能力。

    关键特性:
    1. 基于源路由: 源节点决定数据包的完整路径
    2. 维护多条路径: 为每个目的地保存多条路径
    3. 路径分离度量: 选择相互独立程度高的路径
    4. 按需路由发现: 仅在需要时发现路径
    """

    def __init__(self, simulator, drone):
        """
        初始化MP-DSR协议实例

        Args:
            simulator: 仿真器实例
            drone: 安装该协议的无人机
        """
        self.simulator = simulator
        self.my_drone = drone
        self.env = drone.env

        # 路由表: {目的地ID: [路径列表]}
        # 每个路径是一个节点ID列表
        self.route_cache = {}

        # 已处理的路由请求缓存: {(源ID, 目的ID, 请求ID): True}
        # 用于避免重复处理相同的路由请求
        self.request_cache = {}

        # 路由请求超时: {请求ID: 过期时间}
        self.request_timeout = {}

        # 当前活跃的路由发现: {目的地ID: (请求ID, 开始时间)}
        self.active_route_discovery = {}

        # 最大路径数
        self.max_paths = config.MAX_PATHS if hasattr(config, 'MAX_PATHS') else 3

        # 路由请求ID计数器
        self.req_id_counter = 0

        # 路由回复ID计数器
        self.reply_id_counter = 0

        # 正在等待路由的数据包: {目的地ID: [数据包列表]}
        self.pending_packets = defaultdict(list)

        # 每个流使用的当前路径索引: {(源ID, 目的ID): 当前索引}
        self.current_path_index = {}

        # 记录路径使用情况: {(源ID, 目的ID): {路径哈希: 使用次数}}
        self.path_usage = defaultdict(lambda: defaultdict(int))

        # 启动路由维护进程
        self.env.process(self.maintain_routes())

    def next_hop_selection(self, packet):
        """
        为数据包选择下一跳节点

        Args:
            packet: 需要路由的数据包

        Returns:
            (has_route, modified_packet, enquire):
                - has_route: 是否找到路由
                - modified_packet: 可能被修改的数据包
                - enquire: 是否需要发送路由请求
        """
        if not isinstance(packet, DataPacket):
            # 非数据包(如控制包)，直接使用预设的下一跳
            return True, packet, False

        dst_id = packet.dst_drone.identifier

        # 添加检查 - 确保每个数据包都有next_hop_id属性
        if not hasattr(packet, 'next_hop_id'):
            packet.next_hop_id = None

        # 如果目的地是自己，不需要路由
        if dst_id == self.my_drone.identifier:
            return True, packet, False

        # 检查路由缓存中是否有到目的地的路径
        if dst_id in self.route_cache and self.route_cache[dst_id]:
            # 选择一条路径
            path = self.select_path(dst_id, packet)

            if path:
                # 找到当前节点在路径中的位置
                try:
                    current_index = path.index(self.my_drone.identifier)

                    # 检查是否是路径的最后一个节点
                    if current_index < len(path) - 1:
                        next_hop = path[current_index + 1]

                        # 设置数据包的下一跳和完整路径
                        packet.next_hop_id = next_hop
                        packet.routing_path = path
                        packet.transmission_mode = 0  # 单播模式

                        logging.info(
                            f"MP-DSR: 节点 {self.my_drone.identifier} 选择路径 {path} 发送至 {dst_id}, 下一跳 {next_hop}")

                        # 更新路径使用统计
                        path_hash = self.path_to_hash(path)
                        flow_id = (self.my_drone.identifier, dst_id)
                        self.path_usage[flow_id][path_hash] += 1

                        return True, packet, False
                    else:
                        logging.warning(
                            f"MP-DSR: 节点 {self.my_drone.identifier} 是路径 {path} 的最后节点，但目的地是 {dst_id}")
                except ValueError:
                    logging.warning(f"MP-DSR: 节点 {self.my_drone.identifier} 不在路径 {path} 中")

        # 如果没有找到路径或路径无效，启动路由发现
        self.pending_packets[dst_id].append(packet)

        # 检查是否已经有正在进行的路由发现
        if dst_id in self.active_route_discovery:
            req_id, start_time = self.active_route_discovery[dst_id]

            # 如果路由发现已经超时，启动新的路由发现
            if self.env.now - start_time > config.PACKET_LIFETIME * 0.2:  # 20%的包寿命作为超时时间
                logging.info(f"MP-DSR: 节点 {self.my_drone.identifier} 路由发现至 {dst_id} 超时，重新发起")
                self.initiate_route_discovery(dst_id)
            else:
                logging.info(f"MP-DSR: 节点 {self.my_drone.identifier} 等待路由发现至 {dst_id} 完成")
        else:
            # 启动新的路由发现
            self.initiate_route_discovery(dst_id)

        return False, packet, True

    def initiate_route_discovery(self, dst_id):
        """
        启动路由发现过程

        Args:
            dst_id: 目的地节点ID
        """
        # 增加请求ID计数器
        self.req_id_counter += 1
        req_id = self.req_id_counter

        # 记录活跃的路由发现
        self.active_route_discovery[dst_id] = (req_id, self.env.now)

        # 创建路由请求包
        rreq = RouteRequestPacket(
            src_drone=self.my_drone,
            dst_drone=self.simulator.drones[dst_id],
            req_id=req_id,
            creation_time=self.env.now,
            simulator=self.simulator
        )

        # 将请求添加到已处理缓存
        req_key = (self.my_drone.identifier, dst_id, req_id)
        self.request_cache[req_key] = True

        # 设置请求超时
        self.request_timeout[req_key] = self.env.now + config.PACKET_LIFETIME * 0.5  # 50%的包寿命作为超时时间

        # 广播路由请求
        logging.info(f"MP-DSR: 节点 {self.my_drone.identifier} 发起路由发现至 {dst_id}, 请求ID {req_id}")
        self.my_drone.transmitting_queue.put(rreq)

    def packet_reception(self, packet, sender):
        """
        处理接收到的数据包

        Args:
            packet: 接收到的数据包
            sender: 发送该数据包的节点ID
        """
        # 处理不同类型的包
        if isinstance(packet, RouteRequestPacket):
            self.handle_route_request(packet, sender)
        elif isinstance(packet, RouteReplyPacket):
            self.handle_route_reply(packet, sender)
        elif isinstance(packet, RouteErrorPacket):
            self.handle_route_error(packet, sender)
        elif isinstance(packet, DataPacket):
            self.handle_data_packet(packet, sender)
        else:
            # 其他类型的包，交给基类处理
            pass

    def handle_route_request(self, rreq, sender):
        """
        处理路由请求包

        Args:
            rreq: 路由请求包
            sender: 发送者节点ID
        """
        src_id = rreq.src_drone.identifier
        dst_id = rreq.dst_drone.identifier
        req_id = rreq.req_id

        # 检查是否已经处理过该请求
        req_key = (src_id, dst_id, req_id)
        if req_key in self.request_cache:
            return

        # 将请求添加到已处理缓存
        self.request_cache[req_key] = True

        # 设置请求超时
        self.request_timeout[req_key] = self.env.now + config.PACKET_LIFETIME * 0.5

        # 更新路由记录
        new_route_record = rreq.route_record.copy()
        new_route_record.append(self.my_drone.identifier)

        # 检查是否达到目的地
        if self.my_drone.identifier == dst_id:
            # 创建路由回复
            self.reply_id_counter += 1
            rrep = RouteReplyPacket(
                src_drone=self.my_drone,
                dst_drone=self.simulator.drones[src_id],
                reply_id=self.reply_id_counter,
                route=new_route_record,
                creation_time=self.env.now,
                simulator=self.simulator
            )

            logging.info(f"MP-DSR: 节点 {self.my_drone.identifier} (目的地) 回复路由请求，路径: {new_route_record}")

            # 发送路由回复
            self.my_drone.transmitting_queue.put(rrep)
            return

        # 检查是否知道到目的地的路径
        if dst_id in self.route_cache and self.route_cache[dst_id]:
            for stored_path in self.route_cache[dst_id]:
                # 创建完整路径
                complete_path = new_route_record + stored_path[1:]  # 跳过存储路径的第一个节点(当前节点)

                # 检查路径中是否有循环
                if len(complete_path) == len(set(complete_path)):
                    # 创建路由回复
                    self.reply_id_counter += 1
                    rrep = RouteReplyPacket(
                        src_drone=self.my_drone,
                        dst_drone=self.simulator.drones[src_id],
                        reply_id=self.reply_id_counter,
                        route=complete_path,
                        creation_time=self.env.now,
                        simulator=self.simulator
                    )

                    logging.info(
                        f"MP-DSR: 节点 {self.my_drone.identifier} 使用缓存路径回复路由请求，完整路径: {complete_path}")

                    # 发送路由回复
                    self.my_drone.transmitting_queue.put(rrep)
                    return

        # 检查是否达到最大跳数限制
        if len(new_route_record) >= rreq.max_hop_count:
            logging.info(f"MP-DSR: 节点 {self.my_drone.identifier} 丢弃路由请求，已达最大跳数")
            return

        # 转发修改后的路由请求
        new_rreq = RouteRequestPacket(
            src_drone=rreq.src_drone,
            dst_drone=rreq.dst_drone,
            req_id=req_id,
            creation_time=self.env.now,
            simulator=self.simulator,
            route_record=new_route_record
        )

        logging.info(f"MP-DSR: 节点 {self.my_drone.identifier} 转发路由请求，路径记录: {new_route_record}")

        # 延迟转发以减轻广播风暴
        yield self.env.timeout(10)  # 10微秒延迟
        self.my_drone.transmitting_queue.put(new_rreq)

    def handle_route_reply(self, rrep, sender):
        """
        处理路由回复包

        Args:
            rrep: 路由回复包
            sender: 发送者节点ID
        """
        src_id = rrep.src_drone.identifier
        dst_id = rrep.dst_drone.identifier
        route = rrep.route

        # 检查是否是给自己的回复
        if self.my_drone.identifier == dst_id:
            # 到达目的地，处理回复
            src_dst_pair = (rrep.route[0], rrep.route[-1])

            # 初始化路由缓存项
            if rrep.route[-1] not in self.route_cache:
                self.route_cache[rrep.route[-1]] = []

            # 添加新路径
            self.add_route(rrep.route[-1], rrep.route)

            # 检查是否有待处理的数据包
            if rrep.route[-1] in self.pending_packets and self.pending_packets[rrep.route[-1]]:
                logging.info(
                    f"MP-DSR: 节点 {self.my_drone.identifier} 处理 {len(self.pending_packets[rrep.route[-1]])} 个待发送数据包")

                # 处理待发送的数据包
                packets_to_send = self.pending_packets[rrep.route[-1]].copy()
                self.pending_packets[rrep.route[-1]] = []

                for packet in packets_to_send:
                    # 重新尝试发送
                    self.my_drone.transmitting_queue.put(packet)

            # 清除活跃路由发现
            if rrep.route[-1] in self.active_route_discovery:
                del self.active_route_discovery[rrep.route[-1]]

            logging.info(f"MP-DSR: 节点 {self.my_drone.identifier} 接收路由回复，路径: {route}")
        else:
            # 转发路由回复
            # 找到当前节点在路径中的位置
            try:
                current_index = route.index(self.my_drone.identifier)

                # 检查是否是路径的最后一个节点
                if current_index > 0:
                    # 为源节点存储路径信息
                    if route[-1] not in self.route_cache:
                        self.route_cache[route[-1]] = []

                    # 从当前节点到目的地的子路径
                    sub_path = route[current_index:]
                    self.add_route(route[-1], sub_path)

                    # 检查是否还有路径可走
                    if current_index < len(route) - 1:
                        # 设置下一跳
                        next_hop = route[current_index - 1]  # 沿着回复路径反向转发
                        rrep.next_hop_id = next_hop

                        logging.info(
                            f"MP-DSR: 节点 {self.my_drone.identifier} 转发路由回复给 {next_hop}，完整路径: {route}")

                        # 转发回复
                        self.my_drone.transmitting_queue.put(rrep)
                    else:
                        logging.warning(f"MP-DSR: 节点 {self.my_drone.identifier} 是路由回复路径中的最后节点，无法转发")
                else:
                    logging.warning(f"MP-DSR: 节点 {self.my_drone.identifier} 是路由回复路径中的第一个节点，无法转发")
            except ValueError:
                logging.warning(f"MP-DSR: 节点 {self.my_drone.identifier} 不在回复路径 {route} 中")

    def handle_route_error(self, rerr, sender):
        """
        处理路由错误包

        Args:
            rerr: 路由错误包
            sender: 发送者节点ID
        """
        broken_link = rerr.broken_link

        # 检查并删除包含断开链路的路径
        routes_removed = False

        for dst_id, paths in list(self.route_cache.items()):
            updated_paths = []

            for path in paths:
                # 检查路径是否包含断开链路
                contains_broken_link = False

                for i in range(len(path) - 1):
                    if path[i] == broken_link[0] and path[i + 1] == broken_link[1]:
                        contains_broken_link = True
                        break

                if not contains_broken_link:
                    updated_paths.append(path)
                else:
                    routes_removed = True
                    logging.info(
                        f"MP-DSR: 节点 {self.my_drone.identifier} 移除包含断开链路 {broken_link} 的路径: {path}")

            # 更新该目的地的路径
            if updated_paths:
                self.route_cache[dst_id] = updated_paths
            else:
                # 如果没有路径，删除该目的地的条目
                del self.route_cache[dst_id]

        # 如果是给自己的错误包，处理完成
        if self.my_drone.identifier == rerr.dst_drone.identifier:
            return

        # 如果移除了路径，继续传播错误
        if routes_removed:
            # 创建新的错误包
            new_rerr = RouteErrorPacket(
                src_drone=self.my_drone,
                dst_drone=rerr.dst_drone,
                broken_link=broken_link,
                creation_time=self.env.now,
                simulator=self.simulator
            )

            logging.info(f"MP-DSR: 节点 {self.my_drone.identifier} 转发路由错误，断开链路: {broken_link}")

            # 转发错误包
            self.my_drone.transmitting_queue.put(new_rerr)

    def handle_data_packet(self, packet, sender):
        """
        处理数据包

        Args:
            packet: 数据包
            sender: 发送者节点ID
        """
        # 检查是否是目的地
        if packet.dst_drone.identifier == self.my_drone.identifier:
            # 数据包到达目的地，交给上层处理
            return

        # 检查是否有源路由信息
        if hasattr(packet, 'routing_path') and packet.routing_path:
            path = packet.routing_path

            # 找到当前节点在路径中的位置
            try:
                current_index = path.index(self.my_drone.identifier)

                # 检查是否还有下一跳
                if current_index < len(path) - 1:
                    next_hop = path[current_index + 1]
                    packet.next_hop_id = next_hop

                    logging.info(
                        f"MP-DSR: 节点 {self.my_drone.identifier} 转发数据包 {packet.packet_id} 给 {next_hop}，路径: {path}")

                    # 转发数据包
                    self.my_drone.transmitting_queue.put(packet)
                else:
                    logging.warning(f"MP-DSR: 节点 {self.my_drone.identifier} 是路径 {path} 的最后节点，无法转发数据包")
            except ValueError:
                logging.warning(f"MP-DSR: 节点 {self.my_drone.identifier} 不在路径 {path} 中")
        else:
            logging.warning(f"MP-DSR: 数据包 {packet.packet_id} 没有源路由信息")

    def add_route(self, dst_id, path):
        """
        添加或更新路由缓存中的路径

        Args:
            dst_id: 目的地节点ID
            path: 路径(节点ID列表)
        """
        # 检查路径是否有效
        if not path or len(path) < 2:
            return False

        # 检查路径中是否有循环
        if len(path) != len(set(path)):
            return False

        # 初始化目的地路径列表
        if dst_id not in self.route_cache:
            self.route_cache[dst_id] = []

        # 计算路径哈希用于比较
        path_hash = self.path_to_hash(path)

        # 检查是否已经有完全相同的路径
        for existing_path in self.route_cache[dst_id]:
            existing_hash = self.path_to_hash(existing_path)
            if existing_hash == path_hash:
                return False  # 已经有相同的路径

        # 检查是否需要删除较差的路径
        if len(self.route_cache[dst_id]) >= self.max_paths:
            # 计算新路径的质量
            new_path_quality = self.evaluate_path_quality(path)

            # 找出质量最差的路径
            worst_path = None
            worst_quality = float('inf')

            for existing_path in self.route_cache[dst_id]:
                quality = self.evaluate_path_quality(existing_path)
                if quality < worst_quality:
                    worst_quality = quality
                    worst_path = existing_path

            # 如果新路径质量更好，替换最差的路径
            if new_path_quality > worst_quality:
                self.route_cache[dst_id].remove(worst_path)
                logging.info(f"MP-DSR: 节点 {self.my_drone.identifier} 用质量更好的路径替换最差路径")
            else:
                return False  # 新路径质量不够好

        # 添加新路径
        self.route_cache[dst_id].append(path)

        # 对路径进行排序，优质路径排在前面
        self.route_cache[dst_id].sort(key=self.evaluate_path_quality, reverse=True)

        logging.info(f"MP-DSR: 节点 {self.my_drone.identifier} 添加到 {dst_id} 的新路径: {path}")

        return True

    def evaluate_path_quality(self, path):
        """
        评估路径质量

        Args:
            path: 路径(节点ID列表)

        Returns:
            float: 路径质量评分(越高越好)
        """
        # 基础质量指标 - 跳数倒数
        hop_count = len(path) - 1
        base_quality = 1.0 / max(1, hop_count)

        # TODO: 可以添加更多质量指标，如:
        # - 链路稳定性
        # - 能量水平
        # - 节点拥塞情况

        return base_quality

    def select_path(self, dst_id, packet=None):
        """
        为给定目的地选择最佳路径

        Args:
            dst_id: 目的地节点ID
            packet: 可选，要发送的数据包

        Returns:
            list: 选择的路径，如果没有可用路径则返回None
        """
        if dst_id not in self.route_cache or not self.route_cache[dst_id]:
            return None

        # 获取所有可用路径
        available_paths = self.route_cache[dst_id]

        # 如果只有一条路径，直接返回
        if len(available_paths) == 1:
            return available_paths[0]

        # 流ID
        flow_id = (self.my_drone.identifier, dst_id)

        # 使用轮询策略选择路径
        if flow_id not in self.current_path_index:
            self.current_path_index[flow_id] = 0

        # 获取当前索引并递增
        index = self.current_path_index[flow_id]
        self.current_path_index[flow_id] = (index + 1) % len(available_paths)

        selected_path = available_paths[index]

        # 检查路径是否包含当前节点
        if self.my_drone.identifier not in selected_path:
            logging.warning(f"MP-DSR: 选择的路径 {selected_path} 不包含当前节点 {self.my_drone.identifier}")
            return None

        return selected_path

    def path_to_hash(self, path):
        """
        将路径转换为哈希值，用于路径比较

        Args:
            path: 路径(节点ID列表)

        Returns:
            str: 路径的哈希值
        """
        return '_'.join(map(str, path))

    def calculate_path_disjointness(self, path1, path2):
        """
        计算两条路径的分离度 (0-1范围，1表示完全分离)

        Args:
            path1: 第一条路径
            path2: 第二条路径

        Returns:
            float: 路径分离度
        """
        # 去除源节点和目的节点(它们必然相同)
        internal_path1 = set(path1[1:-1])
        internal_path2 = set(path2[1:-1])

        # 计算共享节点数
        shared_nodes = len(internal_path1.intersection(internal_path2))

        # 计算分离度
        if not internal_path1 and not internal_path2:
            return 1.0  # 两条路径都只有源和目的，视为完全分离

        max_internal_length = max(len(internal_path1), len(internal_path2))
        disjointness = 1.0 - (shared_nodes / max_internal_length) if max_internal_length > 0 else 1.0

        return disjointness

    def maintain_routes(self):
        """
        周期性地维护路由表的进程

        - 清理过期的请求缓存
        - 检测并处理失效的路由
        """
        while True:
            # 每隔一段时间执行一次
            yield self.env.timeout(config.HELLO_PACKET_LIFETIME)  # 使用Hello包的生命周期作为检查间隔

            # 清理过期的请求缓存
            current_time = self.env.now
            expired_requests = []

            for req_key, timeout in list(self.request_timeout.items()):
                if current_time > timeout:
                    expired_requests.append(req_key)

            for req_key in expired_requests:
                if req_key in self.request_cache:
                    del self.request_cache[req_key]
                if req_key in self.request_timeout:
                    del self.request_timeout[req_key]

            # 检测失效路由
            for dst_id, paths in list(self.route_cache.items()):
                valid_paths = []

                for path in paths:
                    # 检查路径上的每个链路是否有效
                    path_valid = True

                    for i in range(len(path) - 1):
                        # 检查两个相邻节点是否仍在通信范围内
                        node1 = path[i]
                        node2 = path[i + 1]

                        if not self.is_link_valid(node1, node2):
                            path_valid = False

                            # 如果当前节点是断开链路的源节点，发送路由错误
                            if node1 == self.my_drone.identifier:
                                self.send_route_error(node2, path[-1])

                            break

                    if path_valid:
                        valid_paths.append(path)

                # 更新路由缓存
                if valid_paths:
                    self.route_cache[dst_id] = valid_paths
                else:
                    # 如果没有有效路径，删除该目的地的条目
                    if dst_id in self.route_cache:
                        del self.route_cache[dst_id]

    def is_link_valid(self, node1_id, node2_id):
        """
        检查两个节点之间的链路是否有效

        Args:
            node1_id: 第一个节点ID
            node2_id: 第二个节点ID

        Returns:
            bool: 链路是否有效
        """
        # 检查节点是否存在
        if node1_id >= len(self.simulator.drones) or node2_id >= len(self.simulator.drones):
            return False

        # 获取节点对象
        node1 = self.simulator.drones[node1_id]
        node2 = self.simulator.drones[node2_id]

        # 检查节点是否休眠
        if node1.sleep or node2.sleep:
            return False

        # 计算节点间距离
        pos1 = node1.coords
        pos2 = node2.coords
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))

        # 检查是否在通信范围内
        return distance <= config.SENSING_RANGE

    def send_route_error(self, broken_link_dest, final_dest):
        """
        发送路由错误包

        Args:
            broken_link_dest: 断开链路的目的节点ID
            final_dest: 最终目的地节点ID
        """
        # 创建断开链路描述
        broken_link = (self.my_drone.identifier, broken_link_dest)

        # 创建路由错误包
        rerr = RouteErrorPacket(
            src_drone=self.my_drone,
            dst_drone=self.simulator.drones[final_dest],
            broken_link=broken_link,
            creation_time=self.env.now,
            simulator=self.simulator
        )

        logging.info(f"MP-DSR: 节点 {self.my_drone.identifier} 发送路由错误，断开链路: {broken_link}")

        # 发送错误包
        self.my_drone.transmitting_queue.put(rerr)

        # 从路由缓存中删除包含该链路的所有路径
        for dst_id, paths in list(self.route_cache.items()):
            updated_paths = []

            for path in paths:
                # 检查路径是否包含断开链路
                contains_broken_link = False

                for i in range(len(path) - 1):
                    if path[i] == broken_link[0] and path[i + 1] == broken_link[1]:
                        contains_broken_link = True
                        break

                if not contains_broken_link:
                    updated_paths.append(path)

            # 更新该目的地的路径
            if updated_paths:
                self.route_cache[dst_id] = updated_paths
            else:
                # 如果没有路径，删除该目的地的条目
                del self.route_cache[dst_id]

    def calculate_cost_matrix(self, jitter=0):
        """
        计算当前网络的代价矩阵，兼容DSDV接口

        Args:
            jitter: 可选的抖动参数，用于生成不同的路径

        Returns:
            numpy.ndarray: 代价矩阵
        """
        n_drones = len(self.simulator.drones)
        cost_matrix = np.full((n_drones, n_drones), np.inf)

        # 设置对角线元素为0
        np.fill_diagonal(cost_matrix, 0)

        # 为每对节点计算代价
        for i in range(n_drones):
            node_i = self.simulator.drones[i]
            pos_i = node_i.coords

            for j in range(n_drones):
                if i == j:
                    continue

                node_j = self.simulator.drones[j]
                pos_j = node_j.coords

                # 如果节点休眠，设置为无穷大
                if node_i.sleep or node_j.sleep:
                    cost_matrix[i, j] = np.inf
                    continue

                # 计算距离
                distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos_i, pos_j)))

                # 如果在通信范围内，设置代价
                if distance <= config.SENSING_RANGE:
                    # 基本代价是距离
                    base_cost = distance

                    # 添加抖动以便生成不同的路径
                    if jitter > 0:
                        # 添加一些随机扰动，但保持道路的基本结构
                        random_factor = 1 + (np.random.random() - 0.5) * jitter
                        cost = base_cost * random_factor
                    else:
                        cost = base_cost

                    cost_matrix[i, j] = cost

        return cost_matrix

    def dijkstra(self, cost_matrix, src_id, dst_id, path_index=0):
        """
        使用Dijkstra算法找到从源到目的地的最短路径，兼容DSDV接口

        Args:
            cost_matrix: 代价矩阵
            src_id: 源节点ID
            dst_id: 目的地节点ID
            path_index: 要返回的第几条路径(0表示最短路径)

        Returns:
            list: 路径(节点ID列表)，如果没有路径则返回空列表
        """
        n_drones = len(cost_matrix)

        # 检查源和目的地是否相同
        if src_id == dst_id:
            return [src_id]

        # 初始化
        distance = np.full(n_drones, np.inf)
        distance[src_id] = 0
        visited = np.zeros(n_drones, dtype=bool)
        previous = np.full(n_drones, -1, dtype=int)

        # Dijkstra算法
        for _ in range(n_drones):
            # 找出未访问节点中距离最小的
            min_distance = np.inf
            min_index = -1

            for i in range(n_drones):
                if not visited[i] and distance[i] < min_distance:
                    min_distance = distance[i]
                    min_index = i

            # 如果没有可达节点，结束
            if min_index == -1:
                break

            # 标记为已访问
            visited[min_index] = True

            # 检查是否到达目的地
            if min_index == dst_id:
                break

            # 更新相邻节点的距离
            for i in range(n_drones):
                if not visited[i] and cost_matrix[min_index, i] < np.inf:
                    new_distance = distance[min_index] + cost_matrix[min_index, i]

                    # 如果找到更短的路径，更新
                    if new_distance < distance[i]:
                        distance[i] = new_distance
                        previous[i] = min_index

        # 如果目的地不可达
        if distance[dst_id] == np.inf:
            return []

        # 重建路径
        path = []
        current = dst_id

        while current != -1:
            path.append(current)
            current = previous[current]

        # 反转路径，使其从源到目的地
        path.reverse()

        return path

    def get_paths_to_destination(self, dst_id):
        """
        获取到指定目的地的所有路径

        Args:
            dst_id: 目的地节点ID

        Returns:
            list: 路径列表，每个路径是节点ID列表
        """
        if dst_id in self.route_cache:
            return self.route_cache[dst_id].copy()
        return []

    def get_path_metrics(self):
        """
        获取路径指标统计

        Returns:
            dict: 路径统计信息
        """
        metrics = {
            'destinations': len(self.route_cache),
            'total_paths': sum(len(paths) for paths in self.route_cache.values()),
            'average_paths_per_destination': 0,
            'path_length_stats': {
                'min': float('inf'),
                'max': 0,
                'avg': 0
            },
            'path_usage': dict(self.path_usage)
        }

        # 计算平均路径数
        if metrics['destinations'] > 0:
            metrics['average_paths_per_destination'] = metrics['total_paths'] / metrics['destinations']

        # 计算路径长度统计
        path_lengths = []
        for paths in self.route_cache.values():
            for path in paths:
                path_length = len(path) - 1  # 跳数
                path_lengths.append(path_length)

                # 更新最大最小值
                metrics['path_length_stats']['min'] = min(metrics['path_length_stats']['min'], path_length)
                metrics['path_length_stats']['max'] = max(metrics['path_length_stats']['max'], path_length)

        # 计算平均路径长度
        if path_lengths:
            metrics['path_length_stats']['avg'] = sum(path_lengths) / len(path_lengths)
        else:
            metrics['path_length_stats']['min'] = 0

        return metrics