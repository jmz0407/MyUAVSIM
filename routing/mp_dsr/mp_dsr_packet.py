from utils import config
from entities.packet import Packet


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
            pass

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