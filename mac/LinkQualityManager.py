class LinkQualityManager:
    """
    链路质量管理器
    用于跟踪和评估网络中各节点对之间的链路质量
    """

    def __init__(self, window_size=10):
        self.link_quality_history = {}  # 存储链路质量历史记录
        self.window_size = window_size  # 滑动窗口大小

    def update_link_quality(self, src_id, dst_id, sinr):
        """
        更新链路质量历史记录

        Args:
            src_id: 发送节点ID
            dst_id: 接收节点ID
            sinr: 当前测量的SINR值
        """
        pair_key = tuple(sorted([src_id, dst_id]))
        if pair_key not in self.link_quality_history:
            self.link_quality_history[pair_key] = []

        history = self.link_quality_history[pair_key]
        history.append(sinr)

        # 保持窗口大小固定
        if len(history) > self.window_size:
            history.pop(0)

    def get_link_quality(self, src_id, dst_id):
        """
        获取两个节点之间的链路质量评分

        Returns:
            float: 链路质量评分（0-1之间），-1表示无历史数据
        """
        pair_key = tuple(sorted([src_id, dst_id]))
        history = self.link_quality_history.get(pair_key, [])

        if not history:
            return -1

        # 计算近期SINR的平均值，并归一化到0-1范围
        avg_sinr = sum(history) / len(history)
        normalized_quality = min(max(avg_sinr / 20.0, 0), 1)  # 假设20dB为最佳SINR

        return normalized_quality

    def can_share_slot(self, node1_id, node2_id, quality_threshold=0.7):
        """
        判断两个节点是否可以共享时隙
        """
        link_quality = self.get_link_quality(node1_id, node2_id)
        return link_quality == -1 or link_quality >= quality_threshold