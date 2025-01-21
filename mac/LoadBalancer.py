class LoadBalancer:
    """
    负载均衡管理器
    负责监控和管理网络中各节点的流量负载情况
    """

    def __init__(self):
        # 记录每个节点的队列长度历史
        self.queue_history = {}
        # 记录每个节点的吞吐量历史
        self.throughput_history = {}
        # 记录每个节点的平均传输延迟
        self.delay_history = {}
        # 负载评分的权重参数
        self.weights = {
            'queue_length': 0.4,  # 队列长度权重
            'throughput': 0.3,  # 吞吐量权重
            'delay': 0.3  # 延迟权重
        }

    def update_node_stats(self, node_id, queue_length, throughput, delay):
        """
        更新节点的统计信息
        """
        # 初始化该节点的历史记录
        if node_id not in self.queue_history:
            self.queue_history[node_id] = []
            self.throughput_history[node_id] = []
            self.delay_history[node_id] = []

        # 更新历史记录，保持最近100个样本
        self.queue_history[node_id].append(queue_length)
        self.throughput_history[node_id].append(throughput)
        self.delay_history[node_id].append(delay)

        # 限制历史记录长度
        max_history = 100
        if len(self.queue_history[node_id]) > max_history:
            self.queue_history[node_id].pop(0)
            self.throughput_history[node_id].pop(0)
            self.delay_history[node_id].pop(0)

    def calculate_load_score(self, node_id):
        """
        计算节点的负载评分
        返回一个0-1之间的值，1表示负载最重
        """
        if node_id not in self.queue_history:
            return 0

        # 计算队列长度评分
        avg_queue = sum(self.queue_history[node_id]) / len(self.queue_history[node_id])
        queue_score = min(avg_queue / 20, 1)  # 假设最大队列长度为20

        # 计算吞吐量评分
        avg_throughput = sum(self.throughput_history[node_id]) / len(self.throughput_history[node_id])
        throughput_score = min(avg_throughput / 1000000, 1)  # 假设最大吞吐量为1Mbps

        # 计算延迟评分
        avg_delay = sum(self.delay_history[node_id]) / len(self.delay_history[node_id])
        delay_score = min(avg_delay / 1000000, 1)  # 假设最大可接受延迟为1s

        # 计算加权总分
        total_score = (
                self.weights['queue_length'] * queue_score +
                self.weights['throughput'] * throughput_score +
                self.weights['delay'] * delay_score
        )

        return total_score

    def get_high_load_nodes(self, threshold=0.8):
        """
        获取负载较重的节点列表
        """
        high_load_nodes = []
        for node_id in self.queue_history.keys():
            if self.calculate_load_score(node_id) > threshold:
                high_load_nodes.append(node_id)
        return high_load_nodes

    def get_load_distribution(self):
        """
        获取整个网络的负载分布情况
        """
        distribution = {}
        for node_id in self.queue_history.keys():
            distribution[node_id] = {
                'load_score': self.calculate_load_score(node_id),
                'queue_length': self.queue_history[node_id][-1],
                'throughput': self.throughput_history[node_id][-1],
                'delay': self.delay_history[node_id][-1]
            }
        return distribution