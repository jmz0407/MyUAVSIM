import logging
from utils.ieee_802_11 import IEEE_802_11

IEEE_802_11 = IEEE_802_11().g  # IEEE 802.11g

# --------------------- simulation parameters --------------------- #
MAP_LENGTH = 500  # m, length of the map
MAP_WIDTH = 500  # m, width of the map
MAP_HEIGHT = 500  # m, height of the map
SIM_TIME = 5 * 1e6  # us, total simulation time
NUMBER_OF_DRONES = 10  # number of drones in the network
STATIC_CASE = 1  # whether to simulate a static network
HETEROGENEOUS = 0  # heterogeneous network support (in terms of speed)
LOGGING_LEVEL = logging.INFO  # whether to print the detail information during simulation

# ---------- hardware parameters of drone (rotary-wing) -----------#
PROFILE_DRAG_COEFFICIENT = 0.012
AIR_DENSITY = 1.225  # kg/m^3
ROTOR_SOLIDITY = 0.05  # defined as the ratio of the total blade area to disc area
ROTOR_DISC_AREA = 0.79  # m^2
BLADE_ANGULAR_VELOCITY = 400  # radians/second
ROTOR_RADIUS = 0.5  # m
INCREMENTAL_CORRECTION_FACTOR = 0.1
AIRCRAFT_WEIGHT = 100  # Newton
ROTOR_BLADE_TIP_SPEED = 500
MEAN_ROTOR_VELOCITY = 7.2  # mean rotor induced velocity in hover
FUSELAGE_DRAG_RATIO = 0.3
INITIAL_ENERGY = 20 * 1e3  # in joule
ENERGY_THRESHOLD = 200  # in joule
MAX_QUEUE_SIZE = 200  # maximum size of drone's queue

# ----------------------- radio parameters ----------------------- #
TRANSMITTING_POWER = 0.1  # in Watt
LIGHT_SPEED = 3 * 1e8  # light speed (m/s)
CARRIER_FREQUENCY = IEEE_802_11['carrier_frequency']  # carrier frequency (Hz)
NOISE_POWER = 6 * 1e-11  # noise power (Watt)
RADIO_SWITCHING_TIME = 100  # us, the switching time of the transceiver mode
SNR_THRESHOLD = 6 # dB

# ---------------------- packet parameters ----------------------- #
MAX_TTL = NUMBER_OF_DRONES + 1  # maximum time-to-live value
PACKET_LIFETIME = 10 * 1e6  # 10s
IP_HEADER_LENGTH = 20 * 8  # header length in network layer, 20 byte
MAC_HEADER_LENGTH = 14 * 8  # header length in mac layer, 14 byte

# ---------------------- physical layer -------------------------- #
PATH_LOSS_EXPONENT = 2  # for large-scale fading
PLCP_PREAMBLE = 128 + 16  # including synchronization and SFD (start frame delimiter)
PLCP_HEADER = 8 + 8 + 16 + 16  # including signal, service, length and HEC (header error check)
PHY_HEADER_LENGTH = PLCP_PREAMBLE + PLCP_HEADER  # header length in physical layer, PLCP preamble + PLCP header

ACK_HEADER_LENGTH = 16 * 8  # header length of ACK packet, 16 byte

DATA_PACKET_PAYLOAD_LENGTH = 1024 * 8  # 1024 byte
DATA_PACKET_LENGTH = IP_HEADER_LENGTH + MAC_HEADER_LENGTH + PHY_HEADER_LENGTH + DATA_PACKET_PAYLOAD_LENGTH

ACK_PACKET_LENGTH = ACK_HEADER_LENGTH + 14 * 8  # bit

HELLO_PACKET_PAYLOAD_LENGTH = 256  # bit
HELLO_PACKET_LENGTH = IP_HEADER_LENGTH + MAC_HEADER_LENGTH + PHY_HEADER_LENGTH + HELLO_PACKET_PAYLOAD_LENGTH

# define the range of packet_id of different types of packets
GL_ID_HELLO_PACKET = 10000
GL_ID_VF_PACKET = 30000
GL_ID_GRAD_MESSAGE = 40000
GL_ID_CHIRP_PACKET = 50000
GL_ID_TRAFFIC_REQUIREMENT = 60000
# ------------------ physical layer parameters ------------------- #
BIT_RATE = IEEE_802_11['bit_rate']
BIT_TRANSMISSION_TIME = 1/BIT_RATE * 1e6
BANDWIDTH = IEEE_802_11['bandwidth']
SENSING_RANGE = 200  # in meter, defines the area where a sending node can disturb a transmission from a third node

# --------------------- mac layer parameters --------------------- #
# SLOT_DURATION = IEEE_802_11['slot_duration']
SLOT_DURATION = 162  # in microsecond
FRAME_DURATION = SLOT_DURATION * NUMBER_OF_DRONES * 2  # 适应节点数的动态帧长
SIFS_DURATION = IEEE_802_11['SIFS']
DIFS_DURATION = SIFS_DURATION + (2 * SLOT_DURATION)
CW_MIN = 201  # initial contention window size
# ACK_TIMEOUT = ACK_PACKET_LENGTH / BIT_RATE * 1e6 + SIFS_DURATION + 50  # maximum waiting time for ACK (0.1 s)
ACK_TIMEOUT = 0
# MAX_RETRANSMISSION_ATTEMPT = 5
MAX_RETRANSMISSION_ATTEMPT = 5

# 新增STDMA参数
CONTROL_FRAME_SLOTS = 5
QUEUE_MONITOR_INTERVAL = 1000  # μs, 队列状态监控间隔
BROADCAST_FRAME_SLOTS = 2
MIN_INFO_FRAME_SLOTS = 10
DATA_CHANNEL_RATE = BIT_RATE  # bps, 数据通道速率
CONTROL_CHANNEL_RATE = BIT_RATE  # bps, 控制通道速率
SLOT_ALLOCATION_DELAY = 1000  # μs, 时隙分配延迟

REUSE_DISTANCE = 200  # 复用同一时隙的最小间距(米)
CLUSTER_SIZE = 2      # 分簇复用组数
TOTAL_CLUSTERS = 5


# 路由参数
# config.py

# GNN-DRL参数
GNN_HIDDEN_DIM = 64
DRL_GAMMA = 0.99
DRL_EPSILON = 0.1
DRL_BUFFER_SIZE = 10000
DRL_BATCH_SIZE = 32
DRL_LEARNING_RATE = 0.001

# 性能监控参数
MONITOR_INTERVAL = 1000000  # 1s

# 多径传输配置
MULTIPATH_ENABLED = True
MAX_PATHS = 3  # 最大路径数
PATH_SELECTION_STRATEGY = 'parallel'  # 'parallel' 或 'backup'
# 添加路由协议选择
ROUTING_PROTOCOL = 'MP-DSR'  # 可选: 'DSDV', 'GPSR', 'OPAR', 'MP-DSR', 'AMLB-UAV'等
HELLO_PACKET_LIFETIME = 1e6
# 例如，DSDV 更新间隔设置为 1e6 时间单位（具体单位根据您的仿真环境确定）
DSDV_UPDATE_INTERVAL = 1e6

# utils/config.py (添加)

# 多路径路由配置
MULTIPATH_ENABLED = True
MAX_PATHS = 3  # 最大路径数
USE_GAT = True    # 启用GAT
PATH_SELECTION_STRATEGY = 'adaptive'  # 'weighted', 'round_robin', 'adaptive', 'best_quality'
# config.py中的完整GAT选项
GAT_ENABLED = True                # 是否启用GAT
GAT_UPDATE_INTERVAL = 1 * 1e6     # GAT模型更新间隔(微秒)
GAT_HIDDEN_DIM = 32               # GAT隐藏层维度
GAT_HEADS = 2                     # 注意力头数量
GAT_TRAINING_EPOCHS = 3           # 每次训练的轮次
GAT_LEARNING_RATE = 0.001         # 学习率
GAT_DROPOUT = 0.1                 # Dropout率
PATH_SELECTION_STRATEGY = 'gat'   # 使用GAT策略
USE_ATTENTION_WEIGHTS = True      # 是否使用注意力权重