#!/bin/bash
# 运行MAC协议性能测试的脚本

# 创建输出目录
OUTPUT_DIR="mac_protocol_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo "=================================================================="
echo "        无人机网络MAC协议性能测试"
echo "=================================================================="
echo "开始测试..."
echo "结果将保存到: $OUTPUT_DIR"
echo ""

# 运行基本测试 - 所有协议, 10-20-30架无人机, CBR流量
echo "执行基础测试场景 - CBR流量..."
python mac_test.py \
  --protocols STDMA TraTDMA BasicSTDMA \
  --drone-counts 10 20 30 \
  --traffic-patterns CBR \
  --flow-counts 3 5 \
  --seeds 42 \
  --sim-time 5000000 \
  --output-dir $OUTPUT_DIR/basic_test
echo "基础测试完成"
echo ""

# 运行高级测试 - 多种业务模式
echo "执行高级测试场景 - 多种业务模式..."
python mac_test.py \
  --protocols STDMA TraTDMA BasicSTDMA \
  --drone-counts 20 \
  --traffic-patterns CBR VBR BURST PERIODIC \
  --flow-counts 5 \
  --seeds 42 \
  --sim-time 5000000 \
  --output-dir $OUTPUT_DIR/traffic_pattern_test
echo "多种业务模式测试完成"
echo ""

# 运行密度测试 - 不同无人机密度
echo "执行密度测试场景 - 不同无人机数量..."
python mac_test.py \
  --protocols STDMA TraTDMA \
  --drone-counts 5 10 15 20 25 30 \
  --traffic-patterns CBR \
  --flow-counts 3 \
  --seeds 42 \
  --sim-time 5000000 \
  --output-dir $OUTPUT_DIR/density_test
echo "密度测试完成"
echo ""

# 运行高负载测试 - 大量业务流
echo "执行高负载测试场景 - 大量业务流..."
python mac_test.py \
  --protocols STDMA TraTDMA BasicSTDMA \
  --drone-counts 15 \
  --traffic-patterns CBR \
  --flow-counts 5 10 15 \
  --seeds 42 \
  --sim-time 5000000 \
  --output-dir $OUTPUT_DIR/high_load_test
echo "高负载测试完成"
echo ""

echo "所有测试完成，查看 $OUTPUT_DIR 目录获取详细报告"
echo "=================================================================="