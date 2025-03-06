#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MP_OLSR和AMLB_OPAR路由协议比较测试脚本
用于对比两种多路径路由协议的性能
"""

import os
import logging
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端，避免需要图形界面

from utils import config
from routing_compare import RoutingProtocolTester


def main():
    # 配置测试参数
    base_config = {
        'NUMBER_OF_DRONES': 10,  # 无人机数量
        'SIM_TIME': 5 * 1e6,  # 仿真时间 (5秒)
        'MULTIPATH_ENABLED': True,  # 启用多路径
        'MAX_PATHS': 3,  # 最大路径数
        'PATH_SELECTION_STRATEGY': 'adaptive'  # 路径选择策略
    }

    # 创建测试目录
    os.makedirs('test_results', exist_ok=True)

    # 创建测试框架
    tester = RoutingProtocolTester(base_config)

    # 运行所有测试
    logging.info("开始路由协议比较测试")
    results = tester.run_all_tests()
    logging.info("测试完成，开始生成结果")

    # 可视化结果
    tester.visualize_results()

    # 生成报告
    report = tester.generate_report()
    with open('test_results/routing_comparison_report.md', 'w', encoding='utf-8') as f:
        f.write(report)

    logging.info(f"测试报告已保存到: test_results/routing_comparison_report.md")
    logging.info(f"测试图表已保存到: test_results/ 目录")


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('routing_test.log', mode='w'),
            # logging.StreamHandler()
        ]
    )

    try:
        main()
        print("测试完成！结果保存在test_results目录下。")
    except Exception as e:
        logging.error(f"测试过程中发生错误: {str(e)}", exc_info=True)
        print(f"测试失败: {str(e)}")