#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MP_OLSR与AMLB_OPAR路由协议比较整合启动脚本
执行测试、生成报告和分析结果
"""

import os
import sys
import time
import logging
import pickle
import argparse
from datetime import datetime

# 配置命令行参数
parser = argparse.ArgumentParser(description='无人机网络多路径路由协议比较测试工具')
parser.add_argument('--only-analyze', action='store_true', help='仅分析现有结果，不运行新测试')
parser.add_argument('--drones', type=int, default=10, help='无人机数量')
parser.add_argument('--sim-time', type=float, default=5, help='仿真时间(秒)')
parser.add_argument('--max-paths', type=int, default=3, help='最大路径数')
parser.add_argument('--scenarios', nargs='+', default=['basic', 'dense', 'mobile', 'failure'],
                    help='要测试的场景: basic, dense, mobile, failure')
args = parser.parse_args()

# 配置日志
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'routing_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        # logging.StreamHandler(sys.stdout)
    ]
)

# 创建结果目录
results_dir = 'test_results'
os.makedirs(results_dir, exist_ok=True)


def run_tests():
    """运行测试并保存结果"""
    logging.info("开始路由协议比较测试")
    start_time = time.time()

    # 导入测试模块
    from utils import config
    from routing_compare import RoutingProtocolTester

    # 配置测试参数
    base_config = {
        'NUMBER_OF_DRONES': args.drones,
        'SIM_TIME': args.sim_time * 1e6,  # 转换为微秒
        'MULTIPATH_ENABLED': True,
        'MAX_PATHS': args.max_paths,
        'PATH_SELECTION_STRATEGY': 'adaptive'
    }

    # 创建测试框架
    tester = RoutingProtocolTester(base_config)

    # 运行测试
    logging.info(f"测试配置: 无人机数量={args.drones}, 仿真时间={args.sim_time}秒, 最大路径数={args.max_paths}")
    logging.info(f"测试场景: {', '.join(args.scenarios)}")

    # 根据选择的场景运行测试
    selected_scenarios = set(args.scenarios)
    results = {}

    if 'basic' in selected_scenarios:
        logging.info("运行基本场景测试")
        results['mp_olsr_basic'] = tester.run_protocol_test('MP_OLSR', 'basic')
        results['amlb_opar_basic'] = tester.run_protocol_test('AMLB_OPAR', 'basic')

    if 'dense' in selected_scenarios:
        logging.info("运行高密度场景测试")
        results['mp_olsr_dense'] = tester.run_protocol_test(
            'MP_OLSR', 'dense', override_config={'NUMBER_OF_DRONES': 10}
        )
        results['amlb_opar_dense'] = tester.run_protocol_test(
            'AMLB_OPAR', 'dense', override_config={'NUMBER_OF_DRONES': 10}
        )

    if 'mobile' in selected_scenarios:
        logging.info("运行高移动性场景测试")
        results['mp_olsr_mobile'] = tester.run_protocol_test(
            'MP_OLSR', 'mobile', override_config={'MOBILITY_SPEED': 10}
        )
        results['amlb_opar_mobile'] = tester.run_protocol_test(
            'AMLB_OPAR', 'mobile', override_config={'MOBILITY_SPEED': 10}
        )

    if 'failure' in selected_scenarios:
        logging.info("运行故障恢复场景测试")
        results['mp_olsr_failure'] = tester.run_failure_test('MP_OLSR')
        results['amlb_opar_failure'] = tester.run_failure_test('AMLB_OPAR')

    # 保存结果
    results_file = os.path.join(results_dir, 'results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)

    logging.info(f"测试结果已保存到: {results_file}")

    # 可视化结果
    tester.visualize_results()

    # 生成报告
    report = tester.generate_report()
    report_file = os.path.join(results_dir, 'routing_comparison_report.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    logging.info(f"测试报告已保存到: {report_file}")
    logging.info(f"测试图表已保存到: {results_dir}/ 目录")

    end_time = time.time()
    logging.info(f"测试完成，耗时: {end_time - start_time:.2f} 秒")

    return results


def analyze_results():
    """分析测试结果"""
    logging.info("开始分析测试结果")

    try:
        # 导入分析器
        from result_analyzer import ResultAnalyzer

        # 创建分析器
        analyzer = ResultAnalyzer(os.path.join(results_dir, 'results.pkl'))

        # 运行所有分析
        analyzer.analyze_all()

        logging.info(f"分析结果已保存到: {results_dir}/analysis/ 目录")

    except Exception as e:
        logging.error(f"分析过程中发生错误: {str(e)}", exc_info=True)


def main():
    """主函数"""
    print("=" * 80)
    print(" MP_OLSR与AMLB_OPAR路由协议比较测试工具 ".center(80, '='))
    print("=" * 80)

    # 创建结果目录
    if not args.only_analyze:
        # 运行测试
        run_tests()
    else:
        logging.info("跳过测试，仅分析现有结果")

    # 分析结果
    analyze_results()

    print("=" * 80)
    print(" 测试与分析完成 ".center(80, '='))
    print("=" * 80)
    print(f"日志文件: {log_file}")
    print(f"测试报告: {os.path.join(results_dir, 'routing_comparison_report.md')}")
    print(f"详细分析: {os.path.join(results_dir, 'analysis/detailed_analysis_report.md')}")
    print(f"图表结果: {results_dir}/ 和 {results_dir}/analysis/ 目录")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        logging.info("程序被用户中断")
    except Exception as e:
        print(f"\n程序执行出错: {str(e)}")
        logging.error(f"程序执行出错: {str(e)}", exc_info=True)
    finally:
        print("\n程序结束")