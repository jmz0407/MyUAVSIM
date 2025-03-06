# 示例使用代码

from routing.multipath.multipath_test import MultipathTestSuite

# 创建测试套件
test_suite = MultipathTestSuite()

# 运行所有测试场景
results = test_suite.run_test_scenarios()

# 可视化结果
test_suite.visualize_results()

# 针对故障恢复场景的可视化
test_suite.visualize_failure_recovery()

# 生成报告
report = test_suite.generate_report()
with open('multipath_test_report.md', 'w') as f:
    f.write(report)