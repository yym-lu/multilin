#!/bin/bash

# 检查测试结果是否存在
if [ ! -f "test_results/all_metrics.json" ]; then
    echo "测试结果文件 test_results/all_metrics.json 不存在。"
    echo "请等待 test_models.py 运行完成。"
    exit 1
fi

echo "开始生成可视化图表..."
python3 plot_gain_cost.py

echo "可视化完成！图表保存在 visualization 目录中。"
