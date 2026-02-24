#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
绘制增益曲线图和成本效益图
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

def load_test_results(results_file):
    """
    加载测试结果
    """
    if not os.path.exists(results_file):
        print(f"错误: 找不到测试结果文件: {results_file}")
        return None
    
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_gain_curve(results, output_dir):
    """
    绘制增益曲线图
    """
    print("绘制增益曲线图...")
    
    # 整理数据
    # 中文数据量 (4k, 8k, 20k)
    chinese_data_sizes = ["4k", "8k", "20k"]
    
    # 提取仅中文模型的F1分数
    chinese_only_f1 = []
    model_names = [f"colda_{size}" for size in chinese_data_sizes]
    for model_name in model_names:
        model_result = next((item for item in results if item["model_name"] == model_name), None)
        if model_result:
            chinese_only_f1.append(model_result["f1"])
    
    # 提取混合模型的F1分数（包括4k、8k、20k的混合模型）
    hybrid_f1 = []
    for size in chinese_data_sizes:
        model_name = f"colda_{size}_olid_13k"
        model_result = next((item for item in results if item["model_name"] == model_name), None)
        if model_result:
            hybrid_f1.append(model_result["f1"])
        else:
            hybrid_f1.append(None)
    
    # 设置中文字体
    # 优先使用 macOS 常见中文字体
    font_names = ['Arial Unicode MS', 'Heiti TC', 'PingFang HK', 'STHeiti', 'SimHei']
    
    # 配置 matplotlib 全局字体
    plt.rcParams['font.sans-serif'] = font_names + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 同时也尝试创建一个 FontProperties 对象
    zh_font = None
    for name in font_names:
        try:
             # 简单地创建 FontProperties，让 matplotlib 去找
            zh_font = fm.FontProperties(family=name, size=12)
            zh_font_large = fm.FontProperties(family=name, size=14)
            # 简单测试一下是否能找到
            if fm.findfont(zh_font) != fm.findfont('default_impossible_font'):
                print(f"使用字体: {name}")
                break
        except:
            continue

    if zh_font is None:
        print("警告: 未找到推荐的中文字体，尝试使用系统默认")
        zh_font = fm.FontProperties(size=12)
        zh_font_large = fm.FontProperties(size=14)
    
    # 创建图表
    plt.figure(figsize=(10, 6))
    
    # 绘制曲线
    x = np.arange(len(chinese_data_sizes))
    width = 0.35
    
    # 绘制仅中文数据的柱状图和曲线
    plt.bar(x - width/2, chinese_only_f1, width, label='仅中文数据', alpha=0.8, color='blue')
    plt.plot(x, chinese_only_f1, 'b-', linewidth=2, marker='o', markersize=6)
    
    # 绘制混合数据的柱状图和曲线（仅4k和20k）
    hybrid_f1_filtered = [f1 for f1 in hybrid_f1 if f1 is not None]
    x_hybrid = [i for i, f1 in enumerate(hybrid_f1) if f1 is not None]
    plt.bar([x + width/2 for x in x_hybrid], hybrid_f1_filtered, width, label='中文+英文混合数据', alpha=0.8, color='green')
    plt.plot(x_hybrid, hybrid_f1_filtered, 'g-', linewidth=2, marker='s', markersize=6)
    
    # 设置标题和标签
    plt.title('中文数据量对模型性能的影响', fontproperties=zh_font_large)
    plt.xlabel('中文数据量', fontproperties=zh_font)
    plt.ylabel('测试F1分数', fontproperties=zh_font)
    
    # 设置横轴刻度
    plt.xticks(x, chinese_data_sizes, fontproperties=zh_font)
    
    # 减小纵轴的跨度，使对比更明显
    min_f1 = min(chinese_only_f1 + hybrid_f1_filtered)
    max_f1 = max(chinese_only_f1 + hybrid_f1_filtered)
    f1_range_padding = (max_f1 - min_f1) * 0.05
    plt.ylim(min_f1 - f1_range_padding, max_f1 + f1_range_padding)
    
    # 添加网格
    plt.grid(True, alpha=0.3)
    
    # 添加图例
    plt.legend(prop=zh_font)
    
    # 保存图表
    output_file = os.path.join(output_dir, 'gain_curve.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"增益曲线图已保存到: {output_file}")

def plot_cost_benefit(results, output_dir):
    """
    绘制成本效益图
    """
    print("绘制成本效益图...")
    
    # 计算成本数据
    # 中文标注成本: $0.5每条
    cost_per_sample = 0.5
    
    # 纯中文标注方案
    # 数据量: 4k, 8k, 20k
    chinese_sizes = [4000, 8000, 20000]
    chinese_costs = [size * cost_per_sample for size in chinese_sizes]
    
    # 提取纯中文模型的F1分数
    chinese_only_f1 = []
    model_names = [f"colda_{size//1000}k" for size in chinese_sizes]
    for model_name in model_names:
        model_result = next((item for item in results if item["model_name"] == model_name), None)
        if model_result:
            chinese_only_f1.append(model_result["f1"])
    
    # 英文迁移+少量中文方案
    # 中文数据: 4k, 8k, 20k（去掉中文数据为0的点）
    mixed_sizes = [4000, 8000, 20000]
    mixed_costs = [size * cost_per_sample for size in mixed_sizes]
    
    # 提取混合模型的F1分数
    mixed_f1 = []
    mixed_costs_filtered = []
    model_names = [f"colda_{size//1000}k_olid_13k" for size in mixed_sizes]
    for i, model_name in enumerate(model_names):
        model_result = next((item for item in results if item["model_name"] == model_name), None)
        if model_result:
            mixed_f1.append(model_result["f1"])
            mixed_costs_filtered.append(mixed_costs[i])
    
    # 设置中文字体
    # 优先使用 macOS 常见中文字体
    font_names = ['Arial Unicode MS', 'Heiti TC', 'PingFang HK', 'STHeiti', 'SimHei']
    
    # 配置 matplotlib 全局字体
    plt.rcParams['font.sans-serif'] = font_names + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 同时也尝试创建一个 FontProperties 对象
    zh_font = None
    for name in font_names:
        try:
             # 简单地创建 FontProperties，让 matplotlib 去找
            zh_font = fm.FontProperties(family=name, size=12)
            zh_font_large = fm.FontProperties(family=name, size=14)
            # 简单测试一下是否能找到
            if fm.findfont(zh_font) != fm.findfont('default_impossible_font'):
                print(f"使用字体: {name}")
                break
        except:
            continue

    if zh_font is None:
        print("警告: 未找到推荐的中文字体，尝试使用系统默认")
        zh_font = fm.FontProperties(size=12)
        zh_font_large = fm.FontProperties(size=14)
    
    # 创建图表
    plt.figure(figsize=(10, 6))
    
    # 绘制曲线
    plt.plot(chinese_costs, chinese_only_f1, 'b-', linewidth=2, marker='o', markersize=6, label='纯中文标注')
    plt.plot(mixed_costs_filtered, mixed_f1, 'g-', linewidth=2, marker='s', markersize=6, label='英文迁移+中文标注')
    
    # 设置标题和标签
    plt.title('中文标注成本与模型性能的关系', fontproperties=zh_font_large)
    plt.xlabel(f'中文标注成本 (美元，${cost_per_sample}/条)', fontproperties=zh_font)
    plt.ylabel('测试F1分数', fontproperties=zh_font)
    
    # 缩短纵轴的跨度，使对比更明显
    # 获取所有F1分数的范围
    all_f1 = chinese_only_f1 + mixed_f1
    min_f1 = min(all_f1)
    max_f1 = max(all_f1)
    f1_range_padding = (max_f1 - min_f1) * 0.03  # 仅保留3%的边距
    plt.ylim(min_f1 - f1_range_padding, max_f1 + f1_range_padding)
    
    # 添加网格
    plt.grid(True, alpha=0.3)
    
    # 添加图例
    plt.legend(prop=zh_font)
    
    # 保存图表
    output_file = os.path.join(output_dir, 'cost_benefit_curve.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"成本效益图已保存到: {output_file}")

def main():
    # 获取当前脚本所在目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 设置结果文件路径
    results_file = os.path.join(base_dir, 'test_results', 'all_metrics.json')
    output_dir = os.path.join(base_dir, 'visualization')
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载测试结果
    print("加载测试结果...")
    results = load_test_results(results_file)
    
    if results:
        # 绘制增益曲线图
        plot_gain_curve(results, output_dir)
        
        # 绘制成本效益图
        plot_cost_benefit(results, output_dir)
        
        print("\n所有图表绘制完成！")
    else:
        print("错误: 无法加载测试结果，图表绘制失败")

if __name__ == '__main__':
    main()
