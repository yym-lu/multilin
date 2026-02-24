#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成训练损失和验证F1分数随step变化的曲线图
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import argparse

def calculate_epoch(step, total_steps, total_epochs):
    """
    将step转换为对应的epoch
    
    Args:
        step: 当前训练步数
        total_steps: 总训练步数
        total_epochs: 总训练轮次
    
    Returns:
        epoch: 对应的epoch数
    """
    return (step / total_steps) * total_epochs

def plot_metrics(model_path, output_dir):
    """
    为指定模型绘制训练损失和验证F1分数曲线图
    
    Args:
        model_path: 模型目录路径
        output_dir: 图表保存目录
    """
    model_name = os.path.basename(model_path)
    print(f"处理模型: {model_name}")
    
    # 加载训练损失数据
    loss_file = os.path.join(model_path, "training_loss.json")
    if not os.path.exists(loss_file):
        print(f"错误: 找不到训练损失文件: {loss_file}")
        return
    
    with open(loss_file, 'r', encoding='utf-8') as f:
        loss_data = json.load(f)
    
    # 加载评估指标数据
    metrics_file = os.path.join(model_path, "eval_metrics.json")
    if not os.path.exists(metrics_file):
        print(f"错误: 找不到评估指标文件: {metrics_file}")
        return
    
    with open(metrics_file, 'r', encoding='utf-8') as f:
        metrics_data = json.load(f)
    
    # 提取训练损失数据
    train_steps = []
    train_losses = []
    for item in loss_data:
        train_steps.append(item["step"])
        train_losses.append(item["loss"])
    
    # 提取验证F1数据
    eval_steps = []
    eval_f1 = []
    for item in metrics_data:
        eval_steps.append(item["step"])
        eval_f1.append(item["f1"])
    
    print(f"训练步数: {len(train_steps)}")
    print(f"评估步数: {len(eval_steps)}")
    
    # 计算总训练步数和总epoch数
    total_steps = max(train_steps)
    total_epochs = 5  # 根据训练配置，总epoch数为5
    
    # 将step转换为epoch
    train_epochs = [calculate_epoch(step, total_steps, total_epochs) for step in train_steps]
    eval_epochs = [calculate_epoch(step, total_steps, total_epochs) for step in eval_steps]
    
    # 计算F1分数的范围，设置合理的纵轴范围
    min_f1 = min(eval_f1)
    max_f1 = max(eval_f1)
    # 扩展5%的范围，确保数据点不会贴在图表边缘
    f1_range_min = max(0.6, min_f1 - 0.01)  # 将最小值改为0.6，确保所有F1值都能显示
    f1_range_max = min(1.0, max_f1 + 0.01)
    
    # 寻找最佳模型位置（early stopping点）
    if metrics_data:
        best_idx = eval_f1.index(max(eval_f1))
        best_epoch = eval_epochs[best_idx]
        best_f1 = eval_f1[best_idx]
    else:
        best_epoch = None
        best_f1 = None
    
    # 只在横轴的11个标签位置标注训练损失
    target_epochs = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    filtered_train_epochs = []
    filtered_train_losses = []
    
    # 对于每个目标epoch，找到最接近的训练点
    for target_epoch in target_epochs:
        # 找到最接近目标epoch的训练点
        closest_idx = min(range(len(train_epochs)), key=lambda i: abs(train_epochs[i] - target_epoch))
        filtered_train_epochs.append(train_epochs[closest_idx])
        filtered_train_losses.append(train_losses[closest_idx])
    
    # 创建图表
    plt.figure(figsize=(12, 10))
    
    # 直接指定中文字体文件路径
    try:
        # 尝试使用MacOS系统字体
        font_names = ['Arial Unicode MS', 'Heiti TC', 'STHeiti', 'PingFang SC', 'Hiragino Sans GB']
        zh_font = None
        zh_font_large = None
        
        # 查找可用字体
        available_fonts = set(f.name for f in fm.fontManager.ttflist)
        for font_name in font_names:
            if font_name in available_fonts:
                zh_font = fm.FontProperties(family=font_name, size=12)
                zh_font_large = fm.FontProperties(family=font_name, size=14)
                print(f"使用字体: {font_name}")
                break
                
        if zh_font is None:
             # 如果找不到常用中文字体，尝试使用系统默认sans-serif并设置fallback
            print("警告: 未找到常用中文字体，尝试使用默认配置")
            zh_font = fm.FontProperties(size=12)
            zh_font_large = fm.FontProperties(size=14)
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False

    except Exception as e:
        # 如果失败，使用默认字体
        print(f"警告: 加载字体失败: {e}，将使用默认字体")
        zh_font = fm.FontProperties(size=12)
        zh_font_large = fm.FontProperties(size=14)
    
    # 绘制训练损失曲线
    ax1 = plt.subplot(211)
    # 使用过滤后的数据绘制曲线
    ax1.plot(filtered_train_epochs, filtered_train_losses, 'b-', linewidth=2, marker='o', markersize=5)
    ax1.set_ylabel('训练损失', fontproperties=zh_font)
    ax1.set_title(f'{model_name} 训练损失和验证F1分数随Epoch变化', fontproperties=zh_font_large)
    ax1.grid(True, alpha=0.3)
    
    # 添加图例
    ax1.legend(['训练损失'], prop=zh_font)
    
    # 设置合适的纵轴范围
    loss_min = min(filtered_train_losses)
    loss_max = max(filtered_train_losses)
    loss_range_padding = (loss_max - loss_min) * 0.1
    ax1.set_ylim(max(0, loss_min - loss_range_padding), loss_max + loss_range_padding)
    
    # 设置横轴刻度
    ax1.set_xticks(target_epochs)
    ax1.set_xticklabels([f'{tick:.1f}' for tick in target_epochs], fontproperties=zh_font)
    
    # 绘制验证F1曲线
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(eval_epochs, eval_f1, 'r-', linewidth=2, label='验证F1')
    ax2.set_xlabel('训练轮次 (Epoch)', fontproperties=zh_font)
    ax2.set_ylabel('F1分数', fontproperties=zh_font)
    ax2.grid(True, alpha=0.3)
    
    # 设置F1分数的纵轴范围
    ax2.set_ylim(f1_range_min, f1_range_max)
    
    # 添加图例
    ax2.legend(prop=zh_font)
    
    # 添加early stopping点
    if best_epoch is not None:
        ax2.axvline(x=best_epoch, color='orange', linestyle='--', linewidth=1.5, label='Early Stopping')
        ax2.scatter(best_epoch, best_f1, color='red', s=50, zorder=5)
        ax2.text(best_epoch + 0.1, best_f1 - 0.001, 
                 f'最佳F1: {best_f1:.4f}', 
                 ha='left', va='top', 
                 fontproperties=zh_font, 
                 bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))
        ax2.legend(prop=zh_font)
    
    # 设置横轴刻度
    ax2.set_xticks(target_epochs)
    ax2.set_xticklabels([f'{tick:.1f}' for tick in target_epochs], fontproperties=zh_font)
    
    # 保存图表
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{model_name}_metrics.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    print(f"图表已保存到: {output_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='生成训练损失和验证F1分数曲线图')
    parser.add_argument('--models_dir', type=str, 
                      default='models',
                      help='模型保存目录')
    parser.add_argument('--output_dir', type=str, 
                      default='visualization',
                      help='图表保存目录')
    parser.add_argument('--model_names', type=str, nargs='+',
                      default=['colda_8k', 'colda_4k', 'colda_20k', 'olid_13k', 'colda_4k_olid_13k', 'colda_20k_olid_13k', 'colda_8k_olid_13k'],
                      help='要处理的模型名称列表')
    
    args = parser.parse_args()
    
    print("开始生成训练损失和验证F1分数曲线图")
    print(f"模型目录: {args.models_dir}")
    print(f"图表保存目录: {args.output_dir}")
    print(f"处理的模型: {args.model_names}")
    print("=" * 60)
    
    for model_name in args.model_names:
        model_path = os.path.join(args.models_dir, model_name)
        if os.path.exists(model_path):
            plot_metrics(model_path, args.output_dir)
        else:
            print(f"警告: 模型目录不存在: {model_path}")
    
    print("=" * 60)
    print("所有模型的图表生成完成！")

if __name__ == "__main__":
    main()