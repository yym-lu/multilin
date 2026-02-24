#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
计算多个种子训练结果的平均值
"""

import json
import os
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='计算多个种子训练结果的平均值')
    parser.add_argument('model_path', type=str, help='模型保存路径')
    return parser.parse_args()

def calculate_average_results(model_path):
    """
    计算多个种子的平均结果
    """
    print(f"计算模型 {os.path.basename(model_path)} 的平均结果...")
    
    # 查找所有种子目录
    seed_dirs = []
    for item in os.listdir(model_path):
        item_path = os.path.join(model_path, item)
        if os.path.isdir(item_path) and item.startswith('seed_'):
            seed_dirs.append(item_path)
    
    if not seed_dirs:
        print(f"错误: 未找到种子目录在 {model_path}")
        return False
    
    print(f"找到 {len(seed_dirs)} 个种子目录: {[os.path.basename(d) for d in seed_dirs]}")
    
    # 加载所有种子的训练损失和评估指标
    all_training_loss = []
    all_eval_metrics = []
    
    for seed_dir in seed_dirs:
        # 加载训练损失
        loss_file = os.path.join(seed_dir, 'training_loss.json')
        if os.path.exists(loss_file):
            with open(loss_file, 'r', encoding='utf-8') as f:
                loss_data = json.load(f)
                all_training_loss.append(loss_data)
        else:
            print(f"警告: 未找到训练损失文件: {loss_file}")
            return False
        
        # 加载评估指标
        eval_file = os.path.join(seed_dir, 'eval_metrics.json')
        if os.path.exists(eval_file):
            with open(eval_file, 'r', encoding='utf-8') as f:
                eval_data = json.load(f)
                all_eval_metrics.append(eval_data)
        else:
            print(f"警告: 未找到评估指标文件: {eval_file}")
            return False
    
    # 计算平均训练损失
    print("计算平均训练损失...")
    avg_training_loss = []
    
    # 假设所有种子的训练步数相同
    if all_training_loss:
        num_steps = len(all_training_loss[0])
        for step in range(num_steps):
            step_losses = []
            step_epochs = []
            step_lrs = []
            
            for seed_loss in all_training_loss:
                if step < len(seed_loss):
                    step_losses.append(seed_loss[step]['loss'])
                    step_epochs.append(seed_loss[step]['epoch'])
                    step_lrs.append(seed_loss[step]['learning_rate'])
            
            avg_loss = np.mean(step_losses)
            avg_epoch = np.mean(step_epochs)
            avg_lr = np.mean(step_lrs)
            
            avg_training_loss.append({
                'step': step + 1,
                'epoch': avg_epoch,
                'loss': avg_loss,
                'learning_rate': avg_lr
            })
    
    # 计算平均评估指标
    print("计算平均评估指标...")
    avg_eval_metrics = []
    
    if all_eval_metrics:
        num_epochs = len(all_eval_metrics[0])
        for epoch in range(num_epochs):
            epoch_precisions = []
            epoch_recalls = []
            epoch_f1s = []
            epoch_accuracies = []
            epoch_losses = []
            epoch_steps = []
            epoch_epochs = []
            
            for seed_eval in all_eval_metrics:
                if epoch < len(seed_eval):
                    epoch_precisions.append(seed_eval[epoch]['precision'])
                    epoch_recalls.append(seed_eval[epoch]['recall'])
                    epoch_f1s.append(seed_eval[epoch]['f1'])
                    epoch_accuracies.append(seed_eval[epoch]['accuracy'])
                    epoch_losses.append(seed_eval[epoch]['loss'])
                    epoch_steps.append(seed_eval[epoch]['step'])
                    epoch_epochs.append(seed_eval[epoch]['epoch'])
            
            avg_precision = np.mean(epoch_precisions)
            avg_recall = np.mean(epoch_recalls)
            avg_f1 = np.mean(epoch_f1s)
            avg_accuracy = np.mean(epoch_accuracies)
            avg_loss = np.mean(epoch_losses)
            avg_step = np.mean(epoch_steps)
            avg_epoch_num = np.mean(epoch_epochs)
            
            avg_eval_metrics.append({
                'step': avg_step,
                'epoch': avg_epoch_num,
                'precision': avg_precision,
                'recall': avg_recall,
                'f1': avg_f1,
                'accuracy': avg_accuracy,
                'loss': avg_loss
            })
    
    # 保存平均结果
    print("保存平均结果...")
    
    # 保存平均训练损失
    avg_loss_file = os.path.join(model_path, 'training_loss.json')
    with open(avg_loss_file, 'w', encoding='utf-8') as f:
        json.dump(avg_training_loss, f, ensure_ascii=False, indent=2)
    print(f"  平均训练损失已保存到: {avg_loss_file}")
    
    # 保存平均评估指标
    avg_eval_file = os.path.join(model_path, 'eval_metrics.json')
    with open(avg_eval_file, 'w', encoding='utf-8') as f:
        json.dump(avg_eval_metrics, f, ensure_ascii=False, indent=2)
    print(f"  平均评估指标已保存到: {avg_eval_file}")
    
    # 复制最佳模型（使用F1分数最高的种子）
    print("复制最佳模型...")
    
    # 找出F1分数最高的种子
    best_f1 = 0.0
    best_seed_dir = None
    
    for seed_dir in seed_dirs:
        eval_file = os.path.join(seed_dir, 'eval_metrics.json')
        with open(eval_file, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
            # 取最后一个epoch的F1分数
            last_f1 = eval_data[-1]['f1'] if eval_data else 0.0
            
            if last_f1 > best_f1:
                best_f1 = last_f1
                best_seed_dir = seed_dir
    
    if best_seed_dir:
        # 复制最佳模型
        best_model_src = os.path.join(best_seed_dir, 'best_model')
        best_model_dst = os.path.join(model_path, 'best_model')
        
        if os.path.exists(best_model_dst):
            # 删除旧的最佳模型
            import shutil
            shutil.rmtree(best_model_dst)
        
        # 复制新的最佳模型
        import shutil
        shutil.copytree(best_model_src, best_model_dst)
        print(f"  最佳模型已从 {os.path.basename(best_seed_dir)} (F1={best_f1:.4f}) 复制到: {best_model_dst}")
    
    print("平均结果计算完成！")
    return True

def main():
    args = parse_args()
    success = calculate_average_results(args.model_path)
    if not success:
        exit(1)

if __name__ == '__main__':
    main()
