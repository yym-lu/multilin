#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试脚本
用于在测试集上评估训练好的模型
"""

import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer
from peft import PeftModel
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class HateSpeechDataset(Dataset):
    """
    仇恨言论检测数据集类
    """
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = f"{item['instruction']} {item['input']}"
        label = 1 if item['output'] == "Yes" or item['output'] == "是" else 0
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
            "raw_text": text,
            "raw_output": item['output'],
            "instruction": item['instruction'],
            "input": item['input']
        }

def parse_args():
    parser = argparse.ArgumentParser(description='在测试集上评估训练好的模型')
    parser.add_argument('--test_file', type=str, 
                      default='dataset/COLDATAtest_1k.json',
                      help='测试集文件路径')
    parser.add_argument('--model_names', type=str, nargs='+',
                      default=['colda_8k', 'colda_4k', 'colda_20k', 'olid_13k', 'colda_4k_olid_13k', 'colda_20k_olid_13k', 'colda_8k_olid_13k'],
                      help='要测试的模型名称列表')
    parser.add_argument('--models_dir', type=str, 
                      default='models',
                      help='模型保存目录')
    parser.add_argument('--base_model_path', type=str, 
                      default='models/xlm-roberta-base',
                      help='基础模型路径')
    parser.add_argument('--output_dir', type=str, 
                      default='test_results',
                      help='测试结果保存目录')
    parser.add_argument('--batch_size', type=int, default=16, help='测试批次大小')
    parser.add_argument('--max_length', type=int, default=128, help='最大序列长度')
    return parser.parse_args()

def load_model(model_path, base_model_path):
    """
    加载训练好的模型
    """
    # 加载基础模型
    tokenizer = XLMRobertaTokenizer.from_pretrained(base_model_path)
    model = XLMRobertaForSequenceClassification.from_pretrained(
        base_model_path,
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    
    # 加载LoRA权重
    model = PeftModel.from_pretrained(model, os.path.join(model_path, "best_model"))
    
    # 设置为评估模式
    model.eval()
    
    return model, tokenizer

def test_model(model, tokenizer, test_data, args):
    """
    在测试集上测试模型
    """
    # 创建数据集和数据加载器
    test_dataset = HateSpeechDataset(test_data, tokenizer, max_length=args.max_length)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 初始化评估结果
    all_predictions = []
    all_labels = []
    all_raw_texts = []
    all_raw_outputs = []
    all_instructions = []
    all_inputs = []
    
    # 进行预测
    with torch.no_grad():
        for batch in test_dataloader:
            # 准备数据
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # 前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # 获取预测结果
            predictions = torch.argmax(logits, dim=-1)
            
            # 保存结果
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_raw_texts.extend(batch["raw_text"])
            all_raw_outputs.extend(batch["raw_output"])
            all_instructions.extend(batch["instruction"])
            all_inputs.extend(batch["input"])
    
    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, pos_label=1, zero_division=0)
    recall = recall_score(all_labels, all_predictions, pos_label=1, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, pos_label=1, zero_division=0)
    
    # 生成预测结果列表
    results = []
    for i in range(len(test_data)):
        results.append({
            "instruction": all_instructions[i],
            "input": all_inputs[i],
            "true_output": all_raw_outputs[i],
            "true_label": int(all_labels[i]),
            "predicted_label": int(all_predictions[i]),
            "predicted_output": "是" if all_predictions[i] == 1 else "否"
        })
    
    return {
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "total_samples": len(test_data)
        },
        "predictions": results
    }

def main():
    args = parse_args()
    
    print("开始在测试集上评估模型")
    print(f"测试集文件: {args.test_file}")
    print(f"模型目录: {args.models_dir}")
    print(f"处理的模型: {args.model_names}")
    print(f"结果保存目录: {args.output_dir}")
    print("=" * 70)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载测试数据集
    print(f"\n加载测试数据集...")
    with open(args.test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    print(f"测试集数据量: {len(test_data)}条")
    
    # 统计测试集标签分布
    test_yes = sum(1 for item in test_data if item['output'] == 'Yes' or item['output'] == '是')
    test_no = sum(1 for item in test_data if item['output'] == 'No' or item['output'] == '否')
    print(f"测试集标签分布：")
    print(f"  仇恨言论 (yes): {test_yes} ({test_yes/len(test_data):.2%})")
    print(f"  非仇恨言论 (no): {test_no} ({test_no/len(test_data):.2%})")
    
    # 测试所有模型
    all_metrics = []
    
    for model_name in args.model_names:
        print(f"\n" + "=" * 70)
        print(f"测试模型: {model_name}")
        print("=" * 70)
        
        # 构建模型路径
        model_path = os.path.join(args.models_dir, model_name)
        
        # 检查模型是否存在
        if not os.path.exists(model_path):
            print(f"错误: 找不到模型目录: {model_path}")
            continue
            
        # 检查最佳模型是否存在
        best_model_path = os.path.join(model_path, "best_model")
        if not os.path.exists(best_model_path):
            print(f"错误: 找不到最佳模型: {best_model_path}")
            continue
        
        # 加载模型
        print(f"加载模型...")
        try:
            model, tokenizer = load_model(model_path, args.base_model_path)
        except Exception as e:
            print(f"错误: 加载模型失败: {e}")
            continue
        
        # 测试模型
        print(f"开始测试...")
        test_result = test_model(model, tokenizer, test_data, args)
        
        # 保存测试结果
        model_output_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # 保存详细预测结果
        predictions_file = os.path.join(model_output_dir, "predictions.json")
        with open(predictions_file, "w", encoding="utf-8") as f:
            json.dump(test_result["predictions"], f, ensure_ascii=False, indent=2)
        
        # 保存评估指标
        metrics_file = os.path.join(model_output_dir, "metrics.json")
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(test_result["metrics"], f, ensure_ascii=False, indent=2)
        
        # 保存到总指标列表
        model_metrics = test_result["metrics"]
        model_metrics["model_name"] = model_name
        all_metrics.append(model_metrics)
        
        # 打印评估结果
        print(f"\n模型: {model_name}")
        print(f"准确率: {model_metrics['accuracy']:.4f}")
        print(f"精确率: {model_metrics['precision']:.4f}")
        print(f"召回率: {model_metrics['recall']:.4f}")
        print(f"F1分数: {model_metrics['f1']:.4f}")
        print(f"测试样本数: {model_metrics['total_samples']}")
        
        print(f"\n预测结果已保存到: {predictions_file}")
        print(f"评估指标已保存到: {metrics_file}")
    
    # 保存所有模型的评估指标
    all_metrics_file = os.path.join(args.output_dir, "all_metrics.json")
    with open(all_metrics_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    
    # 打印所有模型的评估结果对比
    print(f"\n" + "=" * 70)
    print("所有模型的评估结果对比")
    print("=" * 70)
    print(f"{'模型名称':<25} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10}")
    print("-" * 70)
    
    for metrics in all_metrics:
        print(f"{metrics['model_name']:<25} {metrics['accuracy']:.4f}    {metrics['precision']:.4f}    {metrics['recall']:.4f}    {metrics['f1']:.4f}")
    
    print(f"\n所有模型的评估结果已保存到: {all_metrics_file}")
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)

if __name__ == '__main__':
    main()
