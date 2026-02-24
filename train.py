import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import time
import argparse
import csv
import os

# 1. 数据集类
class HateSpeechDataset(Dataset):
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
            "raw_output": item['output']
        }

# 解析命令行参数
parser = argparse.ArgumentParser(description='训练德语/英语仇恨言论检测模型')
parser.add_argument('--config', type=str, required=True, choices=['config1', 'config2', 'config3', 'config4', 'config5', 'config6', 'config7'], 
                    help='训练配置')
parser.add_argument('--save_path', type=str, required=True, help='模型保存路径')
parser.add_argument('--seed', type=int, default=42, choices=[42, 43, 44, 45, 46], help='随机种子')
args = parser.parse_args()

# 创建保存目录
os.makedirs(args.save_path, exist_ok=True)

# 定义七种配置（七个模型）
CONFIGS = {
    'config1': {
        'name': 'colda_8k',
        'datasets': ['dataset/COLDATAtrain_8k.json'],
        'description': 'COLDATAtrain_8k.json (8k数据)'
    },
    'config2': {
        'name': 'colda_4k',
        'datasets': ['dataset/COLDATAtrain_4k.json'],
        'description': 'COLDATAtrain_4k.json (4k数据)'
    },
    'config3': {
        'name': 'colda_20k',
        'datasets': ['dataset/COLDATAtrain_20k.json'],
        'description': 'COLDATAtrain_20k.json (20k数据)'
    },
    'config4': {
        'name': 'olid_13k',
        'datasets': ['dataset/olid_13k.json'],
        'description': 'olid_13k.json (13k数据)'
    },
    'config5': {
        'name': 'colda_4k_olid_13k',
        'datasets': ['dataset/COLDATAtrain_4k.json', 
                   'dataset/olid_13k.json'],
        'description': 'COLDATAtrain_4k.json + olid_13k.json (4k+13k数据)'
    },
    'config6': {
        'name': 'colda_20k_olid_13k',
        'datasets': ['dataset/COLDATAtrain_20k.json', 
                   'dataset/olid_13k.json'],
        'description': 'COLDATAtrain_20k.json + olid_13k.json (20k+13k数据)'
    },
    'config7': {
        'name': 'colda_8k_olid_13k',
        'datasets': ['dataset/COLDATAtrain_8k.json', 
                   'dataset/olid_13k.json'],
        'description': 'COLDATAtrain_8k.json + olid_13k.json (8k+13k数据)'
    }
}

# 获取当前配置
config = CONFIGS[args.config]
print(f"使用配置: {args.config} ({config['name']})")
print(f"配置描述: {config['description']}")
print(f"数据集: {config['datasets']}")

# 2. 加载训练集和验证集
print("\n加载数据集...")

# 加载所有数据集并合并
all_data = []
for dataset_path in config['datasets']:
    print(f"  加载: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        all_data.extend(data)
        print(f"    数据量: {len(data)}条")

print(f"\n合并后总数据量: {len(all_data)}条")

# 划分训练集和验证集（80%训练，20%验证）
train_size = int(len(all_data) * 0.8)
val_size = len(all_data) - train_size
train_data, val_data = random_split(all_data, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))

# 转换为列表
train_data = list(train_data)
val_data = list(val_data)

print(f"\n使用数据集：")
print(f"  训练集：{len(train_data)}条")
print(f"  验证集：{len(val_data)}条")

# 统计训练集标签分布
train_yes = sum(1 for item in train_data if item['output'] == 'Yes' or item['output'] == '是')
train_no = sum(1 for item in train_data if item['output'] == 'No' or item['output'] == '否')
print(f"  训练集标签分布：")
print(f"    仇恨言论 (yes): {train_yes} ({train_yes/len(train_data):.2%})")
print(f"    非仇恨言论 (no): {train_no} ({train_no/len(train_data):.2%})")
if train_yes > 0:
    print(f"    比例: 1:{train_no/train_yes:.2f} (yes:no)")
else:
    print(f"    比例: 0:1 (yes:no)")

# 3. 加载模型和分词器
model_path = "models/xlm-roberta-base"
print(f"\n加载模型：{model_path}")
tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
model = XLMRobertaForSequenceClassification.from_pretrained(
    model_path,
    num_labels=2,
    ignore_mismatched_sizes=True
)

# 4. 配置LoRA
lora_config = LoraConfig(
     r=16,                  
     lora_alpha=32,          
     target_modules=["query", "key", "value"],  
     lora_dropout=0.1, 
     bias="none", 
     task_type="SEQ_CLS"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 5. 准备训练数据
train_dataset = HateSpeechDataset(train_data, tokenizer)
test_dataset = HateSpeechDataset(val_data, tokenizer)

# 6. 配置训练参数
batch_size = 16
num_epochs = 5
learning_rate = 2e-4
gradient_accumulation_steps = 4
seed = args.seed

# 设置所有随机种子
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)

# 早停机制参数
early_stopping_patience = 2
best_eval_f1 = 0.0
best_model_state = None
no_improvement_count = 0
stop_training = False

# 创建数据加载器
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

test_dataloader = DataLoader(
    test_dataset, 
    batch_size=batch_size,
    num_workers=4,
    pin_memory=True
)

# 设置优化器和损失函数
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"使用设备：{device}")

# 初始化保存列表
training_loss = []
eval_metrics = []

# 7. 训练循环
print("\n" + "="*70)
print(f"准备开始完整数据集训练")
print(f"训练集：{len(train_dataset)}条")
print(f"验证集：{len(test_dataset)}条")
print(f"Batch size：{batch_size}")
print(f"训练轮次：{num_epochs}")
print(f"总训练步数：{len(train_dataloader) * num_epochs}")
print("="*70)

# 记录总开始时间
total_start_time = time.time()

total_steps = 0

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print("-" * 40)
    
    # 训练模式
    model.train()
    epoch_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    # 记录当前epoch开始时间
    epoch_start_time = time.time()
    
    for step, batch in enumerate(train_dataloader):
        # 准备数据
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # 前向传播
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # 计算损失（梯度累积需要将损失除以累积步数）
        loss = loss_fn(logits, labels) / gradient_accumulation_steps
        epoch_loss += loss.item() * gradient_accumulation_steps
        
        # 计算准确率
        predictions = torch.argmax(logits, dim=-1)
        correct_predictions += (predictions == labels).sum().item()
        total_predictions += labels.size(0)
        
        # 反向传播
        loss.backward()
        
        # 当达到累积步数或最后一步时更新参数
        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
            optimizer.step()
            optimizer.zero_grad()
        
        # 保存训练损失
        total_steps += 1
        training_loss.append({
            "step": total_steps,
            "epoch": epoch + 1,
            "loss": loss.item() * gradient_accumulation_steps,
            "learning_rate": learning_rate
        })
        
        # 实时打印进度（每10步或最后一步）
        if (step + 1) % 10 == 0 or (step + 1) == len(train_dataloader):
            avg_loss = epoch_loss / (step + 1)
            accuracy = correct_predictions / total_predictions
            elapsed_time = time.time() - epoch_start_time
            
            # 计算进度百分比
            epoch_progress = (step + 1) / len(train_dataloader) * 100
            
            print(f"Step {step+1:3d}/{len(train_dataloader)} | "
                  f"Progress: {epoch_progress:5.1f}% | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Accuracy: {accuracy:.4f} | "
                  f"Time: {elapsed_time:.2f}s")
    
    # 当前epoch训练完成
    epoch_time = time.time() - epoch_start_time
    avg_train_loss = epoch_loss / len(train_dataloader)
    train_accuracy = correct_predictions / total_predictions
    
    print(f"\nEpoch {epoch+1} 训练完成")
    print(f"训练时间: {epoch_time:.2f}s")
    print(f"平均训练损失: {avg_train_loss:.4f}")
    print(f"训练准确率: {train_accuracy:.4f}")
    
    # 评估模式
    print("开始评估...")
    model.eval()
    eval_loss = 0
    eval_correct = 0
    eval_total = 0
    
    # 收集所有预测和真实标签用于计算详细指标
    all_predictions = []
    all_labels = []
    all_raw_texts = []
    all_raw_outputs = []
    
    eval_start_time = time.time()
    
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            raw_texts = batch["raw_text"]
            raw_outputs = batch["raw_output"]
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            loss = loss_fn(logits, labels)
            eval_loss += loss.item()
            
            predictions = torch.argmax(logits, dim=-1)
            eval_correct += (predictions == labels).sum().item()
            eval_total += labels.size(0)
            
            # 收集预测和标签
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_raw_texts.extend(raw_texts)
            all_raw_outputs.extend(raw_outputs)
    
    eval_time = time.time() - eval_start_time
    avg_eval_loss = eval_loss / len(test_dataloader)
    eval_accuracy = eval_correct / eval_total
    
    # 计算精确率、召回率和F1分数
    import numpy as np
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    # 转换为numpy数组
    all_predictions_np = np.array(all_predictions)
    all_labels_np = np.array(all_labels)
    
    # 计算指标（pos_label=1表示仇恨言论为正类）
    precision = precision_score(all_labels_np, all_predictions_np, pos_label=1, zero_division=0)
    recall = recall_score(all_labels_np, all_predictions_np, pos_label=1, zero_division=0)
    f1 = f1_score(all_labels_np, all_predictions_np, pos_label=1, zero_division=0)
    
    # 保存评估指标
    eval_metrics.append({
        "step": total_steps,
        "epoch": epoch + 1,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": eval_accuracy,
        "loss": avg_eval_loss
    })
    
    print(f"评估时间: {eval_time:.2f}s")
    print(f"测试损失: {avg_eval_loss:.4f}")
    print(f"测试准确率: {eval_accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数 (F1-Score): {f1:.4f}")
    print("-" * 40)
    
    # 早停机制检查
    if f1 > best_eval_f1:
        best_eval_f1 = f1
        best_model_state = model.state_dict()
        no_improvement_count = 0
        print(f"✓ 验证F1分数提升到 {best_eval_f1:.4f}，重置早停计数")
    else:
        no_improvement_count += 1
        print(f"✗ 验证F1分数没有提升，连续 {no_improvement_count}/{early_stopping_patience} 轮")
        
        if no_improvement_count >= early_stopping_patience:
            print(f"\n⚠️  连续 {early_stopping_patience} 轮验证F1分数没有提升，触发早停机制")
            stop_training = True
            break

# 8. 训练完成
print("\n" + "="*70)
total_time = time.time() - total_start_time
print(f"完整训练完成！")
print(f"总训练时间: {total_time:.2f}s")

# 9. 保存所有要求的文件

# 保存训练损失
loss_file_path = os.path.join(args.save_path, "training_loss.json")
print(f"\n保存训练损失到: {loss_file_path}")
with open(loss_file_path, "w", encoding="utf-8") as f:
    json.dump(training_loss, f, ensure_ascii=False, indent=2)

# 保存评估指标
eval_file_path = os.path.join(args.save_path, "eval_metrics.json")
print(f"保存评估指标到: {eval_file_path}")
with open(eval_file_path, "w", encoding="utf-8") as f:
    json.dump(eval_metrics, f, ensure_ascii=False, indent=2)

# 保存最佳模型
best_model_path = os.path.join(args.save_path, "best_model")
print(f"保存最佳LoRA模型到: {best_model_path}")
os.makedirs(best_model_path, exist_ok=True)

if best_model_state is not None:
    print(f"\n恢复最佳模型状态（F1分数: {best_eval_f1:.4f}）")
    model.load_state_dict(best_model_state)

model.save_pretrained(best_model_path)
tokenizer.save_pretrained(best_model_path)

# 保存预测结果
predictions_file_path = os.path.join(args.save_path, "predictions.csv")
print(f"保存预测结果到: {predictions_file_path}")

with open(predictions_file_path, "w", newline='', encoding="utf-8") as csvfile:
    fieldnames = ['raw_text', 'raw_output', 'true_label', 'predicted_label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for i in range(len(all_predictions)):
        writer.writerow({
            'raw_text': all_raw_texts[i],
            'raw_output': all_raw_outputs[i],
            'true_label': 'Yes' if all_labels[i] == 1 else 'No',
            'predicted_label': 'Yes' if all_predictions[i] == 1 else 'No'
        })

print(f"\n所有文件已保存到: {args.save_path}")
print(f"  - training_loss.json: 每个step的训练损失")
print(f"  - eval_metrics.json: 每个eval step的dev集F1/Precision/Recall")
print(f"  - best_model/: LoRA权重 (~30MB)")
print(f"  - predictions.csv: 测试集每条样本的预测结果 + 真实标签")

print("\n" + "="*70)
print("模型训练和所有文件保存完成！")
print("="*70)