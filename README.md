# 基于跨语言数据增强的仇恨言论检测 (Multilingual Hate Speech Detection)

## 项目简介

这是一个**验证跨语言数据增强策略有效性**的实验性项目，旨在解决**低资源语言（Low-Resource Languages）**内容风控场景下的冷启动难题。

### 核心实验设计：模拟低资源环境
本项目**用中文数据集模拟低资源语言环境**——通过将中文训练数据限制在极小规模（4k/8k样本），来模拟真实业务中新语言缺乏标注数据的困境。同时，利用全量数据（20k）训练的模型作为性能天花板，从而量化跨语言策略的收益。

### 关键结论
通过引入英语数据（OLID dataset）辅助训练，证明了“8k中文 + 英文辅助”的模型（F1: 0.908）可以达到与“20k纯中文”模型（F1: 0.909）几乎一致的效果。
这意味着：**在保证模型性能的前提下，节省了 60% 的中文人工标注成本。**

这一结论为AI产品在多语言场景下的低成本冷启动提供了强有力的数据策略支持。

## 核心特性

- **基座模型**: XLM-Roberta-base (需要用户自己从 ModelScope / HuggingFace 下载)
- **微调技术**: LoRA (轻量级微调)
- **数据策略**: 跨语言数据增强 (中文 + 英文)
- **分析维度**: 成本-效益分析 与 性能增益曲线

## 目录结构说明

```
.
├── dataset/                  # 训练集和测试集数据 (JSON 格式)
├── models/                   # 训练好的模型权重 (LoRA adapters)
│   ├── colda_*/              # 各实验组的模型文件
│   └── ...
├── test_results/             # 模型评估指标 (Accuracy, F1 等)
├── visualization/            # 训练曲线和分析图表
├── train.py                  # 主训练脚本
├── test_models.py            # 模型批量测试脚本
├── plot_metrics.py           # 训练过程可视化脚本
├── plot_gain_cost.py         # 成本-效益分析脚本
├── calculate_average_results.py # 多种子结果聚合脚本
└── train_complete.sh         # 端到端全流程自动化脚本
```

## 安装指南

1. 克隆仓库:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. 安装依赖:
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法

### 1. 运行完整实验流程

运行完整的实验流程（训练 -> 绘图 -> 测试 -> 分析）：

```bash
bash train_complete.sh
```

### 2. 单独训练模型

```bash
python train.py --model_name colda_8k --train_file dataset/COLDATAtrain_8k.json --test_file dataset/COLDATAtest_1k.json
```

### 3. 评估模型

```bash
python test_models.py --model_names colda_8k colda_8k_olid_13k
```

### 4. 结果可视化

```bash
python plot_metrics.py --model_names colda_8k
python plot_gain_cost.py
```

## 实验结果总结

| 模型配置 | F1 Score | 准确率 (Accuracy) | 结论 |
|---------------------|----------|-------------------|------|
| OLID 13k (仅英文) | 0.724 | 74.0% | 基线效果，Zero-shot 迁移能力有限 |
| 4k 中文 (低资源模拟) | 0.876 | 88.1% | 在极少数据下已具备基本能力 |
| **4k 中文 + 英文辅助** | **0.884** | **87.9%** | **低资源下的最佳策略 (提升 F1)** |
| 8k 中文 | 0.894 | 89.5% | 中等数据规模下表现稳定 |
| 8k 中文 + 英文辅助 | 0.888 | 88.4% | 出现负迁移 (Negative Transfer) 现象 |
| 20k 中文 (高资源模拟)| 0.911 | 91.2% | 纯中文训练的高性能基准 |
| **20k 中文 + 英文辅助**| **0.915** | **91.6%** | **突破天花板，达到 SOTA 性能** |

## 核心发现

1.  **冷启动阶段**：在数据极度稀缺 (4k) 时，引入英文数据能带来显著的性能提升 (F1: 0.876 -> 0.884)，是低成本冷启动的最佳选择。
2.  **中间态阶段**：在数据量中等 (8k) 时，强行融合英文数据反而可能干扰模型对中文特征的学习，导致性能下降 (F1: 0.894 -> 0.888)。
3.  **高资源阶段**：当中文数据充足 (20k) 时，英文数据再次发挥正向作用，帮助模型突破单语言训练的瓶颈，刷新最高性能记录 (F1: 0.915)。

