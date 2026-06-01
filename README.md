# IGNet AI Train

一个灵活的深度学习模型训练工具包，支持多种神经网络架构和数据集类型。

## 项目概述

IGNet AI Train 是一个基于 PyTorch 的模型训练框架，提供了模块化的设计，支持：
- 多种神经网络（MobileNet V1、ResShift 等）
- 多种数据集类型（图像、音频、NPY 格式）
- 灵活的模型配置系统
- 优化器和训练工具集


## 项目结构

```
ignet_ai_train/
├── __main__.py           # 项目入口
├── cli.py               # 命令行接口
├── config.py            # 配置管理
├── model_builder.py     # 模型构建器
├── dataset/             # 数据集模块
│   ├── create.py        # 数据集创建
│   └── modules/         # 数据集类型（音频、图像、NPY）
├── nn/                  # 神经网络模块
│   ├── create.py        # 网络创建
│   └── modules/         # 网络层（MobileNet、CReLU、Roll Mix 等）
├── optim/               # 优化器模块
│   └── modules/         # 优化器实现（Muon 等）
└── utils/               # 工具函数
    ├── train.py         # 训练工具
    ├── eval.py          # 评估工具
    └── ...
```

## 快速开始

### 安装依赖

本项目使用UV管理依赖，Python版本3.11+

```bash
uv sync
```

### 基础用法

- 参考训练[配置说明](TRAIN_CONFIG.md)文档，生产训练配置

- 使用 CLI 训练模型

```bash
python -m ignet_ai_train --config test/train.yaml
```

- 集成到Python项目

```python
import torch

from ignet_ai_train.utils import train
from ignet_ai_train.utils.eval import evaluate_classification
from ignet_ai_train.dataset.create import create_dataset
from ignet_ai_train.model_builder import build_model_by_yaml, create_model
from ignet_ai_train.optim.create import create_optimizer

model:torch.nn.Module = build_model_by_yaml("model.yaml")
# liner:torch.nn.Module = create_model("Linear", in_features=10, out_features=1)
optimizer = create_optimizer("MuSGD", model.parameters(), lr=0.01)
train_dataset = create_dataset("NpyFolder","/root/train")
eval_dataset = create_dataset("NpyFolder","/root/eval")

train(model, optimizer, train_dataset, eval_dataset,device="cpu", epochs=10)
# torch.save(model.state_dict(), "model.pt")

eval_info:dict = evaluate_classification(model, eval_dataset,num_classes=2, device="cpu")
print(eval_info)
```

### 配置文件

项目使用 YAML 配置文件定义模型和训练参数，示例配置位于 `test/models/` 目录。

## 核心模块

| 模块 | 功能 |
|------|------|
| **dataset** | 支持多种数据格式（图像、音频、NPY）的数据集加载 |
| **nn** | 神经网络层和架构（MobileNet V1、ResShift、自定义层） |
| **optim** | 优化器实现（Muon 等） |
| **utils** | 训练、评估、图像处理等工具函数 |

## 主要功能

- 模块化模型构建
- 多数据源支持
- 配置驱动的训练
- 模型导出功能
- 评估工具集

## 示例

查看 `test/` 目录获取更多示例：
- `train.py` - 训练脚本示例
- `evaluate_classify.py` - 分类评估
- `export.py` - 模型导出
- 预定义配置文件（分类、超分辨率等）

## 文档

更多详细信息，请查看各模块的源代码注释和 `test/` 目录中的示例。
