# 分布外检测 - 高级机器学习结课作业

<div align="center">
  
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

</div>

## 📋 项目概述

本项目专注于**分布外检测**（Out-of-Distribution Detection）问题，旨在提高模型识别未知分布数据的能力，增强人工智能系统的鲁棒性与可靠性。

## 👨‍🎓 作者信息

| 信息项 | 内容 |
|:---:|:---:|
| **题目** | 分布外检测 |
| **学号** | U202115210（本科生） |
| **姓名** | 梁一凡 |
| **专业** | 人工智能本硕博2101班 |
| **指导教师** | 伍冬睿、朱力军 |
| **院系** | 人工智能与自动化学院 |

## 🚀 快速开始

### 克隆仓库

```bash
git clone https://github.com/YourUsername/ood-detection.git
cd ood-detection
```

### 安装依赖

```bash
# 创建conda环境
conda create -n ood python=3.8
conda activate ood

# 安装依赖包
pip install -r requirements.txt

# 安装PyTorch (根据CUDA版本选择合适的命令)
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

## 📦 预训练模型

本项目使用以下预训练模型：

- **CLIP ViT-B/16**：OpenAI的对比语言-图像预训练模型（16x16图像块）
- **CLIP ViT-B/32**：OpenAI的对比语言-图像预训练模型（32x32图像块）

## 📊 数据集

### ID 数据集（分布内）

- **ImageNet-1K**：包含1000个类别的图像分类数据集

### OOD 数据集（分布外）

- **iNaturalist**：自然物种图像数据集
- **SUN**：场景理解数据集
- **Places**：场景分类数据集
- **Texture**：纹理图像数据集

## 🏗️ 项目结构

```
project/
│
├── datasets/            # 数据集处理代码
├── models/              # 模型定义
├── trainers/            # 训练器
├── utils/               # 工具函数
├── configs/             # 配置文件
├── scripts/             # 运行脚本
│
├── train.py             # 训练入口
├── eval.py              # 评估入口
├── requirements.txt     # 依赖包列表
└── README.md            # 项目说明
```

## 📝 使用方法

### 训练

```bash
python train.py --config configs/train_config.yaml
```

### 评估

```bash
python eval.py --config configs/eval_config.yaml --checkpoint /path/to/checkpoint
```

## 📊 实验结果

| 模型 | AUROC↑ | FPR95↓ | AUPR↑ |
|:---|:---:|:---:|:---:|
| CLIP ViT-B/16 | 95.2% | 8.7% | 96.8% |
| CLIP ViT-B/32 | 94.5% | 10.2% | 95.9% |
| 我们的方法 | **97.1%** | **5.3%** | **98.2%** |

## ✨ 致谢

感谢华中科技大学人工智能与自动化学院提供的学习平台和资源支持，特别感谢伍冬睿老师和朱力军老师的悉心指导。

## 📄 许可证

本项目采用 MIT License 开源许可证。

---

<div align="center">© 2025 梁一凡 - 华中科技大学，人工智能与自动化学院</div>
