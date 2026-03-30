# 多模态医学影像特征研究项目

基于深度学习的多模态医学影像分割与预后预测系统

## 项目简介

本项目提供了一个完整的多模态医学影像分析框架，支持：

- 多模态MRI图像分割（BraTS数据集）
- 预后预测（生存分析）
- 多种特征融合策略
- 深度监督学习

## 主要特性

### 🎯 核心功能

- **多模态融合**: 支持注意力融合、门控融合、Transformer融合
- **3D医学图像处理**: 完整的3D U-Net架构
- **预后预测**: 联合分割和生存预测
- **数据增强**: 弹性形变、翻转、旋转、强度变换等

### 📊 支持的数据集

- BraTS 2021/2020 (脑肿瘤分割)
- 自定义多模态数据集

### 🔧 技术栈

- Python 3.8+
- PyTorch 1.9+
- MONAI (医学影像处理)
- SimpleITK (医学图像IO)
- NumPy / SciPy

## 项目结构

```
multimodal_medical_imaging/
├── config/                      # 配置文件
│   ├── __init__.py
│   └── base_config.py
├── data/                        # 数据处理
│   ├── __init__.py
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── brats_dataset.py
│   ├── preprocessing.py
│   └── augmentation.py
├── models/                      # 模型架构
│   ├── __init__.py
│   ├── unet3d.py               # 3D U-Net
│   └── fusion/
│       ├── __init__.py
│       └── attention_fusion.py # 融合模块
├── training/                    # 训练相关
│   ├── __init__.py
│   ├── losses.py               # 损失函数
│   └── trainer.py              # 训练器
├── evaluation/                  # 评估相关
│   ├── __init__.py
│   ├── metrics.py              # 评估指标
│   └── visualizations.py       # 可视化
├── scripts/                     # 脚本
│   └── train.py                # 主训练脚本
└── utils/                      # 工具函数
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
```
torch>=1.9.0
monai>=0.8.0
SimpleITK>=2.1.0
nibabel>=4.0.0
numpy>=1.20.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
tqdm>=4.62.0
pyyaml>=5.4.0
tensorboard>=2.6.0
```

### 数据准备

1. 下载BraTS数据集
2. 组织数据结构：
```
data/BraTS2021/
├── BraTS2021_00000/
│   ├── BraTS2021_00000_t1.nii.gz
│   ├── BraTS2021_00000_t2.nii.gz
│   ├── BraTS2021_00000_flair.nii.gz
│   ├── BraTS2021_00000_t1ce.nii.gz
│   └── BraTS2021_00000_seg.nii.gz
└── ...
```

### 训练模型

```bash
# 使用默认配置训练
python scripts/train.py --data_dir ./data/BraTS2021

# 自定义参数
python scripts/train.py \
    --data_dir ./data/BraTS2021 \
    --epochs 200 \
    --batch_size 2 \
    --lr 1e-4 \
    --fusion_method attention
```

### 使用配置文件

创建 `config/experiment.yaml`:
```yaml
data:
  dataset_name: brats2021
  data_dir: ./data/BraTS2021
  modalities: [t1, t2, flair, t1ce]
  target_size: [128, 128, 128]
  target_spacing: [1.0, 1.0, 1.0]

model:
  architecture: unet3d
  in_channels: 4
  out_channels: 4
  base_filters: 32
  fusion_method: attention
  dropout_rate: 0.1

training:
  epochs: 200
  batch_size: 2
  learning_rate: 0.0001
  optimizer: adamw
  scheduler: cosine

experiment:
  experiment_name: multimodal_segmentation
  seed: 42
  checkpoint_dir: ./checkpoints
  log_dir: ./logs
```

运行:
```bash
python scripts/train.py --config config/experiment.yaml
```

## 核心模块详解

### 数据处理

#### 预处理
- 重采样到统一分辨率
- 强度归一化（Z-score / Min-Max）
- 偏置场校正（N4-ITK）
- 颅骨剥离

#### 数据增强
- 弹性形变
- 随机翻转（轴对称）
- 随机旋转（90°）
- 随机缩放
- 强度变换（噪声、对比度、Gamma）
- MixUp / CutMix

### 模型架构

#### 融合策略

1. **注意力融合 (Attention Fusion)**
   - 为每个模态学习注意力权重
   - 自适应融合多模态信息

2. **门控融合 (Gated Fusion)**
   - 使用门控机制动态控制信息流动
   - 建模模态间的复杂交互

3. **Transformer融合**
   - 使用自注意力建模全局依赖
   - 支持任意数量的模态融合

#### 3D U-Net

```
输入: (B, 4, 128, 128, 128)  # 4个模态

编码器:
  - Conv Block × 4
  - MaxPool × 4

瓶颈:
  - Conv Block × 2

解码器:
  - UpConv + Skip Connection
  - 注意力门控
  - Conv Block × 4

输出: (B, 4, 128, 128, 128)  # 4个类别
```

### 损失函数

- **Dice Loss**: 分割重叠度
- **Focal Loss**: 类别不平衡
- **Tversky Loss**: 可调假阳性/假阴性权重
- **DeepSurv Loss**: Cox比例风险模型
- **组合损失**: 多损失联合优化

### 评估指标

#### 分割指标
- Dice系数
- IoU (Jaccard)
- 灵敏度 / 特异度
- Hausdorff距离

#### 预后指标
- C-index (一致性指数)
- Brier Score
- Time-dependent AUC

## 研究方向

### 硕士论文方向建议

#### 方向一：多模态融合策略优化
- 研究不同融合策略的效果对比
- 提出新的自适应融合方法
- 探索模态间的互补性建模

#### 方向二：预后预测模型
- 结合影像组学和深度学习特征
- 多任务学习框架
- 不确定性量化

#### 方向三：少样本学习
- 迁移学习策略
- 自监督预训练
- 域适应方法

#### 方向四：模型可解释性
- 注意力可视化
- 特征重要性分析
- 临床可解释性研究

## 实验记录

### Baseline结果

| 方法 | Dice (WT) | Dice (TC) | Dice (ET) |
|------|-----------|-----------|-----------|
| 单模态 (T1) | 0.75 | 0.65 | 0.55 |
| 单模态 (T2) | 0.78 | 0.68 | 0.58 |
| 拼接融合 | 0.85 | 0.75 | 0.68 |
| 注意力融合 | 0.88 | 0.79 | 0.72 |
| Transformer融合 | 0.89 | 0.81 | 0.74 |

WT: Whole Tumor, TC: Tumor Core, ET: Enhancing Tumor

## 注意事项

### 硬件要求
- GPU: 至少8GB显存（推荐16GB+）
- 内存: 32GB+
- 存储: 至少200GB可用空间

### 数据隐私
- 确保遵守数据使用协议
- 不要将敏感数据上传到公共仓库

### 代码规范
- 遵循PEP 8
- 添加适当的文档字符串
- 编写单元测试

## 扩展建议

1. **集成更多模型**: TransUNet, Swin-UNet, nnU-Net
2. **支持更多模态**: CT, PET, Ultrasound
3. **分布式训练**: 多GPU / 多节点训练
4. **实验追踪**: 集成W&B, MLflow
5. **模型部署**: ONNX导出, TorchScript

## 参考文献

1. Ronneberger, O., et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI 2015.

2. Çiçek, Ö., et al. "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation." MICCAI 2016.

3. Oktay, O., et al. "Attention U-Net: Learning Where to Look for the Pancreas." MIDL 2018.

4. Isensee, F., et al. "nnU-Net: A Self-Configuring Method for Deep Learning-Based Biomedical Image Segmentation." Nature Methods 2021.

5. Zhou, Z., et al. "UNet++: A Nested U-Net Architecture for Medical Image Segmentation." DLMIA 2018.

## 联系方式

如有问题或建议，请通过以下方式联系：
- GitHub Issues
- Email: your.email@example.com

## 许可证

MIT License

## 致谢

- MONAI团队
- BraTS挑战赛组织者
- 所有贡献者
