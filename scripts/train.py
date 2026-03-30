"""
主训练脚本

多模态医学影像分割与预后预测模型训练
"""

import os
import sys
import argparse
import torch
import torch.optim as optim
from pathlib import Path
import random
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import (
    DataConfig,
    ModelConfig,
    TrainingConfig,
    ExperimentConfig,
    load_config,
    create_config_from_dict,
    print_config
)
from data import BrATSDataModule
from models.fusion import (
    ModalAttentionFusion,
    GatedMultimodalFusion,
    TransformerFusion
)
from training import (
    CombinedLoss,
    DeepSurvivalLoss,
    MultiTaskLoss,
    SegmentationTrainer
)
from evaluation import SegmentationMetrics


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimizer(model, config: TrainingConfig):
    """
    获取优化器
    
    Args:
        model: 模型
        config: 训练配置
    
    Returns:
        优化器
    """
    if config.optimizer.lower() == "adam":
        return optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif config.optimizer.lower() == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif config.optimizer.lower() == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"不支持的优化器: {config.optimizer}")


def get_scheduler(optimizer, config: TrainingConfig):
    """
    获取学习率调度器
    
    Args:
        optimizer: 优化器
        config: 训练配置
    
    Returns:
        调度器
    """
    if config.scheduler.lower() == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
            eta_min=config.learning_rate * 0.01
        )
    elif config.scheduler.lower() == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.epochs // 3,
            gamma=0.1
        )
    elif config.scheduler.lower() == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
    else:
        return None


def create_model(config: ModelConfig):
    """
    创建模型
    
    Args:
        config: 模型配置
    
    Returns:
        模型
    """
    # 这里需要导入实际的模型
    # 由于模型代码较长，这里提供一个占位符
    from models.unet3d import AttentionUNet3D
    
    # 创建融合模块
    if config.fusion_method == "attention":
        fusion = ModalAttentionFusion
    elif config.fusion_method == "gate":
        fusion = GatedMultimodalFusion
    elif config.fusion_method == "transformer":
        fusion = TransformerFusion
    else:
        fusion = None
    
    model = AttentionUNet3D(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        base_filters=config.base_filters,
        dropout=config.dropout_rate,
        fusion_method=config.fusion_method,
        num_modalities=len(config.in_channels) if isinstance(config.in_channels, int) else 4
    )
    
    return model


def train(args):
    """
    训练函数
    
    Args:
        args: 命令行参数
    """
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载配置
    if args.config:
        config_dict = load_config(args.config)
        data_config, model_config, training_config, experiment_config = create_config_from_dict(config_dict)
    else:
        # 使用默认配置
        data_config = DataConfig(
            data_dir=args.data_dir,
            modalities=args.modalities.split(',') if args.modalities else ["t1", "t2", "flair", "t1ce"]
        )
        model_config = ModelConfig(
            in_channels=len(data_config.modalities),
            fusion_method=args.fusion_method
        )
        training_config = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )
        experiment_config = ExperimentConfig(
            experiment_name=args.experiment_name,
            seed=args.seed,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir
        )
    
    # 打印配置
    print_config(data_config, model_config, training_config, experiment_config)
    
    # 设置设备
    device = torch.device(experiment_config.device if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    # 创建数据模块
    print("\n准备数据集...")
    data_module = BrATSDataModule(
        data_dir=data_config.data_dir,
        batch_size=training_config.batch_size,
        num_workers=data_config.num_workers,
        modalities=data_config.modalities,
        crop_size=data_config.target_size,
        target_spacing=data_config.target_spacing
    )
    
    try:
        data_module.setup()
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        print(f"训练集: {len(train_loader.dataset)} 样本")
        print(f"验证集: {len(val_loader.dataset)} 样本")
    except Exception as e:
        print(f"数据加载失败: {e}")
        print("将使用合成数据进行演示...")
        train_loader = None
        val_loader = None
    
    # 创建模型
    print("\n创建模型...")
    model = create_model(model_config)
    model = model.to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 创建优化器和调度器
    optimizer = get_optimizer(model, training_config)
    scheduler = get_scheduler(optimizer, training_config)
    
    # 创建损失函数
    loss_fn = CombinedLoss(
        num_classes=model_config.out_channels,
        dice_weight=training_config.dice_weight,
        focal_weight=training_config.focal_weight
    )
    
    # 创建评估指标
    metrics_fn = SegmentationMetrics(
        num_classes=model_config.out_channels,
        class_names=["Background", "Necrotic", "Edema", "Enhancing"]
    )
    
    # 创建训练器
    trainer = SegmentationTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        scheduler=scheduler,
        device=str(device),
        log_dir=experiment_config.log_dir,
        checkpoint_dir=experiment_config.checkpoint_dir,
        gradient_clip=training_config.gradient_clip,
        deep_supervision_weight=0.4 if model_config.use_deep_supervision else 0.0
    )
    
    # 加载检查点（如果指定）
    if experiment_config.resume_checkpoint:
        print(f"\n加载检查点: {experiment_config.resume_checkpoint}")
        trainer.load_checkpoint(experiment_config.resume_checkpoint)
    
    # 开始训练
    if train_loader and val_loader:
        print("\n开始训练...")
        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=training_config.epochs,
            start_epoch=trainer.current_epoch
        )
    else:
        print("\n跳过训练（无有效数据）")
    
    # 保存训练历史
    trainer.save_history()
    print("\n训练完成!")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="多模态医学影像分割与预后预测训练"
    )
    
    # 配置相关
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--experiment_name", type=str, default="multimodal_segmentation")
    
    # 数据相关
    parser.add_argument("--data_dir", type=str, default="./data/BraTS2021")
    parser.add_argument("--modalities", type=str, default="t1,t2,flair,t1ce")
    
    # 模型相关
    parser.add_argument("--fusion_method", type=str, default="attention",
                       choices=["concat", "attention", "gate", "transformer"])
    
    # 训练相关
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    
    # 其他
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    train(args)


if __name__ == "__main__":
    main()
