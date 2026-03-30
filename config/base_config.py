"""
基础配置模块

定义数据、模型、训练和实验的基础配置类
支持从YAML文件加载和保存配置
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import yaml
import os


@dataclass
class DataConfig:
    """数据配置"""
    dataset_name: str = "brats2021"
    data_dir: str = "./data/BraTS2021"
    modalities: List[str] = field(default_factory=lambda: ["t1", "t2", "flair", "t1ce"])
    target_spacing: tuple = (1.0, 1.0, 1.0)
    target_size: tuple = (128, 128, 128)
    train_split: float = 0.7
    val_split: float = 0.15
    num_workers: int = 4
    preload_data: bool = False


@dataclass
class ModelConfig:
    """模型配置"""
    architecture: str = "unet3d"
    in_channels: int = 4
    out_channels: int = 4
    base_filters: int = 32
    encoder_name: str = "resnet34"
    pretrained: bool = True
    fusion_method: str = "attention"  # concat, attention, gate, transformer
    use_deep_supervision: bool = True
    dropout_rate: float = 0.1
    use_auxiliary_heads: bool = False


@dataclass
class TrainingConfig:
    """训练配置"""
    epochs: int = 200
    batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 10
    gradient_clip: float = 1.0
    mixed_precision: bool = True
    accumulation_steps: int = 4
    label_smoothing: float = 0.0
    dice_weight: float = 0.5
    focal_weight: float = 0.5


@dataclass
class ExperimentConfig:
    """实验配置"""
    experiment_name: str = "multimodal_segmentation"
    seed: int = 42
    device: str = "cuda"
    log_interval: int = 10
    save_interval: int = 5
    val_interval: int = 1
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "medical-imaging"
    resume_checkpoint: Optional[str] = None


@dataclass
class SurvivalConfig:
    """预后预测配置"""
    enable_survival: bool = True
    survival_weight: float = 1.0
    segmentation_weight: float = 1.0
    use_clinical_features: bool = True
    clinical_features: List[str] = field(default_factory=lambda: ["age", "gender"])
    survival_head_dim: int = 128
    use_uncertainty_weighting: bool = True


def load_config(config_path: str) -> Dict:
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict, save_path: str):
    """
    保存配置到YAML文件
    
    Args:
        config: 配置字典
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def create_config_from_dict(config_dict: Dict) -> tuple:
    """
    从配置字典创建配置对象
    
    Args:
        config_dict: 配置字典
    
    Returns:
        (DataConfig, ModelConfig, TrainingConfig, ExperimentConfig) 元组
    """
    data_config = DataConfig(**config_dict.get('data', {}))
    model_config = ModelConfig(**config_dict.get('model', {}))
    training_config = TrainingConfig(**config_dict.get('training', {}))
    experiment_config = ExperimentConfig(**config_dict.get('experiment', {}))
    
    return data_config, model_config, training_config, experiment_config


def print_config(data_config: DataConfig, model_config: ModelConfig, 
                 training_config: TrainingConfig, experiment_config: ExperimentConfig):
    """
    打印配置信息
    
    Args:
        data_config: 数据配置
        model_config: 模型配置
        training_config: 训练配置
        experiment_config: 实验配置
    """
    print("=" * 80)
    print("实验配置信息")
    print("=" * 80)
    
    print("\n【数据配置】")
    for key, value in vars(data_config).items():
        print(f"  {key}: {value}")
    
    print("\n【模型配置】")
    for key, value in vars(model_config).items():
        print(f"  {key}: {value}")
    
    print("\n【训练配置】")
    for key, value in vars(training_config).items():
        print(f"  {key}: {value}")
    
    print("\n【实验配置】")
    for key, value in vars(experiment_config).items():
        print(f"  {key}: {value}")
    
    print("=" * 80)
