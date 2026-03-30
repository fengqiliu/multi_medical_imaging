"""
BraTS多模态脑肿瘤分割数据集

该数据集支持多模态MRI图像的加载、预处理和增强，
适用于脑肿瘤分割任务。
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import monai.transforms as mtransforms
import SimpleITK as sitk


class MultiModalBrATS(Dataset):
    """
    BraTS多模态脑肿瘤分割数据集
    
    数据结构:
        data_dir/
            BraTS2021_00000/
                BraTS2021_00000_t1.nii.gz
                BraTS2021_00000_t2.nii.gz
                BraTS2021_00000_flair.nii.gz
                BraTS2021_00000_t1ce.nii.gz
                BraTS2021_00000_seg.nii.gz
    """
    
    CLASSES = {
        0: "background",
        1: "necrotic_core", 
        2: "edema",
        4: "enhancing_tumor"
    }
    
    # 合并肿瘤子区域为统一标签
    TUMOR_CLASSES = {
        0: "background",
        1: "tumor"  # 合并所有肿瘤类别用于二分类
    }
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        modalities: List[str] = ["t1", "t2", "flair", "t1ce"],
        transform=None,
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        crop_size: Tuple[int, int, int] = (128, 128, 128),
        preload: bool = False
    ):
        """
        初始化BraTS数据集
        
        Args:
            data_dir: 数据根目录
            split: 数据集划分 ('train', 'val', 'test')
            modalities: 使用的模态列表
            transform: 数据增强变换
            target_spacing: 目标体素间距 (mm)
            crop_size: 裁剪尺寸
            preload: 是否预加载数据到内存
        """
        self.data_dir = data_dir
        self.split = split
        self.modalities = modalities
        self.transform = transform
        self.target_spacing = target_spacing
        self.crop_size = crop_size
        self.preload = preload
        
        # 扫描数据目录
        self.case_ids = self._scan_cases()
        
        # 预处理变换
        self.preprocessing = self._get_preprocessing_transforms()
        
        # 预加载数据（可选）
        if self.preload:
            self.data_cache = {}
            print(f"预加载 {len(self.case_ids)} 个案例到内存...")
            for case_id in self.case_ids:
                self.data_cache[case_id] = self._load_case(case_id)
    
    def _scan_cases(self) -> List[str]:
        """扫描数据目录获取所有案例ID"""
        case_ids = []
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
        
        for folder in sorted(os.listdir(self.data_dir)):
            if folder.startswith("BraTS"):
                case_ids.append(folder)
        
        return case_ids
    
    def _load_nifti(self, file_path: str) -> Tuple[np.ndarray, dict]:
        """加载NIfTI文件"""
        image = sitk.ReadImage(file_path)
        array = sitk.GetArrayFromImage(image)
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        direction = image.GetDirection()
        
        metadata = {
            "spacing": spacing,
            "origin": origin,
            "direction": direction
        }
        
        return array, metadata
    
    def _load_case(self, case_id: str) -> Dict[str, np.ndarray]:
        """加载单个案例的所有模态"""
        case_dir = os.path.join(self.data_dir, case_id)
        
        data = {}
        for mod in self.modalities:
            file_path = os.path.join(case_dir, f"{case_id}_{mod}.nii.gz")
            if os.path.exists(file_path):
                data[mod], _ = self._load_nifti(file_path)
            else:
                raise FileNotFoundError(f"模态文件不存在: {file_path}")
        
        # 加载标签
        seg_path = os.path.join(case_dir, f"{case_id}_seg.nii.gz")
        if os.path.exists(seg_path):
            data["label"], _ = self._load_nifti(seg_path)
        else:
            data["label"] = None
        
        return data
    
    def _get_preprocessing_transforms(self):
        """获取预处理变换"""
        return mtransforms.Compose([
            mtransforms.LoadImageD(keys=self.modalities + ["label"]),
            mtransforms.AddChannelD(keys=self.modalities),
            mtransforms.SpacingD(
                keys=self.modalities + ["label"],
                pixdim=self.target_spacing,
                mode=("bilinear", "nearest")
            ),
            mtransforms.Orientationd(
                keys=self.modalities + ["label"],
                axcodes="RAS"
            ),
        ])
    
    def _normalize_intensity(
        self, 
        image: np.ndarray, 
        percentiles: Tuple[float, float] = (0.5, 99.5)
    ) -> np.ndarray:
        """
        强度归一化
        
        Args:
            image: 输入图像
            percentiles: 用于裁剪的百分位数
        
        Returns:
            归一化后的图像
        """
        # 裁剪异常值
        lower, upper = np.percentile(image, percentiles)
        image = np.clip(image, lower, upper)
        
        # Z-score标准化
        mean = image.mean()
        std = image.std()
        if std > 0:
            image = (image - mean) / std
        
        return image.astype(np.float32)
    
    def _crop_or_pad(
        self, 
        volume: np.ndarray, 
        target_size: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        裁剪或填充到目标大小
        
        Args:
            volume: 输入体数据
            target_size: 目标尺寸
        
        Returns:
            调整后的体数据
        """
        current_shape = volume.shape
        
        # 计算裁剪起点
        starts = [(c - t) // 2 for c, t in zip(current_shape, target_size)]
        starts = [max(0, s) for s in starts]
        
        # 裁剪
        ends = [s + t for s, t in zip(starts, target_size)]
        ends = [min(e, c) for e, c in zip(ends, current_shape)]
        
        cropped = volume[
            starts[0]:ends[0],
            starts[1]:ends[1],
            starts[2]:ends[2]
        ]
        
        # 填充（如需要）
        if cropped.shape != tuple(target_size):
            padded = np.zeros(target_size, dtype=volume.dtype)
            p_starts = [(t - c) // 2 for c, t in zip(cropped.shape, target_size)]
            p_ends = [p + c for p, c in zip(p_starts, cropped.shape)]
            padded[p_starts[0]:p_ends[0], p_starts[1]:p_ends[1], p_starts[2]:p_ends[2]] = cropped
            return padded
        
        return cropped
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.case_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
        
        Returns:
            包含图像和标签的字典
        """
        case_id = self.case_ids[idx]
        
        # 加载数据
        if self.preload and case_id in self.data_cache:
            data = self.data_cache[case_id]
        else:
            data = self._load_case(case_id)
        
        # 堆叠多模态数据
        images = np.stack([
            self._normalize_intensity(data[mod]) for mod in self.modalities
        ], axis=0)  # (C, H, W, D)
        
        # 处理标签
        if data["label"] is not None:
            label = data["label"].astype(np.int64)
        else:
            label = np.zeros_like(images[0], dtype=np.int64)
        
        # 裁剪或填充
        images = self._crop_or_pad(images, (len(self.modalities),) + self.crop_size)
        label = self._crop_or_pad(label, self.crop_size)
        
        # 应用增强变换
        if self.transform:
            # MONAI期望通道在最后
            images = np.moveaxis(images, 0, -1)
            label = np.moveaxis(label, 0, -1)
            
            transformed = self.transform({
                "image": images,
                "label": label
            })
            
            images = np.moveaxis(transformed["image"], -1, 0)
            label = np.moveaxis(transformed["label"], -1, 0)
        
        # 转换为张量
        images = torch.from_numpy(images).float()
        label = torch.from_numpy(label).long()
        
        return {
            "case_id": case_id,
            "image": images,
            "label": label
        }


class BrATSDataModule:
    """
    BraTS数据模块
    
    用于PyTorch Lightning或自定义训练循环的数据管理
    """
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 2,
        num_workers: int = 4,
        modalities: List[str] = ["t1", "t2", "flair", "t1ce"],
        crop_size: Tuple[int, int, int] = (128, 128, 128),
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ):
        """
        初始化数据模块
        
        Args:
            data_dir: 数据目录
            batch_size: 批大小
            num_workers: 数据加载线程数
            modalities: 使用的模态
            crop_size: 裁剪尺寸
            target_spacing: 目标间距
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.modalities = modalities
        self.crop_size = crop_size
        self.target_spacing = target_spacing
        
        self.train_transform = self._get_train_transform()
        self.val_transform = self._get_val_transform()
    
    def _get_train_transform(self):
        """训练数据增强"""
        return mtransforms.Compose([
            mtransforms.RandRotated(
                keys=["image", "label"],
                range_x=15,
                range_y=15,
                range_z=15,
                prob=0.5,
                mode=("bilinear", "nearest")
            ),
            mtransforms.RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0, 1, 2],
                prob=0.5
            ),
            mtransforms.RandGaussianNoised(
                keys=["image"],
                prob=0.3,
                mean=0.0,
                std=0.1
            ),
            mtransforms.RandAdjustContrastd(
                keys=["image"],
                prob=0.3,
                gamma=(0.7, 1.5)
            ),
            mtransforms.RandZoomd(
                keys=["image", "label"],
                prob=0.2,
                min_zoom=0.8,
                max_zoom=1.2,
                mode=("trilinear", "nearest")
            ),
            mtransforms.RandGibbsNoised(
                keys=["image"],
                prob=0.2,
                alpha=(0.5, 1.0)
            ),
            mtransforms.RandKSpaceSpikeNoised(
                keys=["image"],
                prob=0.1,
                intensity_range=(0.5, 1.5)
            )
        ])
    
    def _get_val_transform(self):
        """验证数据变换（仅基础预处理）"""
        return None
    
    def setup(self, stage: Optional[str] = None):
        """
        设置训练、验证、测试数据集
        
        Args:
            stage: 当前阶段 ('fit', 'validate', 'test', 或 None)
        """
        # 划分数据集
        full_dataset = MultiModalBrATS(
            data_dir=self.data_dir,
            split="full",
            modalities=self.modalities,
            target_spacing=self.target_spacing,
            crop_size=self.crop_size
        )
        
        total_size = len(full_dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # 设置变换
        self.train_dataset.dataset.transform = self.train_transform
        self.val_dataset.dataset.transform = self.val_transform
        self.test_dataset.dataset.transform = self.val_transform
    
    def train_dataloader(self):
        """训练数据加载器"""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        """验证数据加载器"""
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        """测试数据加载器"""
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
