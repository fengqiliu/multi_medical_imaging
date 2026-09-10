"""
BraTS多模态脑肿瘤分割数据集

该数据集支持多模态MRI图像的加载、预处理和增强，
适用于脑肿瘤分割任务。
"""

import os
import random
import json
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
        preload: bool = False,
        case_ids: Optional[List[str]] = None,
        require_label: bool = True,
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
        self.require_label = require_label
        
        # 扫描数据目录
        self.case_ids = list(case_ids) if case_ids is not None else self._scan_cases()
        
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
    
    def _load_nifti(
        self,
        file_path: str,
        is_label: bool = False,
    ) -> Tuple[np.ndarray, dict]:
        """加载、统一方向和体素间距后的NIfTI数组。"""
        image = sitk.ReadImage(file_path)
        image = sitk.DICOMOrient(image, "RAS")
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()

        if tuple(original_spacing) != tuple(self.target_spacing):
            output_size = [
                max(1, int(round(size * spacing / target)))
                for size, spacing, target in zip(
                    original_size, original_spacing, self.target_spacing
                )
            ]
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(tuple(self.target_spacing))
            resampler.SetSize(output_size)
            resampler.SetOutputDirection(image.GetDirection())
            resampler.SetOutputOrigin(image.GetOrigin())
            resampler.SetTransform(sitk.Transform())
            resampler.SetInterpolator(
                sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
            )
            image = resampler.Execute(image)

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
                data[mod], _ = self._load_nifti(file_path, is_label=False)
            else:
                raise FileNotFoundError(f"模态文件不存在: {file_path}")
        
        # 加载标签
        seg_path = os.path.join(case_dir, f"{case_id}_seg.nii.gz")
        if os.path.exists(seg_path):
            data["label"], _ = self._load_nifti(seg_path, is_label=True)
        elif self.require_label:
            raise FileNotFoundError(f"分割标签不存在: {seg_path}")
        else:
            data["label"] = None
        
        return data
    
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
        if len(current_shape) != len(target_size):
            raise ValueError(
                f"体数据维度 {current_shape} 与目标维度 {target_size} 不一致"
            )
        
        # 计算裁剪起点
        starts = [(c - t) // 2 for c, t in zip(current_shape, target_size)]
        starts = [max(0, s) for s in starts]
        
        # 裁剪
        ends = [s + t for s, t in zip(starts, target_size)]
        ends = [min(e, c) for e, c in zip(ends, current_shape)]
        
        cropped = volume[tuple(slice(start, end) for start, end in zip(starts, ends))]
        
        # 填充（如需要）
        if cropped.shape != tuple(target_size):
            padded = np.zeros(target_size, dtype=volume.dtype)
            p_starts = [(t - c) // 2 for c, t in zip(cropped.shape, target_size)]
            p_ends = [p + c for p, c in zip(p_starts, cropped.shape)]
            destination = tuple(
                slice(start, end) for start, end in zip(p_starts, p_ends)
            )
            padded[destination] = cropped
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
            unique_labels = set(np.unique(label).tolist())
            unexpected_labels = unique_labels.difference({0, 1, 2, 4})
            if unexpected_labels:
                raise ValueError(
                    f"{case_id} 标签包含未支持的值: {sorted(unexpected_labels)}"
                )
            # BraTS 原始标签为 0/1/2/4，四分类损失需要连续索引 0/1/2/3。
            label = np.where(label == 4, 3, label).astype(np.int64)
        else:
            label = np.zeros_like(images[0], dtype=np.int64)
        
        # 裁剪或填充
        images = self._crop_or_pad(images, (len(self.modalities),) + self.crop_size)
        label = self._crop_or_pad(label, self.crop_size)
        
        # 应用增强变换
        if self.transform:
            # MONAI 字典变换使用通道优先；标签补一个通道后再移除。
            transformed = self.transform({
                "image": images,
                "label": label[None, ...],
            })
            images = np.asarray(transformed["image"])
            label = np.asarray(transformed["label"])[0]
        
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
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        train_split: float = 0.7,
        val_split: float = 0.15,
        seed: int = 42,
        preload: bool = False,
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
        self.train_split = train_split
        self.val_split = val_split
        self.seed = seed
        self.preload = preload
        
        self.train_transform = self._get_train_transform()
        self.val_transform = self._get_val_transform()
    
    def _get_train_transform(self):
        """训练数据增强"""
        return mtransforms.Compose([
            mtransforms.RandRotateD(
                keys=["image", "label"],
                range_x=np.deg2rad(15),
                range_y=np.deg2rad(15),
                range_z=np.deg2rad(15),
                prob=0.5,
                mode=("bilinear", "nearest")
            ),
            mtransforms.RandFlipD(
                keys=["image", "label"],
                spatial_axis=[0, 1, 2],
                prob=0.5
            ),
            mtransforms.RandGaussianNoiseD(
                keys=["image"],
                prob=0.3,
                mean=0.0,
                std=0.1
            ),
            mtransforms.RandAdjustContrastD(
                keys=["image"],
                prob=0.3,
                gamma=(0.7, 1.5)
            ),
            mtransforms.RandZoomD(
                keys=["image", "label"],
                prob=0.2,
                min_zoom=0.8,
                max_zoom=1.2,
                mode=("trilinear", "nearest")
            ),
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
        # 先按病例ID拆分，再为每个子集创建独立Dataset，避免共享transform。
        discovery_dataset = MultiModalBrATS(
            data_dir=self.data_dir,
            split="full",
            modalities=self.modalities,
            target_spacing=self.target_spacing,
            crop_size=self.crop_size,
            require_label=True,
        )
        case_ids = list(discovery_dataset.case_ids)
        total_size = len(case_ids)
        if total_size < 4:
            raise ValueError("训练/验证/测试三份拆分至少需要4个带标签病例")

        rng = random.Random(self.seed)
        rng.shuffle(case_ids)
        train_size = max(1, int(self.train_split * total_size))
        val_size = max(1, int(self.val_split * total_size))
        if train_size + val_size >= total_size:
            val_size = max(1, total_size - train_size - 1)
        test_size = total_size - train_size - val_size

        train_ids = case_ids[:train_size]
        val_ids = case_ids[train_size:train_size + val_size]
        test_ids = case_ids[train_size + val_size:]
        self.split_case_ids = {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        }

        common = dict(
            data_dir=self.data_dir,
            modalities=self.modalities,
            target_spacing=self.target_spacing,
            crop_size=self.crop_size,
            preload=self.preload,
            require_label=True,
        )
        self.train_dataset = MultiModalBrATS(
            transform=self.train_transform, case_ids=train_ids, **common
        )
        self.val_dataset = MultiModalBrATS(
            transform=self.val_transform, case_ids=val_ids, **common
        )
        self.test_dataset = MultiModalBrATS(
            transform=self.val_transform, case_ids=test_ids, **common
        )
    
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

    def save_split_manifest(self, path: str):
        """保存病例级拆分清单，便于复现实验和审计患者边界。"""
        if not hasattr(self, "split_case_ids"):
            raise RuntimeError("请先调用 setup() 创建数据拆分")
        manifest_path = os.fspath(path)
        parent = os.path.dirname(manifest_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as file:
            json.dump(self.split_case_ids, file, ensure_ascii=False, indent=2)
    
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
