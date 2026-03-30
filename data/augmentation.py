"""
医学图像数据增强模块

提供适用于3D医学影像的数据增强方法
包括弹性形变、翻转、旋转、强度变换等
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import random


class MedicalImageAugmentation:
    """
    医学图像增强器
    
    提供多种3D医学影像专用增强方法
    """
    
    def __init__(
        self,
        elastic_alpha: float = 30.0,
        elastic_sigma: float = 10.0,
        rotation_range: float = 15.0,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        intensity_shift_range: float = 0.1,
        intensity_scale_range: Tuple[float, float] = (0.9, 1.1),
        flip_probability: float = 0.5,
        noise_std: float = 0.05,
        gamma_range: Tuple[float, float] = (0.7, 1.5)
    ):
        """
        初始化增强器
        
        Args:
            elastic_alpha: 弹性形变alpha参数
            elastic_sigma: 弹性形变sigma参数
            rotation_range: 旋转角度范围（度）
            scale_range: 缩放范围
            intensity_shift_range: 强度平移范围
            intensity_scale_range: 强度缩放范围
            flip_probability: 翻转概率
            noise_std: 噪声标准差
            gamma_range: Gamma变换范围
        """
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.intensity_shift_range = intensity_shift_range
        self.intensity_scale_range = intensity_scale_range
        self.flip_probability = flip_probability
        self.noise_std = noise_std
        self.gamma_range = gamma_range
    
    def apply_elastic_deformation(
        self,
        image: np.ndarray,
        label: Optional[np.ndarray] = None,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        应用弹性形变
        
        Args:
            image: 输入图像 (C, H, W, D) 或 (H, W, D)
            label: 标签图像（可选）
            seed: 随机种子
        
        Returns:
            变形后的图像和标签
        """
        if seed is not None:
            np.random.seed(seed)
        
        if image.ndim == 4:
            # 多通道图像
            channels, depth, height, width = image.shape
            img_shape = (depth, height, width)
        else:
            channels = None
            img_shape = image.shape
        
        # 生成位移场
        dx = self._elastic_deformation_3d(img_shape, self.elastic_alpha, self.elastic_sigma)
        dy = self._elastic_deformation_3d(img_shape, self.elastic_alpha, self.elastic_sigma)
        dz = self._elastic_deformation_3d(img_shape, self.elastic_alpha, self.elastic_sigma)
        
        # 应用位移场（简化的双线性插值）
        deformed_image = self._apply_displacement(image, dx, dy, dz)
        
        if label is not None:
            deformed_label = self._apply_displacement_label(label, dx, dy, dz)
            return deformed_image, deformed_label
        
        return deformed_image, None
    
    def _elastic_deformation_3d(
        self,
        shape: Tuple[int, int, int],
        alpha: float,
        sigma: float
    ) -> np.ndarray:
        """
        生成3D弹性位移场
        """
        from scipy.ndimage import gaussian_filter
        
        size = shape[0] * shape[1] * shape[2]
        random_state = np.random.RandomState(None)
        
        displacement = random_state.randn(3, size).astype(np.float32)
        
        # 使用高斯滤波平滑位移场
        for i in range(3):
            displacement[i] = gaussian_filter(
                displacement[i].reshape(shape),
                sigma,
                mode="constant"
            ).flatten() * alpha / max(shape)
        
        return displacement
    
    def _apply_displacement(
        self,
        image: np.ndarray,
        dx: np.ndarray,
        dy: np.ndarray,
        dz: np.ndarray
    ) -> np.ndarray:
        """
        应用位移场到图像
        """
        from scipy.ndimage import map_coordinates
        
        if image.ndim == 4:
            channels = image.shape[0]
            shape = image.shape[1:]
            deformed = np.zeros_like(image)
            
            for c in range(channels):
                coords = self._get_coords(shape, dx, dy, dz)
                deformed[c] = map_coordinates(
                    image[c],
                    coords,
                    order=1,
                    mode='reflect'
                ).reshape(shape)
            
            return deformed
        else:
            coords = self._get_coords(image.shape, dx, dy, dz)
            return map_coordinates(
                image,
                coords,
                order=1,
                mode='reflect'
            ).reshape(image.shape)
    
    def _apply_displacement_label(
        self,
        label: np.ndarray,
        dx: np.ndarray,
        dy: np.ndarray,
        dz: np.ndarray
    ) -> np.ndarray:
        """
        应用位移场到标签（使用最近邻插值）
        """
        from scipy.ndimage import map_coordinates
        
        coords = self._get_coords(label.shape, dx, dy, dz)
        return map_coordinates(
            label,
            coords,
            order=0,  # 最近邻插值
            mode='reflect'
        ).reshape(label.shape)
    
    def _get_coords(
        self,
        shape: Tuple,
        dx: np.ndarray,
        dy: np.ndarray,
        dz: np.ndarray
    ) -> np.ndarray:
        """
        生成网格坐标
        """
        d, h, w = shape
        x, y, z = np.meshgrid(
            np.arange(w),
            np.arange(h),
            np.arange(d),
            indexing='ij'
        )
        
        coords = np.array([
            (x + dx).flatten() / (w - 1) * 2 - 1,
            (y + dy).flatten() / (h - 1) * 2 - 1,
            (z + dz).flatten() / (d - 1) * 2 - 1
        ])
        
        return coords
    
    def apply_random_flip(
        self,
        image: np.ndarray,
        label: Optional[np.ndarray] = None,
        axes: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        随机翻转
        
        Args:
            image: 输入图像
            label: 标签（可选）
            axes: 翻转轴列表
        
        Returns:
            翻转后的图像和标签
        """
        if axes is None:
            axes = [0, 1, 2]
        
        deformed_image = image.copy()
        deformed_label = label.copy() if label is not None else None
        
        for axis in axes:
            if random.random() < self.flip_probability:
                deformed_image = np.flip(deformed_image, axis=axis + 1 if deformed_image.ndim == 4 else axis)
                if deformed_label is not None:
                    deformed_label = np.flip(deformed_label, axis=axis)
        
        return deformed_image, deformed_label
    
    def apply_random_rotation(
        self,
        image: np.ndarray,
        label: Optional[np.ndarray] = None,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        随机旋转（90度）
        
        Args:
            image: 输入图像
            label: 标签（可选）
            seed: 随机种子
        
        Returns:
            旋转后的图像和标签
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        k = random.randint(0, 3)  # 旋转0, 90, 180, 270度
        
        if k == 0:
            return image, label
        
        deformed_image = np.rot90(image, k=k, axes=(-2, -1))
        deformed_label = np.rot90(label, k=k, axes=(-2, -1)) if label is not None else None
        
        return deformed_image, deformed_label
    
    def apply_intensity_augmentation(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        强度增强
        
        Args:
            image: 输入图像
        
        Returns:
            增强后的图像
        """
        # 强度缩放
        scale = np.random.uniform(*self.intensity_scale_range)
        image = image * scale
        
        # 强度平移
        shift = np.random.uniform(-self.intensity_shift_range, self.intensity_shift_range)
        image = image + shift
        
        # Gamma变换
        gamma = np.random.uniform(*self.gamma_range)
        image = np.power(
            np.clip(image / (image.max() + 1e-8), 1e-8, 1.0),
            gamma
        ) * image.max()
        
        # 高斯噪声
        if np.random.random() < 0.3:
            noise = np.random.normal(0, self.noise_std, image.shape)
            image = image + noise
        
        return image
    
    def apply_zoom(
        self,
        image: np.ndarray,
        label: Optional[np.ndarray] = None,
        zoom_factor: Optional[float] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        随机缩放
        
        Args:
            image: 输入图像
            label: 标签（可选）
            zoom_factor: 缩放因子（可选）
        
        Returns:
            缩放后的图像和标签
        """
        from scipy.ndimage import zoom as scipy_zoom
        
        if zoom_factor is None:
            zoom_factor = np.random.uniform(*self.scale_range)
        
        if image.ndim == 4:
            zoom_factors = (1, zoom_factor, zoom_factor, zoom_factor)
        else:
            zoom_factors = (zoom_factor, zoom_factor, zoom_factor)
        
        deformed_image = scipy_zoom(image, zoom_factors, order=1)
        deformed_label = scipy_zoom(label, zoom_factors[1:], order=0) if label is not None else None
        
        return deformed_image, deformed_label
    
    def __call__(
        self,
        image: np.ndarray,
        label: Optional[np.ndarray] = None,
        augment_prob: float = 0.8
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        应用完整增强流程
        
        Args:
            image: 输入图像
            label: 标签（可选）
            augment_prob: 增强概率
        
        Returns:
            增强后的图像和标签
        """
        aug_image = image.copy()
        aug_label = label.copy() if label is not None else None
        
        if random.random() < augment_prob:
            # 弹性形变
            if random.random() < 0.5:
                aug_image, aug_label = self.apply_elastic_deformation(
                    aug_image, aug_label
                )
            
            # 翻转
            aug_image, aug_label = self.apply_random_flip(aug_image, aug_label)
            
            # 旋转
            if random.random() < 0.5:
                aug_image, aug_label = self.apply_random_rotation(aug_image, aug_label)
            
            # 缩放
            if random.random() < 0.3:
                aug_image, aug_label = self.apply_zoom(aug_image, aug_label)
            
            # 强度增强
            aug_image = self.apply_intensity_augmentation(aug_image)
        
        return aug_image, aug_label


class MixUpAugmentation:
    """
    MixUp数据增强
    
    通过混合两个样本来生成新的训练样本
    """
    
    def __init__(self, alpha: float = 0.4):
        """
        Args:
            alpha: Beta分布参数
        """
        self.alpha = alpha
    
    def __call__(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        label1: torch.Tensor,
        label2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        应用MixUp
        
        Args:
            image1, image2: 两个样本的图像
            label1, label2: 两个样本的标签
        
        Returns:
            混合图像和标签
        """
        lam = np.random.beta(self.alpha, self.alpha)
        
        mixed_image = lam * image1 + (1 - lam) * image2
        mixed_label = lam * label1 + (1 - lam) * label2
        
        return mixed_image, mixed_label, lam


class CutMixAugmentation:
    """
    CutMix数据增强
    
    通过裁剪和混合两个样本来生成新的训练样本
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha: Beta分布参数
        """
        self.alpha = alpha
    
    def __call__(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        label1: torch.Tensor,
        label2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        应用CutMix
        
        Args:
            image1, image2: 两个样本的图像
            label1, label2: 两个样本的标签
        
        Returns:
            混合图像和标签
        """
        lam = np.random.beta(self.alpha, self.alpha)
        
        B, C, H, W, D = image1.shape
        
        # 生成裁剪区域
        cut_ratios = np.sqrt(1 - lam)
        cut_h = int(H * cut_ratios)
        cut_w = int(W * cut_ratios)
        cut_d = int(D * cut_ratios)
        
        # 随机裁剪位置
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        cz = np.random.randint(D)
        
        # 裁剪区域边界
        x1 = np.clip(cx - cut_w // 2, 0, W)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        y2 = np.clip(cy + cut_h // 2, 0, H)
        z1 = np.clip(cz - cut_d // 2, 0, D)
        z2 = np.clip(cz + cut_d // 2, 0, D)
        
        # 应用CutMix
        mixed_image = image1.clone()
        mixed_image[:, :, y1:y2, x1:x2, z1:z2] = image2[:, :, y1:y2, x1:x2, z1:z2]
        
        # 计算实际混合比例
        actual_lam = 1 - ((x2 - x1) * (y2 - y1) * (z2 - z1)) / (H * W * D)
        mixed_label = actual_lam * label1 + (1 - actual_lam) * label2
        
        return mixed_image, mixed_label, actual_lam
