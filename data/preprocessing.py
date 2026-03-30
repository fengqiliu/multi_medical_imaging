"""
图像预处理模块

提供医学影像的标准化预处理流程
包括重采样、强度归一化、偏置场校正等功能
"""

import numpy as np
import SimpleITK as sitk
from typing import Tuple, Optional, List
import nibabel as nib


class ImagePreprocessor:
    """
    医学图像预处理器
    
    提供完整的医学图像预处理流程
    """
    
    def __init__(
        self,
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        normalize_method: str = "zscore",
        bias_correction: bool = True
    ):
        """
        初始化预处理器
        
        Args:
            target_spacing: 目标体素间距 (mm)
            normalize_method: 归一化方法 ('zscore', 'minmax', 'histogram')
            bias_correction: 是否进行偏置场校正
        """
        self.target_spacing = target_spacing
        self.normalize_method = normalize_method
        self.bias_correction = bias_correction
    
    def load_nifti(self, file_path: str) -> Tuple[np.ndarray, dict]:
        """
        加载NIfTI文件
        
        Args:
            file_path: 文件路径
        
        Returns:
            (图像数组, 元数据字典)
        """
        img = nib.load(file_path)
        data = img.get_fdata()
        header = img.header
        affine = img.affine
        
        metadata = {
            "affine": affine,
            "header": header,
            "shape": data.shape
        }
        
        return data, metadata
    
    def save_nifti(
        self,
        data: np.ndarray,
        file_path: str,
        metadata: Optional[dict] = None
    ):
        """
        保存NIfTI文件
        
        Args:
            data: 图像数据
            file_path: 保存路径
            metadata: 元数据（可选）
        """
        if metadata is not None and "affine" in metadata:
            img = nib.Nifti1Image(data, metadata["affine"])
        else:
            img = nib.Nifti1Image(data, np.eye(4))
        
        nib.save(img, file_path)
    
    def resample(
        self,
        image: sitk.Image,
        target_spacing: Optional[Tuple[float, float, float]] = None,
        interpolator: int = sitk.sitkLinear,
        default_value: float = 0.0
    ) -> sitk.Image:
        """
        重采样图像到目标间距
        
        Args:
            image: SimpleITK图像
            target_spacing: 目标间距（默认使用初始化时的值）
            interpolator: 插值器类型
            default_value: 默认填充值
        
        Returns:
            重采样后的图像
        """
        if target_spacing is None:
            target_spacing = self.target_spacing
        
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        
        # 计算新的尺寸
        target_size = [
            int(round(osz * osp / tsp))
            for osz, osp, tsp in zip(original_size, original_spacing, target_spacing)
        ]
        
        # 创建重采样过滤器
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(target_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(default_value)
        resampler.SetInterpolator(interpolator)
        
        return resampler.Execute(image)
    
    def normalize_zscore(
        self,
        image: np.ndarray,
        percentiles: Tuple[float, float] = (0.5, 99.5)
    ) -> np.ndarray:
        """
        Z-score标准化
        
        Args:
            image: 输入图像
            percentiles: 裁剪百分位
        
        Returns:
            标准化后的图像
        """
        # 裁剪异常值
        lower = np.percentile(image, percentiles[0])
        upper = np.percentile(image, percentiles[1])
        image = np.clip(image, lower, upper)
        
        # 标准化
        mean = image.mean()
        std = image.std()
        
        if std > 0:
            image = (image - mean) / std
        
        return image.astype(np.float32)
    
    def normalize_minmax(
        self,
        image: np.ndarray,
        output_range: Tuple[float, float] = (0, 1)
    ) -> np.ndarray:
        """
        Min-Max归一化
        
        Args:
            image: 输入图像
            output_range: 输出范围
        
        Returns:
            归一化后的图像
        """
        min_val = image.min()
        max_val = image.max()
        
        if max_val > min_val:
            image = (image - min_val) / (max_val - min_val)
        else:
            image = np.zeros_like(image)
        
        # 缩放到目标范围
        out_min, out_max = output_range
        image = image * (out_max - out_min) + out_min
        
        return image.astype(np.float32)
    
    def normalize_histogram(
        self,
        image: np.ndarray,
        reference_hist: Optional[np.ndarray] = None,
        n_bins: int = 256
    ) -> np.ndarray:
        """
        直方图匹配
        
        Args:
            image: 输入图像
            reference_hist: 参考直方图（可选）
            n_bins: 直方图bin数量
        
        Returns:
            归一化后的图像
        """
        # 转换为SimpleITK图像进行直方图匹配
        img_sitk = sitk.GetImageFromArray(image)
        
        if reference_hist is None:
            # 使用参考百分位数进行匹配
            lower = np.percentile(image, 1)
            upper = np.percentile(image, 99)
            matched = sitk.HistogramMatchingImageFilter()
            matched.SetNumberOfHistogramLevels(n_bins)
            matched.SetMatchPoints(10)
            matched.SetThresholdAtMeanIntensity(True)
        else:
            matched = sitk.HistogramMatchingImageFilter()
            matched.SetNumberOfHistogramLevels(n_bins)
            matched.SetMatchPoints(10)
        
        result = matched.Execute(img_sitk, reference_hist)
        return sitk.GetArrayFromImage(result).astype(np.float32)
    
    def bias_correction_n4(self, image: sitk.Image) -> sitk.Image:
        """
        N4偏置场校正
        
        Args:
            image: SimpleITK图像
        
        Returns:
            校正后的图像
        """
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrected = corrector.Execute(image)
        return corrected
    
    def skull_strip_bet(
        self,
        image: sitk.Image,
        mask: Optional[sitk.Image] = None
    ) -> Tuple[sitk.Image, sitk.Image]:
        """
        颅骨剥离（使用BET算法概念的实现）
        
        Args:
            image: 输入图像
            mask: 脑组织掩码（可选）
        
        Returns:
            (剥离后的图像, 脑组织掩码)
        """
        # Otsu阈值分离脑组织和背景
        otsu = sitk.OtsuThresholdImageFilter()
        otsu.SetInsideValue(1)
        otsu.SetOutsideValue(0)
        brain_mask = otsu.Execute(image)
        
        # 形态学开运算去除小的非脑组织
        brain_mask = sitk.BinaryMorphologicalOpening(
            brain_mask,
            sitk.sitkBall,
            3
        )
        
        # 填充小的空洞
        brain_mask = sitk.BinaryMorphologicalClosing(
            brain_mask,
            sitk.sitkBall,
            2
        )
        
        # 应用掩码
        stripped = sitk.Mask(image, brain_mask)
        
        return stripped, brain_mask
    
    def preprocess_volume(
        self,
        image: np.ndarray,
        spacing: Tuple[float, float, float],
        apply_bias_correction: bool = False
    ) -> np.ndarray:
        """
        完整预处理流程
        
        Args:
            image: 输入图像数组
            spacing: 当前图像间距
            apply_bias_correction: 是否应用偏置场校正
        
        Returns:
            预处理后的图像
        """
        # 转换为SimpleITK图像
        img_sitk = sitk.GetImageFromArray(image)
        img_sitk.SetSpacing(spacing)
        
        # 偏置场校正
        if apply_bias_correction and self.bias_correction:
            img_sitk = self.bias_correction_n4(img_sitk)
        
        # 重采样
        img_sitk = self.resample(img_sitk, self.target_spacing)
        
        # 转回numpy
        image = sitk.GetArrayFromImage(img_sitk)
        
        # 强度归一化
        if self.normalize_method == "zscore":
            image = self.normalize_zscore(image)
        elif self.normalize_method == "minmax":
            image = self.normalize_minmax(image)
        
        return image


class IntensityNormalizer:
    """
    强度归一化工具类
    
    提供多种归一化方法用于医学影像处理
    """
    
    @staticmethod
    def normalize_case(
        images: List[np.ndarray],
        method: str = "zscore"
    ) -> List[np.ndarray]:
        """
        对齐归一化多模态图像
        
        Args:
            images: 多模态图像列表
            method: 归一化方法
        
        Returns:
            归一化后的图像列表
        """
        normalizer = ImagePreprocessor(normalize_method=method)
        normalized = []
        
        for img in images:
            if method == "zscore":
                normalized.append(normalizer.normalize_zscore(img))
            elif method == "minmax":
                normalized.append(normalizer.normalize_minmax(img))
            else:
                normalized.append(img.astype(np.float32))
        
        return normalized
    
    @staticmethod
    def compute_radiomics_normalization(
        image: np.ndarray
    ) -> Tuple[float, float]:
        """
        计算影像组学归一化参数
        
        Args:
            image: 输入图像
        
        Returns:
            (均值, 标准差)
        """
        return float(image.mean()), float(image.std())
