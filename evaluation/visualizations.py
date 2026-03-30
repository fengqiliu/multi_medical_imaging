"""
可视化模块

提供医学影像和模型结果的可视化功能
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import SimpleITK as sitk
import os


class Visualizer:
    """
    医学影像可视化工具
    
    提供切片可视化、3D渲染、对比图等功能
    """
    
    def __init__(
        self,
        num_classes: int,
        class_colors: Optional[List[str]] = None,
        slice_axis: int = 2
    ):
        """
        Args:
            num_classes: 分割类别数
            class_colors: 类别颜色列表
            slice_axis: 默认切片轴 (0=x, 1=y, 2=z)
        """
        self.num_classes = num_classes
        self.slice_axis = slice_axis
        
        # 默认颜色映射
        if class_colors is None:
            self.class_colors = [
                'black',      # 背景
                'red',       # 肿瘤核心
                'yellow',    # 水肿
                'green'      # 增强肿瘤
            ][:num_classes]
        else:
            self.class_colors = class_colors
        
        self.cmap = ListedColormap(self.class_colors)
    
    def create_overlay(
        self,
        image: np.ndarray,
        prediction: np.ndarray,
        target: Optional[np.ndarray] = None,
        slice_idx: Optional[int] = None,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        创建预测覆盖图
        
        Args:
            image: 图像切片
            prediction: 预测掩码
            target: 目标掩码（可选）
            slice_idx: 切片索引
            alpha: 透明度
        
        Returns:
            覆盖图
        """
        # 归一化图像到[0, 1]
        if image.max() > image.min():
            image_norm = (image - image.min()) / (image.max() - image.min())
        else:
            image_norm = image
        
        # 创建彩色掩码
        overlay = np.zeros((*image.shape, 3))
        
        for c in range(1, self.num_classes):  # 跳过背景
            color = self._hex_to_rgb(self.class_colors[c])
            mask = (prediction == c).astype(float)
            overlay += np.array(color) * mask[:, :, np.newaxis] * alpha
        
        # 组合图像和掩码
        result = image_norm[:, :, np.newaxis] * (1 - alpha) + overlay
        
        return np.clip(result, 0, 1)
    
    def plot_slice_comparison(
        self,
        images: List[np.ndarray],
        predictions: Optional[List[np.ndarray]] = None,
        targets: Optional[List[np.ndarray]] = None,
        slice_idx: Optional[int] = None,
        titles: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """
        绘制切片对比图
        
        Args:
            images: 图像列表
            predictions: 预测列表
            targets: 目标列表
            slice_idx: 切片索引
            titles: 标题列表
            save_path: 保存路径
        """
        n_cols = 1 + (predictions is not None) + (targets is not None)
        n_rows = len(images)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for row, img in enumerate(images):
            col = 0
            
            # 显示图像
            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].axis('off')
            if titles and row < len(titles):
                axes[row, col].set_title(titles[row])
            col += 1
            
            # 显示预测
            if predictions is not None:
                axes[row, col].imshow(predictions[row], cmap=self.cmap, vmin=0, vmax=self.num_classes-1)
                axes[row, col].axis('off')
                axes[row, col].set_title('Prediction')
                col += 1
            
            # 显示目标
            if targets is not None:
                axes[row, col].imshow(targets[row], cmap=self.cmap, vmin=0, vmax=self.num_classes-1)
                axes[row, col].axis('off')
                axes[row, col].set_title('Target')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图像已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_multimodal_slices(
        self,
        image_dict: Dict[str, np.ndarray],
        slice_idx: int,
        save_path: Optional[str] = None
    ):
        """
        绘制多模态切片
        
        Args:
            image_dict: 模态名称到图像的字典
            slice_idx: 切片索引
            save_path: 保存路径
        """
        n_modalities = len(image_dict)
        fig, axes = plt.subplots(1, n_modalities, figsize=(4 * n_modalities, 4))
        
        if n_modalities == 1:
            axes = [axes]
        
        for ax, (mod_name, img) in zip(axes, image_dict.items()):
            # 获取指定切片
            if self.slice_axis == 0:
                img_slice = img[slice_idx, :, :]
            elif self.slice_axis == 1:
                img_slice = img[:, slice_idx, :]
            else:
                img_slice = img[:, :, slice_idx]
            
            # 归一化并显示
            if img_slice.max() > img_slice.min():
                img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
            
            ax.imshow(img_slice, cmap='gray')
            ax.set_title(mod_name.upper())
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图像已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        save_path: Optional[str] = None
    ):
        """
        绘制训练曲线
        
        Args:
            history: 训练历史 {'train_loss': [...], 'val_loss': [...]}
            save_path: 保存路径
        """
        fig, axes = plt.subplots(1, len(history), figsize=(6 * len(history), 4))
        
        if len(history) == 1:
            axes = [axes]
        
        for ax, (metric_name, values) in zip(axes, history.items()):
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, marker='o')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name)
            ax.set_title(metric_name)
            ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"训练曲线已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_segmentation_results(
        self,
        image: np.ndarray,
        prediction: np.ndarray,
        target: np.ndarray,
        slice_idx: int,
        save_path: Optional[str] = None
    ):
        """
        绘制分割结果对比
        
        Args:
            image: 输入图像
            prediction: 预测掩码
            target: 目标掩码
            slice_idx: 切片索引
            save_path: 保存路径
        """
        # 获取切片
        if self.slice_axis == 0:
            img_slice = image[slice_idx, :, :]
            pred_slice = prediction[slice_idx, :, :]
            target_slice = target[slice_idx, :, :]
        elif self.slice_axis == 1:
            img_slice = image[:, slice_idx, :]
            pred_slice = prediction[:, slice_idx, :]
            target_slice = target[:, slice_idx, :]
        else:
            img_slice = image[:, :, slice_idx]
            pred_slice = prediction[:, :, slice_idx]
            target_slice = target[:, :, slice_idx]
        
        # 创建覆盖图
        pred_overlay = self.create_overlay(img_slice, pred_slice)
        target_overlay = self.create_overlay(img_slice, target_slice)
        
        # 绘图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_slice, cmap='gray')
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        axes[1].imshow(pred_overlay)
        axes[1].set_title('Prediction')
        axes[1].axis('off')
        
        axes[2].imshow(target_overlay)
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"分割结果已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_survival_curve(
        self,
        times: np.ndarray,
        events: np.ndarray,
        risk_groups: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ):
        """
        绘制生存曲线
        
        Args:
            times: 生存时间
            events: 事件指示器
            risk_groups: 风险分组（可选）
            save_path: 保存路径
        """
        from lifelines import KaplanMeierFitter
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if risk_groups is not None:
            for group in np.unique(risk_groups):
                mask = risk_groups == group
                kmf = KaplanMeierFitter()
                kmf.fit(times[mask], events[mask], label=f'Group {group}')
                kmf.plot_survival_function(ax=ax)
        else:
            kmf = KaplanMeierFitter()
            kmf.fit(times, events, label='Overall')
            kmf.plot_survival_function(ax=ax)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Survival Probability')
        ax.set_title('Kaplan-Meier Survival Curve')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"生存曲线已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _hex_to_rgb(self, color: str) -> Tuple[float, float, float]:
        """将十六进制颜色转换为RGB"""
        if color.startswith('#'):
            color = color[1:]
        return tuple(int(color[i:i+2], 16) / 255 for i in (0, 2, 4))


def create_comparison_gif(
    image_slices: List[np.ndarray],
    save_path: str,
    duration: int = 100
):
    """
    创建对比GIF动画
    
    Args:
        image_slices: 图像切片列表
        save_path: 保存路径
        duration: 每帧持续时间（毫秒）
    """
    from PIL import Image
    
    images = []
    for img_slice in image_slices:
        # 归一化
        if img_slice.max() > img_slice.min():
            img_norm = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
        else:
            img_norm = img_slice
        
        # 转换为PIL图像
        img_pil = Image.fromarray((img_norm * 255).astype(np.uint8))
        images.append(img_pil)
    
    # 保存GIF
    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    print(f"GIF已保存: {save_path}")


def save_nifti_visualization(
    volume: np.ndarray,
    save_path: str,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
):
    """
    保存NIfTI可视化
    
    Args:
        volume: 3D体数据
        save_path: 保存路径
        spacing: 体素间距
    """
    img = sitk.GetImageFromArray(volume)
    img.SetSpacing(spacing)
    sitk.WriteImage(img, save_path)
    print(f"NIfTI已保存: {save_path}")
