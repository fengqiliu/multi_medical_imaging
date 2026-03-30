"""
评估指标模块

提供分割和预后预测任务的评估指标计算
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


class SegmentationMetrics:
    """
    分割任务评估指标
    
    计算Dice系数、IoU、灵敏度、特异度等指标
    """
    
    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None
    ):
        """
        Args:
            num_classes: 类别数量
            class_names: 类别名称列表
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        
        self.reset()
    
    def reset(self):
        """重置累积的指标"""
        self.dice_scores = []
        self.iou_scores = []
        self.sensitivities = []
        self.specificities = []
        self.predictions = []
        self.targets = []
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ):
        """
        更新指标
        
        Args:
            predictions: 预测结果 (B, H, W, D)
            targets: 目标标签 (B, H, W, D)
        """
        self.predictions.append(predictions.cpu().numpy())
        self.targets.append(targets.cpu().numpy())
        
        # 转换为numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        # 计算每个样本的指标
        for pred, target in zip(predictions, targets):
            for c in range(self.num_classes):
                pred_c = (pred == c).astype(np.float32)
                target_c = (target == c).astype(np.float32)
                
                if target_c.sum() == 0:
                    continue
                
                # Dice系数
                intersection = (pred_c * target_c).sum()
                union = pred_c.sum() + target_c.sum()
                dice = 2 * intersection / (union + 1e-8)
                self.dice_scores.append(dice)
                
                # IoU
                iou = intersection / (union - intersection + 1e-8)
                self.iou_scores.append(iou)
                
                # 灵敏度（召回率）
                sensitivity = intersection / (target_c.sum() + 1e-8)
                self.sensitivities.append(sensitivity)
                
                # 特异度
                tn = ((pred_c == 0) & (target_c == 0)).sum()
                fp = ((pred_c == 1) & (target_c == 0)).sum()
                specificity = tn / (tn + fp + 1e-8)
                self.specificities.append(specificity)
    
    def compute(self) -> Dict[str, float]:
        """
        计算平均指标
        
        Returns:
            指标字典
        """
        metrics = {}
        
        if len(self.dice_scores) > 0:
            metrics["dice_mean"] = np.mean(self.dice_scores)
            metrics["dice_std"] = np.std(self.dice_scores)
            metrics["iou_mean"] = np.mean(self.iou_scores)
            metrics["sensitivity_mean"] = np.mean(self.sensitivities)
            metrics["specificity_mean"] = np.mean(self.specificities)
        
        return metrics
    
    def compute_per_class(self) -> Dict[str, List[float]]:
        """
        计算每个类别的指标
        
        Returns:
            每个类别的指标
        """
        if len(self.predictions) == 0:
            return {}
        
        predictions = np.concatenate(self.predictions, axis=0)
        targets = np.concatenate(self.targets, axis=0)
        
        per_class_metrics = {}
        
        for c in range(self.num_classes):
            pred_c = (predictions == c).astype(np.float32)
            target_c = (targets == c).astype(np.float32)
            
            if target_c.sum() == 0:
                continue
            
            # Dice
            intersection = (pred_c * target_c).sum(axis=(1, 2, 3))
            union = pred_c.sum(axis=(1, 2, 3)) + target_c.sum(axis=(1, 2, 3))
            dice_per_sample = 2 * intersection / (union + 1e-8)
            
            class_name = self.class_names[c] if c < len(self.class_names) else f"Class_{c}"
            per_class_metrics[f"{class_name}_dice"] = float(np.mean(dice_per_sample))
        
        return per_class_metrics


class HausdorffDistance:
    """
    Hausdorff距离计算
    
    用于评估分割边界距离
    """
    
    @staticmethod
    def compute(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        percentile: int = 95
    ) -> float:
        """
        计算Hausdorff距离
        
        Args:
            predictions: 预测掩码
            targets: 目标掩码
            percentile: 百分位数（95或100）
        
        Returns:
            Hausdorff距离
        """
        from scipy.spatial.distance import directed_hausdorff
        
        # 转换为numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        # 获取前景点
        pred_points = np.argwhere(predictions > 0)
        target_points = np.argwhere(targets > 0)
        
        if len(pred_points) == 0 or len(target_points) == 0:
            return float('inf')
        
        # 计算有向Hausdorff距离
        hd1 = directed_hausdorff(pred_points, target_points)[0]
        hd2 = directed_hausdorff(target_points, pred_points)[0]
        
        # 返回最大距离
        return max(hd1, hd2)


class SurvivalMetrics:
    """
    生存分析评估指标
    
    计算C-index、AUC、Brier Score等
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置累积值"""
        self.risk_scores = []
        self.times = []
        self.events = []
    
    def update(
        self,
        risk_scores: torch.Tensor,
        times: torch.Tensor,
        events: torch.Tensor
    ):
        """
        更新指标
        
        Args:
            risk_scores: 预测风险分数
            times: 生存时间
            events: 事件指示器
        """
        self.risk_scores.extend(risk_scores.cpu().numpy().flatten())
        self.times.extend(times.cpu().numpy().flatten())
        self.events.extend(events.cpu().numpy().flatten())
    
    def c_index(self) -> float:
        """
        计算C-index（一致性指数）
        
        Returns:
            C-index值
        """
        risk_scores = np.array(self.risk_scores)
        times = np.array(self.times)
        events = np.array(self.events)
        
        n = len(risk_scores)
        concordant = 0
        tied = 0
        comparable = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                # 只比较可比较的对
                # i发生事件且j未删失且Ti > Tj
                if events[i] == 1 and events[j] == 0 and times[i] < times[j]:
                    comparable += 1
                    if risk_scores[i] > risk_scores[j]:
                        concordant += 1
                    elif risk_scores[i] == risk_scores[j]:
                        tied += 0.5
                # 或者两者都发生事件
                elif events[i] == 1 and events[j] == 1:
                    comparable += 1
                    if times[i] < times[j] and risk_scores[i] > risk_scores[j]:
                        concordant += 1
                    elif times[i] > times[j] and risk_scores[i] < risk_scores[j]:
                        concordant += 1
                    elif risk_scores[i] == risk_scores[j]:
                        tied += 0.5
        
        if comparable > 0:
            return (concordant + tied) / comparable
        else:
            return 0.5
    
    def brier_score(self, time_bins: Optional[np.ndarray] = None) -> float:
        """
        计算Brier Score
        
        Args:
            time_bins: 时间分箱
        
        Returns:
            Brier Score
        """
        # 简化版本：使用所有样本的均方误差
        # 实际应用中应该使用Kaplan-Meier加权
        risk_scores = np.array(self.risk_scores)
        events = np.array(self.events)
        
        # 预测概率 = 1 - 累积风险（简化）
        pred_probs = np.clip(1 - np.exp(-risk_scores), 0, 1)
        
        # 均方误差
        brier = np.mean((pred_probs - events) ** 2)
        
        return brier
    
    def compute(self) -> Dict[str, float]:
        """
        计算所有指标
        
        Returns:
            指标字典
        """
        return {
            "c_index": self.c_index(),
            "brier_score": self.brier_score()
        }


class MultiTaskMetrics:
    """
    多任务评估指标
    
    同时评估分割和预后预测
    """
    
    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None
    ):
        """
        Args:
            num_classes: 分割类别数
            class_names: 类别名称
        """
        self.seg_metrics = SegmentationMetrics(num_classes, class_names)
        self.surv_metrics = SurvivalMetrics()
    
    def reset(self):
        """重置所有指标"""
        self.seg_metrics.reset()
        self.surv_metrics.reset()
    
    def update(
        self,
        seg_preds: Optional[torch.Tensor] = None,
        seg_targets: Optional[torch.Tensor] = None,
        risk_scores: Optional[torch.Tensor] = None,
        times: Optional[torch.Tensor] = None,
        events: Optional[torch.Tensor] = None
    ):
        """
        更新指标
        
        Args:
            seg_preds: 分割预测
            seg_targets: 分割目标
            risk_scores: 风险分数
            times: 生存时间
            events: 事件指示器
        """
        if seg_preds is not None and seg_targets is not None:
            self.seg_metrics.update(seg_preds, seg_targets)
        
        if risk_scores is not None and times is not None and events is not None:
            self.surv_metrics.update(risk_scores, times, events)
    
    def compute(self) -> Dict[str, float]:
        """
        计算所有指标
        
        Returns:
            指标字典
        """
        metrics = self.seg_metrics.compute()
        metrics.update(self.surv_metrics.compute())
        return metrics


def compute_dice_coefficient(
    prediction: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-5
) -> float:
    """
    计算Dice系数
    
    Args:
        prediction: 预测掩码
        target: 目标掩码
        smooth: 平滑项
    
    Returns:
        Dice系数
    """
    intersection = np.sum(prediction * target)
    dice = (2.0 * intersection + smooth) / (
        np.sum(prediction) + np.sum(target) + smooth
    )
    return dice


def compute_iou(
    prediction: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-5
) -> float:
    """
    计算IoU（Jaccard指数）
    
    Args:
        prediction: 预测掩码
        target: 目标掩码
        smooth: 平滑项
    
    Returns:
        IoU
    """
    intersection = np.sum(prediction * target)
    union = np.sum(prediction) + np.sum(target) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou
