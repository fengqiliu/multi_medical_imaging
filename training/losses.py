"""
医学影像分割与预后预测损失函数

提供多种损失函数：
- 分割损失：Dice Loss, Focal Loss, 组合损失
- 预后损失：Cox Loss, C-index Loss
- 多任务损失：自动加权多任务学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import numpy as np


class DiceLoss(nn.Module):
    """
    Dice损失函数
    
    用于评估分割预测与真实标签的重叠程度
    特别适用于类别不平衡情况
    """
    
    def __init__(
        self,
        num_classes: int,
        smooth: float = 1e-5,
        ignore_index: Optional[int] = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: 预测 logits (B, C, H, W, D) 或 (B, C, H, W)
            target: 目标标签 (B, H, W, D) 或 (B, H, W)
        
        Returns:
            Dice损失
        """
        # 计算softmax概率
        pred = F.softmax(pred, dim=1)
        
        total_loss = 0.0
        num_valid = 0
        
        for c in range(self.num_classes):
            if self.ignore_index is not None and c == self.ignore_index:
                continue
            
            pred_c = pred[:, c]  # (B, H, W, D)
            target_c = (target == c).float()  # (B, H, W, D)
            
            # 计算交集和并集
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            # Dice系数
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            total_loss += (1.0 - dice)
            num_valid += 1
        
        return total_loss / num_valid


class FocalLoss(nn.Module):
    """
    Focal损失函数
    
    通过降低易分类样本的权重，解决类别不平衡问题
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: 预测 logits (B, C, ...)
            target: 目标标签 (B, ...)
        
        Returns:
            Focal损失
        """
        # 计算交叉熵
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        
        # 计算pt（正确类的概率）
        pt = torch.exp(-ce_loss)
        
        # Focal权重
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # 类别权重
        if self.alpha is not None:
            alpha_t = self.alpha[target]
            focal_loss = alpha_t * focal_loss
        
        # 归约
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class TverskyLoss(nn.Module):
    """
    Tversky损失函数
    
    Dice Loss的推广，允许调整假阳性和假阴性的权重
    """
    
    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1e-5
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha  # 假阳性权重
        self.beta = beta    # 假阴性权重
        self.smooth = smooth
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: 预测 logits
            target: 目标标签
        
        Returns:
            Tversky损失
        """
        pred = F.softmax(pred, dim=1)
        
        total_loss = 0.0
        num_valid = 0
        
        for c in range(self.num_classes):
            pred_c = pred[:, c]
            target_c = (target == c).float()
            
            # True Positive, False Positive, False Negative
            TP = (pred_c * target_c).sum()
            FP = (pred_c * (1 - target_c)).sum()
            FN = ((1 - pred_c) * target_c).sum()
            
            # Tversky指数
            tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
            total_loss += (1.0 - tversky)
            num_valid += 1
        
        return total_loss / num_valid


class CombinedLoss(nn.Module):
    """
    组合损失函数
    
    结合Dice Loss和Focal Loss的优势
    """
    
    def __init__(
        self,
        num_classes: int,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        self.dice_loss = DiceLoss(num_classes)
        self.focal_loss = FocalLoss(gamma=focal_gamma)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred: 预测 logits
            target: 目标标签
        
        Returns:
            包含总损失和各分项损失的字典
        """
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        
        total = self.dice_weight * dice + self.focal_weight * focal
        
        return {
            "total": total,
            "dice": dice,
            "focal": focal
        }


class BoundaryLoss(nn.Module):
    """
    边界损失函数
    
    强调预测边界与真实边界的对齐
    """
    
    def __init__(self, theta0: float = 2.0, theta: float = 5.0):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: 预测概率 (B, C, H, W, D)
            target: 目标标签 (B, H, W, D)
        
        Returns:
            边界损失
        """
        pred = F.softmax(pred, dim=1)
        
        # 计算距离变换
        loss = 0.0
        for c in range(pred.shape[1]):
            pred_c = pred[:, c]
            target_c = (target == c).float()
            
            # 简化的边界损失
            pred_boundary = pred_c * (1 - pred_c)
            target_boundary = target_c * (1 - target_c)
            
            loss += (pred_boundary * target_boundary).mean()
        
        return loss


class DeepSurvivalLoss(nn.Module):
    """
    DeepSurv损失函数
    
    基于Cox比例风险模型的对数偏似然
    用于生存分析预后预测
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        risk_pred: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            risk_pred: 预测的对数风险比 (B, 1)
            time: 生存时间 (B,)
            event: 事件指示器 (1=事件, 0=删失) (B,)
        
        Returns:
            负对数偏似然
        """
        # 按时间降序排序
        sorted_idx = torch.argsort(time, descending=True)
        risk_pred = risk_pred[sorted_idx].squeeze()
        event = event[sorted_idx]
        
        # 计算对数风险和累积风险
        log_risk = risk_pred
        risk_sum = torch.cumsum(torch.exp(log_risk), dim=0)
        
        # 计算负对数偏似然
        # 只考虑发生事件的用户
        log_pl = log_risk - torch.log(risk_sum + 1e-10)
        
        # 只保留事件用户
        events = event.float()
        loss = -log_pl * events
        loss = loss.sum() / (events.sum() + 1e-10)
        
        return loss


class SurvivalCIndexLoss(nn.Module):
    """
    一致性指数（C-index）损失
    
    近似计算C-index用于端到端优化
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        risk_pred: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            risk_pred: (B, 1) 预测风险分数
            time: (B,) 生存时间
            event: (B,) 事件指示器
        
        Returns:
            1 - C-index（用于最小化）
        """
        risk_pred = risk_pred.squeeze()
        B = len(risk_pred)
        
        concordant = 0.0
        tied = 0.0
        comparable = 0.0
        
        for i in range(B):
            for j in range(i + 1, B):
                # 只比较可比较的对
                # i发生事件且j未删失且Ti > Tj
                if event[i] == 1 and event[j] == 0 and time[i] < time[j]:
                    comparable += 1
                    if risk_pred[i] > risk_pred[j]:
                        concordant += 1
                    elif risk_pred[i] == risk_pred[j]:
                        tied += 0.5
                # 或者两者都发生事件
                elif event[i] == 1 and event[j] == 1:
                    comparable += 1
                    if time[i] < time[j] and risk_pred[i] > risk_pred[j]:
                        concordant += 1
                    elif time[i] > time[j] and risk_pred[i] < risk_pred[j]:
                        concordant += 1
                    elif risk_pred[i] == risk_pred[j]:
                        tied += 0.5
        
        if comparable > 0:
            c_index = (concordant + tied) / comparable
        else:
            c_index = 0.5
        
        return 1.0 - c_index


class MultiTaskLoss(nn.Module):
    """
    多任务损失（自动加权）
    
    使用不确定度学习自动调整任务权重
    来自论文 "Multi-Task Learning Using Uncertainty to Weigh Losses"
    """
    
    def __init__(
        self,
        segmentation_weight: float = 1.0,
        survival_weight: float = 1.0,
        num_classes: int = 4
    ):
        super().__init__()
        
        self.seg_weight = segmentation_weight
        self.surv_weight = survival_weight
        
        # 可学习的日志标准差（用于不确定度加权）
        self.log_sigma1 = nn.Parameter(torch.zeros(1))
        self.log_sigma2 = nn.Parameter(torch.zeros(1))
        
        self.seg_loss_fn = CombinedLoss(num_classes=num_classes)
        self.surv_loss_fn = DeepSurvivalLoss()
    
    def forward(
        self,
        seg_pred: torch.Tensor,
        surv_pred: torch.Tensor,
        seg_target: torch.Tensor,
        surv_target: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            seg_pred: 分割预测
            surv_pred: 预后预测
            seg_target: 分割目标
            surv_target: 包含time和event的字典
        
        Returns:
            各任务损失和总损失
        """
        # 分割损失
        seg_losses = self.seg_loss_fn(seg_pred, seg_target)
        seg_loss = seg_losses["total"]
        
        # 预后损失
        surv_loss = self.surv_loss_fn(
            surv_pred,
            surv_target["time"],
            surv_target["event"]
        )
        
        # 不确定度加权
        precision1 = torch.exp(-self.log_sigma1) ** 2
        precision2 = torch.exp(-self.log_sigma2) ** 2
        
        loss1 = precision1 * seg_loss + self.log_sigma1
        loss2 = precision2 * surv_loss + self.log_sigma2
        
        total_loss = self.seg_weight * loss1 + self.surv_weight * loss2
        
        return {
            "total": total_loss,
            "segmentation": seg_loss,
            "survival": surv_loss,
            "sigma1": self.log_sigma1,
            "sigma2": self.log_sigma2
        }


class AsymmetricUncertaintyLoss(nn.Module):
    """
    非对称不确定性损失
    
    根据任务难度自适应调整权重
    """
    
    def __init__(self, task_importance: Optional[Dict[str, float]] = None):
        super().__init__()
        self.task_importance = task_importance or {}
        self.log_vars = nn.ParameterDict({
            k: nn.Parameter(torch.zeros(1))
            for k in task_importance.keys()
        })
    
    def forward(
        self,
        losses: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            losses: 各任务损失字典
        
        Returns:
            加权总损失
        """
        total_loss = 0.0
        
        for task_name, task_loss in losses.items():
            if task_name in self.log_vars:
                precision = torch.exp(-self.log_vars[task_name])
                importance = self.task_importance.get(task_name, 1.0)
                total_loss += importance * precision * task_loss + self.log_vars[task_name]
            else:
                total_loss += task_loss
        
        return total_loss
