"""
训练器模块

提供模型训练的基础设施，包括：
- 训练循环管理
- 验证循环
- 指标计算和日志记录
- 模型检查点保存和恢复
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Any
import numpy as np
from tqdm import tqdm
import time
import os
from pathlib import Path
import json


class BaseTrainer:
    """
    基础训练器类
    
    管理训练和验证流程的核心组件
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: str = "cuda",
        log_dir: str = "./logs",
        checkpoint_dir: str = "./checkpoints",
        gradient_clip: float = 1.0
    ):
        """
        初始化训练器
        
        Args:
            model: 神经网络模型
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 计算设备
            log_dir: 日志目录
            checkpoint_dir: 检查点目录
            gradient_clip: 梯度裁剪阈值
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.gradient_clip = gradient_clip
        
        # 创建目录
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard日志
        self.writer = SummaryWriter(self.log_dir)
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        self.history = {"train": [], "val": []}
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
        
        Returns:
            训练指标字典
        """
        self.model.train()
        epoch_losses = {}
        total_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # 数据移到设备
            batch = self._move_batch_to_device(batch)
            
            # 前向传播
            outputs = self.training_step(batch)
            
            # 计算损失
            if isinstance(outputs, dict) and "loss" in outputs:
                loss = outputs["loss"]
            else:
                loss = outputs
            
            # 检查损失有效性
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告: 检测到NaN或Inf损失，跳过此批次")
                continue
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
            
            self.optimizer.step()
            
            # 更新进度条
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # 记录到TensorBoard
            if self.global_step % 10 == 0:
                self.writer.add_scalar("train/loss", loss.item(), self.global_step)
            
            self.global_step += 1
        
        # 计算epoch平均损失
        return {"loss": loss.item()}
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        验证模型
        
        Args:
            val_loader: 验证数据加载器
        
        Returns:
            验证指标字典
        """
        self.model.eval()
        total_loss = 0.0
        total_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = self._move_batch_to_device(batch)
                
                # 前向传播
                outputs = self.validation_step(batch)
                
                if isinstance(outputs, dict) and "loss" in outputs:
                    loss = outputs["loss"]
                else:
                    loss = outputs
                
                total_loss += loss.item()
        
        avg_loss = total_loss / total_batches
        
        # 记录到TensorBoard
        self.writer.add_scalar("val/loss", avg_loss, self.current_epoch)
        
        return {"val_loss": avg_loss}
    
    def training_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        单步训练（需要子类实现）
        
        Args:
            batch: 批次数据
        
        Returns:
            损失字典
        """
        raise NotImplementedError("子类必须实现training_step方法")
    
    def validation_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        单步验证（需要子类实现）
        
        Args:
            batch: 批次数据
        
        Returns:
            损失字典
        """
        raise NotImplementedError("子类必须实现validation_step方法")
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """
        将批次数据移到计算设备
        
        Args:
            batch: 批次数据字典
        
        Returns:
            移动后的批次数据
        """
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        start_epoch: int = 0
    ):
        """
        执行完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            start_epoch: 起始epoch
        """
        print(f"开始训练，共 {epochs} 个epoch")
        print(f"日志保存到: {self.log_dir}")
        print(f"检查点保存到: {self.checkpoint_dir}")
        
        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch
            
            # 训练
            train_metrics = self.train_epoch(train_loader)
            
            # 验证
            if (epoch + 1) % 1 == 0:  # 每个epoch验证一次
                val_metrics = self.validate(val_loader)
                
                # 合并指标
                metrics = {**train_metrics, **val_metrics}
            else:
                metrics = train_metrics
            
            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar("train/learning_rate", current_lr, epoch)
            
            # 保存历史
            self.history["train"].append(train_metrics)
            self.history["val"].append(metrics)
            
            # 保存最佳模型
            if "val_loss" in metrics and metrics["val_loss"] < self.best_metric:
                self.best_metric = metrics["val_loss"]
                self.save_checkpoint("best_model.pt", metrics)
            
            # 定期保存检查点
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt", metrics)
            
            # 打印epoch总结
            self._print_epoch_summary(epoch, metrics)
        
        print("训练完成!")
        self.writer.close()
    
    def _print_epoch_summary(self, epoch: int, metrics: Dict[str, float]):
        """
        打印epoch总结
        
        Args:
            epoch: 当前epoch
            metrics: 指标字典
        """
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"Epoch {epoch+1}/{self.history['train'].__len__() + epoch} - {metrics_str}")
    
    def save_checkpoint(
        self,
        filename: str,
        metrics: Dict[str, float],
        save_optimizer: bool = True
    ):
        """
        保存模型检查点
        
        Args:
            filename: 保存文件名
            metrics: 当前指标
            save_optimizer: 是否保存优化器状态
        """
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "metrics": metrics,
            "best_metric": self.best_metric,
            "history": self.history
        }
        
        if save_optimizer:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        print(f"检查点已保存: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        加载模型检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.current_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        self.best_metric = checkpoint.get("best_metric", 0.0)
        self.history = checkpoint.get("history", {"train": [], "val": []})
        
        print(f"检查点已加载: {checkpoint_path}")
        print(f"从Epoch {checkpoint['epoch']} 继续训练")
    
    def save_history(self, filename: str = "training_history.json"):
        """
        保存训练历史
        
        Args:
            filename: 保存文件名
        """
        history_path = self.log_dir / filename
        
        # 转换numpy数组为列表
        history_serializable = {
            "train": [
                {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                 for k, v in epoch_metrics.items()}
                for epoch_metrics in self.history["train"]
            ],
            "val": [
                {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                 for k, v in epoch_metrics.items()}
                for epoch_metrics in self.history["val"]
            ]
        }
        
        with open(history_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        
        print(f"训练历史已保存: {history_path}")


class SegmentationTrainer(BaseTrainer):
    """
    分割任务训练器
    
    专门用于图像分割任务的训练
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        metrics_fn,
        scheduler: Optional[Any] = None,
        device: str = "cuda",
        log_dir: str = "./logs",
        checkpoint_dir: str = "./checkpoints",
        gradient_clip: float = 1.0,
        deep_supervision_weight: float = 0.4
    ):
        """
        初始化分割训练器
        
        Args:
            model: 分割模型
            optimizer: 优化器
            loss_fn: 损失函数
            metrics_fn: 评估指标函数
            scheduler: 学习率调度器
            device: 计算设备
            log_dir: 日志目录
            checkpoint_dir: 检查点目录
            gradient_clip: 梯度裁剪阈值
            deep_supervision_weight: 深度监督权重
        """
        super().__init__(
            model, optimizer, scheduler, device,
            log_dir, checkpoint_dir, gradient_clip
        )
        
        self.loss_fn = loss_fn
        self.metrics_fn = metrics_fn
        self.deep_supervision_weight = deep_supervision_weight
    
    def training_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        单步训练
        
        Args:
            batch: 批次数据
        
        Returns:
            损失字典
        """
        images = batch["image"]
        labels = batch["label"]
        
        # 前向传播
        outputs = self.model(images)
        
        # 计算主损失
        if isinstance(outputs, dict) and "main" in outputs:
            main_output = outputs["main"]
            
            # 如果有深度监督损失
            if "aux" in outputs and self.deep_supervision_weight > 0:
                aux_loss = 0.0
                for i, aux_output in enumerate(outputs["aux"]):
                    aux_loss += self.loss_fn(aux_output, labels)["total"]
                aux_loss /= len(outputs["aux"])
                
                main_loss = self.loss_fn(main_output, labels)["total"]
                loss = main_loss + self.deep_supervision_weight * aux_loss
                
                return {
                    "loss": loss,
                    "main_loss": main_loss,
                    "aux_loss": aux_loss
                }
            else:
                loss = self.loss_fn(main_output, labels)["total"]
        else:
            loss = self.loss_fn(outputs, labels)["total"]
        
        return {"loss": loss}
    
    def validation_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        单步验证
        
        Args:
            batch: 批次数据
        
        Returns:
            损失和指标字典
        """
        images = batch["image"]
        labels = batch["label"]
        
        with torch.no_grad():
            outputs = self.model(images)
            
            if isinstance(outputs, dict) and "main" in outputs:
                main_output = outputs["main"]
            else:
                main_output = outputs
            
            loss = self.loss_fn(main_output, labels)["total"]
            
            # 计算指标
            preds = torch.argmax(main_output, dim=1)
            metrics = self.metrics_fn(preds, labels)
        
        return {"loss": loss, **metrics}


class MultiTaskTrainer(BaseTrainer):
    """
    多任务训练器
    
    同时处理分割和预后预测任务
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        metrics_fn,
        scheduler: Optional[Any] = None,
        device: str = "cuda",
        log_dir: str = "./logs",
        checkpoint_dir: str = "./checkpoints",
        gradient_clip: float = 1.0
    ):
        super().__init__(
            model, optimizer, scheduler, device,
            log_dir, checkpoint_dir, gradient_clip
        )
        
        self.loss_fn = loss_fn
        self.metrics_fn = metrics_fn
    
    def training_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        单步训练
        
        Args:
            batch: 批次数据
        
        Returns:
            损失字典
        """
        images = batch["image"]
        seg_labels = batch["label"]
        surv_labels = {
            "time": batch["survival_time"].to(self.device),
            "event": batch["survival_event"].to(self.device)
        }
        
        # 前向传播
        outputs = self.model(images)
        
        # 计算多任务损失
        if isinstance(outputs, dict) and "segmentation" in outputs:
            seg_pred = outputs["segmentation"]
            surv_pred = outputs["survival_risk"]
        else:
            seg_pred = outputs
            surv_pred = None
        
        losses = self.loss_fn(
            seg_pred=seg_pred,
            surv_pred=surv_pred,
            seg_target=seg_labels,
            surv_target=surv_labels
        )
        
        return losses
    
    def validation_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        单步验证
        
        Args:
            batch: 批次数据
        
        Returns:
            损失和指标字典
        """
        with torch.no_grad():
            images = batch["image"]
            seg_labels = batch["label"]
            
            outputs = self.model(images)
            
            if isinstance(outputs, dict) and "segmentation" in outputs:
                seg_pred = outputs["segmentation"]
                surv_pred = outputs.get("survival_risk")
            else:
                seg_pred = outputs
                surv_pred = None
            
            # 分割损失
            seg_losses = self.loss_fn.seg_loss_fn(seg_pred, seg_labels)
            
            # 分割指标
            preds = torch.argmax(seg_pred, dim=1)
            seg_metrics = self.metrics_fn["segmentation"](preds, seg_labels)
            
            result = {
                "loss": seg_losses["total"],
                "seg_loss": seg_losses["total"],
                "dice": seg_metrics.get("dice", 0.0)
            }
            
            if surv_pred is not None:
                surv_labels = {
                    "time": batch["survival_time"].to(self.device),
                    "event": batch["survival_event"].to(self.device)
                }
                surv_loss = self.loss_fn.surv_loss_fn(
                    surv_pred,
                    surv_labels["time"],
                    surv_labels["event"]
                )
                result["surv_loss"] = surv_loss
                result["loss"] = result["loss"] + surv_loss
            
            return result
