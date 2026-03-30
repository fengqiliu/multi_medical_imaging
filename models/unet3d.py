"""
3D注意力U-Net模型

用于多模态医学图像分割的完整模型架构
包含编码器、解码器、融合模块和分割头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import math


class DoubleConv3D(nn.Module):
    """
    3D双卷积块
    
    由两个连续的卷积层组成，每个卷积层后跟BatchNorm和ReLU
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class EncoderBlock3D(nn.Module):
    """
    3D编码器块（下采样）
    
    包含特征提取和下采样操作
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.conv = DoubleConv3D(in_channels, out_channels, dropout)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入张量
        
        Returns:
            skip: 用于跳跃连接的特征
            x: 下采样后的特征
        """
        skip = self.conv(x)
        x = self.pool(skip)
        return skip, x


class DecoderBlock3D(nn.Module):
    """
    3D解码器块（上采样）
    
    包含上采样、特征融合和特征提取
    """
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        use_attention: bool = False
    ):
        super().__init__()
        
        self.upconv = nn.ConvTranspose3d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        
        if use_attention:
            self.attention = AttentionGate(
                gate_channels=in_channels // 2,
                skip_channels=skip_channels,
                inter_channels=skip_channels
            )
            self.conv = DoubleConv3D(
                in_channels // 2 + skip_channels,
                out_channels,
                dropout
            )
        else:
            self.attention = None
            self.conv = DoubleConv3D(
                in_channels // 2 + skip_channels,
                out_channels,
                dropout
            )
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 上层特征
            skip: 跳跃连接的特征
        """
        x = self.upconv(x)
        
        if self.attention is not None:
            skip = self.attention(x, skip)
        
        # 处理尺寸不匹配
        x, skip = self._match_sizes(x, skip)
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        
        return x
    
    def _match_sizes(self, x: torch.Tensor, skip: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """匹配张量尺寸"""
        # 计算尺寸差异
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        diff_d = skip.size(4) - x.size(4)
        
        # 如果尺寸不同，进行调整
        if diff_h != 0 or diff_w != 0 or diff_d != 0:
            # 对skip进行裁剪
            skip = skip[:, :,
                       diff_h // 2: skip.size(2) - diff_h + diff_h // 2,
                       diff_w // 2: skip.size(3) - diff_w + diff_w // 2,
                       diff_d // 2: skip.size(4) - diff_d + diff_d // 2]
        
        return x, skip


class AttentionGate(nn.Module):
    """
    注意力门控模块
    
    用于在跳跃连接时强调重要特征，抑制无关区域
    """
    
    def __init__(
        self,
        gate_channels: int,
        skip_channels: int,
        inter_channels: int
    ):
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv3d(gate_channels, inter_channels, kernel_size=1),
            nn.BatchNorm3d(inter_channels)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(skip_channels, inter_channels, kernel_size=1),
            nn.BatchNorm3d(inter_channels)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(
        self,
        gate: torch.Tensor,
        skip: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            gate: 来自解码器上层的特征（控制信号）
            skip: 来自编码器的跳跃连接特征
        
        Returns:
            加权后的跳跃连接特征
        """
        # 1x1卷积降维
        g1 = self.W_g(gate)
        x1 = self.W_x(skip)
        
        # 注意力权重
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # 加权
        return skip * psi


class AttentionUNet3D(nn.Module):
    """
    3D注意力U-Net用于多模态医学图像分割
    
    架构特点：
    - 编码器-解码器结构捕获多尺度特征
    - 注意力门控增强边界定位
    - 支持多种多模态融合方法
    
    Args:
        in_channels: 输入通道数（模态数）
        out_channels: 输出通道数（类别数）
        base_filters: 基础滤波器数量
        dropout: Dropout比率
        fusion_method: 融合方法 ('attention', 'gate', 'transformer', 'concat')
        num_modalities: 模态数量
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_filters: int = 32,
        dropout: float = 0.1,
        fusion_method: str = "attention",
        num_modalities: int = 4
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fusion_method = fusion_method
        
        # 多模态融合模块
        if fusion_method == "attention":
            from models.fusion import ModalAttentionFusion
            self.modal_fusion = ModalAttentionFusion(
                num_modalities=num_modalities,
                feature_dim=in_channels
            )
        elif fusion_method == "gate":
            from models.fusion import GatedMultimodalFusion
            self.modal_fusion = GatedMultimodalFusion(
                num_modalities=num_modalities,
                feature_dim=in_channels
            )
        elif fusion_method == "transformer":
            from models.fusion import TransformerFusion
            self.modal_fusion = TransformerFusion(
                num_modalities=num_modalities,
                feature_dim=in_channels
            )
        else:
            self.modal_fusion = None
        
        # 初始卷积
        self.init_conv = DoubleConv3D(in_channels, base_filters, dropout)
        
        # 编码器
        self.enc1 = EncoderBlock3D(base_filters, base_filters * 2, dropout)
        self.enc2 = EncoderBlock3D(base_filters * 2, base_filters * 4, dropout)
        self.enc3 = EncoderBlock3D(base_filters * 4, base_filters * 8, dropout)
        self.enc4 = EncoderBlock3D(base_filters * 8, base_filters * 16, dropout)
        
        # 瓶颈层
        self.bottleneck = DoubleConv3D(base_filters * 16, base_filters * 32, dropout)
        
        # 解码器（带注意力门控）
        self.dec4 = DecoderBlock3D(
            base_filters * 32, base_filters * 8, base_filters * 8, dropout, use_attention=True
        )
        self.dec3 = DecoderBlock3D(
            base_filters * 8, base_filters * 4, base_filters * 4, dropout, use_attention=True
        )
        self.dec2 = DecoderBlock3D(
            base_filters * 4, base_filters * 2, base_filters * 2, dropout, use_attention=True
        )
        self.dec1 = DecoderBlock3D(
            base_filters * 2, base_filters, base_filters, dropout, use_attention=False
        )
        
        # 最终输出
        self.out_conv = nn.Conv3d(base_filters, out_channels, kernel_size=1)
        
        # 深度监督的辅助头
        self.aux_heads = nn.ModuleList([
            nn.Conv3d(base_filters * (2 ** i), out_channels, 1)
            for i in range(4)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: 输入 (B, num_modalities, H, W, D)
            return_features: 是否返回中间特征
        
        Returns:
            预测结果和可选的特征
        """
        outputs = {}
        
        # 编码路径
        x0 = self.init_conv(x)      # (B, 32, H, W, D)
        
        s1, x1 = self.enc1(x0)       # s1: (B, 64, H/2, W/2, D/2)
        s2, x2 = self.enc2(x1)       # s2: (B, 128, H/4, W/4, D/4)
        s3, x3 = self.enc3(x2)       # s3: (B, 256, H/8, W/8, D/8)
        s4, x4 = self.enc4(x3)       # s4: (B, 512, H/16, W/16, D/16)
        
        # 瓶颈
        x = self.bottleneck(x4)      # (B, 1024, H/16, W/16, D/16)
        
        # 解码路径
        x = self.dec4(x, s4)          # (B, 256, H/8, W/8, D/8)
        x = self.dec3(x, s3)          # (B, 128, H/4, W/4, D/4)
        x = self.dec2(x, s2)          # (B, 64, H/2, W/2, D/2)
        x = self.dec1(x, s1)          # (B, 32, H, W, D)
        
        # 主输出
        outputs["main"] = self.out_conv(x)
        
        # 辅助输出（深度监督）
        outputs["aux"] = [
            aux(x) for aux in self.aux_heads
        ]
        
        if return_features:
            outputs["features"] = {
                "enc1": s1, "enc2": s2, "enc3": s3, "enc4": s4,
                "dec4": x, "bottleneck": x4
            }
        
        return outputs


class UNet3DWithSurvival(nn.Module):
    """
    联合分割和预后预测网络
    
    共享编码器学习多模态特征表示，分叉为分割头和预后预测头
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        base_filters: int = 32,
        dropout: float = 0.1,
        survival_head_dim: int = 128
    ):
        super().__init__()
        
        # 共享编码器
        self.init_conv = DoubleConv3D(in_channels, base_filters, dropout)
        self.enc1 = EncoderBlock3D(base_filters, base_filters * 2, dropout)
        self.enc2 = EncoderBlock3D(base_filters * 2, base_filters * 4, dropout)
        self.enc3 = EncoderBlock3D(base_filters * 4, base_filters * 8, dropout)
        self.enc4 = EncoderBlock3D(base_filters * 8, base_filters * 16, dropout)
        self.bottleneck = DoubleConv3D(base_filters * 16, base_filters * 32, dropout)
        
        # 分割解码器
        self.dec4 = DecoderBlock3D(base_filters * 32, base_filters * 8, base_filters * 8, dropout)
        self.dec3 = DecoderBlock3D(base_filters * 8, base_filters * 4, base_filters * 4, dropout)
        self.dec2 = DecoderBlock3D(base_filters * 4, base_filters * 2, base_filters * 2, dropout)
        self.dec1 = DecoderBlock3D(base_filters * 2, base_filters, base_filters, dropout)
        self.seg_head = nn.Conv3d(base_filters, num_classes, kernel_size=1)
        
        # 预后预测头
        self.survival_global_pool = nn.AdaptiveAvgPool3d(1)
        self.survival_fc = nn.Sequential(
            nn.Linear(base_filters * 32 + num_classes * 2, survival_head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(survival_head_dim, survival_head_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(survival_head_dim // 2, 1)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        clinical_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: 图像输入 (B, C, H, W, D)
            clinical_features: 临床特征 (B, F)
        
        Returns:
            分割和预后预测结果
        """
        # 编码
        x0 = self.init_conv(x)
        s1, x1 = self.enc1(x0)
        s2, x2 = self.enc2(x1)
        s3, x3 = self.enc3(x2)
        s4, x4 = self.enc4(x3)
        bottleneck = self.bottleneck(x4)
        
        # 分割解码
        d4 = self.dec4(bottleneck, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        seg_output = self.seg_head(d1)
        
        # 预后预测
        global_feat = self.survival_global_pool(bottleneck).flatten(1)
        
        # 计算分割统计特征
        seg_probs = torch.softmax(seg_output, dim=1)
        tumor_volume = seg_probs[:, 1:].sum(dim=[2, 3, 4])  # 肿瘤体积分数
        tumor_focus = (seg_probs[:, 1:] > 0.5).sum(dim=[2, 3, 4]).float()  # 肿瘤灶数量
        
        # 融合全局特征和分割特征
        survival_feat = torch.cat([global_feat, tumor_volume, tumor_focus], dim=1)
        
        if clinical_features is not None:
            survival_feat = torch.cat([survival_feat, clinical_features], dim=1)
        
        survival_output = self.survival_fc(survival_feat)
        
        return {
            "segmentation": seg_output,
            "survival_risk": survival_output
        }


class ResNet3DEncoder(nn.Module):
    """
    3D ResNet编码器
    
    使用3D ResNet作为编码器的backbone
    """
    
    def __init__(
        self,
        in_channels: int,
        layers: List[int] = [2, 2, 2, 2],
        base_filters: int = 64
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.base_filters = base_filters
        
        # 初始卷积
        self.conv1 = nn.Conv3d(in_channels, base_filters, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(base_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # ResNet层
        self.layer1 = self._make_layer(base_filters, base_filters, layers[0])
        self.layer2 = self._make_layer(base_filters, base_filters * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_filters * 2, base_filters * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_filters * 4, base_filters * 8, layers[3], stride=2)
    
    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        """创建ResNet层"""
        layers = []
        
        # 第一个块可能有下采样
        layers.append(ResBlock3D(in_channels, out_channels, stride))
        
        # 后续块
        for _ in range(1, blocks):
            layers.append(ResBlock3D(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: (B, C, H, W, D)
        
        Returns:
            多尺度特征列表
        """
        features = []
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x); features.append(x)
        x = self.layer2(x); features.append(x)
        x = self.layer3(x); features.append(x)
        x = self.layer4(x); features.append(x)
        
        return features


class ResBlock3D(nn.Module):
    """3D残差块"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1
    ):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        x += residual
        x = nn.ReLU(x)
        
        return x
