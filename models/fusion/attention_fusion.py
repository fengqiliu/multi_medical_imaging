"""
多模态特征融合模块

提供多种融合策略：
- 注意力融合
- 门控融合
- Transformer融合
- 交叉注意力融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
import math


class MultiHeadCrossAttention(nn.Module):
    """
    多头交叉注意力机制用于多模态特征融合
    
    核心思想：通过查询-键-值注意力机制建模不同模态间的依赖关系
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # 线性投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            query: 查询张量 (B, N_q, C)
            key_value: 键值张量 (B, N_kv, C)
            mask: 注意力掩码
        
        Returns:
            更新后的查询特征
        """
        B, N_q, C = query.shape
        N_kv = key_value.shape[1]
        
        # 线性投影并分头
        Q = self.q_proj(query).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key_value).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(key_value).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 聚合值
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, N_q, C)
        out = self.out_proj(out)
        
        # 残差连接
        out = self.layer_norm(out + query)
        
        return out


class ModalAttentionFusion(nn.Module):
    """
    模态注意力融合模块
    
    核心思想：为每个模态学习注意力权重，然后加权融合
    """
    
    def __init__(
        self,
        num_modalities: int,
        feature_dim: int,
        hidden_dim: Optional[int] = None
    ):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = feature_dim // 2
        
        self.num_modalities = num_modalities
        self.feature_dim = feature_dim
        
        # 模态特定编码
        self.modality_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, feature_dim)
            )
            for _ in range(num_modalities)
        ])
        
        # 注意力权重生成
        self.attention_weights = nn.Sequential(
            nn.Linear(feature_dim * num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modalities),
            nn.Softmax(dim=-1)
        )
        
        self.layer_norm = nn.LayerNorm(feature_dim)
    
    def forward(self, modality_features: List[torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            modality_features: 各模态特征列表 [(B, C), ...]
        
        Returns:
            融合后的特征
        """
        # 编码各模态特征
        encoded = [
            enc(feat) for enc, feat in zip(self.modality_encoders, modality_features)
        ]
        
        # 拼接并生成注意力权重
        concat_features = torch.cat(encoded, dim=-1)
        attn_weights = self.attention_weights(concat_features)  # (B, num_modalities)
        
        # 加权融合
        fused = torch.zeros_like(encoded[0])
        for i, feat in enumerate(encoded):
            fused += attn_weights[:, i:i+1] * feat
        
        # 层归一化
        fused = self.layer_norm(fused)
        
        return fused


class SpatialChannelAttention(nn.Module):
    """
    空间-通道注意力模块
    
    用于增强多模态特征的表示能力
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        
        Returns:
            增强后的特征
        """
        # 通道注意力
        ch_attn = self.channel_attention(x)
        x = x * ch_attn
        
        # 空间注意力
        sp_attn = self.spatial_attention(x)
        x = x * sp_attn
        
        return x


class GatedMultimodalFusion(nn.Module):
    """
    门控多模态融合
    
    使用门控机制动态控制各模态信息流动
    """
    
    def __init__(
        self,
        num_modalities: int,
        feature_dim: int,
        gate_dim: Optional[int] = None
    ):
        super().__init__()
        
        if gate_dim is None:
            gate_dim = feature_dim
        
        self.num_modalities = num_modalities
        self.feature_dim = feature_dim
        
        # 各模态的嵌入层
        self.embeddings = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim)
            for _ in range(num_modalities)
        ])
        
        # 门控网络
        self.gate_network = nn.Sequential(
            nn.Linear(feature_dim * num_modalities, gate_dim),
            nn.ReLU(),
            nn.Linear(gate_dim, num_modalities * feature_dim),
            nn.Sigmoid()
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim * num_modalities, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )
    
    def forward(self, modality_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            modality_features: 各模态特征列表 [(B, C), ...]
        
        Returns:
            融合后的特征
        """
        B = modality_features[0].shape[0]
        
        # 嵌入各模态
        embedded = [emb(feat) for emb, feat in zip(self.embeddings, modality_features)]
        concat = torch.cat(embedded, dim=-1)
        
        # 生成门控
        gates = self.gate_network(concat)
        gates = gates.view(B, self.num_modalities, self.feature_dim)
        
        # 门控加权
        gated_features = [emb * gates[:, i, :] for i, emb in enumerate(embedded)]
        
        # 融合
        fused = torch.sum(torch.stack(gated_features, dim=1), dim=1)
        fused = self.fusion_layer(fused)
        
        return fused


class TransformerFusion(nn.Module):
    """
    Transformer-based多模态融合
    
    使用自注意力机制建模模态间关系
    """
    
    def __init__(
        self,
        num_modalities: int,
        feature_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_modalities = num_modalities
        self.feature_dim = feature_dim
        
        # 模态嵌入（添加位置编码）
        self.modality_embedding = nn.Parameter(
            torch.randn(1, num_modalities, feature_dim) * 0.02
        )
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 输出投影
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)
    
    def forward(self, modality_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            modality_features: 各模态特征列表 [(B, C), ...]
        
        Returns:
            融合后的特征
        """
        # 堆叠模态特征: (B, num_modalities, feature_dim)
        x = torch.stack(modality_features, dim=1)
        
        # 添加模态嵌入
        x = x + self.modality_embedding
        
        # Transformer处理
        x = self.transformer(x)  # (B, num_modalities, feature_dim)
        
        # 聚合所有模态（简单平均或CLS token）
        fused = x.mean(dim=1)  # (B, feature_dim)
        
        # 输出投影
        fused = self.output_proj(fused)
        fused = self.layer_norm(fused)
        
        return fused


class ConcatFusion(nn.Module):
    """
    简单拼接融合
    
    最基础的融合策略，直接拼接所有模态特征
    """
    
    def __init__(self, num_modalities: int, feature_dim: int):
        super().__init__()
        self.output_dim = num_modalities * feature_dim
    
    def forward(self, modality_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            modality_features: 各模态特征列表
        
        Returns:
            拼接后的特征
        """
        return torch.cat(modality_features, dim=-1)


class AMCF(nn.Module):
    """
    自适应多模态特征融合模块 (Adaptive Multimodal Fusion)
    
    结合通道注意力和自适应权重学习
    """
    
    def __init__(self, channels: int, num_modalities: int):
        super().__init__()
        
        self.num_modalities = num_modalities
        self.channels = channels
        
        # 模态注意力权重
        self.modal_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels * num_modalities, channels, 1),
            nn.ReLU(),
            nn.Conv3d(channels, num_modalities, 1),
            nn.Softmax(dim=1)
        )
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv3d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, modality_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            modality_features: 各模态特征列表 [(B, C, H, W, D), ...]
        
        Returns:
            融合后的特征
        """
        # 拼接模态
        concat = torch.cat(modality_features, dim=1)  # (B, C*num_mod, H, W, D)
        
        # 学习模态权重
        modal_weights = self.modal_attention(concat)  # (B, num_mod, 1, 1, 1)
        
        # 加权融合
        fused = torch.zeros_like(modality_features[0])
        for i, feat in enumerate(modality_features):
            fused += modal_weights[:, i:i+1] * feat
        
        # 通道注意力
        channel_weights = self.channel_attention(fused)
        fused = fused * channel_weights
        
        return fused
