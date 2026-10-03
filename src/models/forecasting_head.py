"""
Multi-Task Forecasting Head and Unified Multi-Modal Graph Network.
Coordinates DGT, Multi-Modal Fusion, and Prediction Heads.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from .dynamic_graph_transformer import DynamicGraphTransformer
from .multimodal_fusion import MultiModalCrossAttentionFusion


class MultiTaskForecastingHead(nn.Module):
    """
    Simultaneously forecasts:
    1. Continuous 1-day return (Regression)
    2. Discrete trend direction: Up vs. Down (Binary Classification)
    """
    def __init__(self, in_dim: int = 64, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.reg_head = nn.Linear(hidden_dim, 1) # Return forecast
        self.cls_head = nn.Linear(hidden_dim, 2) # Direction logits (0: Down, 1: Up)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch_size, num_nodes, in_dim)
        Returns:
            pred_return: (batch_size, num_nodes)
            pred_direction_logits: (batch_size, num_nodes, 2)
        """
        h = self.shared(x)
        pred_return = self.reg_head(h).squeeze(-1)
        pred_direction_logits = self.cls_head(h)
        return pred_return, pred_direction_logits


class ExplainableMultiModalForecaster(nn.Module):
    """
    The unified end-to-end framework combining:
    1. Dynamic Graph Transformer (DGT)
    2. Adaptive Multi-Modal Fusion
    3. Multi-Task Forecasting Head
    """
    def __init__(
        self,
        num_features: int = 13,
        graph_dim: int = 64,
        text_dim: int = 64,
        fusion_dim: int = 64,
        num_spatial_heads: int = 4,
        num_temporal_heads: int = 4,
        dropout: float = 0.15
    ):
        super().__init__()
        self.dgt = DynamicGraphTransformer(
            num_features=num_features,
            hidden_dim=graph_dim,
            num_spatial_heads=num_spatial_heads,
            num_temporal_heads=num_temporal_heads,
            dropout=dropout
        )
        self.fusion = MultiModalCrossAttentionFusion(
            graph_dim=graph_dim,
            text_dim=text_dim,
            fusion_dim=fusion_dim,
            num_heads=num_spatial_heads,
            dropout=dropout
        )
        self.head = MultiTaskForecastingHead(
            in_dim=fusion_dim,
            hidden_dim=fusion_dim // 2,
            dropout=dropout
        )

    def forward(
        self,
        x_numerical: torch.Tensor,
        adj: torch.Tensor,
        x_textual: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_numerical: (batch, num_nodes, seq_len, num_features)
            adj: (batch, num_nodes, num_nodes) or (num_nodes, num_nodes)
            x_textual: (batch, num_nodes, text_dim)
        Returns:
            Dictionary containing:
            - pred_return: (batch, num_nodes)
            - pred_direction_logits: (batch, num_nodes, 2)
            - spatial_attention: (batch, heads, num_nodes, num_nodes)
            - text_gate: (batch, num_nodes, 1)
        """
        # 1. Spatio-temporal graph representation
        h_graph, spatial_attn = self.dgt(x_numerical, adj)
        
        # 2. Adaptive cross-attention multi-modal fusion
        z_fused, text_gate = self.fusion(h_graph, x_textual)
        
        # 3. Dual forecasting heads
        pred_return, pred_dir_logits = self.head(z_fused)
        
        return {
            "pred_return": pred_return,
            "pred_direction_logits": pred_dir_logits,
            "spatial_attention": spatial_attn,
            "text_gate": text_gate,
            "h_graph": h_graph,
            "z_fused": z_fused
        }
