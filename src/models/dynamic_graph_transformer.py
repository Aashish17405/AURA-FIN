"""
Dynamic Graph Transformer (DGT) Architecture.
Captures both spatial (inter-stock correlation & sector contagion) and
temporal (multi-day technical price patterns) dependencies.
Exposes internal attention maps for Explainable AI (XAI) analysis.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialGraphAttentionLayer(nn.Module):
    """
    Multi-Head Spatial Attention across stock nodes.
    Integrates dynamic adjacency matrix A_t as relational structural bias.
    """
    def __init__(self, in_dim: int, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(in_dim, hidden_dim)
        self.k_proj = nn.Linear(in_dim, hidden_dim)
        self.v_proj = nn.Linear(in_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.last_attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, num_nodes, in_dim)
        adj: (batch_size, num_nodes, num_nodes) or (num_nodes, num_nodes)
        """
        b, n, _ = x.shape
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(b, -1, -1)
            
        # Linear projections
        q = self.q_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2) # (b, heads, n, head_dim)
        k = self.k_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5) # (b, heads, n, n)
        
        # Inject dynamic graph structural bias
        adj_bias = adj.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        # Avoid log(0)
        adj_log = torch.log(adj_bias.clamp(min=1e-5))
        scores = scores + 0.5 * adj_log
        
        attn = F.softmax(scores, dim=-1)
        self.last_attn_weights = attn.detach() # Retained for XAI
        
        attn_out = torch.matmul(self.dropout(attn), v) # (b, heads, n, head_dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, n, -1)
        
        out = self.out_proj(attn_out)
        out = self.layer_norm(out + (x if x.shape[-1] == out.shape[-1] else 0.0))
        return out


class TemporalAttentionLayer(nn.Module):
    """
    Multi-Head Temporal Self-Attention across sequence time steps.
    """
    def __init__(self, in_dim: int, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.temporal_mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.linear_in = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size * num_nodes, seq_len, in_dim)
        """
        h = self.linear_in(x)
        attn_out, _ = self.temporal_mha(h, h, h)
        out = self.layer_norm(h + self.dropout(attn_out))
        return out


class DynamicGraphTransformer(nn.Module):
    """
    Complete Spatio-Temporal Graph Transformer (DGT).
    Processes dynamic graphs: (batch_size, num_nodes, seq_len, num_features) + Adjacency
    Outputs node-level spatio-temporal representations.
    """
    def __init__(
        self,
        num_features: int = 13,
        hidden_dim: int = 64,
        num_spatial_heads: int = 4,
        num_temporal_heads: int = 4,
        dropout: float = 0.15
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Temporal Encoder
        self.temporal_encoder = TemporalAttentionLayer(
            in_dim=num_features,
            hidden_dim=hidden_dim,
            num_heads=num_temporal_heads,
            dropout=dropout
        )
        
        # Spatial Graph Attention
        self.spatial_attention = SpatialGraphAttentionLayer(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=num_spatial_heads,
            dropout=dropout
        )
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Node features of shape (batch, num_nodes, seq_len, num_features)
            adj: Dynamic Adjacency matrix of shape (batch, num_nodes, num_nodes) or (num_nodes, num_nodes)
        Returns:
            H_graph: Spatio-temporal node representations (batch, num_nodes, hidden_dim)
            spatial_weights: Attention matrix for XAI (batch, heads, num_nodes, num_nodes)
        """
        b, n, t, f = x.shape
        
        # 1. Temporal Encoding per node
        x_reshaped = x.view(b * n, t, f)
        temporal_out = self.temporal_encoder(x_reshaped) # (b * n, t, hidden_dim)
        
        # Take the most recent time-step representation
        h_t = temporal_out[:, -1, :].view(b, n, self.hidden_dim) # (b, n, hidden_dim)
        
        # 2. Spatial Graph Attention across stock nodes
        h_spatial = self.spatial_attention(h_t, adj) # (b, n, hidden_dim)
        
        # 3. Residual Feed-Forward
        h_out = self.final_norm(h_spatial + self.ffn(h_spatial))
        
        spatial_weights = self.spatial_attention.last_attn_weights
        return h_out, spatial_weights
