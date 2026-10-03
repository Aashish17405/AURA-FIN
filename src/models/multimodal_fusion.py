"""
Adaptive Cross-Attention Multi-Modal Fusion Layer.
Dynamically balances numerical graph-temporal representations with
textual financial reasoning tokens via gated cross-attention.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalCrossAttentionFusion(nn.Module):
    """
    Fuses:
    1. H_graph: Spatio-temporal embedding from Dynamic Graph Transformer (batch, num_nodes, graph_dim)
    2. H_text: Semantic reasoning embedding from Financial LLM RAG (batch, num_nodes, text_dim)
    
    Includes a volatility & confidence gating mechanism:
    If news signals are missing or neutral, gate g approaches 0, preserving pure technical graph signals.
    """
    def __init__(self, graph_dim: int = 64, text_dim: int = 64, fusion_dim: int = 64, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.fusion_dim = fusion_dim
        
        # Projections to common fusion dimension
        self.proj_graph = nn.Linear(graph_dim, fusion_dim)
        self.proj_text = nn.Linear(text_dim, fusion_dim)
        
        # Cross-Attention: Query from Graph, Key/Value from Text
        self.cross_attn = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        # Adaptive Confidence Gate: learns when text should override/augment technicals
        self.gate_net = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.layer_norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)
        self.last_gate_weight: Optional[torch.Tensor] = None

    def forward(self, h_graph: torch.Tensor, h_text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_graph: (batch_size, num_nodes, graph_dim)
            h_text: (batch_size, num_nodes, text_dim)
        Returns:
            z_fused: (batch_size, num_nodes, fusion_dim)
            gate: (batch_size, num_nodes, 1) - weight allocated to textual catalysts
        """
        # Linear project
        q = self.proj_graph(h_graph) # (b, n, fusion_dim)
        kv = self.proj_text(h_text)  # (b, n, fusion_dim)
        
        # Cross-attention
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        attn_out = self.dropout(attn_out)
        
        # Compute dynamic gate g in [0, 1] per node
        gate_input = torch.cat([q, attn_out], dim=-1)
        gate = self.gate_net(gate_input) # (b, n, 1)
        self.last_gate_weight = gate.detach()
        
        # Adaptive residual blend: z = (1 - g) * q + g * attn_out
        fused = ((1.0 - gate) * q) + (gate * attn_out)
        z_fused = self.layer_norm(fused)
        
        return z_fused, gate
