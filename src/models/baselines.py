"""
Benchmark Baseline Models for Empirical Review Evaluation:
1. MLPBaseline: Multi-layer Perceptron operating on flattened temporal features.
2. LSTMBaseline: Classical Recurrent Neural Network for single-stock time series.
"""

import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    """
    Standard Multi-Layer Perceptron baseline.
    Ignores inter-stock graph connections and textual documents.
    """
    def __init__(self, num_features: int = 13, seq_len: int = 15, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features * seq_len, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, num_nodes, seq_len, num_features)
        Returns: pred_return (batch, num_nodes)
        """
        b, n, t, f = x.shape
        flat_x = x.view(b * n, t * f)
        out = self.net(flat_x).view(b, n)
        return out


class LSTMBaseline(nn.Module):
    """
    Standard LSTM baseline for financial time-series forecasting.
    Captures temporal patterns per stock independently, without graph or text.
    """
    def __init__(self, num_features: int = 13, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, num_nodes, seq_len, num_features)
        Returns: pred_return (batch, num_nodes)
        """
        b, n, t, f = x.shape
        flat_x = x.view(b * n, t, f)
        lstm_out, _ = self.lstm(flat_x)
        last_step = lstm_out[:, -1, :] # (b * n, hidden_dim)
        out = self.fc(last_step).view(b, n)
        return out
