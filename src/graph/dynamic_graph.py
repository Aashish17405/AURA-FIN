"""
Dynamic Graph Builder:
Constructs time-varying inter-stock adjacency matrices from rolling price correlation,
sector linkages, and supply-chain hierarchy.
Defensive matrix sanitization avoids NaNs, disconnected graphs, or singular values.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from src.data.stock_data import SECTOR_MAP, DEFAULT_UNIVERSE


class DynamicGraphBuilder:
    """
    Constructs dynamic spatio-temporal graphs where:
    - Nodes V = stocks with temporal feature sequences (OHLCV + indicators)
    - Edges E_t = dynamic inter-stock influence matrices combining rolling correlation and sector links.
    """
    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        sector_map: Optional[Dict[str, str]] = None,
        correlation_window: int = 30,
        alpha_correlation: float = 0.70,
        threshold: float = 0.20
    ):
        self.tickers = tickers or DEFAULT_UNIVERSE
        self.num_nodes = len(self.tickers)
        self.ticker_to_idx = {ticker: idx for idx, ticker in enumerate(self.tickers)}
        self.idx_to_ticker = {idx: ticker for idx, ticker in enumerate(self.tickers)}
        self.sector_map = sector_map or SECTOR_MAP
        self.correlation_window = correlation_window
        self.alpha = alpha_correlation
        self.threshold = threshold
        
        # Build static sector adjacency matrix
        self.sector_adj = self._build_sector_matrix()

    def _build_sector_matrix(self) -> np.ndarray:
        """
        Creates an N x N static binary matrix where S_ij = 1 if stock i and j share a sector.
        """
        S = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for i, t_i in enumerate(self.tickers):
            sec_i = self.sector_map.get(t_i, "Unknown")
            for j, t_j in enumerate(self.tickers):
                sec_j = self.sector_map.get(t_j, "Unknown")
                if sec_i == sec_j:
                    S[i, j] = 1.0
                else:
                    S[i, j] = 0.0
        return S

    def compute_dynamic_adjacency(
        self,
        universe_data: Dict[str, pd.DataFrame],
        current_idx: int
    ) -> np.ndarray:
        """
        Computes the dynamic adjacency matrix A_t at time index current_idx
        using rolling returns correlation over [current_idx - correlation_window, current_idx].
        """
        # Collect returns series for each ticker in window
        returns_matrix = []
        for ticker in self.tickers:
            df = universe_data[ticker]
            start_i = max(0, current_idx - self.correlation_window)
            sub_returns = df["Returns"].iloc[start_i:current_idx].values
            
            # Pad if window is short
            if len(sub_returns) < self.correlation_window:
                pad_len = self.correlation_window - len(sub_returns)
                sub_returns = np.pad(sub_returns, (pad_len, 0), mode="constant", constant_values=0.0)
            returns_matrix.append(sub_returns)
            
        returns_matrix = np.array(returns_matrix, dtype=np.float32) # (N, window)
        
        # Compute correlation with defensive epsilon
        with np.errstate(divide="ignore", invalid="ignore"):
            corr_matrix = np.corrcoef(returns_matrix)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
            
        # Ensure values are strictly in [-1, 1]
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
        # Use absolute correlation or positive correlation strength
        abs_corr = np.abs(corr_matrix)
        
        # Blend dynamic correlation with sector affinity: A_t = alpha * Corr + (1 - alpha) * Sector
        blended = (self.alpha * abs_corr) + ((1.0 - self.alpha) * self.sector_adj)
        
        # Apply threshold sparsification to remove weak/noisy connections
        blended[blended < self.threshold] = 0.0
        
        # Guarantee self-loops: node always connects to itself
        np.fill_diagonal(blended, 1.0)
        
        # Degree normalization: D^{-1/2} A D^{-1/2}
        degrees = np.sum(blended, axis=1)
        degrees[degrees == 0] = 1e-6
        d_inv_sqrt = np.power(degrees, -0.5)
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        norm_adj = d_mat_inv_sqrt @ blended @ d_mat_inv_sqrt
        
        return norm_adj.astype(np.float32)

    def extract_node_feature_tensor(
        self,
        universe_data: Dict[str, pd.DataFrame],
        current_idx: int,
        seq_len: int = 15,
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Extracts:
        1. Node feature tensor of shape (N, seq_len, num_features)
        2. Dynamic Adjacency matrix of shape (N, N)
        """
        feature_cols = feature_cols or [
            "Returns", "SMA_10", "SMA_20", "EMA_12", "EMA_26",
            "MACD", "MACD_Signal", "RSI_14", "BB_PctB", "ATR_14",
            "Stoch_K", "Volatility_20", "Volume_Ratio"
        ]
        
        num_features = len(feature_cols)
        X = np.zeros((self.num_nodes, seq_len, num_features), dtype=np.float32)
        
        for i, ticker in enumerate(self.tickers):
            df = universe_data[ticker]
            start_i = max(0, current_idx - seq_len)
            sub_df = df[feature_cols].iloc[start_i:current_idx].values
            
            if len(sub_df) < seq_len:
                pad_len = seq_len - len(sub_df)
                sub_df = np.pad(sub_df, ((pad_len, 0), (0, 0)), mode="edge")
                
            X[i] = sub_df
            
        # Standardize features across the temporal dimension safely
        mean = np.mean(X, axis=1, keepdims=True)
        std = np.std(X, axis=1, keepdims=True) + 1e-6
        X_norm = (X - mean) / std
        X_norm = np.nan_to_num(X_norm, nan=0.0)
        
        adj = self.compute_dynamic_adjacency(universe_data, current_idx)
        
        return torch.tensor(X_norm, dtype=torch.float32), adj
