"""
Explainable AI (XAI) Engine:
1. Feature Attribution: Identifies which technical indicators (RSI, MACD, Volume) drove the forecast.
2. Dynamic Graph Attention Attribution: Identifies which peer stocks in the graph exerted cross-asset influence.
3. Multi-Modal Gate Attribution: Measures the percentage contribution of news/filings vs. technical price action.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from src.models.forecasting_head import ExplainableMultiModalForecaster
from src.rag.fin_llm_reasoner import FinancialReasoningResult

@dataclass
class StockExplanation:
    ticker: str
    predicted_return: float
    predicted_direction: str          # "UP" or "DOWN"
    direction_probability: float      # e.g., 0.78
    top_features: List[Tuple[str, float]] # Top-5 technical features & percentage contribution
    top_influential_neighbors: List[Tuple[str, float]] # Top-3 correlated peer stocks & attention weights
    textual_contribution_pct: float   # Gating percentage attributed to news/filings
    fin_reasoning: FinancialReasoningResult


class XAIEngine:
    """
    Computes exact attribution scores and extracts internal transformer attention weights.
    """
    def __init__(self, model: ExplainableMultiModalForecaster, feature_names: List[str], tickers: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.tickers = tickers
        self.ticker_to_idx = {t: i for i, t in enumerate(tickers)}

    def explain_prediction(
        self,
        target_ticker: str,
        x_numerical: torch.Tensor,
        adj: torch.Tensor,
        x_textual: torch.Tensor,
        fin_reasoning_map: Dict[str, FinancialReasoningResult]
    ) -> StockExplanation:
        """
        Computes multi-level explanations for the target stock.
        """
        self.model.eval()
        target_idx = self.ticker_to_idx[target_ticker]
        
        # Clone tensor with gradient enabled for saliency / attribution
        x_num_grad = x_numerical.clone().detach().requires_grad_(True)
        
        # Forward pass
        outputs = self.model(x_num_grad, adj, x_textual)
        pred_return = float(outputs["pred_return"][0, target_idx].item())
        
        # Direction probabilities
        dir_logits = outputs["pred_direction_logits"][0, target_idx]
        dir_probs = torch.softmax(dir_logits, dim=-1)
        up_prob = float(dir_probs[1].item())
        predicted_dir = "UP" if up_prob >= 0.5 else "DOWN"
        dir_confidence = up_prob if predicted_dir == "UP" else (1.0 - up_prob)
        
        # 1. Feature Attribution via Gradient * Input (Saliency)
        # Target the predicted return
        outputs["pred_return"][0, target_idx].backward(retain_graph=True)
        grad = x_num_grad.grad[0, target_idx] # (seq_len, num_features)
        val = x_num_grad[0, target_idx]
        
        # Attribution per feature averaged across sequence steps
        attribution = torch.mean(torch.abs(grad * val), dim=0).detach().cpu().numpy()
        total_attr = np.sum(attribution) + 1e-9
        norm_attribution = (attribution / total_attr) * 100.0
        
        # Rank top features
        feature_scores = list(zip(self.feature_names, norm_attribution))
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        top_features = [(f, float(s)) for f, s in feature_scores[:5]]
        
        # 2. Graph Attention Influence: Which stocks affected target_ticker?
        # spatial_attention: (batch, heads, N, N)
        spatial_attn = outputs["spatial_attention"][0].mean(dim=0).detach().cpu().numpy() # (N, N)
        incoming_attention = spatial_attn[target_idx] # (N,) weights from all stocks j into target_idx
        
        neighbor_scores = []
        for j, ticker_j in enumerate(self.tickers):
            if ticker_j != target_ticker: # Exclude self-loop for clearer neighbor ranking
                neighbor_scores.append((ticker_j, float(incoming_attention[j])))
        neighbor_scores.sort(key=lambda x: x[1], reverse=True)
        top_neighbors = neighbor_scores[:3]
        
        # 3. Multi-Modal Gate Attribution (News vs. Graph)
        text_gate = float(outputs["text_gate"][0, target_idx, 0].item())
        text_pct = round(text_gate * 100.0, 1)
        
        reasoning = fin_reasoning_map.get(
            target_ticker,
            FinancialReasoningResult(
                ticker=target_ticker,
                sentiment_score=0.0,
                confidence=0.5,
                market_impact="NEUTRAL",
                reasoning_summary="No specific news catalysts recorded.",
                retrieved_documents=[],
                is_fallback=True,
                validation_status="FALLBACK_TRIGGERED"
            )
        )
        
        return StockExplanation(
            ticker=target_ticker,
            predicted_return=pred_return,
            predicted_direction=predicted_dir,
            direction_probability=dir_confidence,
            top_features=top_features,
            top_influential_neighbors=top_neighbors,
            textual_contribution_pct=text_pct,
            fin_reasoning=reasoning
        )
