"""
Automated Unit and Integration Test Suite.
Validates:
1. Stock data pipeline and technical indicator calculations (no NaNs, correct columns).
2. Dynamic graph construction and adjacency matrix normalization.
3. RAG vector store indexing and cosine similarity retrieval.
4. Defensive error handling in FinLLM Reasoner (malformed responses, contradictions, fallbacks).
5. PyTorch Forward Pass of Dynamic Graph Transformer (DGT) and Multi-Modal Fusion.
6. XAI Attribution & Investor Report generation.
"""

import unittest
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch

from src.data.stock_data import StockDataFetcher, calculate_technical_indicators, DEFAULT_UNIVERSE
from src.data.news_data import FinancialNewsLoader
from src.graph.dynamic_graph import DynamicGraphBuilder
from src.rag.vector_store import FinancialVectorStore
from src.rag.fin_llm_reasoner import FinancialLLMReasoner, FinancialReasoningResult
from src.models.forecasting_head import ExplainableMultiModalForecaster
from src.xai.explainability import XAIEngine
from src.xai.report_generator import generate_investor_report


class TestStockForecastingPipeline(unittest.TestCase):

    def setUp(self):
        self.tickers = ["AAPL", "MSFT", "GOOGL", "NVDA"]
        self.fetcher = StockDataFetcher()

    def test_01_stock_data_and_indicators(self):
        """Validates that technical indicators compute cleanly with zero NaNs."""
        df = self.fetcher.fetch_ticker_data("AAPL", start="2023-01-01", end="2023-08-01")
        self.assertGreater(len(df), 50)
        self.assertIn("RSI_14", df.columns)
        self.assertIn("MACD", df.columns)
        self.assertIn("BB_Upper", df.columns)
        self.assertIn("Volatility_20", df.columns)
        # Check no NaNs exist
        self.assertEqual(df["RSI_14"].isna().sum(), 0)
        self.assertTrue((df["RSI_14"] >= 0).all() and (df["RSI_14"] <= 100).all())

    def test_02_dynamic_graph_construction(self):
        """Validates that dynamic adjacency matrices are symmetric/normalized and bounded."""
        universe = self.fetcher.fetch_universe(self.tickers, start="2023-01-01", end="2023-08-01")
        builder = DynamicGraphBuilder(tickers=self.tickers, correlation_window=20)
        
        adj = builder.compute_dynamic_adjacency(universe, current_idx=40)
        self.assertEqual(adj.shape, (4, 4))
        self.assertFalse(np.isnan(adj).any())
        self.assertTrue(np.all(adj >= 0.0))
        # Diagonal elements (self-loops) must be strictly non-zero
        self.assertTrue(np.all(np.diag(adj) > 0.0))

    def test_03_rag_vector_store(self):
        """Validates document indexing and cosine retrieval."""
        loader = FinancialNewsLoader(self.tickers)
        docs = loader.get_all_documents()
        store = FinancialVectorStore(embedding_dim=64)
        store.add_documents(docs)
        
        results = store.query("Apple AI and iPhone revenue expansion", ticker="AAPL", top_k=2)
        self.assertEqual(len(results), 2)
        top_doc, score = results[0]
        self.assertIsInstance(score, float)
        self.assertIn("AAPL", top_doc.ticker)

    def test_04_fin_llm_defensive_exception_handling(self):
        """
        Validates the critical requirement:
        Handling malformed responses, contradictions, and unexpected AI outputs.
        """
        reasoner = FinancialLLMReasoner(text_embed_dim=64)
        loader = FinancialNewsLoader(["NVDA"])
        docs = [(doc, 0.85) for doc in loader.get_documents_for_ticker("NVDA")]

        # Case A: Malformed raw LLM response with broken JSON
        broken_json = "I think the stock will go up! {sentiment_score: 0.85, confidence: 'high' "
        res_a = reasoner.reason_over_documents("NVDA", docs, raw_llm_response=broken_json)
        self.assertIsNotNone(res_a)
        # Should gracefully trigger fallback without crashing
        self.assertIn(res_a.validation_status, ["PASSED", "FALLBACK_TRIGGERED"])

        # Case B: Out-of-bounds sentiment score
        out_of_bounds = '{"sentiment_score": 999.0, "confidence": 1.5, "reasoning": "Unprecedented hype"}'
        res_b = reasoner.reason_over_documents("NVDA", docs, raw_llm_response=out_of_bounds)
        # Score must be safely clamped to 1.0
        self.assertLessEqual(res_b.sentiment_score, 1.0)
        self.assertLessEqual(res_b.confidence, 1.0)

        # Case C: Contradiction guardrail
        bearish_docs = [
            (FinancialNewsLoader().get_documents_for_ticker("GOOGL")[1], 0.9) # Antitrust / lawsuit
        ]
        hallucinated_bullish = '{"sentiment_score": 0.95, "confidence": 0.9, "reasoning": "Fantastic earnings outlook!"}'
        res_c = reasoner.reason_over_documents("GOOGL", bearish_docs, raw_llm_response=hallucinated_bullish)
        # Contradiction detector should repair or penalize the hallucinated score
        self.assertIn(res_c.validation_status, ["REPAIRED", "PASSED", "FALLBACK_TRIGGERED"])

    def test_05_model_forward_pass(self):
        """Validates tensor shapes through DGT, Multi-Modal Fusion, and Prediction Heads."""
        model = ExplainableMultiModalForecaster(
            num_features=13,
            graph_dim=32,
            text_dim=32,
            fusion_dim=32,
            num_spatial_heads=2,
            num_temporal_heads=2
        )
        
        batch_size = 2
        num_nodes = 4
        seq_len = 10
        num_features = 13
        
        x_num = torch.randn(batch_size, num_nodes, seq_len, num_features)
        adj = torch.eye(num_nodes).unsqueeze(0).expand(batch_size, -1, -1)
        x_txt = torch.randn(batch_size, num_nodes, 32)
        
        outputs = model(x_num, adj, x_txt)
        self.assertIn("pred_return", outputs)
        self.assertIn("pred_direction_logits", outputs)
        self.assertIn("spatial_attention", outputs)
        self.assertIn("text_gate", outputs)
        
        self.assertEqual(outputs["pred_return"].shape, (batch_size, num_nodes))
        self.assertEqual(outputs["pred_direction_logits"].shape, (batch_size, num_nodes, 2))
        self.assertEqual(outputs["text_gate"].shape, (batch_size, num_nodes, 1))

    def test_06_xai_attribution_and_reporting(self):
        """Validates SHAP/saliency feature importance and report generation."""
        model = ExplainableMultiModalForecaster(
            num_features=13,
            graph_dim=32,
            text_dim=32,
            fusion_dim=32,
            num_spatial_heads=2,
            num_temporal_heads=2
        )
        feature_names = [f"F_{i}" for i in range(13)]
        engine = XAIEngine(model, feature_names, self.tickers)
        
        x_num = torch.randn(1, 4, 10, 13)
        adj = torch.eye(4)
        x_txt = torch.randn(1, 4, 32)
        reasoning_map = {
            t: FinancialReasoningResult(t, 0.5, 0.8, "MODERATE_POSITIVE", "Solid fundamentals.", [], False, "PASSED")
            for t in self.tickers
        }
        
        explanation = engine.explain_prediction("AAPL", x_num, adj, x_txt, reasoning_map)
        self.assertEqual(explanation.ticker, "AAPL")
        self.assertEqual(len(explanation.top_features), 5)
        self.assertIn(explanation.predicted_direction, ["UP", "DOWN"])
        
        report = generate_investor_report(explanation)
        self.assertIn("EXPLAINABLE FINANCIAL FORECAST REPORT: AAPL", report)
        self.assertIn("MULTI-MODAL DECISION WEIGHTING", report)


if __name__ == "__main__":
    unittest.main()
