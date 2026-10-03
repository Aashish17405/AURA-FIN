"""
Master One-Click Execution Pipeline.
Runs the entire Explainable Multi-Modal Stock Market Forecasting Framework:
1. Fetching & Feature Engineering (13+ Indicators)
2. Dynamic Correlation Graph Generation
3. Financial RAG Ingestion & FinLLM Reasoning
4. Dynamic Graph Transformer (DGT) Model Training
5. Benchmark Evaluation against MLP and LSTM baselines
6. Explainable AI (XAI) Attribution & Report Generation
"""

import os
import sys
import argparse
import logging
import torch

from src.data.stock_data import StockDataFetcher, DEFAULT_UNIVERSE
from src.train import train_model
from src.evaluate import evaluate_all_models
from src.xai.explainability import XAIEngine
from src.xai.report_generator import generate_investor_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MASTER_PIPELINE")


def run_full_pipeline(epochs: int = 10, batch_size: int = 8, demo_ticker: str = "AAPL"):
    print("\n" + "="*80)
    print(" EXPLAINABLE MULTI-MODAL STOCK MARKET FORECASTING FRAMEWORK")
    print(" Team 15 | KMIT - Department of Information Technology")
    print("="*80 + "\n")
    
    # 1. Train Model & Save Checkpoint
    logger.info(">>> STEP 1: Running Model Training Pipeline (DGT + Multi-Modal Fusion)...")
    model, artifacts = train_model(epochs=epochs, batch_size=batch_size, save_path="models/best_multimodal_dgt.pt")
    
    # 2. Run Benchmark Evaluation
    logger.info(">>> STEP 2: Running Benchmark Evaluation against Baseline Models...")
    test_data = artifacts["test_data"]
    eval_df = evaluate_all_models(
        model_path="models/best_multimodal_dgt.pt",
        test_data=test_data
    )
    print("\n" + "-"*80)
    print(" EMPIRICAL BENCHMARK EVALUATION (Out-of-Sample Test Set):")
    print(eval_df.to_string(index=False))
    print("-"*80 + "\n")
    
    # 3. Generate Explainable AI Report for Demo Ticker
    logger.info(f">>> STEP 3: Generating Explainable AI (XAI) Attribution for {demo_ticker}...")
    sample_x_num, sample_adj, sample_x_txt, _, _ = test_data[-1]
    sample_x_num = sample_x_num.unsqueeze(0)
    sample_adj = torch.tensor(sample_adj, dtype=torch.float32)
    sample_x_txt = sample_x_txt.unsqueeze(0)
    
    feature_names = [
        "Returns", "SMA_10", "SMA_20", "EMA_12", "EMA_26",
        "MACD", "MACD_Signal", "RSI_14", "BB_PctB", "ATR_14",
        "Stoch_K", "Volatility_20", "Volume_Ratio"
    ]
    xai = XAIEngine(model, feature_names, DEFAULT_UNIVERSE)
    explanation = xai.explain_prediction(
        demo_ticker, sample_x_num, sample_adj, sample_x_txt, artifacts["fin_reasoning_map"]
    )
    report = generate_investor_report(explanation)
    print(report)
    
    print("\n" + "="*80)
    print(" PIPELINE EXECUTION COMPLETE!")
    print(" To launch the Interactive Web Dashboard, execute:")
    print("     streamlit run dashboard/app.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full stock forecasting pipeline.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Target demo ticker for XAI report")
    args = parser.parse_args()
    
    run_full_pipeline(epochs=args.epochs, demo_ticker=args.ticker)
