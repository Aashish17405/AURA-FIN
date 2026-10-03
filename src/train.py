"""
End-to-End Training and Validation Pipeline.
Orchestrates:
1. Data Fetching & Feature Engineering
2. Dynamic Graph Sequence Generation
3. RAG Retrieval & Defensive FinLLM Reasoning
4. DGT + Fusion Multi-Task Training with Early Stopping & Checkpoint Saving
"""

import os
import time
import logging
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.data.stock_data import StockDataFetcher, DEFAULT_UNIVERSE
from src.data.news_data import FinancialNewsLoader
from src.graph.dynamic_graph import DynamicGraphBuilder
from src.rag.vector_store import FinancialVectorStore
from src.rag.fin_llm_reasoner import FinancialLLMReasoner
from src.models.forecasting_head import ExplainableMultiModalForecaster

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class MultiModalStockDataset(Dataset):
    """
    PyTorch Dataset providing (x_numerical, adj, x_textual, y_return, y_direction) per time step.
    """
    def __init__(
        self,
        samples: List[Tuple[torch.Tensor, np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]]
    ):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_num, adj, x_txt, y_ret, y_dir = self.samples[idx]
        return x_num, torch.tensor(adj, dtype=torch.float32), x_txt, y_ret, y_dir


def prepare_multimodal_dataset(
    tickers: List[str] = DEFAULT_UNIVERSE,
    seq_len: int = 15,
    start_date: str = "2023-01-01",
    end_date: str = "2026-03-01"
):
    logger.info("Initializing Data Fetcher & Technical Indicators...")
    fetcher = StockDataFetcher()
    universe_data = fetcher.fetch_universe(tickers, start=start_date, end=end_date)
    
    logger.info("Initializing Financial Vector Store & News Ingestion...")
    news_loader = FinancialNewsLoader(tickers)
    all_docs = news_loader.get_all_documents()
    
    vector_store = FinancialVectorStore(embedding_dim=64)
    vector_store.add_documents(all_docs)
    
    reasoner = FinancialLLMReasoner(text_embed_dim=64)
    
    # Precompute textual reasoning embeddings per ticker
    text_embeddings = {}
    fin_reasoning_map = {}
    for ticker in tickers:
        retrieved = vector_store.query(f"{ticker} earnings quarterly revenue performance", ticker=ticker, top_k=3)
        res = reasoner.reason_over_documents(ticker, retrieved)
        fin_reasoning_map[ticker] = res
        text_embeddings[ticker] = reasoner.get_text_embedding_tensor(res)
        
    # Stack textual embeddings across nodes: (num_nodes, text_dim)
    node_text_tensor = torch.stack([text_embeddings[t] for t in tickers], dim=0)
    
    logger.info("Constructing Dynamic Graph sequences...")
    graph_builder = DynamicGraphBuilder(tickers=tickers, correlation_window=30)
    
    # Find common sequence length
    min_len = min(len(df) for df in universe_data.values())
    logger.info(f"Common aligned time steps: {min_len} trading days across {len(tickers)} stocks.")
    
    samples = []
    # Start after correlation_window + seq_len to ensure full historical window
    start_idx = 45
    for t_idx in range(start_idx, min_len - 1):
        x_num, adj = graph_builder.extract_node_feature_tensor(universe_data, t_idx, seq_len=seq_len)
        
        # Targets for all stocks at step t_idx: 1-day forward return & direction
        y_ret = torch.tensor([universe_data[t]["Target_Return_1d"].iloc[t_idx] for t in tickers], dtype=torch.float32)
        y_dir = torch.tensor([universe_data[t]["Target_Direction_1d"].iloc[t_idx] for t in tickers], dtype=torch.long)
        
        samples.append((x_num, adj, node_text_tensor, y_ret, y_dir))
        
    return samples, universe_data, fin_reasoning_map


def train_model(
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 0.001,
    save_path: str = "models/best_multimodal_dgt.pt"
) -> Tuple[ExplainableMultiModalForecaster, Dict]:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    samples, universe_data, fin_reasoning_map = prepare_multimodal_dataset()
    total_samples = len(samples)
    
    train_split = int(0.70 * total_samples)
    val_split = int(0.85 * total_samples)
    
    train_data = samples[:train_split]
    val_data = samples[train_split:val_split]
    test_data = samples[val_split:]
    
    logger.info(f"Dataset splits: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    train_loader = DataLoader(MultiModalStockDataset(train_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(MultiModalStockDataset(val_data), batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")
    
    model = ExplainableMultiModalForecaster(
        num_features=13,
        graph_dim=64,
        text_dim=64,
        fusion_dim=64,
        num_spatial_heads=4,
        num_temporal_heads=4,
        dropout=0.15
    ).to(device)
    
    criterion_reg = nn.SmoothL1Loss() # Robust Huber loss
    criterion_cls = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    
    logger.info("Beginning training loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for x_num, adj, x_txt, y_ret, y_dir in train_loader:
            x_num, adj, x_txt = x_num.to(device), adj.to(device), x_txt.to(device)
            y_ret, y_dir = y_ret.to(device), y_dir.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_num, adj, x_txt)
            
            # Multi-task loss: Regression Loss + Direction Classification Loss
            loss_reg = criterion_reg(outputs["pred_return"], y_ret)
            # Reshape for cross entropy: (batch * num_nodes, 2) vs (batch * num_nodes)
            loss_cls = criterion_cls(
                outputs["pred_direction_logits"].view(-1, 2),
                y_dir.view(-1)
            )
            total_loss = loss_reg + 0.5 * loss_cls
            
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            train_loss += total_loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct_dir = 0
        total_predictions = 0
        
        with torch.no_grad():
            for x_num, adj, x_txt, y_ret, y_dir in val_loader:
                x_num, adj, x_txt = x_num.to(device), adj.to(device), x_txt.to(device)
                y_ret, y_dir = y_ret.to(device), y_dir.to(device)
                
                outputs = model(x_num, adj, x_txt)
                l_reg = criterion_reg(outputs["pred_return"], y_ret)
                l_cls = criterion_cls(outputs["pred_direction_logits"].view(-1, 2), y_dir.view(-1))
                val_loss += (l_reg + 0.5 * l_cls).item()
                
                # Direction accuracy
                preds = torch.argmax(outputs["pred_direction_logits"], dim=-1)
                correct_dir += (preds == y_dir).sum().item()
                total_predictions += y_dir.numel()
                
        val_loss /= len(val_loader)
        val_acc = (correct_dir / total_predictions) * 100.0 if total_predictions > 0 else 0.0
        scheduler.step(val_loss)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        logger.info(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dir Acc: {val_acc:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "history": history
            }, save_path)
            logger.info(f"--> Saved best model checkpoint to {save_path}")
            
    return model, {
        "history": history,
        "test_data": test_data,
        "universe_data": universe_data,
        "fin_reasoning_map": fin_reasoning_map
    }


if __name__ == "__main__":
    train_model(epochs=8)
