# Explainable Multi-Modal Stock Market Forecasting

### Using Dynamic Graph Transformers, Financial Large Language Models, and Retrieval-Augmented Generation

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Academic Milestone:** Project Stage I — Review 1  
> **Institution:** Keshav Memorial Institute of Technology (KMIT, Hyderabad)  
> **Department:** Department of Information Technology (IT - B)  
> **Team No:** 15 | **Academic Year:** 2026–2027  

---

## 👥 Team 15 Member Roles

| S. No. | Roll Number | Student Name | Role & Core Ownership |
| :---: | :---: | :---: | :--- |
| **1** | **23BD1A12A1** | **M SUMANTH** | **Data Engineering & Dynamic Graph Construction** (`src/data/stock_data.py`, `src/graph/dynamic_graph.py`) |
| **2** | **23BD1A12C9** | **YEDIDA SASHANK** | **Financial RAG Pipeline & FinLLM Reasoner** (`src/data/news_data.py`, `src/rag/vector_store.py`, `src/rag/fin_llm_reasoner.py`) |
| **3** | **23BD1A12CA** | **YELLASIRI SERENE RAJIV** | **Dynamic Graph Transformer (DGT) & Multi-Modal Fusion** (`src/models/dynamic_graph_transformer.py`, `src/models/multimodal_fusion.py`) |
| **4** | **23BD1A1287** | **JARPULA CHARAN** | **Explainable AI (XAI) Engine & Web Dashboard** (`src/xai/explainability.py`, `src/xai/report_generator.py`, `dashboard/app.py`) |

---

## 🚀 Key Architectural Innovations

1. **Dynamic Spatio-Temporal Graph Transformer (DGT):**  
   Stocks are modeled as dynamic nodes in an evolving financial network. Edges are computed via rolling 30-day cross-correlation matrices and sector affiliations, allowing the spatial graph attention mechanism to capture inter-stock momentum contagion.
2. **Financial Knowledge Retrieval (RAG):**  
   Financial news wires, earnings reports, and SEC filing summaries are indexed in a normalized dense vector store with top-$k$ semantic retrieval.
3. **Defensive AI Safety Net & Exception Handling:**  
   LLM outputs are strictly validated against JSON contracts, checked for contradiction against lexical financial facts, and gracefully fall back to deterministic weighted heuristics if an external model hallucinates, fails, or produces malformed syntax.
4. **Adaptive Cross-Attention Multi-Modal Fusion:**  
   Dynamically weights numerical time-series indicators against textual financial sentiment via confidence-gated cross-attention.
5. **Explainable AI (XAI):**  
   Extracts internal spatial graph attention weights and input gradient saliencies (SHAP proxy) to provide transparent investor reports.

---

## 📁 Repository Directory Structure

```
yedida/
├── data/
│   ├── raw/                      # Cached OHLCV market files (Parquet / CSV)
│   └── processed/                # Normalized indicators & graph matrices
├── src/
│   ├── data/
│   │   ├── stock_data.py         # YFinance data fetcher + 13 technical indicators
│   │   └── news_data.py          # Financial news & SEC documents loader
│   ├── graph/
│   │   ├── dynamic_graph.py      # Evolving cross-correlation adjacency matrix builder
│   │   └── graph_visualizer.py   # NetworkX & Matplotlib inter-stock network plot
│   ├── rag/
│   │   ├── vector_store.py       # In-memory dense semantic vector database
│   │   └── fin_llm_reasoner.py   # FinLLM reasoner with 3-layer defensive safety net
│   ├── models/
│   │   ├── dynamic_graph_transformer.py  # Spatio-Temporal Graph Transformer (DGT)
│   │   ├── multimodal_fusion.py          # Adaptive Cross-Attention Gated Fusion
│   │   ├── forecasting_head.py           # Multi-task regression & direction heads
│   │   └── baselines.py                  # Standard LSTM and MLP benchmarks
│   ├── xai/
│   │   ├── explainability.py     # Gradient feature attribution & graph attention
│   │   └── report_generator.py   # Human-readable investor summary generator
│   ├── train.py                  # End-to-end training and checkpoint saving
│   └── evaluate.py               # Empirical evaluation suite (MAE, RMSE, DA)
├── dashboard/
│   └── app.py                    # Interactive Streamlit investor intelligence UI
├── docs/
│   ├── PS1_Review1.docx          # Blank official review sheet
│   ├── PS1_Review1_Filled.md     # Completed evaluation dossier
│   └── Review1_Presentation_Deck.md  # 14-slide viva presentation deck
├── tests/
│   └── test_pipeline.py          # Automated unit & integration test suite (6 tests)
├── run_pipeline.py               # Master one-click execution script
├── requirements.txt              # Production dependency specifications
└── PROJECT_PLAN.md               # Detailed academic project plan
```

---

## ⚡ Quick Start Guide

### 1. Installation
Clone or navigate to the project directory and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Automated Test Suite
Run the 6 unit and integration test cases covering data fetching, dynamic graphs, defensive RAG handling, forward passes, and XAI:
```bash
python tests/test_pipeline.py
```
*(Expected output: `Ran 6 tests ... OK`)*

### 3. Run the Master End-to-End Pipeline
Train the DGT model, evaluate against LSTM/MLP baselines, and print an XAI investor report:
```bash
python run_pipeline.py --epochs 5 --ticker AAPL
```

### 4. Launch the Interactive Web Dashboard
Run the Streamlit application for the live review demonstration:
```bash
streamlit run dashboard/app.py
```

---

## 📊 Empirical Benchmark Results

Evaluated on an out-of-sample test split across 10 liquid equities over 790+ trading days:

| Architecture / Model | MAE (Error) | RMSE (Error) | Directional Accuracy (%) |
| :--- | :---: | :---: | :---: |
| **Baseline 1: Multi-Layer Perceptron (MLP)** | 0.0166 | 0.0217 | 50.18% |
| **Baseline 2: Standard LSTM (Time-Series Only)** | 0.0140 | 0.0192 | 53.04% |
| **Proposed: Multi-Modal DGT + RAG** | **0.0114** | **0.0159** | **63.80% ⭐** |

---

## 🛡️ Defensive Exception Handling Architecture

Because real-world AI and LLM APIs can hallucinate, time out, or produce malformed syntax, this project enforces strict defensive boundaries:
1. **Schema Validation & Clamping:** Outputs are parsed with regular expression fallbacks, and scores are clamped strictly into $[-1.0, 1.0]$.
2. **Contradiction Detection:** If an AI model asserts extreme bullish optimism while retrieved documents explicitly report SEC lawsuits or revenue collapse, the system flags the contradiction (`REPAIRED`) and bounds the sentiment.
3. **Deterministic Fallbacks:** If an LLM is unreachable or offline, the RAG reasoner automatically calculates similarity-weighted lexical sentiment, ensuring the pipeline and live demo never crash.
4. **Data Fault Tolerance:** Stock data fetching includes local parquet/csv file caching and automatic geometric Brownian motion synthetic fallback if external finance servers are unreachable.
