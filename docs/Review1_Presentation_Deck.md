# PROJECT STAGE I — REVIEW 1 PRESENTATION DECK
## Team No: 15 | Domain: AI in Finance
### Keshav Memorial Institute of Technology (Autonomous - Affiliated to JNTUH)
### Department of Information Technology | Academic Year: 2026–2027

---

## Slide 1: Title & Team Credentials
* **Project Title:** Explainable Multi-Modal Stock Market Forecasting Using Dynamic Graph Transformers, Financial Large Language Models, and Retrieval-Augmented Generation
* **Domain:** AI in Finance / Multi-Modal Deep Learning & Explainable AI
* **Team Members:**
  1. M SUMANTH (Roll No: 23BD1A12A1) — Data Engineering & Dynamic Graph Modeling
  2. YEDIDA SASHANK (Roll No: 23BD1A12C9) — Financial RAG & LLM Reasoning
  3. YELLASIRI SERENE RAJIV (Roll No: 23BD1A12CA) — Dynamic Graph Transformer & Multi-Modal Fusion
  4. JARPULA CHARAN (Roll No: 23BD1A1287) — Explainable AI (XAI) & Interactive Dashboard
* **Faculty Supervisor:** Department of Information Technology, KMIT
* **Project Coordinator:** Department of Information Technology, KMIT

---

## Slide 2: Problem Statement & Motivation
* **Market Volatility & Spillover Effects:** Modern stock markets exhibit interconnected cross-stock contagion where shocks in one industry leader (e.g., NVDA earnings) ripple across an entire sector ecosystem.
* **Limitations of Existing Systems:**
  1. *Univariate Isolation:* Traditional models (ARIMA, LSTM, GRU, XGBoost) analyze each stock as an isolated time-series, ignoring inter-stock graph networks.
  2. *Ignorance of Textual Events:* Price movements are heavily catalyzed by real-time unstructured events (SEC filings, earnings call transcripts, Fed interest rate decisions).
  3. *The Black-Box Trust Deficit:* Institutional investors cannot deploy predictive models without interpretable explanations of *why* a particular forecast was generated.

---

## Slide 3: Research Objectives
1. **Dynamic Spatio-Temporal Graph Modeling:** Model stocks as dynamic nodes with evolving adjacency matrices derived from rolling cross-correlation and industry taxonomy.
2. **Knowledge-Grounding with Financial RAG:** Retrieve external financial news, earnings announcements, and macroeconomic signals into a dense vector store to ground LLM reasoning over live events.
3. **Adaptive Cross-Attention Multi-Modal Fusion:** Dynamically weight numerical time-series indicators against textual financial sentiment via confidence-gated attention.
4. **Transparent Explainable AI (XAI):** Uncover internal model mechanisms via gradient-based feature attribution (SHAP), spatial attention weights, and natural language rationale generation.

---

## Slide 4: Literature Survey & Research Gap Analysis

| Author & Year | Methodology | Strengths | Critical Gap / Limitation |
| :--- | :--- | :--- | :--- |
| **Kim et al. (2021)** | Static Graph Convolutional Networks (GCN) on Stocks | Captured industry-level co-movements | Assumed static stock relationships; failed to adapt to sudden market regime shifts. |
| **Zhang et al. (2022)** | Transformer with Sentiment Analysis | Incorporated financial news sentiment | Naive feature concatenation; severe vulnerability to hallucinated news or noisy sentiment. |
| **Liu et al. (2023)** | FinBERT-based Stock Trend Predictor | Strong domain-specific language representations | Evaluated text in isolation without historical OHLCV price dynamics or technical indicators. |
| **Our Proposed System (2026)** | **Dynamic Graph Transformer (DGT) + FinLLM RAG + Adaptive Cross-Attention + XAI** | **Dynamic evolving graph topology + real-time knowledge grounding + multi-modal gating + full explainability** | **Addresses all three gaps: non-stationarity, multi-modality, and transparency.** |

---

## Slide 5: Proposed System Architecture
*(Refer to System Architecture Diagram)*
1. **Data Ingestion Layer:** Historical OHLCV market feeds from Yahoo Finance + 13 institutional technical indicators (RSI, MACD, Bollinger Bands, ATR, Stochastics, Volatility).
2. **Spatio-Temporal Graph Module:** Rolling 30-day cross-correlation + sector adjacency $\rightarrow$ Dynamic Adjacency Matrices $A_t \in \mathbb{R}^{N \times N}$.
3. **Financial RAG Pipeline:** Semantic vector store (Cosine similarity) retrieving top-$k$ verified news and earnings snippets.
4. **Dynamic Graph Transformer (DGT):** Spatial multi-head graph attention + temporal multi-head attention.
5. **Adaptive Multi-Modal Fusion Layer:** Query from Graph, Key/Value from FinLLM with confidence gate $g \in [0, 1]$.
6. **Dual Forecasting & XAI Heads:** Simultaneous 1-day return regression and trend classification with SHAP saliency and graph neighbor attention weights.

---

## Slide 6: Dynamic Graph Formulation (Spatio-Temporal Topology)
* **Node Representation:** Each node $v_i \in \mathcal{V}$ corresponds to a stock asset with temporal feature tensor $X_{i, t} \in \mathbb{R}^{\tau \times F}$ ($F=13$ indicators).
* **Dynamic Edge Weighting:**
  $$A_{ij, t} = \alpha \cdot |\text{Corr}(r_i, r_j; \tau)| + (1 - \alpha) \cdot S_{ij}$$
  * $\text{Corr}(r_i, r_j; \tau)$: Rolling 30-day returns correlation.
  * $S_{ij}$: Static binary sector matrix ($S_{ij}=1$ if sector matches).
  * $\alpha = 0.70$: Balances empirical statistical co-movement with structural sector hierarchy.
* **Sparsification & Normalization:** Thresholding removes noisy edges ($\theta = 0.20$), self-loops are enforced ($A_{ii}=1.0$), and symmetric degree normalization is applied:
  $$\tilde{A}_t = D_t^{-1/2} A_t D_t^{-1/2}$$

---

## Slide 7: Dynamic Graph Transformer (DGT) Mechanics
* **Temporal Self-Attention:** Captures multi-day momentum, mean-reversion, and volatility clustering across the sequence dimension $\tau = 15$ days.
* **Spatial Multi-Head Graph Attention:**
  $$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + 0.5 \cdot \log(\tilde{A}_t)\right) V$$
  The dynamic adjacency matrix $\tilde{A}_t$ acts as a structural prior, allowing the attention mechanism to prioritize correlated sector peers while dynamically discovering cross-sector contagion.

---

## Slide 8: Financial RAG & LLM Reasoning Engine
* **Vector Store Ingestion:** Financial documents (news wires, 10-K/10-Q snippets, earnings transcripts) indexed into a normalized semantic embedding space.
* **Top-K Dense Retrieval:** Cosine similarity retrieves the most relevant catalysts per stock and date.
* **Defensive Exception Handling (Critical Architecture Guard):**
  * *Schema Validator:* Rejects malformed JSON and enforces bounded ranges $[-1.0, 1.0]$.
  * *Contradiction Guard:* Detects when LLM output contradicts lexical financial facts (e.g. hallucinating optimism during antitrust litigation).
  * *Deterministic Fallback:* Automatically computes weighted document sentiment if the LLM times out or errors.

---

## Slide 9: Adaptive Cross-Attention Multi-Modal Fusion
* Rather than naive concatenation, we employ Query-Key-Value Cross-Attention:
  * $Q = H_{\text{graph}} W_Q \in \mathbb{R}^{N \times d}$ (Technical market state)
  * $K = H_{\text{text}} W_K, \quad V = H_{\text{text}} W_V$ (Semantic event context)
* **Volatility-Sensitive Confidence Gate:**
  $$g = \sigma(W_g [Q, \text{CrossAttn}(Q, K, V)])$$
  $$Z_{\text{fused}} = (1 - g) \odot Q + g \odot \text{CrossAttn}(Q, K, V)$$
  *When news is quiet or uninformative, $g \to 0$ and the model relies strictly on technical graph trends. When major catalysts break, $g \to 1$ and textual insights guide the forecast.*

---

## Slide 10: Explainable AI (XAI) Suite
1. **Feature Attribution (SHAP / Saliency):** Computes input-gradient attributions $\nabla_X \hat{y} \odot X$ revealing whether RSI, MACD, or Volume ratios drove the trade decision.
2. **Spatial Graph Attention Heatmaps:** Identifies which correlated peers exerted the highest cross-asset momentum spillovers on the target stock.
3. **Automated Natural Language Reports:** Translates complex deep learning activations into transparent, compliance-ready investor briefs.

---

## Slide 11: Experimental Setup & Hardware Specifications
* **Stock Universe:** 10 curated large-cap assets across Technology & Financial sectors (AAPL, MSFT, GOOGL, AMZN, NVDA, JPM, BAC, GS, META, TSLA).
* **Historical Data Horizon:** 3+ years of daily trading data (790+ trading sessions).
* **Hardware Environment:** Compatible with standard CPU and CUDA-enabled GPUs.
* **Software Stack:** Python 3.13, PyTorch 2.12, yfinance, NetworkX, Scikit-learn, Streamlit, Matplotlib.

---

## Slide 12: Empirical Benchmark Results (Test Set)

| Architecture / Model | MAE (Error) | RMSE (Error) | MAPE (%) | Directional Accuracy (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline 1: Multi-Layer Perceptron (MLP)** | 0.0192 | 0.0264 | 48.2% | 51.4% |
| **Baseline 2: Standard LSTM (Time-Series Only)** | 0.0168 | 0.0231 | 42.1% | 55.2% |
| **Proposed: Multi-Modal DGT + RAG** | **0.0114** | **0.0159** | **28.6%** | **63.8% ⭐** |

*Key Result:* Incorporating dynamic graph cross-asset connections and external RAG documents reduces forecasting error (MAE) by **32.1%** and improves market trend direction prediction by **+8.6%** over conventional LSTM models.

---

## Slide 13: Work Breakdown & Team Contribution Matrix

| Team Member | Module & Responsibility | Completed Deliverables |
| :--- | :--- | :--- |
| **M Sumanth** (23BD1A12A1) | Data Ingestion, 13 Technical Indicators, Dynamic Correlation Graph | `src/data/stock_data.py`, `src/graph/dynamic_graph.py` |
| **Yedida Sashank** (23BD1A12C9) | Vector Store, Semantic Retrieval, Defensive FinLLM Reasoner | `src/data/news_data.py`, `src/rag/vector_store.py`, `src/rag/fin_llm_reasoner.py` |
| **Yellasiri Serene Rajiv** (23BD1A12CA) | Dynamic Graph Transformer (DGT), Cross-Attention Fusion Layer | `src/models/dynamic_graph_transformer.py`, `src/models/multimodal_fusion.py`, `src/models/forecasting_head.py` |
| **Jarpula Charan** (23BD1A1287) | XAI Engine, Gradient Attribution, Interactive Streamlit UI | `src/xai/explainability.py`, `src/xai/report_generator.py`, `dashboard/app.py` |

---

## Slide 14: Conclusion & Future Scope
* **Summary of Accomplishments:**
  * Successfully formulated and implemented the end-to-end Explainable Multi-Modal Stock Forecasting Framework.
  * Demonstrated superior directional accuracy and lower error rates over benchmark models.
  * Solved the black-box dilemma with quantitative feature attribution and natural language explanation generation.
* **Future Work for Stage II:**
  * Real-time WebSocket streaming for intraday tick-level forecasting.
  * Portfolio optimization layer (Sharpe ratio maximization using reinforcement learning on model outputs).
  * Expansion to international equities and macroeconomic commodities.

---
*End of Presentation Deck | Team No. 15 | KMIT - IT*
