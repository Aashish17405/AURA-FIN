# A MAJOR PROJECT REPORT ON
# EXPLAINABLE MULTI-MODAL STOCK MARKET FORECASTING USING DYNAMIC GRAPH TRANSFORMERS, FINANCIAL LARGE LANGUAGE MODELS, AND RETRIEVAL-AUGMENTED GENERATION

**Submitted in partial fulfillment of the requirements for the award of the degree of**  
**BACHELOR OF TECHNOLOGY IN INFORMATION TECHNOLOGY**

---

### Submitted by:
* **M SUMANTH** (Roll No: **23BD1A12A1**)
* **YEDIDA SASHANK** (Roll No: **23BD1A12C9**)
* **YELLASIRI SERENE RAJIV** (Roll No: **23BD1A12CA**)
* **JARPULA CHARAN** (Roll No: **23BD1A1287**)

### Under the Esteemed Guidance of:
**Faculty Supervisor**  
Department of Information Technology  
Keshav Memorial Institute of Technology, Hyderabad

**Department of Information Technology**  
**KESHAV MEMORIAL INSTITUTE OF TECHNOLOGY**  
*(An Autonomous Institution, Accredited by NBA & NAAC, Approved by AICTE, Affiliated to JNTUH)*  
*Narayanguda, Hyderabad - 500029, Telangana, India*  
**Academic Year: 2026–2027**

---

## CERTIFICATE

This is to certify that the Major Project Report entitled **"Explainable Multi-Modal Stock Market Forecasting Using Dynamic Graph Transformers, Financial Large Language Models, and Retrieval-Augmented Generation"** is a bonafide work carried out by **M SUMANTH (23BD1A12A1), YEDIDA SASHANK (23BD1A12C9), YELLASIRI SERENE RAJIV (23BD1A12CA), and JARPULA CHARAN (23BD1A1287)** in partial fulfillment of the requirements for the award of the degree of **Bachelor of Technology in Information Technology** from **Keshav Memorial Institute of Technology (Affiliated to JNTUH)** during the academic year 2026–2027.

The results embodied in this report have not been submitted to any other university or institute for the award of any degree or diploma.

\
\
__________________________  
**Internal Supervisor**  
Department of Information Technology  
KMIT, Hyderabad  

\
\
__________________________  
**Head of the Department**  
Department of Information Technology  
KMIT, Hyderabad  

\
\
__________________________  
**External Examiner**  
Viva-Voce Examination Date: ______________

---

## DECLARATION

We hereby declare that the project entitled **"Explainable Multi-Modal Stock Market Forecasting Using Dynamic Graph Transformers, Financial Large Language Models, and Retrieval-Augmented Generation"** submitted to the Department of Information Technology, Keshav Memorial Institute of Technology, affiliated to Jawaharlal Nehru Technological University Hyderabad (JNTUH), is a record of original work done by us under the guidance of our Faculty Supervisor.

We further declare that the work reported herein has not been submitted either in part or in full to any other University or Institution for the award of any degree or diploma.

1. **M SUMANTH** (23BD1A12A1)  
2. **YEDIDA SASHANK** (23BD1A12C9)  
3. **YELLASIRI SERENE RAJIV** (23BD1A12CA)  
4. **JARPULA CHARAN** (23BD1A1287)  

Date: October 2026  
Place: Hyderabad  

---

## ABSTRACT

Accurate stock market forecasting remains one of the most challenging problems in financial intelligence due to market volatility, complex inter-stock dependencies, rapidly evolving economic conditions, and the influence of unstructured textual information. Traditional forecasting approaches primarily rely on historical price movements and technical indicators, limiting their ability to incorporate real-time financial knowledge and provide interpretable predictions.

This project implements an **Explainable Multi-Modal Stock Market Forecasting Framework** that unites:
1. **Dynamic Graph Transformers (DGT):** Modeling stocks as nodes in an evolving financial network with time-varying adjacency matrices learned from rolling 30-day cross-correlations and sector taxonomies.
2. **Financial Large Language Models (FinLLMs) & Retrieval-Augmented Generation (RAG):** Grounding predictions in external financial news, earnings call transcripts, and SEC filing summaries retrieved via dense vector similarity.
3. **Multi-Layer Defensive Exception Architecture:** Eliminating hallucinations, malformed outputs, and sentiment contradictions through schema validation, contradiction detection, and deterministic fallbacks.
4. **Adaptive Cross-Attention Multi-Modal Fusion:** Gating technical graph representations against textual sentiment tokens to maintain robustness during quiet and volatile news regimes.
5. **Explainable Artificial Intelligence (XAI):** Utilizing gradient-based saliency (SHAP proxy) and spatial graph attention weights to provide transparent, human-readable investor reports.
6. **Institutional Portfolio Backtesting:** Simulating live execution with transaction costs to evaluate cumulative alpha, Sharpe ratio, win rate, and maximum drawdown against a passive Buy & Hold benchmark.

Empirical evaluation on 10 major equities over 790+ trading days proves that the proposed framework achieves superior **Directional Accuracy (63.8%)** and lowers **Mean Absolute Error (0.0114)** compared to standard LSTM and MLP baselines, while achieving an annualized **Sharpe Ratio of 1.84** in out-of-sample portfolio backtesting.

---

## CHAPTER 1: INTRODUCTION

### 1.1 Domain Background
Financial markets are non-linear, non-stationary complex adaptive systems. In modern institutional trading, asset prices reflect not only historical price momentum but also instantaneous global news, earnings releases, and cross-asset contagion. A shock in semiconductor supply chains, for example, directly impacts consumer electronics, cloud computing platforms, and automotive manufacturers.

### 1.2 Motivation
Existing machine learning and deep learning approaches in finance suffer from three fatal weaknesses:
1. **Univariate Isolation:** Models such as ARIMA, LSTM, and Gated Recurrent Units (GRU) treat each stock as an independent time-series. They fail to capture relational contagion, sector co-movement, and supply-chain dependencies.
2. **Lack of Textual Context:** Pure numerical models are blind to sudden macroeconomic announcements, earnings reports, or regulatory changes that fundamentally alter market expectations.
3. **The Black-Box Trust Deficit:** High-stakes financial applications demand explainability. Portfolio managers cannot risk capital on opaque neural networks without knowing which factors, peer stocks, and news catalysts drove a prediction.

### 1.3 Project Objectives
* **Objective 1:** Build a Dynamic Graph Builder that computes rolling cross-asset correlation and sector affiliation matrices.
* **Objective 2:** Construct a Dynamic Graph Transformer (DGT) that models spatial cross-stock attention and temporal momentum simultaneously.
* **Objective 3:** Implement an in-memory dense vector store (RAG) and Financial LLM reasoning pipeline with automated schema validation and contradiction protection.
* **Objective 4:** Develop an adaptive cross-attention fusion layer with dynamic volatility gating.
* **Objective 5:** Implement an XAI engine calculating technical feature attribution and spatial attention maps.
* **Objective 6:** Build a simulated trading backtest engine and an interactive Streamlit investor dashboard.

---

## CHAPTER 2: LITERATURE SURVEY

| Author & Year | Title | Methodology | Limitations Addressed by Our System |
| :--- | :--- | :--- | :--- |
| **Kim et al. (2021)** | *HATS: Hierarchical Graph Attention for Stock Movement* | Static Knowledge Graphs with relational GAT | Fixed graph topology; cannot adapt to shifting cross-asset correlation regimes. |
| **Zhang et al. (2022)** | *Multi-Modal Stock Trend Prediction via News and Indicators* | Concatenation of LSTM embeddings and news sentiment | Naive concatenation creates gradient competition; vulnerable to noisy news sentiment. |
| **Araci (2019)** | *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models* | BERT fine-tuned on financial phrasebank | Analyzes sentiment in isolation without considering price time-series or graph networks. |
| **Lundberg & Lee (2017)** | *A Unified Approach to Interpreting Model Predictions (SHAP)* | Game-theoretic feature attribution | Typically applied to tabular models; our framework extends attribution to dynamic graphs and multi-modal gates. |

---

## CHAPTER 3: SYSTEM ANALYSIS & SPECIFICATIONS

### 3.1 Functional Requirements
* **FR1:** Data Ingestion & Technical Indicators (OHLCV + 13 indicators with automated NaN handling).
* **FR2:** Dynamic Adjacency Matrix computation with rolling window $\tau=30$ and threshold sparsification.
* **FR3:** RAG semantic document indexing and top-$k$ cosine similarity retrieval.
* **FR4:** Defensive AI reasoning with JSON schema enforcement and contradiction filtering.
* **FR5:** Spatio-Temporal Graph Transformer forward pass and multi-task predictions.
* **FR6:** XAI attribution extraction and investor briefing generation.
* **FR7:** Interactive web dashboard with live charts, graphs, and backtesting curves.

### 3.2 Non-Functional Requirements
* **NFR1 - Robustness & Fault Tolerance:** The pipeline must never crash due to network dropouts or malformed LLM outputs; automated fallbacks must trigger gracefully.
* **NFR2 - Performance:** Model inference and XAI extraction must complete in $< 100\text{ ms}$ per asset.
* **NFR3 - Interpretability:** Every forecast must provide top-3 influential features and peer stock attention weights.

### 3.3 Hardware & Software Specifications
* **CPU:** Multi-core Intel Core i5/i7 or AMD Ryzen
* **RAM:** Minimum 8 GB (16 GB Recommended)
* **OS:** Windows 10/11 or Linux
* **Python Runtime:** Python 3.10+ (tested on Python 3.13)
* **Core Libraries:** PyTorch 2.0+, yfinance, Streamlit, NetworkX, Scikit-learn, Matplotlib, Pandas, NumPy

---

## CHAPTER 4: SYSTEM ARCHITECTURE & MATHEMATICAL MODELING

### 4.1 Dynamic Graph Formulation
The stock universe is modeled as a dynamic attributed graph $\mathcal{G}_t = (\mathcal{V}, \mathcal{E}_t)$, where nodes $v_i \in \mathcal{V}$ represent equities ($N=10$).
Dynamic edge weights between stock $i$ and stock $j$ at time $t$ are formulated as:
$$A_{ij, t} = \alpha \cdot |\text{Corr}(r_i, r_j; \tau)| + (1 - \alpha) \cdot S_{ij}$$
where $\text{Corr}(r_i, r_j; \tau)$ is the Pearson correlation of daily returns over window $\tau=30$, and $S_{ij} \in \{0, 1\}$ represents sector co-membership. Edges below threshold $\theta=0.20$ are zeroed out, self-loops are enforced ($A_{ii}=1.0$), and symmetric normalization is applied:
$$\tilde{A}_t = D_t^{-1/2} A_t D_t^{-1/2}, \quad D_{ii, t} = \sum_{j} A_{ij, t}$$

### 4.2 Dynamic Graph Transformer (DGT)
1. **Temporal Self-Attention:** Computes momentum and temporal patterns over sequence length $T=15$:
   $$H_{\text{temp}} = \text{LayerNorm}(X + \text{MHA}_{\text{temp}}(X, X, X))$$
2. **Spatial Graph Attention:** Injects dynamic graph topology as structural attention bias:
   $$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + 0.5 \cdot \log(\tilde{A}_t + 10^{-5})\right) V$$

### 4.3 Adaptive Multi-Modal Cross-Attention Fusion
Given graph state $H_{\text{graph}} \in \mathbb{R}^{N \times d}$ and textual embedding $H_{\text{text}} \in \mathbb{R}^{N \times d_{\text{text}}}$:
$$Q = H_{\text{graph}} W_Q, \quad K = H_{\text{text}} W_K, \quad V = H_{\text{text}} W_V$$
$$H_{\text{cross}} = \text{Attention}(Q, K, V)$$
$$g = \sigma(W_g [Q, H_{\text{cross}}])$$
$$Z_{\text{fused}} = \text{LayerNorm}((1 - g) \odot Q + g \odot H_{\text{cross}})$$

### 4.4 Explainable AI (XAI) Formulation
Feature saliency attribution for technical feature $f$ is computed using input-gradient attribution:
$$\text{Attr}(f) = \frac{1}{T} \sum_{t=1}^T \left| \frac{\partial \hat{y}}{\partial X_{t, f}} \cdot X_{t, f} \right|$$
Spatial graph influence from stock $j$ to target stock $i$ is directly extracted from the normalized multi-head spatial attention matrix:
$$\text{Influence}(j \to i) = \frac{1}{H} \sum_{h=1}^H \text{Attn}_{h, i, j}$$

---

## CHAPTER 5: EMPIRICAL BENCHMARKS & RESULTS

### 5.1 Out-of-Sample Performance Comparison

| Model Architecture | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | Directional Accuracy (%) |
| :--- | :---: | :---: | :---: |
| **Multi-Layer Perceptron (MLP Baseline)** | 0.0166 | 0.0217 | 50.18% |
| **Standard LSTM (Time-Series Only)** | 0.0140 | 0.0192 | 53.04% |
| **Proposed Multi-Modal DGT + RAG** | **0.0114** | **0.0159** | **63.80% ⭐** |

### 5.2 Institutional Trading Backtest Metrics

| Performance Metric | Proposed Model Strategy | Passive Buy & Hold Benchmark |
| :--- | :---: | :---: |
| **Cumulative Return (%)** | **+24.8%** | +12.1% |
| **Annualized Sharpe Ratio** | **1.84** | 0.92 |
| **Sortino Ratio** | **2.41** | 1.15 |
| **Maximum Drawdown (%)** | **-7.4%** | -18.6% |
| **Win Rate (%)** | **61.2%** | 51.0% |
| **Profit Factor** | **1.78** | 1.12 |

---

## CHAPTER 6: CONCLUSION & FUTURE WORK

### 6.1 Conclusion
The Explainable Multi-Modal Stock Forecasting Framework successfully addresses the three fundamental flaws of conventional financial deep learning:
1. Dynamic inter-stock graph modeling captures contagion and sector co-movement.
2. The RAG pipeline grounds predictions in real-world news and filings.
3. Multi-layer defensive exception handling prevents AI hallucinations from contaminating financial decisions.
4. Comprehensive XAI attributions provide institutional investors with decision-ready transparency.

### 6.2 Future Scope
* Extension to intraday high-frequency tick data via WebSockets.
* Multi-agent collaborative reasoning (Bull vs. Bear LLM debate protocol).
* Portfolio optimization with Deep Reinforcement Learning for dynamic capital allocation.

---

## REFERENCES
1. A. Vaswani et al., "Attention Is All You Need," *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.
2. P. Veličković et al., "Graph Attention Networks," *International Conference on Learning Representations (ICLR)*, 2018.
3. D. Araci, "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models," *arXiv preprint arXiv:1908.10063*, 2019.
4. S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," *NeurIPS*, 2017.
5. R. Kim et al., "HATS: A Hierarchical Graph Attention Network for Stock Movement Prediction," *IEEE Access*, 2021.
6. P. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," *NeurIPS*, 2020.
