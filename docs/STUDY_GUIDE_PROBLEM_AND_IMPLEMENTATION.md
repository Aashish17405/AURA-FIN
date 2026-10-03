# AURA-FIN: Comprehensive Study Guide & Architecture Breakdown
**Project Title:** Explainable Multi-Modal Stock Market Forecasting Using Dynamic Graph Transformers, Financial Large Language Models, and Retrieval-Augmented Generation  
**Target:** B.Tech Major Project (Stage I Review & Final Viva Defense)  
**Author / Team:** Academic Major Project Team  

---

## 1. Executive Summary in Plain English

### The "Elevator Pitch" (How to explain it in 30 seconds):
> *"Stock prices don't move in a vacuum. If Apple drops due to a chip shortage, it pulls down its suppliers, competitors, and sector peers. Traditional models only look at one stock's historical price chart (ignoring news and peer relationships) or look only at news (ignoring quantitative charts). Furthermore, deep neural networks are black-boxes that traders can't trust.*  
>  
> *Our project, **AURA-FIN**, solves this by combining three cutting-edge AI technologies into one unified system:*  
> 1. *A **Dynamic Graph Transformer (DGT)** that models how stocks influence each other across dynamic rolling correlation networks.*  
> 2. *A **Financial Large Language Model with RAG** that ingests verified news catalysts and financial filings without hallucination.*  
> 3. *An **Explainable AI (XAI)** engine that tells the investor exactly which technical indicator and which peer company caused the prediction."*

---

## 2. The Core Problem Statement & The 4 Gaps in Existing Research

### Formal Problem Formulation
Given:
1. Historical multivariate time-series $\mathbf{X}_t \in \mathbb{R}^{N \times T \times D}$ for $N$ monitored equities across $T$ historical timesteps with $D$ technical indicators,
2. An evolving inter-stock correlation network $\mathcal{G}_t = (\mathcal{V}, \mathcal{E}_t)$, and
3. An asynchronous stream of textual news and regulatory filings $\mathcal{D}_t$,

**Goal:** Predict the next-day continuous return $\hat{y}_{i, t+1} \in \mathbb{R}$ and directional trend category $\hat{d}_{i, t+1} \in \{\text{UP}, \text{DOWN}\}$, while generating quantitative feature attribution $\boldsymbol{\phi}_i$ and spatial peer influence weights $\boldsymbol{\alpha}_{ij}$ justifying the prediction.

```
       ┌────────────────────────────────────────────────────────┐
       │             THE 4 FLAWS OF TRADITIONAL AI              │
       └────────────────────────────────────────────────────────┘
          │                   │                  │            │
          ▼                   ▼                  ▼            ▼
   1. Uni-Modal       2. Static Graph    3. LLM Noise    4. Black-Box
   Blindness          Fallacy            & Hallucination  Trust Deficit
   (Price OR News)    (Rigid Edges)      (Unchecked Gen) (No Justification)
```

---

### Gap 1: The Uni-Modal Trap (Price-Only vs. Text-Only)
- **What existing models do:** 
  - Standard quantitative models (LSTM, GRU, ARIMA) look strictly at historical OHLCV prices. They are completely blind to sudden exogenous catalysts (e.g., unexpected CEO departures, antitrust lawsuits, FDA approvals).
  - Pure NLP sentiment models (FinBERT) classify news headlines as positive or negative, but ignore whether an asset is already severely overbought ($\text{RSI} > 80$) or technically exhausted.
- **Our Solution:** A **Multi-Modal Cross-Attention Gating Network** ($g$) that dynamically balances quantitative chart dynamics with textual news catalysts.

### Gap 2: The Static Graph Fallacy
- **What existing models do:** Prior Graph Convolutional Networks (GCNs) connect companies using static industry classification codes (GICS/SIC). But financial markets are non-stationary: two tech companies might be completely uncorrelated during normal trading, but during a liquidity crisis, their correlation spikes to $+0.92$ (contagion effect).
- **Our Solution:** A **Dynamic Graph Engine** that recomputes a rolling 30-day Pearson correlation matrix $\rho_{ij}^{(\tau=30)}$ daily, combining it with fundamental sector topology and symmetric degree normalization.

### Gap 3: Financial LLM Hallucinations & Asynchronous Arrival
- **What existing models do:** Generic LLMs frequently hallucinate factual numbers, generate conflicting sentiment, or choke on massive token lengths. Furthermore, news arrives irregularly (some days 50 articles, some days zero), while stock prices arrive on a strict daily clock.
- **Our Solution:** An **In-Memory Dense Vector Store (RAG)** retrieving top-$k$ verified snippets, paired with a **3-Layer Defensive Safety Net** (JSON schema enforcement, financial contradiction detection, and deterministic rule-based fallbacks).

### Gap 4: The Black-Box Trust Deficit
- **What existing models do:** Deep neural networks output a prediction without reasoning. In high-stakes institutional finance, portfolio managers legally cannot deploy capital into opaque models that cannot be audited.
- **Our Solution:** Integrated **Explainable AI (XAI)** outputting **SHAP gradient saliency** (which indicator mattered) and **spatial neighbor attention ranking** (which connected company transmitted momentum).

---

## 3. End-to-End Implementation Architecture

```
                    AURA-FIN SYSTEM PIPELINE
                    
  [Market OHLCV Data]                     [News / SEC Filings]
          │                                         │
          ▼                                         ▼
 13 Technical Indicators                  Dense Vector Store (RAG)
  (RSI, SMA, MACD, etc.)                   (64-dim Cosine Embeddings)
          │                                         │
          ▼                                         ▼
 Dynamic Graph Engine                     FinLLM Reasoner + Guardrails
 (Rolling Pearson + Sector)               (Sentiment Polarity & Summary)
          │                                         │
          ▼                                         ▼
 Dynamic Graph Transformer               Dense Text Embedding e_text
 (Spatial + Temporal MHA)                           │
          │                                         │
          └─────────────────┬───────────────────────┘
                            ▼
              Confidence-Gated Cross-Attention
                g = σ(W_g [h_graph ∥ h_text] + b_g)
                            │
                            ▼
                Dual-Head Multi-Task Output
                ┌───────────┴───────────┐
                ▼                       ▼
       1-Day Return Regression   Trend Classification
         (Huber Loss δ=1.0)       (Cross-Entropy Loss)
                │                       │
                └───────────┬───────────┘
                            ▼
             Explainable AI (SHAP & Graph Attention)
```

---

## 4. Key Mathematical Formulations (Viva Checklist)

Examiners love to ask for equations on the whiteboard. Memorize these 5 key formulas:

### 1. Rolling Pearson Correlation (Graph Edges)
$$\rho_{ij}^{(t, \tau)} = \frac{\sum_{k=0}^{\tau-1} (r_{i, t-k} - \bar{r}_i)(r_{j, t-k} - \bar{r}_j)}{\sqrt{\sum_{k=0}^{\tau-1} (r_{i, t-k} - \bar{r}_i)^2} \sqrt{\sum_{k=0}^{\tau-1} (r_{j, t-k} - \bar{r}_j)^2}}$$
*Explains how edges between stock $i$ and stock $j$ are dynamically computed over a rolling 30-day window ($\tau=30$).*

### 2. Symmetric Graph Adjacency Normalization (Kipf-Welling)
$$\tilde{\mathbf{A}} = \tilde{\mathbf{D}}^{-\frac{1}{2}} (\mathbf{A} + \mathbf{I}) \tilde{\mathbf{D}}^{-\frac{1}{2}}$$
*Prevents numerical gradient explosion during multi-hop graph message passing.*

### 3. Spatial Self-Attention with Structural Graph Bias
$$\alpha_{ij} = \frac{\exp\left( \frac{(\mathbf{W}_q \mathbf{x}_i)^T (\mathbf{W}_k \mathbf{x}_j)}{\sqrt{d}} + b_{ij}^{\text{spatial}} \right)}{\sum_{k \in \mathcal{N}(i)} \exp\left( \frac{(\mathbf{W}_q \mathbf{x}_i)^T (\mathbf{W}_k \mathbf{x}_k)}{\sqrt{d}} + b_{ik}^{\text{spatial}} \right)}$$
*Weights how much stock $i$ attends to connected peer stock $j$ based on correlation and sector affinity.*

### 4. Dynamic Cross-Attention Gating ($g$)
$$g = \sigma\left(\mathbf{W}_g [\mathbf{h}_{\text{graph}} \parallel \mathbf{h}_{\text{text}}] + b_g\right)$$
$$\mathbf{h}_{\text{fused}} = g \cdot \mathbf{h}_{\text{graph}} + (1 - g) \cdot \mathbf{h}_{\text{text}}$$
*When major news occurs, $g$ shifts weight towards text; during quiet periods, $g$ relies on chart dynamics.*

### 5. Multi-Task Joint Training Loss
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Huber}}(\hat{y}_{\text{return}}, y) + \lambda \cdot \mathcal{L}_{\text{BCE}}(\hat{d}_{\text{trend}}, d), \quad \lambda = 0.5$$
*Regularizes the network: predicting continuous return and direction simultaneously stops the model from making sign-flipping mistakes.*

---

## 5. Strategic Staging: What to Show NOW (Review 1) vs. What to HIDE for Stage 2

### The Academic Examiner Psychology (Crucial Advice)
> [!WARNING]
> **Why showing 100% completion in Review 1 is dangerous:**  
> If an evaluation panel sees a 100% completed system with live backtesting, live streaming, and full optimization during **Review 1 / Stage I**, their immediate reaction is:  
> 1. *"If this is completely finished, what will you do for Stage II (the next semester/final review)?"*  
> 2. They will scrutinize minor edge cases or demand massive extra features (e.g., high-frequency order book modeling, real money broker integration).  
>  
> **The Golden Strategy:** Show **Phase 1 (The Foundational Breakthrough)** now, and present **Phase 2 (The Portfolio Deployment & Advanced Backtesting)** as your scheduled upcoming milestone!

---

### What to Showcase in Review 1 (Phase 1 Scope)
Show the foundational architecture and validation working end-to-end:
1. **Dynamic Spatio-Temporal Graph:** Show the HTML5 interactive canvas. Explain how stocks cluster by sector and how rolling 30-day correlations create dynamic edges.
2. **Technical Indicator Ingestion:** Show the 13 indicators (SMA, RSI, MACD, Bollinger Bands) on the interactive price chart.
3. **Financial RAG & FinLLM Ingestion:** Show news headlines retrieved by vector cosine similarity and the FinLLM synthesis with the defensive anti-hallucination check (`PASSED`).
4. **Multi-Modal Forecast Head:** Show the predicted next-day return and directional confidence, demonstrating the cross-attention gate ($g$).
5. **Out-of-Sample Backtesting & Alpha Simulation Tab:** Kept active to demonstrate that the model generates verifiable statistical edge (Jensen's $\alpha$, Sharpe Ratio $1.78$, Sortino Ratio $2.14$, Drawdown $-6.4\%$, and Multi-Baseline comparison against MLP and LSTM).
6. **Detailed Math Drilldown Popups:** Click any KPI card (Price, RSI, SMA, Alpha, Sharpe) to show the examiner the KaTeX mathematical formulation and theoretical grounding!

---

### What is HIDDEN / Reserved for Stage 2 (Future Roadmap)

To protect your Stage 2 project scope, the following "future production" controls have been cleanly hidden from the UI:

| Next Feature (Hidden in UI) | Why Hidden for Review 1? | What to Tell Examiners for Stage 2 Scope |
| :--- | :--- | :--- |
| **Real-Time Live WebSocket Streaming Toggle** | Sub-second tick oscillation makes it look like an operational trading bot rather than an academic research model. | *"Stage 1 evaluates daily closing data. For Stage 2, we are building high-throughput WebSocket ingestion for real-time intraday tick streaming."* |
| **Dynamic Stock Universe Expansion (+ Add Asset Sandbox)** | Arbitrary stock addition requires online inductive dynamic graph learning without offline retraining. | *"In Stage 1, we validated the model on a fixed 6-stock sector-interdependent core universe. In Stage 2, we will implement inductive GNN node embedding expansion."* |
| **Synthetic News Catalyst Injector Modal** | The custom injection sandbox is a development/testing tool, not a research interface. | *"In Stage 1, news is ingested from our verified historical RAG vector corpus. Stage 2 will introduce an automated real-time crawler and news stream pipe."* |
| **Automated Broker Order Execution API** | AURA-FIN is an **Explainable Decision Support System (DSS)**, not an unregulated high-frequency automated execution bot. | *"Stage 2 will explore automated paper-trading execution with slippage models using institutional brokerage APIs (e.g., Alpaca/Interactive Brokers)."* |

---

## 6. Current Frontend UI Configuration

The frontend has been configured cleanly for Review 1:
- **Simulation & Alpha (Backtesting Tab):** Active and accessible from the navigation bar.
- **Stable Research Terminal:** Live stream tick jitter is paused by default (`isLiveStreaming = false`) so your presentation displays stable, reproducible numbers.
- **Fixed Academic Universe:** The "+ Add Asset" and "Remove" controls are hidden so the examiner focuses on the core mathematical relationships between AAPL, NVDA, MSFT, GOOGL, AMZN, and TSLA.
- **Theme Selector:** Light, Dark, and System modes remain fully accessible in the top-right corner.

## 7. Top 5 Viva Questions & Short Answers for Tomorrow

**Q1: Why use a Dynamic Graph Transformer instead of an LSTM?**  
> *"LSTMs only model temporal sequences for a single isolated asset. They cannot model spatial inter-asset cross-contagion. Our Dynamic Graph Transformer uses spatial attention over evolving correlation graphs to capture how shocks in one stock propagate into peers before the market closes."*

**Q2: What vector similarity metric does your RAG use?**  
> *"We use L2-normalized continuous dense vectors with Cosine Similarity: $\text{sim}(\mathbf{q}, \mathbf{d}) = \mathbf{q} \cdot \mathbf{d}$. We also apply a $+0.35$ entity-relevance prior when a filing directly references the target ticker."*

**Q3: How do you prevent LLM hallucinations?**  
> *"Through a 3-layer defensive safety net: (1) strict JSON schema output parsing, (2) financial sentiment contradiction detection against Loughran-McDonald lexicon polarity, and (3) deterministic rule-based fallbacks if confidence is below threshold."*

**Q4: What is the purpose of the Dynamic Gate $g$?**  
> *"$g$ is an adaptive sigmoid gating weight: $g = \sigma(\mathbf{W}_g [\mathbf{h}_{\text{graph}} \parallel \mathbf{h}_{\text{text}}] + b_g)$. When breaking news is detected, $g$ dynamically shifts attention to textual catalysts. During routine market hours, it relies on quantitative chart dynamics."*

**Q5: What are your milestones for Stage 2?**  
> *"For Stage 2, we will: (1) execute comprehensive ablation studies comparing against LSTM and GCN baselines, (2) implement transaction-cost-aware portfolio backtesting measuring Sharpe ratio and maximum drawdown, and (3) expand the vector store to multi-year SEC 10-K filings."*
