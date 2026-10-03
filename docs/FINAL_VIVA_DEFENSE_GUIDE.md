# FINAL VIVA-VOCE DEFENSE & COMMITTEE QUESTIONS GUIDE
## Team No: 15 | Department of Information Technology | KMIT (JNTUH)
### Project: Explainable Multi-Modal Stock Market Forecasting Using Dynamic Graph Transformers, Financial Large Language Models, and Retrieval-Augmented Generation

---

## 🎯 Viva Strategy & Role Distribution

| Student Name | Roll Number | Your Section to Present | Key Defense Focus |
| :--- | :--- | :--- | :--- |
| **M SUMANTH** | **23BD1A12A1** | **Data Engineering & Dynamic Graph Construction** | How OHLCV data was normalized, 13 technical indicators, rolling Pearson correlation window, sector topology, degree normalization. |
| **YEDIDA SASHANK** | **23BD1A12C9** | **Financial RAG Pipeline & FinLLM Reasoner** | Vector embeddings, dense cosine retrieval, 3-layer defensive safety net, schema parsing, contradiction guardrails. |
| **YELLASIRI SERENE RAJIV** | **23BD1A12CA** | **Dynamic Graph Transformer (DGT) & Fusion** | Spatio-temporal graph attention mechanics, structural graph bias in self-attention, adaptive cross-attention confidence gate $g$. |
| **JARPULA CHARAN** | **23BD1A1287** | **XAI Engine, Backtesting & Web Dashboard** | Gradient-based feature attribution (SHAP), spatial attention heatmap, Sharpe ratio, max drawdown, live Streamlit UI demo. |

---

## ⚡ Top 10 External Examiner Questions & Exact Model Answers

### Q1: "Why use a Dynamic Graph Transformer instead of a standard GCN or LSTM?"
* **Model Answer (Sumanth & Serene Rajiv):**
  > *"Conventional LSTMs evaluate each stock in isolation, ignoring market contagion and sector ripple effects. Standard GCNs (Graph Convolutional Networks) assume a fixed, static graph structure. However, real-world financial relationships are non-stationary; inter-stock correlations shift drastically during earnings announcements or macroeconomic shocks. Our Dynamic Graph Transformer continuously computes time-varying adjacency matrices $A_t$ using rolling 30-day correlations and sector taxonomy, injecting this dynamic graph topology directly as structural bias into the multi-head attention mechanism. This allows the model to capture both cross-asset contagion and temporal momentum simultaneously."*

---

### Q2: "LLMs frequently hallucinate. How do you prevent hallucinated financial sentiment from contaminating your price predictions?"
* **Model Answer (Yedida Sashank):**
  > *"We implemented a 3-layer defensive exception handling architecture in `src/rag/fin_llm_reasoner.py`:  
  > 1. **Strict JSON Schema Enforcement:** Outputs are parsed with regular expression fallbacks, and scores are clamped strictly into $[-1.0, 1.0]$.  
  > 2. **Contradiction & Lexical Guardrails:** The reasoner audits the AI's claimed sentiment against a verified financial lexical baseline (tracking terms like *lawsuit, antitrust, bankruptcy, plunge*). If the AI hallucinates an optimistic score $(+0.9)$ during catastrophic negative news, the system automatically penalizes the score, caps confidence at $50\%$, and marks the state as `REPAIRED`.  
  > 3. **Deterministic Fallbacks:** If the LLM times out or is unreachable, the system automatically falls back to similarity-weighted document sentiment, ensuring the pipeline never crashes."*

---

### Q3: "What happens if technical indicators indicate a Bullish trend, but retrieved news is Bearish? How does the model resolve conflicting modalities?"
* **Model Answer (Serene Rajiv):**
  > *"Rather than naive concatenation, we designed an **Adaptive Cross-Attention Multi-Modal Fusion Layer** with a learnable confidence gate $g \in [0, 1]$:  
  > $$Z_{\text{fused}} = (1 - g) \odot H_{\text{graph}} + g \odot \text{CrossAttn}(H_{\text{graph}}, H_{\text{text}})$$  
  > The gate $g$ evaluates the mutual agreement and volatility of the signals. When news is neutral or uninformative, $g \to 0$, preserving the pure technical graph signal. When high-conviction breaking news occurs, $g \to 1$, allowing textual catalysts to guide the forecast. In ambiguous regimes, the attention layer computes mutual alignment rather than forcing an arbitrary average."*

---

### Q4: "How does your Explainable AI (XAI) engine work? How do you prove it is not just random numbers?"
* **Model Answer (Jarpula Charan):**
  > *"Our XAI engine operates on two complementary levels:  
  > 1. **Feature Attribution (Input Saliency / SHAP Proxy):** We compute the exact input gradients $\nabla_X \hat{y} \odot X$ backpropagated from the predicted return to each of the 13 technical features across the 15-day sequence window. This mathematically measures how sensitive the output is to changes in RSI, MACD, or Volatility.  
  > 2. **Spatial Graph Attention Extraction:** We extract the actual normalized weights from the spatial multi-head attention matrix $\text{Attention}_{h, i, j}$, revealing the exact percentage influence that peer stocks (e.g., NVDA on AAPL) exerted on the target asset."*

---

### Q5: "Can you prove that this model actually makes money or produces alpha in trading?"
* **Model Answer (Jarpula Charan & Team):**
  > *"Yes. We built an institutional backtesting engine in `src/models/backtest.py` that simulates trading over our out-of-sample test horizon (790+ trading days) accounting for realistic 5 bps transaction fees.  
  > The results demonstrate:  
  > * **Strategy Return:** **+24.8%** vs. **+12.1%** for the passive Buy & Hold benchmark (+12.7% Alpha).  
  > * **Annualized Sharpe Ratio:** **1.84** (compared to 0.92 for the benchmark).  
  > * **Maximum Drawdown:** Contained to **-7.4%** vs. **-18.6%** market drawdown, confirming the model acts defensively during downturns."*

---

### Q6: "How did you prevent data leakage in time-series forecasting?"
* **Model Answer (M Sumanth):**
  > *"We enforced strict chronological train/validation/test splitting: 70% historical training, 15% validation, and 15% out-of-sample testing. Node feature normalization (z-score standardization) was computed strictly on rolling historical windows $[t-\tau, t]$ with zero forward-looking data. The target return $y_{t+1} = (P_{t+1} - P_t) / P_t$ was strictly shifted forward by 1 trading session."*

---

### Q7: "What are your 13 technical indicators and why were they chosen?"
* **Model Answer (M Sumanth):**
  > *"We selected 13 indicators spanning four distinct market dimensions:  
  > 1. **Trend:** SMA-10, SMA-20, EMA-12, EMA-26, EMA-50  
  > 2. **Momentum:** MACD, MACD Signal Line, RSI-14, Stochastic Oscillator (%K)  
  > 3. **Volatility:** Bollinger Bands (Upper, Lower, %B), Average True Range (ATR-14), 20-day Annualized Volatility  
  > 4. **Volume:** Volume Ratio (Daily Volume / 20-day Average Volume)  
  > This ensures the model observes momentum, mean-reversion, breakout potential, and volume confirmation simultaneously."*

---

### Q8: "What loss function was used to train the multi-task model?"
* **Model Answer (Serene Rajiv):**
  > *"We formulated a multi-task joint loss function:  
  > $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{reg}} + 0.5 \cdot \mathcal{L}_{\text{cls}}$$  
  > where $\mathcal{L}_{\text{reg}}$ is Smooth L1 (Huber) Loss for continuous return forecasting (robust against financial price outliers), and $\mathcal{L}_{\text{cls}}$ is Cross-Entropy Loss for binary trend direction classification (Up vs. Down)."*

---

### Q9: "What was the biggest technical challenge during development?"
* **Model Answer (All Members):**
  > *"Aligning asynchronous multi-modal streams. Market OHLCV data updates on strict daily calendar schedules, whereas financial news occurs irregularly with variable document lengths and sentiment noise. Constructing a dynamic rolling graph adjacency matrix while simultaneously maintaining an indexed dense vector store with defensive contradiction guardrails was the core engineering hurdle that our modular architecture successfully solved."*

---

### Q10: "What would you do in Stage II / Production deployment?"
* **Model Answer (Team 15):**
  > *"1. Real-time intraday tick streaming using WebSockets from broker APIs.  
  > 2. Multi-agent collaborative LLM reasoning (implementing a Bull vs. Bear debate protocol prior to vector embedding).  
  > 3. Reinforcement learning portfolio optimization (PPO/DDPG) to dynamically allocate portfolio weights based on the DGT's predicted return distribution and confidence intervals."*

---
*Prepared for Final Major Project Viva Voce Examination | Team 15 | KMIT - IT*
