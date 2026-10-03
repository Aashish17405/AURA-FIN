/**
 * Institutional Market Data & Model Outputs for AURA-FIN Frontend
 * Mapped to the Python Dynamic Graph Transformer and Financial RAG Pipeline.
 */

export const STOCK_UNIVERSE = [
  {
    ticker: "AAPL",
    name: "Apple Inc.",
    sector: "Technology",
    price: 182.45,
    changePct: +1.84,
    marketCap: "2.84T",
    sma20: 178.60,
    rsi14: 64.2,
    volatility: 22.4,
    forecast: {
      returnPct: +2.15,
      direction: "BULLISH",
      confidence: 76.4,
      textGatePct: 42.9,
      graphGatePct: 57.1,
    },
    topFeatures: [
      { name: "Volatility (20-day)", score: 18.4, color: "#06b6d4" },
      { name: "MACD Signal Line", score: 14.8, color: "#3b82f6" },
      { name: "RSI (14-day)", score: 13.5, color: "#8b5cf6" },
      { name: "Average True Range", score: 11.2, color: "#10b981" },
      { name: "Volume Moving Ratio", score: 9.6, color: "#f59e0b" },
    ],
    graphNeighbors: [
      { ticker: "MSFT", attention: 0.285, correlation: 0.84, sector: "Technology" },
      { ticker: "NVDA", attention: 0.242, correlation: 0.76, sector: "Semiconductors" },
      { ticker: "GOOGL", attention: 0.198, correlation: 0.72, sector: "Technology" },
      { ticker: "AMZN", attention: 0.165, correlation: 0.68, sector: "Consumer/Tech" },
    ],
    ragNews: [
      {
        id: "aapl-1",
        headline: "Apple Unveils Next-Generation AI Siri and M4 Chip Ecosystem",
        source: "Bloomberg Financial",
        similarity: 0.912,
        sentiment: "BULLISH",
        snippet: "Enterprise adoption expectations and upgraded hardware replacement cycles drove strong analyst revisions."
      },
      {
        id: "aapl-2",
        headline: "Services Revenue Accelerates to Record High Operating Margin",
        source: "SEC 10-Q Filing Summary",
        similarity: 0.865,
        sentiment: "BULLISH",
        snippet: "App Store and iCloud monetization expand gross margin trajectory to 46.2%."
      },
      {
        id: "aapl-3",
        headline: "European Regulatory Compliance Adjustments Implemented",
        source: "Reuters Markets",
        similarity: 0.742,
        sentiment: "NEUTRAL",
        snippet: "Core technology fee restructuring limits potential antitrust liabilities in EU jurisdiction."
      }
    ],
    finReasoning: {
      summary: "Multi-modal synthesis verifies bullish convergence. Technical RSI bounce from 58.4 is confirmed by high-conviction AI services monetization. Peer momentum from MSFT (+2.1%) reinforces positive spatio-temporal sector flow.",
      status: "PASSED",
      sentimentScore: +0.68,
      safetyCheck: "Zero Hallucination Contradictions Detected"
    },
    backtest: {
      strategyReturn: +28.4,
      benchmarkReturn: +14.2,
      sharpeRatio: 1.94,
      sortinoRatio: 2.58,
      maxDrawdown: -6.8,
      winRate: 64.5,
      profitFactor: 1.88,
    }
  },
  {
    ticker: "NVDA",
    name: "NVIDIA Corporation",
    sector: "Semiconductors",
    price: 884.20,
    changePct: +4.62,
    marketCap: "2.18T",
    sma20: 842.10,
    rsi14: 72.8,
    volatility: 38.6,
    forecast: {
      returnPct: +3.85,
      direction: "BULLISH",
      confidence: 84.2,
      textGatePct: 52.4,
      graphGatePct: 47.6,
    },
    topFeatures: [
      { name: "Volume Moving Ratio", score: 24.1, color: "#f59e0b" },
      { name: "MACD Histogram", score: 18.2, color: "#3b82f6" },
      { name: "RSI (14-day)", score: 15.6, color: "#8b5cf6" },
      { name: "Bollinger %B", score: 12.3, color: "#06b6d4" },
      { name: "Volatility (20-day)", score: 11.5, color: "#10b981" },
    ],
    graphNeighbors: [
      { ticker: "MSFT", attention: 0.312, correlation: 0.88, sector: "Technology" },
      { ticker: "AAPL", attention: 0.264, correlation: 0.76, sector: "Technology" },
      { ticker: "GOOGL", attention: 0.215, correlation: 0.74, sector: "Technology" },
      { ticker: "AMZN", attention: 0.145, correlation: 0.69, sector: "Consumer/Tech" },
    ],
    ragNews: [
      {
        id: "nvda-1",
        headline: "Blackwell Ultra AI Superchips Surpass Initial Demand Forecasts",
        source: "Financial Times",
        similarity: 0.954,
        sentiment: "BULLISH",
        snippet: "Hyperscaler capex projections for generative AI infrastructure confirm multi-quarter backorders."
      },
      {
        id: "nvda-2",
        headline: "Gross Margins Sustain Historic 76% Run Rate",
        source: "Earnings Conference Transcript",
        similarity: 0.902,
        sentiment: "BULLISH",
        snippet: "Hardware moat and CUDA software ecosystem maintain exceptional unit economics."
      }
    ],
    finReasoning: {
      summary: "Dominant catalyst identified in datacenter hardware spend. Dynamic graph shows NVDA serving as the primary lead node transmitting positive momentum to MSFT and AAPL.",
      status: "PASSED",
      sentimentScore: +0.92,
      safetyCheck: "Defensive Lexical Verification Confirmed"
    },
    backtest: {
      strategyReturn: +42.6,
      benchmarkReturn: +22.8,
      sharpeRatio: 2.15,
      sortinoRatio: 2.89,
      maxDrawdown: -8.4,
      winRate: 68.2,
      profitFactor: 2.12,
    }
  },
  {
    ticker: "MSFT",
    name: "Microsoft Corporation",
    sector: "Technology",
    price: 420.50,
    changePct: +1.25,
    marketCap: "3.12T",
    sma20: 412.30,
    rsi14: 58.6,
    volatility: 20.8,
    forecast: {
      returnPct: +1.72,
      direction: "BULLISH",
      confidence: 74.0,
      textGatePct: 44.5,
      graphGatePct: 55.5,
    },
    topFeatures: [
      { name: "SMA 20", score: 19.2, color: "#06b6d4" },
      { name: "RSI (14-day)", score: 16.4, color: "#8b5cf6" },
      { name: "MACD Signal", score: 14.1, color: "#3b82f6" },
      { name: "ATR 14", score: 10.8, color: "#10b981" },
      { name: "Volume Ratio", score: 9.2, color: "#f59e0b" },
    ],
    graphNeighbors: [
      { ticker: "NVDA", attention: 0.320, correlation: 0.88, sector: "Semiconductors" },
      { ticker: "AAPL", attention: 0.285, correlation: 0.84, sector: "Technology" },
      { ticker: "GOOGL", attention: 0.210, correlation: 0.79, sector: "Technology" },
    ],
    ragNews: [
      {
        id: "msft-1",
        headline: "Azure Cloud Enterprise AI Revenue Surges 31% Year-over-Year",
        source: "Wall Street Journal",
        similarity: 0.932,
        sentiment: "BULLISH",
        snippet: "Copilot commercial seats expanded across 60% of Fortune 500 corporations."
      }
    ],
    finReasoning: {
      summary: "Solid enterprise cloud cash flows. Low volatility regime favors technical trend-following gated with positive cloud momentum.",
      status: "PASSED",
      sentimentScore: +0.75,
      safetyCheck: "Verified"
    },
    backtest: {
      strategyReturn: +24.2,
      benchmarkReturn: +13.5,
      sharpeRatio: 1.82,
      sortinoRatio: 2.34,
      maxDrawdown: -5.9,
      winRate: 62.8,
      profitFactor: 1.76,
    }
  },
  {
    ticker: "GOOGL",
    name: "Alphabet Inc.",
    sector: "Technology",
    price: 154.80,
    changePct: -0.65,
    marketCap: "1.92T",
    sma20: 156.40,
    rsi14: 48.2,
    volatility: 24.5,
    forecast: {
      returnPct: -0.85,
      direction: "BEARISH",
      confidence: 63.8,
      textGatePct: 58.2,
      graphGatePct: 41.8,
    },
    topFeatures: [
      { name: "MACD Histogram", score: 21.4, color: "#3b82f6" },
      { name: "RSI (14-day)", score: 17.8, color: "#8b5cf6" },
      { name: "Bollinger %B", score: 15.2, color: "#06b6d4" },
      { name: "Volatility (20-day)", score: 12.1, color: "#10b981" },
      { name: "SMA 10", score: 9.8, color: "#f59e0b" },
    ],
    graphNeighbors: [
      { ticker: "META", attention: 0.340, correlation: 0.81, sector: "Communication/Tech" },
      { ticker: "MSFT", attention: 0.260, correlation: 0.79, sector: "Technology" },
      { ticker: "AMZN", attention: 0.220, correlation: 0.71, sector: "Consumer/Tech" },
    ],
    ragNews: [
      {
        id: "googl-1",
        headline: "DOJ Antitrust Remedies Focus on Default Search Contracts",
        source: "Bloomberg Law",
        similarity: 0.894,
        sentiment: "BEARISH",
        snippet: "Potential structural remedies may impact default browser revenue share agreements."
      }
    ],
    finReasoning: {
      summary: "Regulatory legal overhang triggers conservative gate allocation. Textual risk overrides mild technical support near SMA-50.",
      status: "REPAIRED",
      sentimentScore: -0.45,
      safetyCheck: "Contradiction Guard Activated & Repaired"
    },
    backtest: {
      strategyReturn: +18.6,
      benchmarkReturn: +9.2,
      sharpeRatio: 1.62,
      sortinoRatio: 2.05,
      maxDrawdown: -7.8,
      winRate: 59.4,
      profitFactor: 1.58,
    }
  },
  {
    ticker: "JPM",
    name: "JPMorgan Chase & Co.",
    sector: "Financials",
    price: 198.40,
    changePct: +1.10,
    marketCap: "565B",
    sma20: 194.20,
    rsi14: 61.4,
    volatility: 18.2,
    forecast: {
      returnPct: +1.35,
      direction: "BULLISH",
      confidence: 71.8,
      textGatePct: 38.6,
      graphGatePct: 61.4,
    },
    topFeatures: [
      { name: "SMA 20", score: 22.1, color: "#06b6d4" },
      { name: "Volatility (20-day)", score: 18.5, color: "#10b981" },
      { name: "MACD Signal", score: 14.8, color: "#3b82f6" },
      { name: "RSI 14", score: 12.3, color: "#8b5cf6" },
      { name: "Volume Ratio", score: 8.6, color: "#f59e0b" },
    ],
    graphNeighbors: [
      { ticker: "BAC", attention: 0.385, correlation: 0.89, sector: "Financials" },
      { ticker: "GS", attention: 0.342, correlation: 0.86, sector: "Financials" },
      { ticker: "AAPL", attention: 0.110, correlation: 0.42, sector: "Technology" },
    ],
    ragNews: [
      {
        id: "jpm-1",
        headline: "Record Net Interest Income and Resilient Consumer Balance Sheets",
        source: "CNBC Markets",
        similarity: 0.885,
        sentiment: "BULLISH",
        snippet: "Higher-for-longer rate environment drives robust institutional trading fee upside."
      }
    ],
    finReasoning: {
      summary: "Financial sector cluster shows tight co-movement with BAC and GS. Stable macroeconomic yield curve spread supports continued upward drift.",
      status: "PASSED",
      sentimentScore: +0.62,
      safetyCheck: "Verified"
    },
    backtest: {
      strategyReturn: +22.4,
      benchmarkReturn: +12.8,
      sharpeRatio: 1.88,
      sortinoRatio: 2.45,
      maxDrawdown: -5.4,
      winRate: 63.1,
      profitFactor: 1.82,
    }
  },
  {
    ticker: "TSLA",
    name: "Tesla, Inc.",
    sector: "Automotive/Tech",
    price: 215.30,
    changePct: -2.40,
    marketCap: "680B",
    sma20: 224.50,
    rsi14: 39.4,
    volatility: 46.2,
    forecast: {
      returnPct: -1.95,
      direction: "BEARISH",
      confidence: 68.5,
      textGatePct: 54.0,
      graphGatePct: 46.0,
    },
    topFeatures: [
      { name: "Volatility (20-day)", score: 26.5, color: "#10b981" },
      { name: "Bollinger %B", score: 20.4, color: "#06b6d4" },
      { name: "RSI (14-day)", score: 17.2, color: "#8b5cf6" },
      { name: "MACD Histogram", score: 14.6, color: "#3b82f6" },
      { name: "ATR 14", score: 11.2, color: "#f59e0b" },
    ],
    graphNeighbors: [
      { ticker: "AMZN", attention: 0.280, correlation: 0.62, sector: "Consumer/Tech" },
      { ticker: "NVDA", attention: 0.265, correlation: 0.58, sector: "Semiconductors" },
      { ticker: "AAPL", attention: 0.210, correlation: 0.54, sector: "Technology" },
    ],
    ragNews: [
      {
        id: "tsla-1",
        headline: "International Delivery Margins Impacted by Price Competition",
        source: "Wall Street Journal",
        similarity: 0.915,
        sentiment: "BEARISH",
        snippet: "Automotive gross margins compress as competitors expand lower-cost product tiers."
      }
    ],
    finReasoning: {
      summary: "High volatility regime coupled with negative margin news. DGT model projects continued downward mean-reversion toward 208 support.",
      status: "PASSED",
      sentimentScore: -0.58,
      safetyCheck: "Verified"
    },
    backtest: {
      strategyReturn: +26.8,
      benchmarkReturn: -4.5,
      sharpeRatio: 1.74,
      sortinoRatio: 2.21,
      maxDrawdown: -10.2,
      winRate: 60.5,
      profitFactor: 1.72,
    }
  }
];

export const BENCHMARK_COMPARISON = [
  { model: "MLP Baseline (Feedforward)", mae: "0.0166", rmse: "0.0217", mape: "48.2%", directionalAcc: "50.18%" },
  { model: "Standard LSTM (Time-Series Only)", mae: "0.0140", rmse: "0.0192", mape: "42.1%", directionalAcc: "53.04%" },
  { model: "Proposed: Dynamic Graph Transformer + FinLLM RAG", mae: "0.0114", rmse: "0.0159", mape: "28.6%", directionalAcc: "63.80% ⭐" },
];

export const TEAM_MEMBERS = [
  { roll: "23BD1A12A1", name: "M SUMANTH", role: "Data Engineering & Dynamic Graph Modeling", desc: "Built OHLCV data pipeline, 13 technical indicators, and rolling cross-correlation adjacency matrix." },
  { roll: "23BD1A12C9", name: "YEDIDA SASHANK", role: "Financial RAG & Defensive FinLLM Reasoner", desc: "Constructed dense vector store, semantic document retrieval, and 3-layer anti-hallucination guardrail." },
  { roll: "23BD1A12CA", name: "YELLASIRI SERENE RAJIV", role: "Dynamic Graph Transformer & Multi-Modal Fusion", desc: "Engineered Spatio-Temporal Graph Transformer, spatial attention structural bias, and gated cross-attention." },
  { roll: "23BD1A1287", name: "JARPULA CHARAN", role: "Explainable AI (XAI) & Interactive Terminal UI", desc: "Implemented SHAP gradient saliency, spatial attention heatmaps, backtesting engine, and React frontend." },
];
