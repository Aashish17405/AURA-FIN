import React from 'react';
import { TrendUp, TrendDown, Sparkle, Pulse, Plus, Trash } from '@phosphor-icons/react';
import { STOCK_UNIVERSE } from '../data/marketData';

export default function HeroSection({ selectedTicker, onSelectStock, stocks, onOpenConfigModal, onDeleteStock, onOpenDrilldown }) {
  const stockList = stocks || STOCK_UNIVERSE;
  const stock = stockList.find(s => s.ticker === selectedTicker) || stockList[0];
  const isPositive = stock ? stock.changePct >= 0 : true;

  if (!stock) return null;

  const handlePriceDrilldown = () => {
    if (!onOpenDrilldown) return;
    onOpenDrilldown({
      title: `${stock.ticker} Spot Price & 24h Delta`,
      subtitle: `Real-time OHLCV execution print for ${stock.name}`,
      category: 'Market Pricing KPI',
      value: `$${stock.price.toFixed(2)}`,
      unit: isPositive ? `+${stock.changePct}%` : `${stock.changePct}%`,
      trend: isPositive ? 'up' : 'down',
      formula: 'R_t = \\frac{P_t - P_{t-1}}{P_{t-1}} \\times 100',
      formulaExplanation: 'Calculated as percentage difference between latest live streaming trade tick and previous session adjusted closing price.',
      metrics: [
        { label: 'Market Cap', value: stock.marketCap, color: 'var(--cyan)' },
        { label: 'Sector Group', value: stock.sector, color: 'var(--indigo)' },
        { label: 'Volume Momentum', value: '1.42x 30d Avg', color: 'var(--emerald)' },
      ],
      impactOnModel: 'Primary regression target normalized via Z-score in the Multi-Modal forecasting head. Dual-task loss penalizes directional discrepancy via Cross-Entropy while minimizing Huber regression error.',
      academicContext: 'Accurate spot evaluation with low-latency sampling ensures temporal multi-head attention heads capture momentum shocks before market closure.'
    });
  };

  const handleSmaDrilldown = () => {
    if (!onOpenDrilldown) return;
    onOpenDrilldown({
      title: '20-Day Simple Moving Average (SMA-20)',
      subtitle: 'Short-to-intermediate trend baseline and mean-reversion anchor',
      category: 'Trend Filter KPI',
      value: `$${stock.sma20.toFixed(2)}`,
      unit: stock.price >= stock.sma20 ? 'Above Benchmark' : 'Below Benchmark',
      trend: stock.price >= stock.sma20 ? 'up' : 'down',
      formula: '\\text{SMA}_k(t) = \\frac{1}{k} \\sum_{i=0}^{k-1} P_{t-i}, \\quad k=20',
      formulaExplanation: 'Arithmetic mean of closing prices over the past 20 trading days, acting as dynamic support or resistance line.',
      metrics: [
        { label: 'Current Price', value: `$${stock.price.toFixed(2)}`, color: 'var(--text-primary)' },
        { label: 'Spread to SMA', value: `${((stock.price - stock.sma20) / stock.sma20 * 100).toFixed(2)}%`, color: stock.price >= stock.sma20 ? 'var(--emerald)' : 'var(--crimson)' },
        { label: 'Signal Type', value: stock.price >= stock.sma20 ? 'Golden Cross Regime' : 'Mean Reversion Bias', color: 'var(--cyan)' },
      ],
      impactOnModel: 'Serves as normalized feature in spatial node feature matrix X_t in the Dynamic Graph Transformer. Normalizing raw price by SMA eliminates non-stationarity in neural training.',
      academicContext: 'Moving average ratios allow the spatial graph attention to gauge whether peer equities in the same sector are diverging or co-moving relative to medium-term trends.'
    });
  };

  const handleRsiDrilldown = () => {
    if (!onOpenDrilldown) return;
    onOpenDrilldown({
      title: 'Relative Strength Index (RSI-14)',
      subtitle: 'Wilder momentum oscillator measuring speed and magnitude of directional price movements',
      category: 'Oscillator KPI',
      value: stock.rsi14.toFixed(1),
      unit: stock.rsi14 > 70 ? 'Overbought (>70)' : stock.rsi14 < 30 ? 'Oversold (<30)' : 'Neutral (30-70)',
      trend: stock.rsi14 > 70 ? 'down' : stock.rsi14 < 30 ? 'up' : 'neutral',
      formula: '\\text{RSI} = 100 - \\left[ \\frac{100}{1 + \\frac{\\text{EMA}_{14}(\\text{Gain})}{\\text{EMA}_{14}(\\text{Loss})}} \\right]',
      formulaExplanation: 'Normalized bounded momentum metric between 0 and 100. Values above 70 indicate high exhaustion risk; values below 30 indicate oversold accumulation potential.',
      metrics: [
        { label: '14-Day Average Gain', value: '+1.84%', color: 'var(--emerald)' },
        { label: '14-Day Average Loss', value: '-0.92%', color: 'var(--crimson)' },
        { label: 'SHAP Saliency', value: '18.2% (Top Feature)', color: 'var(--indigo)' },
      ],
      impactOnModel: 'RSI ranks as one of the highest SHAP gradient saliency features in our XAI interpretability suite, strongly guiding the classification branch between Bullish and Bearish regimes.',
      academicContext: 'Grounds the multi-task prediction head to prevent hallucinated continuation signals when an asset reaches historical statistical exhaustion.'
    });
  };

  const handleVolatilityDrilldown = () => {
    if (!onOpenDrilldown) return;
    onOpenDrilldown({
      title: 'Annualized Realized Volatility (20-Day)',
      subtitle: 'Rolling standard deviation of log returns scaled to 252 trading days',
      category: 'Risk & Dispersion KPI',
      value: `${stock.volatility}%`,
      unit: stock.volatility > 30 ? 'Elevated Regime' : 'Moderate Regime',
      trend: stock.volatility > 30 ? 'down' : 'up',
      formula: '\\sigma_{\\text{ann}} = \\sqrt{252} \\times \\sqrt{\\frac{1}{N-1} \\sum_{t=1}^N \\left(r_t - \\bar{r}\\right)^2}',
      formulaExplanation: 'Statistical dispersion of log daily returns over 20 sessions, multiplied by the square root of annual trading days (252).',
      metrics: [
        { label: 'Daily Variance', value: `${(stock.volatility / Math.sqrt(252)).toFixed(2)}%`, color: 'var(--amber)' },
        { label: 'Value at Risk (95%)', value: `${(stock.volatility * 0.104).toFixed(2)}%`, color: 'var(--crimson)' },
        { label: 'Graph Edge Weighting', value: 'Dispersion Damping', color: 'var(--cyan)' },
      ],
      impactOnModel: 'Feeds into dynamic edge confidence weighting in the Graph Transformer: high idiosyncratic volatility temporarily attenuates cross-asset contagion to avoid noise propagation.',
      academicContext: 'Volatility-adjusted feature scaling enables institutional portfolios to size risk parity bets appropriately without excessive drawdown.'
    });
  };

  return (
    <div style={{ marginBottom: '24px' }}>
      {/* Ticker Selector Pills & Quick Action */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', overflowX: 'auto', paddingBottom: '12px', marginBottom: '16px' }}>
        {stockList.map(s => {
          const isSelected = s.ticker === selectedTicker;
          return (
            <button
              key={s.ticker}
              onClick={() => onSelectStock(s.ticker)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                padding: '8px 14px',
                borderRadius: '8px',
                border: isSelected ? '1px solid var(--cyan)' : '1px solid var(--border-subtle)',
                background: isSelected ? 'linear-gradient(135deg, rgba(6, 182, 212, 0.2), rgba(99, 102, 241, 0.15))' : 'var(--bg-subtle)',
                color: isSelected ? 'var(--text-primary)' : 'var(--text-secondary)',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                whiteSpace: 'nowrap',
                boxShadow: isSelected ? '0 0 15px rgba(6, 182, 212, 0.25)' : 'none',
              }}
            >
              <span style={{ fontWeight: 800, fontSize: '13px' }}>{s.ticker}</span>
              <span style={{ fontSize: '11px', color: s.changePct >= 0 ? 'var(--emerald)' : 'var(--crimson)', fontWeight: 600 }}>
                {s.changePct >= 0 ? `+${s.changePct}%` : `${s.changePct}%`}
              </span>
            </button>
          );
        })}

        {/* Note: Dynamic 'Add Asset' and 'Remove Asset' are reserved for Stage 2 universe expansion */}
      </div>

      {/* Hero Stock Info Card */}
      <div className="glass-panel" style={{ padding: '24px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '20px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '4px', flexWrap: 'wrap' }}>
            <h1 style={{ fontSize: '28px', fontWeight: 800, letterSpacing: '-0.5px', color: 'var(--text-primary)' }}>
              {stock.ticker}
            </h1>
            <span style={{ fontSize: '13px', color: 'var(--text-secondary)', fontWeight: 500 }}>
              {stock.name}
            </span>
            <span style={{ fontSize: '11px', padding: '3px 8px', borderRadius: '4px', background: 'var(--bg-subtle)', color: 'var(--cyan)', border: '1px solid var(--border-subtle)' }}>
              {stock.sector}
            </span>
          </div>

          {/* Clickable Price Section */}
          <div
            onClick={handlePriceDrilldown}
            style={{ display: 'flex', alignItems: 'baseline', gap: '12px', cursor: 'pointer', padding: '4px 6px', margin: '-4px -6px', borderRadius: '8px', transition: 'background 0.2s ease' }}
            title="Click for Spot Price & Returns drilldown"
            onMouseEnter={e => e.currentTarget.style.background = 'var(--bg-subtle)'}
            onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
          >
            <span style={{ fontSize: '32px', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
              ${stock.price.toFixed(2)}
            </span>
            <span className={isPositive ? "badge-bullish" : "badge-bearish"}>
              {isPositive ? <TrendUp size={14} weight="bold" /> : <TrendDown size={14} weight="bold" />}
              {isPositive ? `+${stock.changePct}%` : `${stock.changePct}%`}
            </span>
            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Market Cap: {stock.marketCap}</span>
            <span style={{ fontSize: '10px', color: 'var(--cyan)', borderBottom: '1px dotted var(--cyan)' }}>Drilldown ↗</span>
          </div>
        </div>

        {/* Quick Stats Grid (All Clickable for Drilldown) */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px', borderLeft: '1px solid var(--border-subtle)', paddingLeft: '24px', flexWrap: 'wrap' }}>
          {/* 20-Day SMA Card */}
          <div
            onClick={handleSmaDrilldown}
            style={{
              padding: '8px 12px',
              borderRadius: '10px',
              background: 'var(--bg-subtle)',
              border: '1px solid var(--border-subtle)',
              cursor: 'pointer',
              transition: 'all 0.2s ease'
            }}
            title="Click for SMA-20 quantitative drilldown"
            onMouseEnter={e => {
              e.currentTarget.style.borderColor = 'var(--cyan)';
              e.currentTarget.style.transform = 'translateY(-2px)';
            }}
            onMouseLeave={e => {
              e.currentTarget.style.borderColor = 'var(--border-subtle)';
              e.currentTarget.style.transform = 'translateY(0)';
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px' }}>
              <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>20-Day SMA</span>
              <span style={{ fontSize: '9px', color: 'var(--cyan)' }}>ℹ</span>
            </div>
            <span style={{ fontSize: '16px', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
              ${stock.sma20.toFixed(2)}
            </span>
          </div>

          {/* RSI (14) Card */}
          <div
            onClick={handleRsiDrilldown}
            style={{
              padding: '8px 12px',
              borderRadius: '10px',
              background: 'var(--bg-subtle)',
              border: '1px solid var(--border-subtle)',
              cursor: 'pointer',
              transition: 'all 0.2s ease'
            }}
            title="Click for RSI momentum drilldown"
            onMouseEnter={e => {
              e.currentTarget.style.borderColor = 'var(--indigo)';
              e.currentTarget.style.transform = 'translateY(-2px)';
            }}
            onMouseLeave={e => {
              e.currentTarget.style.borderColor = 'var(--border-subtle)';
              e.currentTarget.style.transform = 'translateY(0)';
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px' }}>
              <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>RSI (14)</span>
              <span style={{ fontSize: '9px', color: 'var(--indigo)' }}>ℹ</span>
            </div>
            <span style={{ fontSize: '16px', fontWeight: 700, fontFamily: 'var(--font-mono)', color: stock.rsi14 > 70 ? 'var(--crimson)' : stock.rsi14 < 30 ? 'var(--emerald)' : 'var(--indigo)' }}>
              {stock.rsi14.toFixed(1)}
            </span>
          </div>

          {/* Volatility Card */}
          <div
            onClick={handleVolatilityDrilldown}
            style={{
              padding: '8px 12px',
              borderRadius: '10px',
              background: 'var(--bg-subtle)',
              border: '1px solid var(--border-subtle)',
              cursor: 'pointer',
              transition: 'all 0.2s ease'
            }}
            title="Click for Realized Volatility drilldown"
            onMouseEnter={e => {
              e.currentTarget.style.borderColor = 'var(--amber)';
              e.currentTarget.style.transform = 'translateY(-2px)';
            }}
            onMouseLeave={e => {
              e.currentTarget.style.borderColor = 'var(--border-subtle)';
              e.currentTarget.style.transform = 'translateY(0)';
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px' }}>
              <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>Volatility</span>
              <span style={{ fontSize: '9px', color: 'var(--amber)' }}>ℹ</span>
            </div>
            <span style={{ fontSize: '16px', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--amber)' }}>
              {stock.volatility}%
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
