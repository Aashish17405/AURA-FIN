import React from 'react';
import { TrendUp, TrendDown, Robot, ShieldCheck, Sparkle, Gauge, Scales } from '@phosphor-icons/react';

export default function ModelForecastCard({ stock, onOpenDrilldown }) {
  const f = stock.forecast;
  const isBullish = f.direction === 'BULLISH';

  const handleReturnDrilldown = () => {
    if (!onOpenDrilldown) return;
    onOpenDrilldown({
      title: `Forecasted Next-Day Return: ${f.returnPct > 0 ? `+${f.returnPct}%` : `${f.returnPct}%`}`,
      subtitle: `Dual-head continuous return expectation for horizon t+1`,
      category: 'Model Output KPI',
      value: f.returnPct > 0 ? `+${f.returnPct}%` : `${f.returnPct}%`,
      unit: f.direction,
      trend: isBullish ? 'up' : 'down',
      formula: '\\hat{y}_t^{\\text{return}} = \\mathbf{W}_r \\left[ \\mathbf{z}_i^{(L)} \\, \\Vert \\, \\mathbf{e}_i^{\\text{text}} \\right] + b_r',
      formulaExplanation: 'Dual regression output layer projecting fused spatio-temporal node embeddings and FinLLM text embeddings into continuous return percentage space.',
      metrics: [
        { label: 'Loss Function', value: 'Huber Loss (δ=1.0)', color: 'var(--cyan)' },
        { label: 'Out-of-Sample MAE', value: '0.0114', color: 'var(--emerald)' },
        { label: 'Target Horizon', value: 'Session t+1 Close', color: 'var(--text-primary)' },
      ],
      impactOnModel: 'Optimized concurrently with the classification head using multi-task weighting: L_total = L_Huber + λ * L_CrossEntropy (λ=0.5).',
      academicContext: 'Simultaneously predicting continuous return and discrete trend category regularizes the latent space and prevents sign-flipping prediction errors.'
    });
  };

  const handleConfidenceDrilldown = () => {
    if (!onOpenDrilldown) return;
    onOpenDrilldown({
      title: 'Directional Probability & Calibration',
      subtitle: 'Softmax probability score of positive vs negative price trajectory',
      category: 'Classification Confidence KPI',
      value: `${f.confidence}%`,
      unit: f.direction,
      trend: isBullish ? 'up' : 'down',
      formula: 'P(\\hat{d}_t = \\text{UP} \\mid \\mathcal{G}, \\mathcal{T}) = \\sigma\\left(\\mathbf{W}_c \\mathbf{h}_{\\text{fused}} + b_c\\right)',
      formulaExplanation: 'Softmax/Sigmoid probability score representing the model confidence in directional trend persistence over the next trading session.',
      metrics: [
        { label: 'Directional Accuracy', value: '64.8%', color: 'var(--emerald)' },
        { label: 'Outperformed Baseline', value: '+10.7% vs LSTM', color: 'var(--cyan)' },
        { label: 'Entropy Temperature', value: 'T = 1.0 (Calibrated)', color: 'var(--indigo)' },
      ],
      impactOnModel: 'High confidence (>70%) triggers larger active risk budget allocation in the backtesting simulator, increasing portfolio alpha while preserving capital on low-confidence regimes.',
      academicContext: 'Evaluated under Matthew Correlation Coefficient (MCC=0.312) and Brier calibration score to ensure probability estimations reflect empirical frequencies.'
    });
  };

  const handleGateDrilldown = () => {
    if (!onOpenDrilldown) return;
    onOpenDrilldown({
      title: 'Multi-Modal Cross-Attention Dynamic Gating (g)',
      subtitle: 'Adaptive balance between quantitative graph dynamics and qualitative FinLLM RAG catalysts',
      category: 'Neural Architecture KPI',
      value: `${f.graphGatePct}% / ${f.textGatePct}%`,
      unit: 'Graph vs Text Weight',
      trend: 'neutral',
      formula: 'g = \\sigma\\left(\\mathbf{W}_g [\\mathbf{h}_{\\text{graph}} \\, \\Vert \\, \\mathbf{h}_{\\text{text}}] + b_g\\right), \\quad \\mathbf{h}_{\\text{fused}} = g \\cdot \\mathbf{h}_{\\text{graph}} + (1-g) \\cdot \\mathbf{h}_{\\text{text}}',
      formulaExplanation: 'Dynamic sigmoid gating mechanism that automatically prioritizes text upon major earnings/news events or structural market graph dynamics during routine trading.',
      metrics: [
        { label: 'Graph Dynamic Weight', value: `${f.graphGatePct}%`, color: 'var(--indigo)' },
        { label: 'RAG Text Weight', value: `${f.textGatePct}%`, color: 'var(--cyan)' },
        { label: 'Contradiction Safety', value: 'Zero-Penalty Active', color: 'var(--emerald)' },
      ],
      impactOnModel: 'When a breaking news catalyst is injected via the RAG panel, the gating vector shifts dynamically toward text, re-weighting predictions based on sentiment polarity.',
      academicContext: 'Solves the multi-modal asynchronous fusion dilemma: structured market time-series update at 1-day intervals while textual events occur intermittently with high informational density.'
    });
  };

  return (
    <div className="glass-panel" style={{ padding: '20px', position: 'relative', overflow: 'hidden' }}>
      {/* Background Accent Glow */}
      <div style={{
        position: 'absolute',
        top: '-40px',
        right: '-40px',
        width: '160px',
        height: '160px',
        borderRadius: '50%',
        background: isBullish ? 'radial-gradient(circle, rgba(16, 185, 129, 0.2), transparent 70%)' : 'radial-gradient(circle, rgba(239, 68, 68, 0.2), transparent 70%)',
        pointerEvents: 'none'
      }}></div>

      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ width: '28px', height: '28px', borderRadius: '6px', background: 'rgba(6, 182, 212, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Robot size={16} weight="duotone" color="var(--cyan)" />
          </div>
          <div>
            <h3 style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text-primary)' }}>
              Multi-Modal Forecast Head
            </h3>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              Dynamic Graph Transformer + FinLLM Cross-Attention
            </span>
          </div>
        </div>

        <span className={isBullish ? "badge-bullish" : "badge-bearish"}>
          {isBullish ? <TrendUp size={14} weight="bold" /> : <TrendDown size={14} weight="bold" />}
          {f.direction}
        </span>
      </div>

      {/* Primary Forecast Metrics Grid (Clickable) */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '14px', marginBottom: '18px' }}>
        <div
          onClick={handleReturnDrilldown}
          style={{
            background: 'var(--bg-subtle)',
            padding: '14px',
            borderRadius: '10px',
            border: '1px solid var(--border-subtle)',
            cursor: 'pointer',
            transition: 'all 0.2s ease'
          }}
          title="Click for Return Forecast drilldown"
          onMouseEnter={e => {
            e.currentTarget.style.borderColor = 'var(--cyan)';
            e.currentTarget.style.transform = 'translateY(-2px)';
          }}
          onMouseLeave={e => {
            e.currentTarget.style.borderColor = 'var(--border-subtle)';
            e.currentTarget.style.transform = 'translateY(0)';
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)', display: 'block' }}>
              Forecasted 1-Day Return
            </span>
            <span style={{ fontSize: '9px', color: 'var(--cyan)' }}>ℹ</span>
          </div>
          <div style={{ fontSize: '24px', fontWeight: 800, fontFamily: 'var(--font-mono)', color: isBullish ? 'var(--emerald)' : 'var(--crimson)', display: 'flex', alignItems: 'center', gap: '6px' }}>
            {f.returnPct > 0 ? `+${f.returnPct}%` : `${f.returnPct}%`}
            {isBullish ? <TrendUp size={22} weight="bold" /> : <TrendDown size={22} weight="bold" />}
          </div>
          <span style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>Horizon: t+1 close</span>
        </div>

        <div
          onClick={handleConfidenceDrilldown}
          style={{
            background: 'var(--bg-subtle)',
            padding: '14px',
            borderRadius: '10px',
            border: '1px solid var(--border-subtle)',
            cursor: 'pointer',
            transition: 'all 0.2s ease'
          }}
          title="Click for Directional Probability drilldown"
          onMouseEnter={e => {
            e.currentTarget.style.borderColor = 'var(--cyan)';
            e.currentTarget.style.transform = 'translateY(-2px)';
          }}
          onMouseLeave={e => {
            e.currentTarget.style.borderColor = 'var(--border-subtle)';
            e.currentTarget.style.transform = 'translateY(0)';
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)', display: 'block' }}>
              Directional Probability
            </span>
            <span style={{ fontSize: '9px', color: 'var(--cyan)' }}>ℹ</span>
          </div>
          <div style={{ fontSize: '24px', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--cyan)' }}>
            {f.confidence}%
          </div>
          <div style={{ width: '100%', height: '5px', background: 'var(--border-subtle)', borderRadius: '3px', marginTop: '6px', overflow: 'hidden' }}>
            <div style={{ width: `${f.confidence}%`, height: '100%', background: 'linear-gradient(90deg, var(--cyan), var(--emerald))', borderRadius: '3px' }}></div>
          </div>
        </div>
      </div>

      {/* Multi-Modal Attention Gate Breakdown (Clickable) */}
      <div
        onClick={handleGateDrilldown}
        style={{
          background: 'var(--bg-subtle)',
          padding: '14px',
          borderRadius: '10px',
          border: '1px solid var(--border-subtle)',
          marginBottom: '16px',
          cursor: 'pointer',
          transition: 'all 0.2s ease'
        }}
        title="Click for Cross-Attention Gating drilldown"
        onMouseEnter={e => {
          e.currentTarget.style.borderColor = 'var(--cyan)';
          e.currentTarget.style.transform = 'translateY(-2px)';
        }}
        onMouseLeave={e => {
          e.currentTarget.style.borderColor = 'var(--border-subtle)';
          e.currentTarget.style.transform = 'translateY(0)';
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px', fontSize: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '5px', color: 'var(--text-primary)', fontWeight: 600 }}>
            <Scales size={15} color="var(--cyan)" />
            <span>Cross-Attention Gating Weight:</span>
          </div>
          <span style={{ fontSize: '11px', color: 'var(--cyan)', fontFamily: 'var(--font-mono)' }}>Dynamic Gate (g) ↗</span>
        </div>

        {/* Dual Progress Bar */}
        <div style={{ width: '100%', height: '8px', background: 'var(--bg-abyss)', borderRadius: '4px', overflow: 'hidden', display: 'flex', marginBottom: '8px', border: '1px solid var(--border-subtle)' }}>
          <div style={{ width: `${f.graphGatePct}%`, background: 'var(--indigo)', transition: 'width 0.5s ease' }}></div>
          <div style={{ width: `${f.textGatePct}%`, background: 'var(--cyan)', transition: 'width 0.5s ease' }}></div>
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px' }}>
          <span style={{ color: 'var(--indigo)', display: 'flex', alignItems: 'center', gap: '4px', fontWeight: 600 }}>
            <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: 'var(--indigo)' }}></span>
            Graph Dynamics: <strong>{f.graphGatePct}%</strong>
          </span>
          <span style={{ color: 'var(--cyan)', display: 'flex', alignItems: 'center', gap: '4px', fontWeight: 600 }}>
            <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: 'var(--cyan)' }}></span>
            News & SEC (RAG): <strong>{f.textGatePct}%</strong>
          </span>
        </div>
      </div>

      {/* Benchmark Badge */}
      <div
        onClick={handleConfidenceDrilldown}
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          fontSize: '11px',
          color: 'var(--text-secondary)',
          borderTop: '1px solid var(--border-subtle)',
          paddingTop: '12px',
          cursor: 'pointer'
        }}
        title="Click for Benchmark Comparison drilldown"
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <ShieldCheck size={16} color="var(--emerald)" weight="fill" />
          <span>Outperforms LSTM Baseline by <strong>+10.7% DA</strong></span>
        </div>
        <span style={{ color: 'var(--cyan)', fontWeight: 600 }}>MAE: 0.0114</span>
      </div>
    </div>
  );
}
