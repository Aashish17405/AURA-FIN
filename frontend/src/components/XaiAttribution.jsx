import React from 'react';
import { ShieldCheck, ChartBar, ShareNetwork, FileText, Sparkle } from '@phosphor-icons/react';

export default function XaiAttribution({ stock }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '20px' }}>
      {/* 1. Feature Attribution (SHAP / Saliency) */}
      <div className="glass-panel" style={{ padding: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '14px' }}>
          <div style={{ width: '28px', height: '28px', borderRadius: '6px', background: 'rgba(59, 130, 246, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <ChartBar size={16} weight="duotone" color="var(--indigo)" />
          </div>
          <div>
            <h3 style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text-primary)' }}>
              Technical Feature Importance (SHAP)
            </h3>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              Gradient Saliency Attribution per Indicator
            </span>
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {stock.topFeatures.map((feat, idx) => (
            <div key={idx}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
                <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>{feat.name}</span>
                <span style={{ color: feat.color, fontFamily: 'var(--font-mono)', fontWeight: 700 }}>{feat.score}%</span>
              </div>
              <div style={{ width: '100%', height: '6px', background: 'var(--border-subtle)', borderRadius: '3px', overflow: 'hidden' }}>
                <div style={{
                  width: `${feat.score * 3.5}%`,
                  height: '100%',
                  background: feat.color,
                  borderRadius: '3px',
                  boxShadow: `0 0 8px ${feat.color}80`,
                  transition: 'width 0.8s cubic-bezier(0.16, 1, 0.3, 1)'
                }}></div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 2. Spatial Graph Attention Influence */}
      <div className="glass-panel" style={{ padding: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '14px' }}>
          <div style={{ width: '28px', height: '28px', borderRadius: '6px', background: 'rgba(16, 185, 129, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <ShareNetwork size={16} weight="duotone" color="var(--emerald)" />
          </div>
          <div>
            <h3 style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text-primary)' }}>
              Cross-Asset Spatial Attention Weights
            </h3>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              Inter-Stock Contagion from Correlated Peers
            </span>
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {stock.graphNeighbors.map((nb, idx) => (
            <div
              key={idx}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '10px 14px',
                background: 'var(--bg-subtle)',
                borderRadius: '8px',
                border: '1px solid var(--border-subtle)'
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <span style={{ width: '32px', height: '32px', borderRadius: '6px', background: 'var(--bg-subtle)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 800, fontSize: '11px', color: 'var(--cyan)', border: '1px solid' }}>
                  {nb.ticker}
                </span>
                <div>
                  <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-primary)' }}>{nb.ticker} Momentum</div>
                  <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>Sector: {nb.sector}</div>
                </div>
              </div>

              <div style={{ textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--emerald)' }}>Attn: {nb.attention.toFixed(3)}</div>
                <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>Corr: {nb.correlation.toFixed(2)}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
