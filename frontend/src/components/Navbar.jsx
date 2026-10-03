import React from 'react';
import {
  TrendUp,
  TrendDown,
  Cpu,
  ShieldCheck,
  Sparkle,
  Pulse,
  Sun,
  Moon,
  Monitor,
  Play,
  Pause,
  SlidersHorizontal,
} from '@phosphor-icons/react';
import { STOCK_UNIVERSE } from '../data/marketData';

export default function Navbar({
  onOpenConfigModal,
  activeTab,
  setActiveTab,
  theme,
  setTheme,
  isLiveStreaming,
  setIsLiveStreaming,
  stocks
}) {
  const stockList = stocks || STOCK_UNIVERSE;

  return (
    <header style={{ position: 'sticky', top: 0, zIndex: 50, backdropFilter: 'blur(20px)', background: 'var(--bg-surface)', borderBottom: '1px solid var(--border-subtle)', opacity: 0.98 }}>
      {/* Real-time Ticker Tape Marquee */}
      <div style={{ background: 'var(--bg-subtle)', borderBottom: '1px solid var(--border-subtle)', overflow: 'hidden', padding: '6px 0', fontSize: '11px', fontFamily: 'var(--font-mono)' }}>
        <div style={{ display: 'flex', width: '200%', animation: 'marquee 30s linear infinite' }}>
          {[...stockList, ...stockList].map((stock, i) => (
            <div key={i} style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', padding: '0 20px', whiteSpace: 'nowrap' }}>
              <span style={{ fontWeight: 700, color: 'var(--text-primary)' }}>{stock.ticker}</span>
              <span style={{ color: 'var(--text-secondary)' }}>${stock.price.toFixed(2)}</span>
              <span style={{ color: stock.changePct >= 0 ? 'var(--emerald)' : 'var(--crimson)', display: 'flex', alignItems: 'center', gap: '2px', fontWeight: 600 }}>
                {stock.changePct >= 0 ? <TrendUp size={12} weight="bold" /> : <TrendDown size={12} weight="bold" />}
                {stock.changePct >= 0 ? `+${stock.changePct.toFixed(2)}%` : `${stock.changePct.toFixed(2)}%`}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Main Navbar */}
      <div style={{ maxWidth: '1440px', margin: '0 auto', padding: '12px 24px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '16px', flexWrap: 'wrap' }}>
        {/* Brand */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            width: '38px',
            height: '38px',
            borderRadius: '10px',
            background: 'linear-gradient(135deg, #06b6d4, #6366f1)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 20px rgba(6, 182, 212, 0.4)'
          }}>
            <Sparkle size={22} weight="fill" color="#ffffff" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '18px', fontWeight: 800, letterSpacing: '-0.5px', color: 'var(--text-primary)' }}>
                AURA-FIN
              </span>
              <span style={{ fontSize: '10px', padding: '2px 6px', borderRadius: '4px', background: 'rgba(6, 182, 212, 0.15)', color: 'var(--cyan)', fontWeight: 700, border: '1px solid rgba(6, 182, 212, 0.3)' }}>
                DGT + RAG
              </span>
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              Explainable Multi-Modal Stock Forecasting
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav style={{ display: 'flex', alignItems: 'center', gap: '4px', background: 'var(--bg-subtle)', padding: '4px', borderRadius: '10px', border: '1px solid var(--border-subtle)' }}>
          {[
            { id: 'overview', label: 'Terminal', icon: Pulse },
            { id: 'graph', label: 'Dynamic Graph', icon: Cpu },
            { id: 'xai', label: 'XAI Attribution', icon: ShieldCheck },
            { id: 'backtest', label: 'Simulation & Alpha', icon: TrendUp },
          ].map(tab => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  padding: '7px 14px',
                  borderRadius: '7px',
                  border: 'none',
                  fontSize: '12px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  background: isActive ? 'linear-gradient(135deg, rgba(6, 182, 212, 0.2), rgba(99, 102, 241, 0.2))' : 'transparent',
                  color: isActive ? 'var(--cyan)' : 'var(--text-secondary)',
                  boxShadow: isActive ? '0 0 12px rgba(6, 182, 212, 0.25)' : 'none',
                }}
              >
                <Icon size={16} weight={isActive ? "fill" : "regular"} />
                {tab.label}
              </button>
            );
          })}
        </nav>

        {/* Controls (Theme Selector Active; Stage 2 Streaming/Config Hidden) */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {/* Note: Live Feed streaming toggle and Dynamic Config Modal are reserved for Stage 2 */}
          {/* <button onClick={() => setIsLiveStreaming(!isLiveStreaming)}>...</button> */}

          {/* Theme Selector (Light, Dark, System) */}
          <div style={{ display: 'flex', background: 'var(--bg-subtle)', borderRadius: '8px', padding: '3px', border: '1px solid var(--border-subtle)' }}>
            {[
              { id: 'light', icon: Sun, title: 'Light Mode' },
              { id: 'dark', icon: Moon, title: 'Dark Mode' },
              { id: 'system', icon: Monitor, title: 'System Default' }
            ].map(t => {
              const Icon = t.icon;
              const isCurr = theme === t.id;
              return (
                <button
                  key={t.id}
                  onClick={() => setTheme(t.id)}
                  style={{
                    background: isCurr ? 'var(--bg-card-hover)' : 'transparent',
                    border: 'none',
                    borderRadius: '6px',
                    padding: '5px 8px',
                    cursor: 'pointer',
                    color: isCurr ? 'var(--cyan)' : 'var(--text-muted)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    transition: 'all 0.15s ease'
                  }}
                  title={t.title}
                >
                  <Icon size={14} weight={isCurr ? "fill" : "regular"} />
                </button>
              );
            })}
          </div>

          {/* Note: Dynamic Universe Configurator reserved for Stage 2 */}
          {/* <button onClick={onOpenConfigModal}>Configure</button> */}
        </div>
      </div>
    </header>
  );
}
