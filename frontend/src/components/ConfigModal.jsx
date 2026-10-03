import React, { useState } from 'react';
import {
  X,
  Plus,
  Trash,
  SlidersHorizontal,
  Sparkle,
  CheckCircle,
  FileText,
  ShareNetwork,
  Buildings
} from '@phosphor-icons/react';

export default function ConfigModal({
  isOpen,
  onClose,
  stocks,
  onAddStock,
  onDeleteStock,
  onInjectNews,
  graphParams,
  onUpdateGraphParams
}) {
  const [activeTab, setActiveTab] = useState('stocks'); // 'stocks' | 'news' | 'graph'

  // New Stock Form State
  const [newTicker, setNewTicker] = useState('');
  const [newName, setNewName] = useState('');
  const [newSector, setNewSector] = useState('Technology');
  const [newPrice, setNewPrice] = useState('150.00');

  // New News Injection State
  const [newsTicker, setNewsTicker] = useState(stocks[0]?.ticker || 'AAPL');
  const [newsHeadline, setNewsHeadline] = useState('');
  const [newsSnippet, setNewsSnippet] = useState('');
  const [newsSentiment, setNewsSentiment] = useState('BULLISH');
  const [newsSuccess, setNewsSuccess] = useState(false);

  if (!isOpen) return null;

  const handleAddStockSubmit = (e) => {
    e.preventDefault();
    if (!newTicker.trim() || !newName.trim()) return;

    onAddStock({
      ticker: newTicker.trim().toUpperCase(),
      name: newName.trim(),
      sector: newSector,
      price: parseFloat(newPrice) || 100.0,
      changePct: +(Math.random() * 4 - 1.8).toFixed(2),
      marketCap: '450B',
      sma20: parseFloat(newPrice) * 0.98,
      rsi14: 55.4,
      volatility: 26.2,
      forecast: {
        returnPct: +(Math.random() * 3.5 - 1.2).toFixed(2),
        direction: Math.random() > 0.4 ? 'BULLISH' : 'BEARISH',
        confidence: +(65 + Math.random() * 20).toFixed(1),
        textGatePct: 45.0,
        graphGatePct: 55.0,
      },
      topFeatures: [
        { name: 'RSI (14-day)', score: 18.2, color: '#8b5cf6' },
        { name: 'MACD Signal Line', score: 16.4, color: '#3b82f6' },
        { name: 'SMA 20', score: 14.1, color: '#06b6d4' },
        { name: 'Volatility (20-day)', score: 12.5, color: '#10b981' },
        { name: 'Volume Ratio', score: 9.8, color: '#f59e0b' },
      ],
      graphNeighbors: [
        { ticker: 'AAPL', attention: 0.32, correlation: 0.74, sector: 'Technology' },
        { ticker: 'MSFT', attention: 0.28, correlation: 0.71, sector: 'Technology' },
      ],
      ragNews: [
        {
          id: `${newTicker.toLowerCase()}-custom`,
          headline: `Initial Coverage: ${newTicker.toUpperCase()} Financial Operations Expand`,
          source: 'Market Ingestion Feed',
          similarity: 0.884,
          sentiment: 'BULLISH',
          snippet: 'Stable operations and solid cash flow generation support baseline forecasts.'
        }
      ],
      finReasoning: {
        summary: `Custom ticker ${newTicker.toUpperCase()} initialized into the Dynamic Graph Transformer. Spatio-temporal edges dynamically linked.`,
        status: 'PASSED',
        sentimentScore: +0.45,
        safetyCheck: 'Automated Ingestion Verified'
      },
      backtest: {
        strategyReturn: +22.5,
        benchmarkReturn: +11.2,
        sharpeRatio: 1.78,
        sortinoRatio: 2.30,
        maxDrawdown: -6.4,
        winRate: 62.0,
        profitFactor: 1.75,
      }
    });

    setNewTicker('');
    setNewName('');
    setNewPrice('150.00');
  };

  const handleInjectNewsSubmit = (e) => {
    e.preventDefault();
    if (!newsHeadline.trim()) return;

    onInjectNews({
      ticker: newsTicker,
      headline: newsHeadline.trim(),
      snippet: newsSnippet.trim() || 'Breaking financial catalyst received via real-time vector feed.',
      source: 'Real-Time User Injection (RAG)',
      sentiment: newsSentiment,
      similarity: 0.945,
    });

    setNewsHeadline('');
    setNewsSnippet('');
    setNewsSuccess(true);
    setTimeout(() => setNewsSuccess(false), 2500);
  };

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      zIndex: 100,
      background: 'rgba(5, 8, 17, 0.85)',
      backdropFilter: 'blur(16px)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '20px'
    }}>
      <div className="glass-panel" style={{
        maxWidth: '720px',
        width: '100%',
        maxHeight: '90vh',
        overflowY: 'auto',
        padding: '26px',
        position: 'relative',
      }}>
        {/* Close Button */}
        <button
          onClick={onClose}
          style={{
            position: 'absolute',
            top: '20px',
            right: '20px',
            background: 'rgba(148, 163, 184, 0.1)',
            border: 'none',
            borderRadius: '50%',
            width: '32px',
            height: '32px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            color: 'var(--text-primary)'
          }}
        >
          <X size={18} />
        </button>

        {/* Modal Header */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
          <div style={{ width: '40px', height: '40px', borderRadius: '10px', background: 'linear-gradient(135deg, #06b6d4, #6366f1)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <SlidersHorizontal size={22} weight="bold" color="#ffffff" />
          </div>
          <div>
            <h2 style={{ fontSize: '18px', fontWeight: 800, color: 'var(--text-primary)' }}>
              System Configuration & Real-Time Injector
            </h2>
            <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
              Add/Delete assets, inject live financial catalysts into RAG, or tune graph hyperparameters
            </p>
          </div>
        </div>

        {/* Tabs */}
        <div style={{ display: 'flex', gap: '8px', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '12px', marginBottom: '20px' }}>
          {[
            { id: 'stocks', label: 'Asset Universe (Add / Delete)', icon: Buildings },
            { id: 'news', label: 'Inject RAG Catalyst', icon: FileText },
            { id: 'graph', label: 'Graph Hyperparameters', icon: ShareNetwork },
          ].map(tab => {
            const Icon = tab.icon;
            const active = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  padding: '8px 14px',
                  borderRadius: '7px',
                  border: 'none',
                  fontSize: '12px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  background: active ? 'rgba(6, 182, 212, 0.15)' : 'transparent',
                  color: active ? 'var(--cyan)' : 'var(--text-secondary)',
                  borderBottom: active ? '2px solid var(--cyan)' : '2px solid transparent',
                  transition: 'all 0.2s ease'
                }}
              >
                <Icon size={16} weight={active ? "fill" : "regular"} />
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* TAB 1: Asset Universe Manager */}
        {activeTab === 'stocks' && (
          <div>
            {/* Add Stock Form */}
            <form onSubmit={handleAddStockSubmit} style={{ background: 'var(--bg-subtle)', padding: '16px', borderRadius: '10px', border: '1px solid var(--border-subtle)', marginBottom: '20px' }}>
              <h4 style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Plus size={16} color="var(--cyan)" weight="bold" />
                <span>Add New Equity to Dynamic Graph</span>
              </h4>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px', marginBottom: '12px' }}>
                <div>
                  <label style={{ fontSize: '11px', color: 'var(--text-secondary)', display: 'block', marginBottom: '4px' }}>Ticker</label>
                  <input
                    type="text"
                    placeholder="e.g. INFY"
                    value={newTicker}
                    onChange={(e) => setNewTicker(e.target.value)}
                    className="form-input"
                    required
                  />
                </div>
                <div>
                  <label style={{ fontSize: '11px', color: 'var(--text-secondary)', display: 'block', marginBottom: '4px' }}>Company Name</label>
                  <input
                    type="text"
                    placeholder="e.g. Infosys Ltd."
                    value={newName}
                    onChange={(e) => setNewName(e.target.value)}
                    className="form-input"
                    required
                  />
                </div>
                <div>
                  <label style={{ fontSize: '11px', color: 'var(--text-secondary)', display: 'block', marginBottom: '4px' }}>Sector</label>
                  <select
                    value={newSector}
                    onChange={(e) => setNewSector(e.target.value)}
                    className="form-input"
                  >
                    <option value="Technology">Technology</option>
                    <option value="Semiconductors">Semiconductors</option>
                    <option value="Financials">Financials</option>
                    <option value="Consumer/Tech">Consumer/Tech</option>
                    <option value="Automotive/Tech">Automotive/Tech</option>
                  </select>
                </div>
                <div>
                  <label style={{ fontSize: '11px', color: 'var(--text-secondary)', display: 'block', marginBottom: '4px' }}>Price ($)</label>
                  <input
                    type="number"
                    step="0.01"
                    placeholder="150.00"
                    value={newPrice}
                    onChange={(e) => setNewPrice(e.target.value)}
                    className="form-input"
                    required
                  />
                </div>
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                <button type="submit" className="btn-primary" style={{ padding: '8px 16px', fontSize: '12px' }}>
                  <Plus size={15} weight="bold" />
                  <span>Add Stock to Universe</span>
                </button>
              </div>
            </form>

            {/* Existing Stocks List */}
            <div>
              <h4 style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '10px' }}>
                Current Portfolio Universe ({stocks.length} Assets)
              </h4>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px' }}>
                {stocks.map(s => (
                  <div
                    key={s.ticker}
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
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{ fontWeight: 800, fontSize: '13px', color: 'var(--text-primary)' }}>{s.ticker}</span>
                        <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{s.sector}</span>
                      </div>
                      <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>${s.price.toFixed(2)}</div>
                    </div>

                    {stocks.length > 2 && (
                      <button
                        onClick={() => onDeleteStock(s.ticker)}
                        style={{
                          background: 'rgba(239, 68, 68, 0.1)',
                          border: '1px solid rgba(239, 68, 68, 0.25)',
                          color: '#ef4444',
                          width: '28px',
                          height: '28px',
                          borderRadius: '6px',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          cursor: 'pointer',
                          transition: 'all 0.2s ease'
                        }}
                        title={`Remove ${s.ticker}`}
                      >
                        <Trash size={14} />
                      </button>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: Custom RAG Catalyst Injector */}
        {activeTab === 'news' && (
          <form onSubmit={handleInjectNewsSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            <div style={{ background: 'rgba(6, 182, 212, 0.08)', padding: '12px', borderRadius: '8px', border: '1px solid rgba(6, 182, 212, 0.2)', fontSize: '12px', color: 'var(--text-secondary)' }}>
              Inject a breaking real-time news item to simulate how the <strong>RAG Vector Store</strong> and <strong>FinLLM Reasoner</strong> instantly rebalance the Multi-Modal Attention Gate.
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '12px' }}>
              <div>
                <label style={{ fontSize: '11px', color: 'var(--text-secondary)', display: 'block', marginBottom: '4px' }}>Target Stock</label>
                <select
                  value={newsTicker}
                  onChange={(e) => setNewsTicker(e.target.value)}
                  className="form-input"
                >
                  {stocks.map(s => (
                    <option key={s.ticker} value={s.ticker}>{s.ticker} — {s.name}</option>
                  ))}
                </select>
              </div>

              <div>
                <label style={{ fontSize: '11px', color: 'var(--text-secondary)', display: 'block', marginBottom: '4px' }}>Catalyst Sentiment</label>
                <select
                  value={newsSentiment}
                  onChange={(e) => setNewsSentiment(e.target.value)}
                  className="form-input"
                >
                  <option value="BULLISH">BULLISH (Positive Catalyst)</option>
                  <option value="BEARISH">BEARISH (Negative Shock / Lawsuit)</option>
                  <option value="NEUTRAL">NEUTRAL (Standard Reporting)</option>
                </select>
              </div>
            </div>

            <div>
              <label style={{ fontSize: '11px', color: 'var(--text-secondary)', display: 'block', marginBottom: '4px' }}>Headline</label>
              <input
                type="text"
                placeholder="e.g. Federal Reserve Announces 50 Bps Rate Cut Amid Strong Growth"
                value={newsHeadline}
                onChange={(e) => setNewsHeadline(e.target.value)}
                className="form-input"
                required
              />
            </div>

            <div>
              <label style={{ fontSize: '11px', color: 'var(--text-secondary)', display: 'block', marginBottom: '4px' }}>Snippet / Excerpt</label>
              <textarea
                rows="3"
                placeholder="Details of the announcement, earnings surprise, or regulatory filing..."
                value={newsSnippet}
                onChange={(e) => setNewsSnippet(e.target.value)}
                className="form-input"
              />
            </div>

            {newsSuccess && (
              <div style={{ color: 'var(--emerald)', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <CheckCircle size={16} weight="fill" />
                <span>Catalyst successfully indexed into Vector Store! Cross-Attention Gate updated.</span>
              </div>
            )}

            <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
              <button type="submit" className="btn-primary">
                <Sparkle size={15} weight="fill" />
                <span>Inject into Vector Store</span>
              </button>
            </div>
          </form>
        )}

        {/* TAB 3: Graph Hyperparameters */}
        {activeTab === 'graph' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <div style={{ background: 'var(--bg-subtle)', padding: '16px', borderRadius: '10px', border: '1px solid var(--border-subtle)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-primary)' }}>
                  Correlation vs. Sector Alpha Weight (α): {Math.round(graphParams.alpha * 100)}% Correlation
                </span>
                <span style={{ fontSize: '12px', fontFamily: 'var(--font-mono)', color: 'var(--cyan)' }}>
                  {graphParams.alpha.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={graphParams.alpha}
                onChange={(e) => onUpdateGraphParams({ ...graphParams, alpha: parseFloat(e.target.value) })}
                style={{ width: '100%', cursor: 'pointer' }}
              />
              <p style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                Higher α prioritizes rolling price correlation; lower α strictly enforces industry sector hierarchy.
              </p>
            </div>

            <div style={{ background: 'var(--bg-subtle)', padding: '16px', borderRadius: '10px', border: '1px solid var(--border-subtle)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-primary)' }}>
                  Adjacency Sparsification Threshold (θ): {graphParams.threshold.toFixed(2)}
                </span>
                <span style={{ fontSize: '12px', fontFamily: 'var(--font-mono)', color: 'var(--cyan)' }}>
                  {graphParams.threshold.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0.05"
                max="0.60"
                step="0.05"
                value={graphParams.threshold}
                onChange={(e) => onUpdateGraphParams({ ...graphParams, threshold: parseFloat(e.target.value) })}
                style={{ width: '100%', cursor: 'pointer' }}
              />
              <p style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                Edges with blended weight below θ are pruned to eliminate noisy inter-stock connections.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
