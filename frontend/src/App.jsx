import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import HeroSection from './components/HeroSection';
import TechnicalChart from './components/TechnicalChart';
import DynamicGraphCanvas from './components/DynamicGraphCanvas';
import ModelForecastCard from './components/ModelForecastCard';
import RagKnowledgeDrawer from './components/RagKnowledgeDrawer';
import XaiAttribution from './components/XaiAttribution';
import BacktestSim from './components/BacktestSim';
import ConfigModal from './components/ConfigModal';
import DrilldownModal from './components/DrilldownModal';
import { STOCK_UNIVERSE } from './data/marketData';

export default function App() {
  const [stocks, setStocks] = useState(STOCK_UNIVERSE);
  const [selectedTicker, setSelectedTicker] = useState('AAPL');
  const [activeTab, setActiveTab] = useState('overview');
  const [isConfigModalOpen, setIsConfigModalOpen] = useState(false);
  const [isLiveStreaming, setIsLiveStreaming] = useState(false);
  const [graphParams, setGraphParams] = useState({ alpha: 0.70, threshold: 0.20 });

  // Detailed KPI & Graph Drilldown State
  const [drilldownDetail, setDrilldownDetail] = useState(null);
  const [isDrilldownOpen, setIsDrilldownOpen] = useState(false);

  const handleOpenDrilldown = (detail) => {
    setDrilldownDetail(detail);
    setIsDrilldownOpen(true);
  };

  // Theme state: 'dark' | 'light' | 'system'
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('aura_fin_theme') || 'dark';
  });

  // Apply Theme to DOM
  useEffect(() => {
    localStorage.setItem('aura_fin_theme', theme);

    const applyTheme = (t) => {
      let resolved = t;
      if (t === 'system') {
        resolved = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
      }
      document.documentElement.setAttribute('data-theme', resolved);
    };

    applyTheme(theme);

    if (theme === 'system') {
      const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
      const listener = (e) => {
        document.documentElement.setAttribute('data-theme', e.matches ? 'dark' : 'light');
      };
      mediaQuery.addEventListener('change', listener);
      return () => mediaQuery.removeEventListener('change', listener);
    }
  }, [theme]);

  // Real-Time Live Streaming Simulation
  useEffect(() => {
    if (!isLiveStreaming) return;

    const interval = setInterval(() => {
      setStocks(prevStocks =>
        prevStocks.map(s => {
          // Slight realistic micro-tick (-0.15% to +0.20%)
          const tickPct = (Math.random() * 0.35 - 0.15);
          const newPrice = Math.max(1, +(s.price * (1 + tickPct / 100)).toFixed(2));
          const newChange = +(s.changePct + tickPct * 0.4).toFixed(2);
          
          return {
            ...s,
            price: newPrice,
            changePct: newChange,
            rsi14: Math.min(88, Math.max(20, +(s.rsi14 + (tickPct * 0.8)).toFixed(1))),
            forecast: {
              ...s.forecast,
              returnPct: +(s.forecast.returnPct + (tickPct * 0.2)).toFixed(2),
            }
          };
        })
      );
    }, 2500);

    return () => clearInterval(interval);
  }, [isLiveStreaming]);

  // Add Stock Handler
  const handleAddStock = (newStock) => {
    setStocks(prev => [...prev, newStock]);
    setSelectedTicker(newStock.ticker);
    setIsConfigModalOpen(false);
  };

  // Delete Stock Handler
  const handleDeleteStock = (tickerToDelete) => {
    setStocks(prev => {
      const filtered = prev.filter(s => s.ticker !== tickerToDelete);
      if (selectedTicker === tickerToDelete && filtered.length > 0) {
        setSelectedTicker(filtered[0].ticker);
      }
      return filtered;
    });
  };

  // Inject Custom News Catalyst Handler (RAG + FinLLM Dynamic Rebalancing)
  const handleInjectNews = ({ ticker, headline, snippet, source, sentiment, similarity }) => {
    setStocks(prev =>
      prev.map(s => {
        if (s.ticker !== ticker) return s;

        const isBull = sentiment === 'BULLISH';
        const isBear = sentiment === 'BEARISH';
        const shiftReturn = isBull ? +0.65 : isBear ? -0.85 : 0.0;
        const shiftScore = isBull ? +0.35 : isBear ? -0.45 : 0.0;
        const newScore = Math.max(-1.0, Math.min(1.0, +(s.finReasoning.sentimentScore + shiftScore).toFixed(2)));

        return {
          ...s,
          forecast: {
            ...s.forecast,
            returnPct: +(s.forecast.returnPct + shiftReturn).toFixed(2),
            direction: newScore >= 0 ? 'BULLISH' : 'BEARISH',
            confidence: Math.min(94, Math.max(55, +(s.forecast.confidence + 4.5).toFixed(1))),
            textGatePct: Math.min(75, Math.max(25, s.forecast.textGatePct + 8)),
            graphGatePct: Math.max(25, Math.min(75, s.forecast.graphGatePct - 8)),
          },
          ragNews: [
            {
              id: `user-${Date.now()}`,
              headline,
              snippet,
              source,
              sentiment,
              similarity: similarity || 0.945,
            },
            ...s.ragNews,
          ],
          finReasoning: {
            ...s.finReasoning,
            summary: `Breaking catalyst injected: "${headline}". FinLLM dynamically rebalanced Multi-Modal gate to ${Math.min(75, s.forecast.textGatePct + 8)}% text weighting.`,
            sentimentScore: newScore,
            status: 'PASSED',
            safetyCheck: 'Real-Time Vector Ingestion Verified'
          }
        };
      })
    );
  };

  const selectedStock = stocks.find(s => s.ticker === selectedTicker) || stocks[0] || STOCK_UNIVERSE[0];

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Top Navigation */}
      <Navbar
        onOpenConfigModal={() => setIsConfigModalOpen(true)}
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        theme={theme}
        setTheme={setTheme}
        isLiveStreaming={isLiveStreaming}
        setIsLiveStreaming={setIsLiveStreaming}
        stocks={stocks}
      />

      {/* Main Container */}
      <main style={{ maxWidth: '1440px', width: '100%', margin: '0 auto', padding: '24px', flex: 1 }}>
        {/* Hero Stock Selector & Overview */}
        <HeroSection
          selectedTicker={selectedTicker}
          onSelectStock={setSelectedTicker}
          stocks={stocks}
          onOpenConfigModal={() => setIsConfigModalOpen(true)}
          onDeleteStock={handleDeleteStock}
          onOpenDrilldown={handleOpenDrilldown}
        />

        {/* Dynamic Tab Views */}
        {activeTab === 'overview' && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(12, 1fr)', gap: '20px' }}>
            {/* Left Column (7 cols): Price Chart + RAG Documents */}
            <div style={{ gridColumn: 'span 7', display: 'flex', flexDirection: 'column', gap: '20px' }}>
              <TechnicalChart stock={selectedStock} onOpenDrilldown={handleOpenDrilldown} />
              <RagKnowledgeDrawer stock={selectedStock} />
            </div>

            {/* Right Column (5 cols): Forecast Head + Dynamic Graph Canvas */}
            <div style={{ gridColumn: 'span 5', display: 'flex', flexDirection: 'column', gap: '20px' }}>
              <ModelForecastCard stock={selectedStock} onOpenDrilldown={handleOpenDrilldown} />
              <DynamicGraphCanvas
                selectedTicker={selectedTicker}
                onSelectStock={setSelectedTicker}
                stocks={stocks}
                graphParams={graphParams}
                onOpenDrilldown={handleOpenDrilldown}
              />
            </div>
          </div>
        )}

        {activeTab === 'graph' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <DynamicGraphCanvas
              selectedTicker={selectedTicker}
              onSelectStock={setSelectedTicker}
              stocks={stocks}
              graphParams={graphParams}
              onOpenDrilldown={handleOpenDrilldown}
            />
            <XaiAttribution stock={selectedStock} />
          </div>
        )}

        {activeTab === 'xai' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <XaiAttribution stock={selectedStock} />
            <RagKnowledgeDrawer stock={selectedStock} />
          </div>
        )}

        {activeTab === 'backtest' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <BacktestSim stock={selectedStock} onOpenDrilldown={handleOpenDrilldown} />
          </div>
        )}
      </main>

      {/* Detailed KPI & Graph Drilldown Modal */}
      <DrilldownModal
        isOpen={isDrilldownOpen}
        onClose={() => setIsDrilldownOpen(false)}
        detail={drilldownDetail}
      />

      {/* System Configuration & Catalyst Injector Modal */}
      <ConfigModal
        isOpen={isConfigModalOpen}
        onClose={() => setIsConfigModalOpen(false)}
        stocks={stocks}
        onAddStock={handleAddStock}
        onDeleteStock={handleDeleteStock}
        onInjectNews={handleInjectNews}
        graphParams={graphParams}
        onUpdateGraphParams={setGraphParams}
      />

      {/* Footer */}
      <footer style={{ borderTop: '1px solid var(--border-subtle)', padding: '20px 24px', background: 'var(--bg-surface)', fontSize: '12px', color: 'var(--text-muted)' }}>
        <div style={{ maxWidth: '1440px', margin: '0 auto', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '10px' }}>
          <div>
            <strong style={{ color: 'var(--text-primary)' }}>AURA-FIN</strong> • Explainable Multi-Modal Stock Market Forecasting Framework
          </div>
          <div>
            Dynamic Graph Transformers & Financial Large Language Models (RAG)
          </div>
        </div>
      </footer>
    </div>
  );
}
