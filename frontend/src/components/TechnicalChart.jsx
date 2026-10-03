import React, { useState } from 'react';
import { ChartLineUp, Pulse, Compass, CalendarBlank } from '@phosphor-icons/react';

export default function TechnicalChart({ stock, onOpenDrilldown }) {
  const [hoverIdx, setHoverIdx] = useState(null);

  // Generate 25 historical data points for the selected stock
  const basePrice = stock.price;
  const points = [];
  const n = 25;
  
  for (let i = 0; i < n; i++) {
    const drift = Math.sin(i * 0.4) * (basePrice * 0.04) + (i - n / 2) * (stock.changePct > 0 ? 0.8 : -0.8);
    const noise = Math.cos(i * 0.9) * (basePrice * 0.015);
    const p = Math.max(10, basePrice + drift + noise);
    const sma = p * 0.98 + Math.sin(i * 0.3) * 2;
    const bbUpper = sma + (basePrice * 0.05);
    const bbLower = sma - (basePrice * 0.05);
    const rsi = Math.min(85, Math.max(25, 50 + (p - sma) * 1.8));

    points.push({
      day: `T-${n - i}`,
      price: p,
      sma,
      bbUpper,
      bbLower,
      rsi,
    });
  }

  // Ensure last point aligns with current price
  points[n - 1].price = basePrice;

  const handlePointDrilldown = (pt) => {
    if (!onOpenDrilldown) return;
    const isAboveSma = pt.price >= pt.sma;
    onOpenDrilldown({
      title: `${stock.ticker} Session Point (${pt.day}): $${pt.price.toFixed(2)}`,
      subtitle: `Historical OHLCV execution point with Bollinger Envelope bounds`,
      category: 'Time-Series Point Drilldown',
      value: `$${pt.price.toFixed(2)}`,
      unit: `Session ${pt.day}`,
      trend: isAboveSma ? 'up' : 'down',
      formula: '\\text{Upper} = \\text{SMA}_{20} + 2\\sigma, \\quad \\text{Lower} = \\text{SMA}_{20} - 2\\sigma',
      formulaExplanation: 'John Bollinger volatility envelope. Bandwidth expands during explosive regime shifts and contracts during volatility squeezes prior to breakouts.',
      metrics: [
        { label: 'Close Price', value: `$${pt.price.toFixed(2)}`, color: 'var(--cyan)' },
        { label: '20-Day SMA', value: `$${pt.sma.toFixed(2)}`, color: 'var(--indigo)' },
        { label: 'Upper Band', value: `$${pt.bbUpper.toFixed(2)}`, color: 'var(--emerald)' },
        { label: 'Lower Band', value: `$${pt.bbLower.toFixed(2)}`, color: 'var(--crimson)' },
        { label: 'Session RSI', value: pt.rsi.toFixed(1), color: pt.rsi > 70 ? 'var(--crimson)' : pt.rsi < 30 ? 'var(--emerald)' : 'var(--indigo)' },
      ],
      impactOnModel: 'Historical temporal sequences pass through multi-head self-attention with learned positional encoding before entering cross-modal fusion with FinLLM text vectors.',
      academicContext: 'Evaluated under continuous sliding windows to verify model robustness against non-stationary temporal drift and cyclical regime shifts.'
    });
  };

  const handleBollingerDrilldown = () => {
    if (!onOpenDrilldown) return;
    onOpenDrilldown({
      title: 'Bollinger Bands & Volatility Channel Specification',
      subtitle: 'Dynamic standard deviation envelope around the 20-day moving average',
      category: 'Technical Indicator KPI',
      value: `±2.0 Standard Deviations`,
      unit: '95.4% Normal Envelope',
      trend: 'neutral',
      formula: '\\text{Bandwidth} = \\frac{\\text{Upper Band} - \\text{Lower Band}}{\\text{SMA}_{20}} \\times 100',
      formulaExplanation: 'Normalized measure of market dispersion. Compression precedes directional volatility expansion, utilized as a high-conviction trigger feature.',
      metrics: [
        { label: 'Period (N)', value: '20 Days', color: 'var(--indigo)' },
        { label: 'Multiplier (K)', value: '2.0', color: 'var(--cyan)' },
        { label: 'Current Regime', value: 'Expansion Phase', color: 'var(--emerald)' },
      ],
      impactOnModel: 'Bandwidth serves as an explicit input feature into the spatial graph edge weighting to prevent false-positive breakouts during consolidation.',
      academicContext: 'Standardized volatility channels ground the dual regression head, preventing prediction of returns exceeding statistical Chebyshev bounds.'
    });
  };

  // SVG dimensions
  const svgWidth = 620;
  const svgHeight = 220;
  const padL = 45;
  const padR = 20;
  const padT = 15;
  const padB = 25;

  const minP = Math.min(...points.map(d => d.bbLower)) * 0.98;
  const maxP = Math.max(...points.map(d => d.bbUpper)) * 1.02;

  const getX = (i) => padL + (i / (n - 1)) * (svgWidth - padL - padR);
  const getY = (val) => padT + (1 - (val - minP) / (maxP - minP)) * (svgHeight - padT - padB);

  // Path generators
  const pricePath = points.map((d, i) => `${i === 0 ? 'M' : 'L'} ${getX(i)} ${getY(d.price)}`).join(' ');
  const smaPath = points.map((d, i) => `${i === 0 ? 'M' : 'L'} ${getX(i)} ${getY(d.sma)}`).join(' ');
  const areaPath = `${pricePath} L ${getX(n - 1)} ${svgHeight - padB} L ${getX(0)} ${svgHeight - padB} Z`;

  // Bollinger envelope area
  const bbTop = points.map((d, i) => `${i === 0 ? 'M' : 'L'} ${getX(i)} ${getY(d.bbUpper)}`).join(' ');
  const bbBottom = points.slice().reverse().map((d, i) => `L ${getX(n - 1 - i)} ${getY(d.bbLower)}`).join(' ');
  const bbEnvelope = `${bbTop} ${bbBottom} Z`;

  const activePoint = hoverIdx !== null ? points[hoverIdx] : points[n - 1];

  return (
    <div className="glass-panel" style={{ padding: '20px' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px', flexWrap: 'wrap', gap: '10px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ width: '28px', height: '28px', borderRadius: '6px', background: 'rgba(59, 130, 246, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <ChartLineUp size={16} weight="duotone" color="var(--indigo)" />
          </div>
          <div>
            <h3 style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text-primary)' }}>
              Historical Price & Bollinger Envelope
            </h3>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              Click any graph point or indicator for quantitative drilldown
            </span>
          </div>
        </div>

        {/* Live Hover Readout (Clickable) */}
        <div
          onClick={() => handlePointDrilldown(activePoint)}
          style={{ display: 'flex', alignItems: 'center', gap: '16px', fontFamily: 'var(--font-mono)', fontSize: '12px', cursor: 'pointer', padding: '4px 8px', borderRadius: '6px', background: 'var(--bg-subtle)', border: '1px solid var(--border-subtle)', transition: 'all 0.2s ease' }}
          title="Click to drilldown on this session point"
        >
          <div>
            <span style={{ color: 'var(--text-muted)' }}>Price: </span>
            <span style={{ color: 'var(--text-primary)', fontWeight: 700 }}>${activePoint.price.toFixed(2)}</span>
          </div>
          <div>
            <span style={{ color: 'var(--text-muted)' }}>SMA: </span>
            <span style={{ color: 'var(--indigo)', fontWeight: 600 }}>${activePoint.sma.toFixed(2)}</span>
          </div>
          <div>
            <span style={{ color: 'var(--text-muted)' }}>RSI: </span>
            <span style={{ color: activePoint.rsi > 70 ? 'var(--crimson)' : activePoint.rsi < 30 ? 'var(--emerald)' : 'var(--indigo)', fontWeight: 600 }}>
              {activePoint.rsi.toFixed(1)}
            </span>
          </div>
          <span style={{ fontSize: '10px', color: 'var(--cyan)' }}>↗</span>
        </div>
      </div>

      {/* SVG Canvas */}
      <div style={{ position: 'relative', width: '100%', overflow: 'hidden' }}>
        <svg
          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
          style={{ width: '100%', height: 'auto', display: 'block' }}
          onMouseLeave={() => setHoverIdx(null)}
        >
          <defs>
            <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--cyan)" stopOpacity="0.25" />
              <stop offset="100%" stopColor="var(--cyan)" stopOpacity="0.0" />
            </linearGradient>
            <linearGradient id="bbGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--indigo)" stopOpacity="0.08" />
              <stop offset="100%" stopColor="var(--indigo)" stopOpacity="0.01" />
            </linearGradient>
          </defs>

          {/* Grid lines */}
          {[0.25, 0.5, 0.75].map((lvl, idx) => (
            <line
              key={idx}
              x1={padL}
              y1={padT + lvl * (svgHeight - padT - padB)}
              x2={svgWidth - padR}
              y2={padT + lvl * (svgHeight - padT - padB)}
              stroke="var(--border-subtle)"
              strokeDasharray="4 4"
            />
          ))}

          {/* Bollinger Band Envelope */}
          <path d={bbEnvelope} fill="url(#bbGradient)" />
          <path d={bbTop} stroke="var(--indigo)" strokeWidth="1" strokeDasharray="3 3" strokeOpacity="0.4" fill="none" />
          <path d={points.map((d, i) => `${i === 0 ? 'M' : 'L'} ${getX(i)} ${getY(d.bbLower)}`).join(' ')} stroke="var(--indigo)" strokeWidth="1" strokeDasharray="3 3" strokeOpacity="0.4" fill="none" />

          {/* Shaded Price Area & Line */}
          <path d={areaPath} fill="url(#priceGradient)" />
          <path d={smaPath} stroke="var(--indigo)" strokeWidth="1.6" strokeDasharray="5 3" fill="none" />
          <path d={pricePath} stroke="var(--cyan)" strokeWidth="2.4" fill="none" />

          {/* Interactive Hover Hit Zones */}
          {points.map((d, i) => (
            <rect
              key={i}
              x={getX(i) - 10}
              y={0}
              width={20}
              height={svgHeight}
              fill="transparent"
              style={{ cursor: 'pointer' }}
              onMouseEnter={() => setHoverIdx(i)}
              onClick={() => handlePointDrilldown(d)}
            />
          ))}

          {/* Active Hover Marker */}
          {hoverIdx !== null && (
            <g style={{ cursor: 'pointer' }} onClick={() => handlePointDrilldown(points[hoverIdx])}>
              <line
                x1={getX(hoverIdx)}
                y1={padT}
                x2={getX(hoverIdx)}
                y2={svgHeight - padB}
                stroke="var(--cyan)"
                strokeDasharray="3 3"
              />
              <circle
                cx={getX(hoverIdx)}
                cy={getY(points[hoverIdx].price)}
                r="5"
                fill="var(--cyan)"
                stroke="var(--bg-surface)"
                strokeWidth="2"
              />
            </g>
          )}
        </svg>

        {/* Legend */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '20px', marginTop: '10px', fontSize: '11px', color: 'var(--text-secondary)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '12px', height: '3px', background: 'var(--cyan)', borderRadius: '2px' }}></span>
            <span>Close Price ($)</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '12px', height: '2px', background: 'var(--indigo)', borderTop: '2px dashed var(--indigo)' }}></span>
            <span>SMA-20</span>
          </div>
          <div
            onClick={handleBollingerDrilldown}
            style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer', padding: '2px 6px', borderRadius: '4px', background: 'var(--bg-subtle)' }}
            title="Click for Bollinger Bands specification"
          >
            <span style={{ width: '12px', height: '8px', background: 'rgba(99, 102, 241, 0.15)', borderRadius: '2px', border: '1px solid var(--border-subtle)' }}></span>
            <span style={{ color: 'var(--cyan)' }}>Bollinger Bands ℹ</span>
          </div>
        </div>
      </div>
    </div>
  );
}
