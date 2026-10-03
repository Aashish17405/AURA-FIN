import React, { useRef, useEffect, useState } from 'react';
import { ShareNetwork, ArrowsClockwise, Lightning, Sparkle, Info } from '@phosphor-icons/react';
import { STOCK_UNIVERSE } from '../data/marketData';

const SECTOR_COLORS = {
  Technology: '#3b82f6',
  Semiconductors: '#8b5cf6',
  Financials: '#f59e0b',
  'Automotive/Tech': '#ef4444',
  'Consumer/Tech': '#10b981',
};

export default function DynamicGraphCanvas({
  selectedTicker,
  onSelectStock,
  stocks,
  graphParams = { alpha: 0.70, threshold: 0.20 },
  onOpenDrilldown
}) {
  const stockList = stocks || STOCK_UNIVERSE;
  const canvasRef = useRef(null);
  const [hoveredNode, setHoveredNode] = useState(null);
  const [pulseSpeed, setPulseSpeed] = useState(1);
  const [showAttentionRays, setShowAttentionRays] = useState(true);

  const handleGraphSpecsDrilldown = () => {
    if (!onOpenDrilldown) return;
    onOpenDrilldown({
      title: 'Dynamic Spatio-Temporal Graph Construction (DGT)',
      subtitle: 'Rolling Pearson correlation + Sector topology with symmetric normalization',
      category: 'Graph Topology KPI',
      value: `${stockList.length} Nodes • ${Math.round(stockList.length * 1.8)} Active Edges`,
      unit: `Threshold θ=${graphParams.threshold.toFixed(2)}`,
      trend: 'neutral',
      formula: '\\mathbf{A}_{ij}^{(t)} = \\alpha \\cdot \\rho_{ij}^{(t, \\tau=30)} + (1 - \\alpha) \\cdot \\mathbf{S}_{ij}, \\quad \\tilde{\\mathbf{A}} = \\tilde{\\mathbf{D}}^{-\\frac{1}{2}} (\\mathbf{A} + \\mathbf{I}) \\tilde{\\mathbf{D}}^{-\\frac{1}{2}}',
      formulaExplanation: 'Adjacency matrix dynamically combines rolling 30-day cross-equity return correlations (weight α=0.70) with fundamental sector classification indicators S_ij (weight 1-α=0.30), normalized symmetrically to prevent spectral explosion.',
      metrics: [
        { label: 'Temporal Window (τ)', value: '30 Trading Days', color: 'var(--cyan)' },
        { label: 'Edge Pruning Cutoff (θ)', value: `${graphParams.threshold.toFixed(2)}`, color: 'var(--indigo)' },
        { label: 'Degree Normalization', value: 'Symmetric Kipf-Welling', color: 'var(--emerald)' },
      ],
      impactOnModel: 'Spatial self-attention heads propagate cross-asset return momentum and risk contagion across correlated nodes, providing spatial contextual bias before temporal sequence processing.',
      academicContext: 'Evaluated against static graphs. Dynamic graph updating captures rapid regime shifts during earnings releases and macroeconomic interest rate announcements.'
    });
  };

  const handleNodeDrilldown = (node) => {
    if (!onOpenDrilldown) return;
    const neighbors = node.graphNeighbors || [];
    onOpenDrilldown({
      title: `Node Topology: ${node.ticker} (${node.name})`,
      subtitle: `Spatio-temporal position in equity market graph`,
      category: 'Graph Node Drilldown',
      value: `$${node.price.toFixed(2)}`,
      unit: `${node.sector} Sector`,
      trend: node.changePct >= 0 ? 'up' : 'down',
      formula: '\\mathbf{z}_i = \\sum_{j \\in \\mathcal{N}(i)} \\alpha_{ij} \\mathbf{W}_v \\mathbf{x}_j, \\quad \\alpha_{ij} = \\frac{\\exp\\left( \\frac{(\\mathbf{W}_q \\mathbf{x}_i)^T (\\mathbf{W}_k \\mathbf{x}_j)}{\\sqrt{d}} + b_{ij}^{\\text{spatial}} \\right)}{\\sum_{k} \\dots}',
      formulaExplanation: `Spatial attention weights computed over neighbor equities in ${node.sector} sector and cross-sector correlation peers with structural graph bias.`,
      metrics: [
        { label: 'Sector Group', value: node.sector, color: 'var(--indigo)' },
        { label: 'Connected Peers', value: `${neighbors.length} Influential Neighbors`, color: 'var(--cyan)' },
        { label: 'Top Peer Attention', value: neighbors[0] ? `${neighbors[0].ticker} (${neighbors[0].attention.toFixed(2)})` : 'N/A', color: 'var(--emerald)' },
      ],
      impactOnModel: `Shocks in ${node.ticker} propagate along high-correlation edges into peers. Graph attention dynamically downweights noisy peers and amplifies leader stocks.`,
      academicContext: `Node features comprise 13 technical indicators concatenated with FinLLM semantic embeddings, maintaining full spatial and multi-modal alignment.`
    });
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let animationFrameId;

    // Set high-DPI resolution
    const width = canvas.parentElement.clientWidth || 600;
    const height = 440;
    canvas.width = width * window.devicePixelRatio;
    canvas.height = height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    // Initialize node positions in a circle layout with slight random offset
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(width, height) * 0.36;

    const nodes = stockList.map((stock, i) => {
      const angle = (i / stockList.length) * 2 * Math.PI - Math.PI / 2;
      return {
        ...stock,
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle),
        baseX: centerX + radius * Math.cos(angle),
        baseY: centerY + radius * Math.sin(angle),
        vx: 0,
        vy: 0,
        radius: stock.ticker === selectedTicker ? 24 : 18,
        color: SECTOR_COLORS[stock.sector] || '#64748b',
      };
    });

    // Particle flow list along edges
    const particles = [];
    nodes.forEach((n1, i) => {
      nodes.forEach((n2, j) => {
        if (i < j && (n1.sector === n2.sector || Math.abs(n1.changePct - n2.changePct) < 2.0)) {
          for (let p = 0; p < 2; p++) {
            particles.push({
              source: n1,
              target: n2,
              progress: Math.random(),
              speed: (0.003 + Math.random() * 0.004) * pulseSpeed,
              color: n1.color,
            });
          }
        }
      });
    });

    let time = 0;

    const render = () => {
      time += 0.02 * pulseSpeed;
      ctx.clearRect(0, 0, width, height);

      // Subtle ambient background grid
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.02)';
      ctx.lineWidth = 1;
      const gridSize = 32;
      for (let x = 0; x < width; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      for (let y = 0; y < height; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      // Gentle floating physics
      nodes.forEach((node, idx) => {
        const floatX = Math.sin(time + idx * 1.5) * 4;
        const floatY = Math.cos(time + idx * 1.2) * 4;
        node.x = node.baseX + floatX;
        node.y = node.baseY + floatY;
      });

      // Draw Dynamic Adjacency Edges
      nodes.forEach((nodeA, i) => {
        nodes.forEach((nodeB, j) => {
          if (i >= j) return;
          const isRelated = nodeA.sector === nodeB.sector || nodeA.graphNeighbors?.some(gn => gn.ticker === nodeB.ticker);
          const isTargetEdge = nodeA.ticker === selectedTicker || nodeB.ticker === selectedTicker;

          if (isRelated) {
            ctx.beginPath();
            ctx.moveTo(nodeA.x, nodeA.y);
            ctx.lineTo(nodeB.x, nodeB.y);

            if (isTargetEdge && showAttentionRays) {
              // Highlighted attention edge
              ctx.strokeStyle = 'rgba(6, 182, 212, 0.65)';
              ctx.lineWidth = 2.4;
              ctx.shadowColor = '#06b6d4';
              ctx.shadowBlur = 8;
            } else {
              ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
              ctx.lineWidth = 1.0;
              ctx.shadowBlur = 0;
            }
            ctx.stroke();
            ctx.shadowBlur = 0;
          }
        });
      });

      // Draw Flowing Spatio-Temporal Momentum Particles
      if (showAttentionRays) {
        particles.forEach(p => {
          p.progress += p.speed;
          if (p.progress > 1) p.progress = 0;

          const px = p.source.x + (p.target.x - p.source.x) * p.progress;
          const py = p.source.y + (p.target.y - p.source.y) * p.progress;

          ctx.beginPath();
          ctx.arc(px, py, 2.5, 0, Math.PI * 2);
          ctx.fillStyle = p.source.ticker === selectedTicker || p.target.ticker === selectedTicker ? '#06b6d4' : 'rgba(255, 255, 255, 0.4)';
          ctx.shadowColor = p.source.ticker === selectedTicker ? '#06b6d4' : '#ffffff';
          ctx.shadowBlur = 6;
          ctx.fill();
          ctx.shadowBlur = 0;
        });
      }

      // Draw Stock Nodes
      nodes.forEach(node => {
        const isSelected = node.ticker === selectedTicker;
        const isHovered = hoveredNode && hoveredNode.ticker === node.ticker;

        // Outer glow
        ctx.beginPath();
        const pulse = Math.sin(time * 2 + node.radius) * 3;
        ctx.arc(node.x, node.y, node.radius + (isSelected ? 8 + pulse : 4), 0, Math.PI * 2);
        ctx.fillStyle = isSelected ? 'rgba(6, 182, 212, 0.25)' : 'rgba(255, 255, 255, 0.03)';
        ctx.fill();

        // Main node body
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        ctx.fillStyle = isSelected ? '#0f172a' : 'rgba(15, 23, 42, 0.9)';
        ctx.fill();
        ctx.lineWidth = isSelected ? 3 : 1.5;
        ctx.strokeStyle = isSelected ? '#06b6d4' : node.color;
        if (isSelected) {
          ctx.shadowColor = '#06b6d4';
          ctx.shadowBlur = 12;
        }
        ctx.stroke();
        ctx.shadowBlur = 0;

        // Node text label
        ctx.font = `700 ${isSelected ? 11 : 10}px 'Plus Jakarta Sans', sans-serif`;
        ctx.fillStyle = '#ffffff';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(node.ticker, node.x, node.y - (isSelected ? 3 : 0));

        if (isSelected) {
          ctx.font = "8px 'JetBrains Mono', monospace";
          ctx.fillStyle = node.forecast.direction === 'BULLISH' ? '#10b981' : '#ef4444';
          ctx.fillText(`${node.forecast.returnPct > 0 ? '+' : ''}${node.forecast.returnPct}%`, node.x, node.y + 9);
        }
      });

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    // Mouse Interaction for Node Clicking & Hover
    const handleMouseMove = (e) => {
      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      const hit = nodes.find(node => {
        const dx = node.x - mouseX;
        const dy = node.y - mouseY;
        return Math.sqrt(dx * dx + dy * dy) <= node.radius + 6;
      });

      setHoveredNode(hit || null);
      canvas.style.cursor = hit ? 'pointer' : 'default';
    };

    const handleClick = (e) => {
      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      const hit = nodes.find(node => {
        const dx = node.x - mouseX;
        const dy = node.y - mouseY;
        return Math.sqrt(dx * dx + dy * dy) <= node.radius + 6;
      });

      if (hit) {
        onSelectStock(hit.ticker);
        handleNodeDrilldown(hit);
      }
    };

    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('click', handleClick);

    return () => {
      cancelAnimationFrame(animationFrameId);
      canvas.removeEventListener('mousemove', handleMouseMove);
      canvas.removeEventListener('click', handleClick);
    };
  }, [selectedTicker, pulseSpeed, showAttentionRays, stockList, graphParams]);

  return (
    <div className="glass-panel" style={{ padding: '20px', position: 'relative', overflow: 'hidden' }}>
      {/* Header controls */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px', flexWrap: 'wrap', gap: '10px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ width: '28px', height: '28px', borderRadius: '6px', background: 'rgba(6, 182, 212, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <ShareNetwork size={16} weight="duotone" color="var(--cyan)" />
          </div>
          <div>
            <h3 style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text-primary)' }}>
              Dynamic Spatio-Temporal Graph (DGT)
            </h3>
            <p style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              Nodes: Equities • Dynamic Edges: Rolling Pearson Correlation (τ=30) + Sector Contagion
            </p>
          </div>
        </div>

        {/* Action Toggles */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <button
            onClick={handleGraphSpecsDrilldown}
            style={{
              padding: '6px 12px',
              fontSize: '11px',
              fontWeight: 600,
              borderRadius: '6px',
              background: 'var(--bg-subtle)',
              color: 'var(--cyan)',
              border: '1px solid var(--border-subtle)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '4px'
            }}
            title="Open Graph Topology Mathematical Specifications"
          >
            <Info size={14} weight="bold" />
            <span>Topology Specs</span>
          </button>

          <button
            onClick={() => setShowAttentionRays(!showAttentionRays)}
            style={{
              padding: '6px 12px',
              fontSize: '11px',
              fontWeight: 600,
              borderRadius: '6px',
              background: showAttentionRays ? 'rgba(6, 182, 212, 0.15)' : 'var(--bg-subtle)',
              color: showAttentionRays ? 'var(--cyan)' : 'var(--text-secondary)',
              border: '1px solid var(--border-subtle)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '4px'
            }}
          >
            <Lightning size={14} weight={showAttentionRays ? "fill" : "regular"} />
            {showAttentionRays ? 'Contagion Flow: ON' : 'Flow: OFF'}
          </button>
        </div>
      </div>

      {/* Canvas */}
      <div style={{ position: 'relative', width: '100%', height: '440px', background: 'var(--bg-subtle)', borderRadius: '12px', overflow: 'hidden', border: '1px solid var(--border-subtle)' }}>
        <canvas ref={canvasRef} style={{ width: '100%', height: '100%', display: 'block' }} />
        
        {/* Sector Legend */}
        <div style={{ position: 'absolute', bottom: '12px', left: '12px', display: 'flex', flexWrap: 'wrap', gap: '10px', background: 'var(--bg-card)', padding: '6px 12px', borderRadius: '8px', border: '1px solid var(--border-subtle)', fontSize: '10px', backdropFilter: 'blur(10px)' }}>
          {Object.entries(SECTOR_COLORS).map(([sec, col]) => (
            <div key={sec} style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: col }}></span>
              <span style={{ color: 'var(--text-secondary)' }}>{sec}</span>
            </div>
          ))}
        </div>

        {/* Hover / Click Prompt */}
        <div style={{ position: 'absolute', top: '12px', right: '12px', fontSize: '11px', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '4px', background: 'var(--bg-card)', padding: '4px 8px', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
          <Sparkle size={13} color="var(--cyan)" />
          <span>Click node to inspect</span>
        </div>
      </div>
    </div>
  );
}
