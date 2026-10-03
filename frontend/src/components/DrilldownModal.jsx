import React, { useEffect } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';
import {
  X,
  Sparkle,
  Cpu,
  Info,
  TrendUp,
  TrendDown,
  ShieldCheck,
  ChartLineUp,
  Scales,
  Clock,
  ArrowRight
} from '@phosphor-icons/react';

export default function DrilldownModal({ isOpen, onClose, detail }) {
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose();
    };
    if (isOpen) {
      window.addEventListener('keydown', handleKeyDown);
    }
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen || !detail) return null;

  const {
    title = 'Metric Drilldown',
    subtitle = 'Detailed Quantitative & Neural Specification',
    category = 'KPI Metric',
    value = '',
    unit = '',
    trend = null, // 'up' | 'down' | 'neutral'
    formula = null,
    formulaExplanation = null,
    academicContext = null,
    metrics = [], // [{ label, value, color }]
    impactOnModel = null,
    institutionalUse = null,
  } = detail;

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 200,
        background: 'rgba(5, 8, 17, 0.85)',
        backdropFilter: 'blur(16px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '20px',
        animation: 'fadeIn 0.2s ease-out'
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="glass-panel"
        style={{
          maxWidth: '680px',
          width: '100%',
          maxHeight: '90vh',
          overflowY: 'auto',
          padding: '28px',
          position: 'relative',
          borderRadius: '16px',
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5), 0 0 30px ',
          animation: 'slideUp 0.25s cubic-bezier(0.16, 1, 0.3, 1)'
        }}
      >
        {/* Close Button */}
        <button
          onClick={onClose}
          style={{
            position: 'absolute',
            top: '20px',
            right: '20px',
            background: 'var(--bg-subtle)',
            border: '1px solid var(--border-subtle)',
            borderRadius: '8px',
            width: '32px',
            height: '32px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--text-muted)',
            cursor: 'pointer',
            transition: 'all 0.15s ease'
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.color = 'var(--text-primary)';
            e.currentTarget.style.borderColor = 'var(--cyan)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.color = 'var(--text-muted)';
            e.currentTarget.style.borderColor = 'var(--border-subtle)';
          }}
        >
          <X size={18} />
        </button>

        {/* Modal Header */}
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: '12px', marginBottom: '20px', paddingRight: '40px' }}>
          <div style={{
            width: '42px',
            height: '42px',
            borderRadius: '10px',
            background: 'linear-gradient(135deg, rgba(6, 182, 212, 0.2), rgba(99, 102, 241, 0.2))',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            border: '1px solid',
            flexShrink: 0
          }}>
            <Sparkle size={24} weight="duotone" color="var(--cyan)" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
              <span style={{
                fontSize: '10px',
                fontWeight: 800,
                textTransform: 'uppercase',
                letterSpacing: '0.6px',
                padding: '2px 8px',
                borderRadius: '4px',
                background: 'rgba(6, 182, 212, 0.12)',
                color: 'var(--cyan)',
                border: '1px solid rgba(6, 182, 212, 0.3)'
              }}>
                {category}
              </span>
              <span style={{ fontSize: '11px', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                <Clock size={12} />
                Real-Time Verified
              </span>
            </div>
            <h2 style={{ fontSize: '20px', fontWeight: 800, color: 'var(--text-primary)', letterSpacing: '-0.3px', margin: 0 }}>
              {title}
            </h2>
            <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '2px', margin: 0 }}>
              {subtitle}
            </p>
          </div>
        </div>

        {/* Main Value Banner */}
        {value !== '' && (
          <div style={{
            background: 'var(--bg-subtle)',
            borderRadius: '12px',
            padding: '16px 20px',
            border: '1px solid var(--border-subtle)',
            display: 'flex',
            alignItems: 'baseline',
            justifyContent: 'space-between',
            marginBottom: '20px'
          }}>
            <div>
              <span style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px', display: 'block', marginBottom: '2px' }}>
                Current Observed Value
              </span>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: '6px' }}>
                <span style={{ fontSize: '32px', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
                  {value}
                </span>
                {unit && (
                  <span style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-muted)' }}>
                    {unit}
                  </span>
                )}
              </div>
            </div>

            {trend && (
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                fontSize: '12px',
                fontWeight: 700,
                color: trend === 'up' ? 'var(--emerald)' : trend === 'down' ? 'var(--crimson)' : 'var(--cyan)',
                background: trend === 'up' ? 'rgba(16, 185, 129, 0.12)' : trend === 'down' ? 'rgba(239, 68, 68, 0.12)' : 'rgba(6, 182, 212, 0.12)',
                padding: '6px 12px',
                borderRadius: '8px',
                border: `1px solid ${trend === 'up' ? 'rgba(16, 185, 129, 0.3)' : trend === 'down' ? 'rgba(239, 68, 68, 0.3)' : 'rgba(6, 182, 212, 0.3)'}`
              }}>
                {trend === 'up' ? <TrendUp size={16} weight="bold" /> : trend === 'down' ? <TrendDown size={16} weight="bold" /> : <ShieldCheck size={16} weight="bold" />}
                <span>{trend === 'up' ? 'Bullish Factor' : trend === 'down' ? 'Bearish Pressure' : 'Calibrated'}</span>
              </div>
            )}
          </div>
        )}

        {/* Micro Metrics Breakdown Grid */}
        {metrics && metrics.length > 0 && (
          <div style={{
            display: 'grid',
            gridTemplateColumns: `repeat(auto-fit, minmax(130px, 1fr))`,
            gap: '10px',
            marginBottom: '20px'
          }}>
            {metrics.map((m, idx) => (
              <div
                key={idx}
                style={{
                  background: 'var(--bg-subtle)',
                  padding: '12px',
                  borderRadius: '10px',
                  border: '1px solid var(--border-subtle)'
                }}
              >
                <span style={{ fontSize: '10px', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>
                  {m.label}
                </span>
                <span style={{
                  fontSize: '15px',
                  fontWeight: 800,
                  fontFamily: 'var(--font-mono)',
                  color: m.color || 'var(--text-primary)'
                }}>
                  {m.value}
                </span>
              </div>
            ))}
          </div>
        )}

        {/* Mathematical Formulation */}
        {formula && (() => {
          let mathHtml = '';
          try {
            mathHtml = katex.renderToString(formula, {
              displayMode: true,
              throwOnError: false,
            });
          } catch (e) {
            mathHtml = '';
          }

          return (
            <div style={{
              background: 'var(--bg-card)',
              border: '1px solid var(--border-subtle)',
              borderRadius: '10px',
              padding: '16px 20px',
              marginBottom: '16px'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '10px', fontSize: '11px', fontWeight: 700, color: 'var(--cyan)' }}>
                <Cpu size={15} weight="duotone" />
                <span>MATHEMATICAL DEFINITION / FORMULATION</span>
              </div>
              <div
                style={{
                  fontSize: '17px',
                  padding: '14px 18px',
                  background: 'var(--bg-abyss)',
                  borderRadius: '10px',
                  border: '1px solid var(--border-subtle)',
                  color: 'var(--text-primary)',
                  overflowX: 'auto',
                  marginBottom: '10px',
                  textAlign: 'center',
                  lineHeight: '1.6'
                }}
                dangerouslySetInnerHTML={{ __html: mathHtml || formula }}
              />
              {formulaExplanation && (
                <p style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: '1.6', margin: 0 }}>
                  {formulaExplanation}
                </p>
              )}
            </div>
          );
        })()}

        {/* Impact on Dynamic Graph Transformer */}
        {impactOnModel && (
          <div style={{
            background: 'linear-gradient(135deg, rgba(6, 182, 212, 0.08), rgba(99, 102, 241, 0.05))',
            border: '1px solid ',
            borderRadius: '10px',
            padding: '14px 18px',
            marginBottom: '16px'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px', fontSize: '12px', fontWeight: 700, color: 'var(--cyan)' }}>
              <Scales size={16} weight="duotone" />
              <span>Role in Neural Model & Spatial Graph Attention:</span>
            </div>
            <p style={{ fontSize: '12px', color: 'var(--text-primary)', lineHeight: '1.6', margin: 0 }}>
              {impactOnModel}
            </p>
          </div>
        )}

        {/* Institutional & Academic Significance */}
        {academicContext && (
          <div style={{
            background: 'var(--bg-subtle)',
            border: '1px solid var(--border-subtle)',
            borderRadius: '10px',
            padding: '14px 18px',
            marginBottom: '20px'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px', fontSize: '12px', fontWeight: 700, color: 'var(--text-primary)' }}>
              <ShieldCheck size={16} weight="fill" color="var(--emerald)" />
              <span>Academic Defense / Research Grounding:</span>
            </div>
            <p style={{ fontSize: '11px', color: 'var(--text-secondary)', lineHeight: '1.6', margin: 0 }}>
              {academicContext}
            </p>
          </div>
        )}

        {/* Action Button */}
        <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
          <button
            onClick={onClose}
            className="btn-primary"
            style={{ padding: '8px 20px', fontSize: '12px' }}
          >
            <span>Close Drilldown</span>
          </button>
        </div>
      </div>
    </div>
  );
}
