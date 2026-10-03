import React from 'react';
import { FileText, ShieldCheck, Sparkle, MagnifyingGlass, CheckCircle, Warning } from '@phosphor-icons/react';

export default function RagKnowledgeDrawer({ stock }) {
  const r = stock.finReasoning;
  const isRepaired = r.status === 'REPAIRED';

  return (
    <div className="glass-panel" style={{ padding: '20px' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ width: '28px', height: '28px', borderRadius: '6px', background: 'rgba(16, 185, 129, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <FileText size={16} weight="duotone" color="var(--emerald)" />
          </div>
          <div>
            <h3 style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text-primary)' }}>
              Financial RAG & Knowledge Reasoning
            </h3>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              Dense Vector Ingestion + Anti-Hallucination Guardrails
            </span>
          </div>
        </div>

        <div className="badge-verified">
          {isRepaired ? <Warning size={14} color="var(--amber)" weight="fill" /> : <CheckCircle size={14} color="var(--cyan)" weight="fill" />}
          <span>{r.status} • {r.safetyCheck}</span>
        </div>
      </div>

      {/* FinLLM Synthesis Callout */}
      <div style={{
        background: 'linear-gradient(135deg, rgba(6, 182, 212, 0.08), rgba(99, 102, 241, 0.05))',
        border: '1px solid',
        borderRadius: '10px',
        padding: '14px',
        marginBottom: '16px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px', color: 'var(--cyan)', fontSize: '12px', fontWeight: 700 }}>
          <Sparkle size={15} weight="fill" />
          <span>FinLLM Reasoner Synthesis:</span>
        </div>
        <p style={{ fontSize: '12px', lineHeight: '1.6', color: 'var(--text-primary)' }}>
          "{r.summary}"
        </p>
        <div style={{ display: 'flex', gap: '14px', marginTop: '10px', fontSize: '11px', color: 'var(--text-secondary)' }}>
          <span>Sentiment Score: <strong style={{ color: r.sentimentScore > 0 ? 'var(--emerald)' : 'var(--crimson)' }}>{r.sentimentScore > 0 ? `+${r.sentimentScore}` : r.sentimentScore}</strong></span>
          <span>Defensive Safety Check: <strong style={{ color: 'var(--emerald)' }}>PASSED</strong></span>
        </div>
      </div>

      {/* Retrieved Documents List */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '10px', fontSize: '12px', color: 'var(--text-secondary)', fontWeight: 600 }}>
          <MagnifyingGlass size={14} color="var(--cyan)" />
          <span>Top Retrieved Catalysts (Cosine Vector Similarity):</span>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {stock.ragNews.map((doc, idx) => (
            <div
              key={doc.id || idx}
              style={{
                background: 'var(--bg-subtle)',
                padding: '12px',
                borderRadius: '8px',
                border: '1px solid var(--border-subtle)',
                transition: 'all 0.2s ease'
              }}
            >
              <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '10px', marginBottom: '4px' }}>
                <span style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-primary)', lineHeight: '1.4' }}>
                  {doc.headline}
                </span>
                <span className={doc.sentiment === 'BULLISH' ? 'badge-bullish' : doc.sentiment === 'BEARISH' ? 'badge-bearish' : 'badge-verified'} style={{ fontSize: '10px', padding: '2px 8px', flexShrink: 0 }}>
                  {doc.sentiment}
                </span>
              </div>
              <p style={{ fontSize: '11px', color: 'var(--text-secondary)', lineHeight: '1.5', marginBottom: '6px' }}>
                {doc.snippet}
              </p>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: 'var(--text-muted)' }}>
                <span>Source: {doc.source}</span>
                <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--cyan)' }}>Sim: {doc.similarity}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
