// EvaluationDashboard.jsx
import { useState, useEffect, useRef } from 'react';
import { Chart } from 'chart.js/auto';

const STRUCTURES = ['liver', 'tumor', 'bloodvessels', 'abdominalaorta'];

export default function EvaluationDashboard({ caseId, structure, setStructure, sliceIdx }) {
  const [metrics, setMetrics]       = useState(null);
  const [baseline, setBaseline]     = useState(null);
  const [loading, setLoading]       = useState(false);
  const [activeTab, setActiveTab]   = useState('metrics');
  const diceChartRef                = useRef(null);
  const diceChartInstance           = useRef(null);

  // Fetch metrics whenever caseId or structure changes
  useEffect(() => {
    if (!caseId) return;
    setLoading(true);
    Promise.all([
      fetch(`/api/evaluate/${caseId}/${structure}`).then(r => r.json()),
      fetch(`/api/evaluate/${caseId}/${structure}?model=unet`).then(r => r.json()),
    ])
      .then(([m, b]) => { setMetrics(m); setBaseline(b); })
      .finally(() => setLoading(false));
  }, [caseId, structure]);

  // Rebuild Dice chart when metrics arrive
  useEffect(() => {
    if (!metrics || !diceChartRef.current) return;
    if (diceChartInstance.current) diceChartInstance.current.destroy();
    diceChartInstance.current = new Chart(diceChartRef.current, {
      type: 'bar',
      data: {
        labels: metrics.per_case_dice.map((_, i) => `C${i + 1}`),
        datasets: [{
          data: metrics.per_case_dice,
          backgroundColor: metrics.per_case_dice.map(v =>
            v >= 0.9 ? '#1D9E75cc' : v >= 0.8 ? '#EF9F27cc' : '#E24B4Acc'
          ),
          borderRadius: 3,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          y: { min: 0.6, max: 1.0 },
          x: { grid: { display: false } }
        }
      }
    });
  }, [metrics]);

  if (!caseId) return (
    <div style={{ padding: 40, color: 'var(--color-text-secondary)' }}>
      Load a case in the viewer first.
    </div>
  );

  if (loading) return <div style={{ padding: 40 }}>Computing metrics…</div>;

  return (
    <div style={{ padding: 20, overflowY: 'auto', height: '100%' }}>
      {/* Structure selector — synced with viewer */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 16 }}>
        {STRUCTURES.map(s => (
          <button
            key={s}
            onClick={() => setStructure(s)}
            style={{
              padding: '4px 14px',
              borderRadius: 20,
              border: '0.5px solid var(--color-border-secondary)',
              background: structure === s ? 'var(--color-text-primary)' : 'transparent',
              color: structure === s ? 'var(--color-background-primary)' : 'var(--color-text-secondary)',
              cursor: 'pointer',
              fontSize: 13,
            }}
          >
            {s.charAt(0).toUpperCase() + s.slice(1)}
          </button>
        ))}
      </div>

      {/* Inner tab bar */}
      <div style={{ display: 'flex', gap: 4, borderBottom: '0.5px solid var(--color-border-tertiary)', marginBottom: 20 }}>
        {['metrics', 'compare', 'boundary'].map(t => (
          <button
            key={t}
            onClick={() => setActiveTab(t)}
            style={{
              padding: '8px 16px',
              fontSize: 13,
              background: 'none',
              border: 'none',
              borderBottom: activeTab === t ? '2px solid var(--color-text-primary)' : '2px solid transparent',
              color: activeTab === t ? 'var(--color-text-primary)' : 'var(--color-text-secondary)',
              cursor: 'pointer',
              fontWeight: activeTab === t ? 500 : 400,
            }}
          >
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {activeTab === 'metrics' && metrics && (
        <>
          {/* KPI cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 10, marginBottom: 20 }}>
            {[
              { label: 'Dice', value: metrics.dice.toFixed(3), delta: baseline ? (metrics.dice - baseline.dice).toFixed(3) : null },
              { label: 'IoU',  value: metrics.iou.toFixed(3) },
              { label: 'HD95', value: `${metrics.hd95.toFixed(1)} mm` },
            ].map(({ label, value, delta }) => (
              <div key={label} style={{ background: 'var(--color-background-secondary)', borderRadius: 8, padding: '12px 14px' }}>
                <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)', marginBottom: 4 }}>{label}</div>
                <div style={{ fontSize: 22, fontWeight: 500 }}>{value}</div>
                {delta && (
                  <div style={{ fontSize: 11, color: parseFloat(delta) > 0 ? 'var(--color-text-success)' : 'var(--color-text-danger)' }}>
                    vs U-Net {parseFloat(delta) > 0 ? '+' : ''}{delta}
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Slice indicator — uses sliceIdx from NiiVue */}
          <div style={{ fontSize: 12, color: 'var(--color-text-tertiary)', marginBottom: 8 }}>
            Viewer slice: {sliceIdx} 
            {metrics.per_slice_dice?.[sliceIdx] !== undefined && (
              <span style={{ marginLeft: 8, color: 'var(--color-text-primary)', fontWeight: 500 }}>
                Dice @ this slice: {metrics.per_slice_dice[sliceIdx].toFixed(3)}
              </span>
            )}
          </div>

          <div style={{ position: 'relative', height: 180 }}>
            <canvas ref={diceChartRef} />
          </div>
        </>
      )}

      {activeTab === 'compare' && metrics && baseline && (
        <ComparePanel metrics={metrics} baseline={baseline} structure={structure} />
      )}

      {activeTab === 'boundary' && (
        <BoundaryPanel caseId={caseId} structure={structure} sliceIdx={sliceIdx} />
      )}
    </div>
  );
}