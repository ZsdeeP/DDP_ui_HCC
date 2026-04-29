// EvaluationDashboard.jsx
import { useState, useEffect, useRef } from 'react';
import { Chart } from 'chart.js/auto'; 


//npm install chart.js, npm install @niivue/nivvue @niivue/dicom-loader, npm install react-router-dom

const STRUCTURES = ['liver', 'tumor', 'bloodvessels', 'abdominalaorta'];


export default function EvaluationDashboard({ caseID, structure, setStructure, sliceIdx, loadErmap, metrics }) {
  //const [baseline, setBaseline]     = useState(null);
  const [activeDashTab, setactiveDashTab]   = useState('metrics');
  const diceChartRef                = useRef(null);
  const diceChartInstance           = useRef(null);
  const iouChartRef                 = useRef(null);
  const iouChartInstance            = useRef(null);
  const [ermapStatus, setErmapStatus] = useState('idle'); // 'idle' | 'loading' | 'done' | 'error'

  // Fetch metrics whenever caseID or structure changes
  // Build per-slice Dice chart

  useEffect(() => {
      setErmapStatus('idle');
    }, [structure]);

  useEffect(() => {
    if (!metrics?.per_slice_dice || !diceChartRef.current || activeDashTab !== 'metrics') return;
    try{
    if (diceChartInstance.current) diceChartInstance.current.destroy();
    diceChartInstance.current = new Chart(diceChartRef.current, {
      type: 'line',
      data: {
        labels: metrics.per_slice_dice.map((_, i) => i),
        datasets: [{
          label: 'Dice',
          data: metrics.per_slice_dice,
          borderColor: '#1D9E75',
          backgroundColor: '#1D9E7522',
          pointRadius: 0,
          fill: true,
          tension: 0.3,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          // Vertical line at current slice
          annotation: {
            annotations: {
              sliceLine: {
                type: 'line',
                xMin: sliceIdx,
                xMax: sliceIdx,
                borderColor: '#E24B4A',
                borderWidth: 1.5,
              }
            }
          }
        },
        scales: {
          y: { min: 0, max: 1, ticks: { font: { size: 11 } } },
          x: { ticks: { font: { size: 11 }, maxTicksLimit: 10 } }
        } 
    }});
    } catch (e) {
    console.error('Chart error',  e);  // this will tell us exactly what's failing
  }
}, [metrics, activeDashTab, sliceIdx]);

    
    // Update slice indicator line without rebuilding chart
  useEffect(() => {
    if (!diceChartInstance.current) return;
    diceChartInstance.current.update('none');
    const ann = diceChartInstance.current.options.plugins?.annotation?.annotations?.sliceLine;
    if (ann) {
      ann.xMin = sliceIdx;
      ann.xMax = sliceIdx; 
    }

  }, [sliceIdx]);

  // Build per-slice IoU chart
  useEffect(() => {
    if (!metrics?.per_slice_iou || !iouChartRef.current || activeDashTab !== 'metrics') return;
    if (iouChartInstance.current) iouChartInstance.current.destroy();
    iouChartInstance.current = new Chart(iouChartRef.current, {
      type: 'line',
      data: {
        labels: metrics.per_slice_iou.map((_, i) => i),
        datasets: [{
          label: 'IoU',
          data: metrics.per_slice_iou,
          borderColor: '#378ADD',
          backgroundColor: '#378ADD22',
          pointRadius: 0,
          fill: true,
          tension: 0.3,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          y: { min: 0, max: 1, ticks: { font: { size: 11 } } },
          x: { ticks: { font: { size: 11 }, maxTicksLimit: 10 } }
        }
      }
    });
  }, [metrics, activeDashTab]);

    async function handleLoadErmap() {
    setErmapStatus('loading');
    try {
      await loadErmap.current?.();
      setErmapStatus('done');
    } catch (e) {
      setErmapStatus('error', e);
    }
  }
  

  if (!caseID) return (
    <div style={{ padding: 40, color: '#888' }}>
      {!caseID ? 'Load a case in the viewer first.' : 'Run load in the viewer to fetch metrics.'}
    </div>
  );
  //if (!metrics)  return null;

  return (
  <div style={{ padding: 20 }}>

    {/* Structure selector */}
    <div style={{ display: 'flex', gap: 6, marginBottom: 16 }}>
      {STRUCTURES.map(s => (
        <button key={s} onClick={() => setStructure(s)} style={{
          padding: '4px 14px', borderRadius: 20, cursor: 'pointer', fontSize: 13,
          border: '0.5px solid #ccc',
          background: structure === s ? '#111' : 'transparent',
          color: structure === s ? '#fff' : '#888',
        }}>
          {s.charAt(0).toUpperCase() + s.slice(1)}
        </button>
      ))}
    </div>

    {/* Inner tabs — ONE copy only */}
    <div style={{ display: 'flex', gap: 4, borderBottom: '1px solid #e5e7eb', marginBottom: 20 }}>
      {['metrics', 'boundary'].map(t => (
        <button key={t} onClick={() => setactiveDashTab(t)} style={{
          padding: '8px 16px', fontSize: 13, background: 'none', border: 'none', cursor: 'pointer',
          borderBottom: activeDashTab === t ? '2px solid #111' : '2px solid transparent',
          color: activeDashTab === t ? '#111' : '#888',
          fontWeight: activeDashTab === t ? 500 : 400,
        }}>
          {t.charAt(0).toUpperCase() + t.slice(1)}
        </button>
      ))}
    </div>

    {/* Empty states */}
    {!caseID && (
      <div style={{ color: '#888', fontSize: 13 }}>Load a case in the viewer first.</div>
    )}
    {caseID && !metrics && (
      <div style={{ color: '#888', fontSize: 13 }}>Load a case in the viewer to see metrics.</div>
    )}

    {/* Metrics tab — only when metrics exists */}
    {activeDashTab === 'metrics' && metrics && (
      <>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 10, marginBottom: 20 }}>
          {[
            { label: 'Dice', value: metrics?.dice?.toFixed(3) ?? '—' },
            { label: 'IoU',  value: metrics?.iou?.toFixed(3)  ?? '—' },
            { label: 'HD',   value: metrics?.hd != null ? `${metrics.hd.toFixed(1)} mm` : '—' },
          ].map(({ label, value }) => (
            <div key={label} style={{ background: '#f9f9f9', borderRadius: 8, padding: '12px 14px' }}>
              <div style={{ fontSize: 11, color: '#999', marginBottom: 4 }}>{label}</div>
              <div style={{ fontSize: 22, fontWeight: 500 }}>{value}</div>
            </div>
          ))}
        </div>

        <div style={{ fontSize: 12, color: '#999', marginBottom: 8 }}>
          Slice {sliceIdx}
          {metrics.per_slice_dice?.[sliceIdx] != null && (
            <span style={{ marginLeft: 8, color: '#111', fontWeight: 500 }}>
              Dice: {metrics.per_slice_dice[sliceIdx].toFixed(3)}
            </span>
          )}
          {metrics.per_slice_iou?.[sliceIdx] != null && (
            <span style={{ marginLeft: 8, color: '#111', fontWeight: 500 }}>
              IoU: {metrics.per_slice_iou[sliceIdx].toFixed(3)}
            </span>
          )}
        </div>

        <div style={{ fontSize: 12, color: '#999', marginBottom: 4 }}>Dice per slice</div>
        <div style={{ position: 'relative', height: 160, marginBottom: 20 }}>
          <canvas ref={diceChartRef} />
        </div>

        <div style={{ fontSize: 12, color: '#999', marginBottom: 4 }}>IoU per slice</div>
        <div style={{ position: 'relative', height: 160 }}>
          <canvas ref={iouChartRef} />
        </div>
      </>
    )}

    {/* Boundary tab — always show button, disable if no case */}
    {activeDashTab === 'boundary' && (
      <div>
        <p style={{ fontSize: 13, color: '#666', marginBottom: 16, lineHeight: 1.6 }}>
          Loads the boundary error map as an overlay in the viewer.
          <br />
          <span style={{ color: '#1D9E75' }}>■</span> True Positive &nbsp;
          <span style={{ color: '#D4A017' }}>■</span> False Positive &nbsp;
          <span style={{ color: '#378ADD' }}>■</span> False Negative
        </p>
        <button
      onClick={handleLoadErmap}
      disabled={!caseID || ermapStatus === 'loading'}
      style={{
        padding: '8px 20px', borderRadius: 6, cursor: 'pointer', fontSize: 13,
        background: ermapStatus === 'done' ? '#378ADD' : '#1D9E75',
        color: '#fff', border: 'none', fontWeight: 500,
        opacity: (!caseID || ermapStatus === 'loading') ? 0.5 : 1,
      }}
    >
      {ermapStatus === 'loading' ? 'Loading…' : ermapStatus === 'done' ? 'Reload in viewer' : 'Load in viewer'}
    </button>

    {/* Status messages */}
    {ermapStatus === 'loading' && (
      <p style={{ fontSize: 12, color: '#888', marginTop: 10 }}>
        Fetching error map, please wait…
      </p>
    )}
    {ermapStatus === 'done' && (
      <p style={{ fontSize: 12, color: '#1D9E75', marginTop: 10 }}>
        ✓ Loaded. Switch to the <strong>Viewer</strong> tab to inspect the overlay.
      </p>
    )}
    {ermapStatus === 'error' && (
      <p style={{ fontSize: 12, color: '#E24B4A', marginTop: 10 }}>
        ✗ Failed to load error map. Check the console for details.
      </p>
    )}

    </div>
    )}

  </div>
  );
}
  // Rebuild Dice chart when metrics arrive
  
/*
  return (
    <div style={{ padding: 20}}>
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

      <div style={{ display: 'flex', gap: 4, borderBottom: '0.5px solid var(--color-border-tertiary)', marginBottom: 20 }}>
        {['metrics', 'compare', 'boundary'].map(t => (
          <button
            key={t}
            onClick={() => setactiveDashTab(t)}
            style={{
              padding: '8px 16px',
              fontSize: 13,
              background: 'none',
              border: 'none',
              borderBottom: activeDashTab === t ? '2px solid var(--color-text-primary)' : '2px solid transparent',
              color: activeDashTab === t ? 'var(--color-text-primary)' : 'var(--color-text-secondary)',
              cursor: 'pointer',
              fontWeight: activeDashTab === t ? 500 : 400,
            }}
          >
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {!caseID && ( //Inline empty states instead of early returns
        <div style={{ color: '#888', fontSize: 13 }}>
          Load a case in the viewer first.
        </div>
      )}
      {caseID && !metrics && (
        <div style={{ color: '#888', fontSize: 13 }}>
          Load a case in the viewer to see metrics here.
        </div>
      )}

      {activeDashTab === 'metrics' && metrics && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 10, marginBottom: 20 }}>
            {[
              { label: 'Dice', value: metrics.dice.toFixed(3), delta: baseline ? (metrics.dice - baseline.dice).toFixed(3) : null },
              { label: 'IoU',  value: metrics.iou.toFixed(3) },
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

      {activeDashTab === 'compare' && metrics && baseline && (
        <ComparePanel metrics={metrics} baseline={baseline} structure={structure} />
      )}

      {activeDashTab === 'boundary' && (
        <BoundaryPanel caseID={caseID} structure={structure} sliceIdx={sliceIdx} />
      )}
    </div>
  );*/
