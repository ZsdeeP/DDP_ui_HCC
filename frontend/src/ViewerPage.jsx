// ViewerPage.jsx (or wherever viewer.jsx lives)
import { useState, useRef } from 'react';
import Viewer from './viewer';
import EvaluationDashboard from './EvaluationDashboard';

function TabBar({ activeTab, onChange }) {
  const tabs = [
    { id: 'viewer',     label: 'Viewer' },
    { id: 'evaluation', label: 'Evaluation' },
  ];
  return (
    <div style={{ display: 'flex', gap: 4, borderBottom: '1px solid #e5e7eb', padding: '0 16px' }}>
      {tabs.map(t => (
        <button
          key={t.id}
          onClick={() => onChange(t.id)}
          style={{
            padding: '10px 18px',
            fontSize: 13,
            background: 'none',
            border: 'none',
            borderBottom: activeTab === t.id ? '2px solid #111' : '2px solid transparent',
            color: activeTab === t.id ? '#111' : '#888',
            cursor: 'pointer',
            fontWeight: activeTab === t.id ? 500 : 400,
          }}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}


export default function ViewerPage() {
  const [caseID, setcaseID]               = useState('');
  const [structure, setStructure]         = useState('liver');
  const [sliceIdx, setSliceIdx]           = useState(0);
  const [activeTab, setActiveTab]         = useState('viewer');
  const [selectedModel, setSelectedModel] = useState("model1");
  const [metrics, setMetrics]             = useState(null);  // ← add
  const nvRef                             = useRef(null);
  const loadErmapRef                      = useRef(null);
  

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', 'overflow': 'hidden'}}>
      <TabBar activeTab={activeTab} onChange={setActiveTab} />
      <div style={{
      visibility: activeTab === 'viewer' ? 'visible' : 'hidden',
      height: activeTab === 'viewer' ? '100%' : '0',
      overflow: 'auto',
      flex: activeTab === 'viewer' ? 1 : 0
        }}>
        <Viewer
          caseID={caseID}
          setcaseID={setcaseID}
          structure={structure}
          //setStructure={setStructure}
          onSliceChange={setSliceIdx}
          nvRef={nvRef}
          loadErmapRef={loadErmapRef}
          selectedModel={selectedModel}
          setSelectedModel={setSelectedModel}
          onMetricsLoaded={setMetrics}       // ← pass setter down
          setActivePageTab={setActiveTab}
        />
      </div>
      
      {activeTab === 'evaluation' && (
      <div style={{ flex: 1, overflowY: 'auto', minHeight: 0 }}> 
      <EvaluationDashboard
          caseID={caseID}
          structure={structure}
          setStructure={setStructure}
          sliceIdx={sliceIdx}
          loadErmap={loadErmapRef}
          //selectedModel={selectedModel}
          metrics={metrics}       
          //setActivePageTab={setActiveTab}           // ← pass data down
      />
      </div>
  )}
  </div>

  );}
