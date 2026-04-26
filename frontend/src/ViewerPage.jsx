// ViewerPage.jsx (or wherever viewer.jsx lives)
import { useState } from 'react';
import Viewer from './Viewer';
import EvaluationDashboard from './EvaluationDashboard';

export default function ViewerPage() {
  const [caseId, setCaseId]         = useState(null);
  const [structure, setStructure]   = useState('liver');
  const [sliceIdx, setSliceIdx]     = useState(0);       // synced from NiiVue
  const [activeTab, setActiveTab]   = useState('viewer');

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <TabBar activeTab={activeTab} onChange={setActiveTab} />
      
      {activeTab === 'viewer' && (
        <Viewer
          caseId={caseId}
          setCaseId={setCaseId}
          structure={structure}
          setStructure={setStructure}
          onSliceChange={setSliceIdx}   // fire this from NiiVue's location callback
        />
      )}
      {activeTab === 'evaluation' && (
        <EvaluationDashboard
          caseId={caseId}
          structure={structure}
          setStructure={setStructure}   // so structure tabs in dashboard sync back
          sliceIdx={sliceIdx}
        />
      )}
    </div>
  );
}