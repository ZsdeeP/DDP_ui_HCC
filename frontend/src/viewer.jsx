import { useEffect, useRef, useState } from "react";
import { Niivue, NVImage } from "@niivue/niivue";
import "./Viewer.css";

export default function Viewer() {

  const canvasRef = useRef(null);
  const nvRef = useRef(null);
  const volumeCache = useRef({});

  const [caseID, setCaseID] = useState("");
  const [opacity, setOpacityValue] = useState(0.4);
  const [selectedModel, setSelectedModel] = useState("model1");
  const [availableModels, setAvailableModels] = useState([]);
  const [radiomicsResult, setRadiomicsResult] = useState(null);
  const [taceResult, setTaceResult] = useState(null);
  const [survivalResult, setSurvivalResult] = useState(null);
  const [outcomeResult, setOutcomeResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    const nv = new Niivue({
      show3Dcrosshair: true
    });
    nvRef.current = nv;
    nv.attachToCanvas(canvasRef.current);
    nv.setSliceType(nv.sliceTypeMultiplanar);
    nv.opts.multiplanarShowRender = "never";

    // Fetch available models
    fetchAvailableModels();

    console.log("Niivue initialized");

  }, []);

  async function fetchAvailableModels() {
    try {
      const res = await fetch(`/models`);
      const data = await res.json();
      setAvailableModels(data || []);
      if (data && data.length > 0) {
        setSelectedModel(data[0]);
      }
    } catch (err) {
      console.error("Error fetching models:", err);
    }
  }

  async function getVolume(url, colormap, opacity){

    if(volumeCache[url]){
      return volumeCache[url]
    }

    const img = await NVImage.loadFromUrl({url:url, colorMap:colormap, opacity:opacity})
    volumeCache[url] = img
    return img
  }

  async function loadCase() {

    const res = await fetch(`/load/${caseID}?model_name=${selectedModel}`);
    const data = await res.json();

    console.log("Volume URLs:");
    console.log(data.image, data.gt);

    nvRef.current.removeAllVolumes;
    
    const volumes = [
      { url: data.image, colorMap: 'gray' },
      { url: data.gt, colorMap: 'blue', opacity: opacity }
    ];
    
    return volumes
  }

  async function segmentWithModel() {
    const res = await fetch(`/segment/${caseID}?model_name=${selectedModel}`);
    const data = await res.json();

    console.log("Segmentation URL:");
    console.log(data.output_path);

    return { url: data.output_path, colorMap: 'hot', opacity: opacity };
  }

  async function extractRadiomics() {
    setIsLoading(true);
    try {
      const res = await fetch(`/extract_radiomics/${caseID}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tissue_class: 2 }) // Default to tumor
      });
      const data = await res.json();
      setRadiomicsResult(data);
    } catch (err) {
      console.error("Error extracting radiomics:", err);
      setRadiomicsResult({ error: err.message });
    } finally {
      setIsLoading(false);
    }
  }

  async function planTACE() {
    setIsLoading(true);
    try {
      const res = await fetch(`/tace_plan/${caseID}`, {
        method: 'POST'
      });
      const data = await res.json();
      setTaceResult(data);
    } catch (err) {
      console.error("Error planning TACE:", err);
      setTaceResult({ error: err.message });
    } finally {
      setIsLoading(false);
    }
  }

  async function runSurvivalAnalysis() {
    setIsLoading(true);
    try {
      const res = await fetch(`/run_survival_analysis`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ clinical_csv: 'clinical.csv', radiomics_dir: 'radiomics', output_dir: 'survival' })
      });
      const data = await res.json();
      setSurvivalResult(data);
    } catch (err) {
      console.error("Error running survival analysis:", err);
      setSurvivalResult({ error: err.message });
    } finally {
      setIsLoading(false);
    }
  }

  async function runOutcomePrediction() {
    setIsLoading(true);
    try {
      const res = await fetch(`/run_outcome_prediction`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ clinical_csv: 'clinical.csv', radiomics_dir: 'radiomics', output_dir: 'outcome_prediction' })
      });
      const data = await res.json();
      setOutcomeResult(data);
    } catch (err) {
      console.error("Error running outcome prediction:", err);
      setOutcomeResult({ error: err.message });
    } finally {
      setIsLoading(false);
    }
  }

  function setOpacity(value) {
    setOpacityValue(value);

    nvRef.current.setOpacity(1, value);
    nvRef.current.setOpacity(2, value);
  }


  function resetSlices() {

    const nv = nvRef.current;

    if (!nv || nv.volumes.length === 0) return;

    const vol = nv.volumes[0];

    const cx = vol.dims[1];
    const cy = vol.dims[2];
    const cz = vol.dims[3];

    nv.moveCrosshairInVox(cx, cy, cz);
    nv.moveCrosshairInVox(-cx/2, -cy/2, -cz/2);
  }

  async function handleSubmit() {
    try {
    
    const volumes = await loadCase();
    const seg = await segmentWithModel();
    volumes.push(seg);

    await nvRef.current.loadVolumes(volumes);
    
    // Ensure overlays are visible with proper opacity
    if (nvRef.current.volumes.length >= 2) {
      nvRef.current.setOpacity(1, opacity);  // Ground truth
    }
    if (nvRef.current.volumes.length >= 3) {
      nvRef.current.setOpacity(2, opacity);  // Segmentation
    }

   }catch (err) {
      console.error("Error loading case:", err);
    }
  }

  return (
    <div>

      <h2>Segmentation Viewer</h2>

      <div className="viewer-container">

      <canvas ref={canvasRef} width={1000} height={800}></canvas>

        <div className="view-label axial">Axial</div>
        <div className="view-label sagittal">Sagittal</div>
        <div className="view-label coronal">Coronal</div>

      </div>

      <br />

      <label>Opacity</label>
      <input
        type="range"
        min="0"
        max="1"
        step="0.05"
        value={opacity}
        onChange={(e) => setOpacity(parseFloat(e.target.value))}
      />

      <br /><br /> 

      <label>Select Model:</label>
      <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
        {availableModels.map((model) => (
          <option key={model} value={model}>
            {model}
          </option>
        ))}
      </select>

      <br /><br />

      <input
        type="text"
        placeholder="Enter image case id"
        value={caseID}
        onChange={(e) => setCaseID(e.target.value)}
      />

      <button onClick={handleSubmit}>
        Submit
      </button>

      <button onClick={resetSlices}>
        Reset Slices
      </button>

      <br /><br />

      <button onClick={extractRadiomics} disabled={isLoading || !caseID}>
        Extract Radiomics
      </button>

      <button onClick={planTACE} disabled={isLoading || !caseID}>
        Plan TACE
      </button>

      <button onClick={runSurvivalAnalysis} disabled={isLoading}>
        Run Survival Analysis
      </button>

      <button onClick={runOutcomePrediction} disabled={isLoading}>
        Run Outcome Prediction
      </button>

      {isLoading && <p>Loading...</p>}

      {survivalResult && (
        <div>
          <h3>Survival Analysis Results</h3>
          {survivalResult.error ? (
            <p>Error: {survivalResult.error}</p>
          ) : (
            <div>
              <p>Status: {survivalResult.status}</p>
              <p>Output Directory: {survivalResult.output_dir}</p>
              {survivalResult.files && survivalResult.files.length > 0 ? (
                <div>
                  <h4>Generated Files</h4>
                  <ul>
                    {survivalResult.files.map((file) => (
                      <li key={file}><a href={file} target="_blank" rel="noopener noreferrer">{file}</a></li>
                    ))}
                  </ul>
                </div>
              ) : (
                <p>No files returned.</p>
              )}
            </div>
          )}
        </div>
      )}

      {outcomeResult && (
        <div>
          <h3>Outcome Prediction Results</h3>
          {outcomeResult.error ? (
            <p>Error: {outcomeResult.error}</p>
          ) : (
            <div>
              <p>Status: {outcomeResult.status}</p>
              <p>Output Directory: {outcomeResult.output_dir}</p>
              {outcomeResult.files && outcomeResult.files.length > 0 ? (
                <div>
                  <h4>Generated Files</h4>
                  <ul>
                    {outcomeResult.files.map((file) => (
                      <li key={file}><a href={file} target="_blank" rel="noopener noreferrer">{file}</a></li>
                    ))}
                  </ul>
                </div>
              ) : (
                <p>No files returned.</p>
              )}
            </div>
          )}
        </div>
      )}

      {radiomicsResult && (
        <div>
          <h3>Radiomics Results</h3>
          {radiomicsResult.error ? (
            <p>Error: {radiomicsResult.error}</p>
          ) : (
            <pre>{JSON.stringify(radiomicsResult, null, 2)}</pre>
          )}
        </div>
      )}

      {taceResult && (
        <div>
          <h3>TACE Planning Results</h3>
          {taceResult.error ? (
            <p>Error: {taceResult.error}</p>
          ) : (
            <div>
              <p>Case ID: {taceResult.case_id}</p>
              <p>Report Path: <a href={taceResult.report_path} target="_blank" rel="noopener noreferrer">View Report</a></p>
              <p>Tumor Volume: {taceResult.summary?.tumor_volume_ml} ml</p>
              <p>Feeding Vessels: {taceResult.summary?.feeding_vessels}</p>
              <p>TACE Feasibility: {taceResult.summary?.tace_feasibility}</p>
              {taceResult.visualizations && (
                <div>
                  <h4>Visualizations</h4>
                  {Object.entries(taceResult.visualizations).map(([key, url]) => (
                    <p key={key}><a href={url} target="_blank" rel="noopener noreferrer">{key}</a></p>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}
      </div>
  );
}