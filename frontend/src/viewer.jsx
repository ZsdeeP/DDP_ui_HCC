import { useEffect, useRef, useState } from "react";
import { Niivue, NVImage } from "@niivue/niivue";
import { Dcm2niix } from '@niivue/dcm2niix';
import "./Viewer.css";

//create a new function that uses loadVolume to load base volume and error maps together - this loads classwise stuff - needs a dropdown option (much like the fetchAvailableModels function)
//reorganize functions as follows: load base volume and ground truth - this replaces the handleSubmit button. loads all four classes together 

export default function Viewer({ caseID, setcaseID, structure, onSliceChange, nvRef, loadErmapRef, selectedModel, setSelectedModel, onMetricsLoaded}) {
  const canvasRef = useRef(null);
  const dcm2niixRef = useRef(null);
  const volumeCache = useRef({});

  const [opacity, setOpacityValue] = useState(0.4);
  const [availableModels, setAvailableModels] = useState([]);
  const [radiomicsResult, setRadiomicsResult] = useState(null);
  const [taceResult, setTaceResult] = useState(null);
  const [survivalResult, setSurvivalResult] = useState(null);
  const [outcomeResult, setOutcomeResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");

  loadErmapRef.current = loadErmap;

  useEffect(() => {
    const nv = new Niivue({
      show3Dcrosshair: true
    });
    nvRef.current = nv;
    nv.attachToCanvas(canvasRef.current);
    nv.setSliceType(nv.sliceTypeMultiplanar);
    nv.opts.multiplanarShowRender = "never";

    const dcm2niix = new Dcm2niix();
    dcm2niix.init()
      .then(() => {
        dcm2niixRef.current = dcm2niix;
      })
      .catch((err) => {
        console.warn("Dcm2niix initialization failed:", err);
      });

    // Fetch available models
    fetchAvailableModels();

    console.log("Niivue initialized");
    if (!nvRef.current) return;
    nvRef.current.onLocationChange = (data) => {
      // data.vox is [x, y, z] in voxel space
      onSliceChange(data.vox[2]);   // z = axial slice index
    };

  }, [nvRef, onSliceChange]);

  const isDicomDir = (url) => typeof url === 'string' && url.endsWith('/');

  async function fetchDicomFileUrls(url) {
    const res = await fetch(`/dicom_files?url=${encodeURIComponent(url)}`);
    if (!res.ok) {
      throw new Error(`Unable to list DICOM series: ${res.statusText}`);
    }
    const data = await res.json();
    return data.files || [];
  }

  async function convertDicomSeries(url) {
    if (!dcm2niixRef.current) {
      throw new Error('Dcm2niix is not initialized');
    }
    setStatusMessage('Fetching DICOM series files...');
    const allFileUrls = await fetchDicomFileUrls(url);
    if (!allFileUrls.length) {
      throw new Error(`No DICOM files found at ${url}`);
    }

    // Filter strategy: try "1-" first, then "2-", then all files
    const getFileName = (fileUrl) => fileUrl.substring(fileUrl.lastIndexOf('/') + 1);
    const prefix1 = allFileUrls.filter(u => getFileName(u).startsWith('1-'));
    const prefix2 = allFileUrls.filter(u => getFileName(u).startsWith('2-'));
    
    let fileUrls;
    if (prefix1.length > 0) {
      console.log(`Using ${prefix1.length} files with prefix "1-"`);
      fileUrls = prefix1;
    } else if (prefix2.length > 0) {
      console.log(`Using ${prefix2.length} files with prefix "2-"`);
      fileUrls = prefix2;
    } else {
      console.log(`No prefix match found, loading all ${allFileUrls.length} DICOM files`);
      fileUrls = allFileUrls;
    }

    setStatusMessage('Converting DICOM series to NIfTI...');
    console.log("DICOM file URLs:", fileUrls);

    const fileObjects = await Promise.all(fileUrls.map(async (fileUrl) => {
      const res = await fetch(fileUrl);
      console.log(`Fetching DICOM: ${fileUrl}`);
      if (!res.ok) {
        throw new Error(`Failed to fetch DICOM file ${fileUrl}`);
      }
      const blob = await res.blob();
      const fileName = getFileName(fileUrl);
      const file = new File([blob], fileName, { type: blob.type || 'application/dicom' });
      file._webkitRelativePath = fileName;
      return file;
    }));

    console.log("File objects for conversion:", fileObjects);
    const convertedFiles = await dcm2niixRef.current.input(fileObjects).run();
    const niftiFile = convertedFiles.find((file) => file.name.endsWith('.nii') || file.name.endsWith('.nii.gz'));
    if (!niftiFile) {
      throw new Error('DICOM conversion produced no NIfTI output');
    }
    console.log("DICOM conversion successful, NIfTI file:", niftiFile);
    setStatusMessage('DICOM conversion complete. Loading volume...');
    return URL.createObjectURL(niftiFile);
  }

  async function getVolume(url, colormap, opacity) {
    if (volumeCache.current[url]) {
      console.log(`Volume for ${url} loaded from cache`);
      return volumeCache.current[url];
    }

    setStatusMessage(`Loading volume: ${url}`);
    console.log(`Loading volume from URL: ${url} with colormap: ${colormap} and opacity: ${opacity}`);
    let loadUrl = url;
    if (isDicomDir(url)) {
      loadUrl = await convertDicomSeries(url);
    }
    console.log(`Final URL for loading volume: ${loadUrl}`);
    let img;
    try {
      img = await NVImage.loadFromUrl({ url: loadUrl, colorMap: colormap, opacity: opacity });
    } catch (err) {
      throw new Error(`Failed to load volume from ${url}: ${err.message}`);
    }
    img.colormap = colormap;   // note: lowercase 'colormap', not 'colorMap'
    img.cal_min = 0.5;         // exclude background label 0
    img.cal_max = img.robust_max || 1.0;
    img.opacity = opacity;
    volumeCache.current[url] = img;
    return img;
  }

  async function fetchAvailableModels() {
    try {
      const res = await fetch(`/models`);
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }
      const data = await res.json();
      setAvailableModels(data || []);
      if (data && data.length > 0) {
        setSelectedModel(data[0]);
      }
    } catch (err) {
      console.error("Error fetching models:", err);
    }
  }

  async function loadCase() {
    setStatusMessage('Requesting case data...');
    const res = await fetch(`/load/${caseID}?model_name=${selectedModel}`);
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    }
    const data = await res.json();

    console.log("Volume URLs:", data.image, data.gt);
    nvRef.current.removeAllVolumes;

    setStatusMessage('Loading base image and ground truth...');
    const imageVol = await getVolume(data.image, 'gray', 1.0);
    
    nvRef.current.addVolume(imageVol);
    nvRef.current.updateGLVolume();
    console.log("Base image loaded");
    const gtVol = await getVolume(data.gt, 'blue', opacity);
    nvRef.current.addVolume(gtVol);
    nvRef.current.updateGLVolume();
    console.log("loadCase - Base image and ground truth loaded");
    //return [imageVol, gtVol];
  }

  async function loadErmap() {
    setStatusMessage('Requesting case data...');
    const res = await fetch(`/load/${caseID}?model_name=${selectedModel}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);

    const resEr = await fetch(`/evaluate_case/${caseID}/${structure}?model_name=${selectedModel}`);
    if (!resEr.ok) throw new Error(`HTTP ${resEr.status}: ${resEr.statusText}`);

    const data   = await res.json();
    const erdata = await resEr.json();

    nvRef.current.removeAllVolumes;
    setStatusMessage('Loading base image and error map...');

    const imageVol = await getVolume(data.image, 'gray', 1.0);
    nvRef.current.addVolume(imageVol);
    nvRef.current.updateGLVolume();

    const erVol = await getVolume(erdata.boundary_error_map, 'hot', opacity);
    nvRef.current.addVolume(erVol);
    nvRef.current.updateGLVolume();

    onMetricsLoaded(erdata);  // ← lift metrics up, erdata has everything
  }

  async function segmentWithModel() {
    setStatusMessage('Requesting segmentation from backend...');
    const res = await fetch(`/segment/${caseID}?model_name=${selectedModel}`);
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    }
    console.log("Segmentation created")
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
      setIsLoading(true);
      setStatusMessage('Preparing to load case...');
      console.log(`Submitting case ID: ${caseID} with model: ${selectedModel}`);
      await loadCase();
      console.log("Base image and ground truth loaded into Niivue");
      await segmentWithModel();
      //await nvRef.current.loadVolumes(volumes);
      console.log("All volumes loaded into Niivue");
      
      setStatusMessage('Case loaded successfully');
    } catch (err) {
      console.error("Error loading case:", err);
      setStatusMessage(`Error: ${err.message}`);
    } finally {
      setIsLoading(false);
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
        onChange={(e) => setcaseID(e.target.value)}
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
      {statusMessage && <p>Status: {statusMessage}</p>}

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