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
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    const nv = new Niivue({
      show3Dcrosshair: true
    });

    nv.attachToCanvas(canvasRef.current);
    nv.setSliceType(nv.sliceTypeMultiplanar);
    nv.opts.multiplanarShowRender = "never";

    nvRef.current = nv;

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
    console.log(data.image, data.gt, data.pred);

    nvRef.current.removeAllVolumes;
    const volumes = [
      { url: data.image },
      { url: data.gt, colormap: "blue", opacity: opacity },
      { url: data.pred, colormap: "gold", opacity: opacity },
    ];

    // Add imageType for DICOM
    volumes.forEach(vol => {
      if (vol.url.endsWith('.dcm')) {
        vol.imageType = "DCM";
      }
    });

    await nvRef.current.loadVolumes(volumes);

    console.log("Volumes:", nvRef.current.volumes.length);
  }

  async function segmentWithModel() {

    const res = await fetch(`/segment/${caseID}?model_name=${selectedModel}`);
    const data = await res.json();

    console.log("Segmentation URL:");
    console.log(data.output_path);

    const predImage = await getVolume(data.output_path, "gold", opacity)

    nvRef.current.addVolume(predImage);

    console.log("Volumes:", nvRef.current.volumes.length);
  }

  async function extractRadiomics() {
    setIsLoading(true);
    try {
      const res = await fetch(`/extract-radiomics/${caseID}`, {
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
      const res = await fetch(`/tace-plan/${caseID}`, {
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
    nv.moveCrosshairInVox(-cx/2, -cy/2, -cz/2)
  }

  async function handleSubmit() {
    try {
    await loadCase();
    await segmentWithModel();
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

    </div>
  );
}