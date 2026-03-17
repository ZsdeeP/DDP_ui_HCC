import { useEffect, useRef, useState } from "react";
import { Niivue, NVImage } from "@niivue/niivue";
import "./Viewer.css";

let fpIndex = null;
let fnIndex = null;

export default function Viewer() {

  const canvasRef = useRef(null);
  const nvRef = useRef(null);
  const volumeCache = useRef({});

  const [caseID, setCaseID] = useState("");
  const [opacity, setOpacityValue] = useState(0.4);
  const [showFP, setShowFP] = useState(true);
  const [showFN, setShowFN] = useState(true);

  useEffect(() => {
    const nv = new Niivue({
      show3Dcrosshair: true
    });

    nv.attachToCanvas(canvasRef.current);
    nv.setSliceType(nv.sliceTypeMultiplanar);
    nv.opts.multiplanarShowRender = "never";

    nvRef.current = nv;

    console.log("Niivue initialized");

  }, []);

  async function getVolume(url, colormap, opacity){

    if(volumeCache[url]){
      return volumeCache[url]
    }

    const img = await NVImage.loadFromUrl({url:url, colorMap:colormap, opacity:opacity})
    volumeCache[url] = img
    return img
  }

  async function loadCase() {

    const volumeCache = {}
    const res = await fetch(`/load/${caseID}`);
    const data = await res.json();

    console.log("Volume URLs:");
    console.log(data.image, data.gt, data.pred);

    nvRef.current.removeAllVolumes;
    await nvRef.current.loadVolumes([
      { url: data.image },
      { url: data.gt, colormap: "gold", opacity: opacity },
      { url: data.pred, colormap: "green", opacity: opacity },
    ]);

    console.log("Volumes:", nvRef.current.volumes.length);
  }

  async function errorMaps() {
    
    const res = await fetch(`/error/${caseID}`);
    const data = await res.json();
    const fpImage = await getVolume(data.fp, "red", 0.7)
    const fnImage = await getVolume(data.fn, "blue", 0.7)


    nvRef.current.addVolume(fpImage);
    nvRef.current.addVolume(fnImage);

    console.log("Volumes:", nvRef.current.volumes.length);
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

  function toggleFP(enabled){
    if (fpIndex !== null){
        nv.setOpacity(fpIndex, enabled ? 0.9 : 0)
    }
  }

  function toggleFN(enabled){
    if (fnIndex !== null){
        nv.setOpacity(fnIndex, enabled ? 0.9 : 0)
    }
  }

  async function handleSubmit() {
    try {
    await loadCase();
    await errorMaps();
   }catch (err) {
      console.error("Error loading case:", err);
    }
  }

  async function jumpToLargestError() {

    const res = await fetch(`/largest_error_slice/${caseID}`);
    const data = await res.json();

    nvRef.current.setSlice(data.slice);
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

      <label>
        <input
          type="checkbox"
          checked={showFP}
          onChange={(e) => toggleFP(e.target.checked)}
        />
        Show False Positives
      </label>

        <br />

      <label>
        <input
          type="checkbox"
          checked={showFN}
          onChange={(e) => toggleFN(e.target.checked)}
        />
        Show False Negatives
      </label>      

      <br /><br />

      <button onClick={jumpToLargestError}>
        Jump to Largest Error
      </button>

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