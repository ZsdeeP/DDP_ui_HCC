import { Niivue } from "/niivue/@niivue/niivue/";

let nv = new Niivue({
  show3Dcrosshair: true
  });
let currentCase = null;

async function init(){
  // Get canvas from HTML
  console.log("script loaded");
  nv.setSliceType(nv.sliceTypeAxial);

  //const currentCase = prompt("Please enter document name:", "example_01");
  const canvas = document.getElementById("viewer");

  nv.attachToCanvas(canvas);
  connectUI();

}

function connectUI() {

  const opacitySlider = document.getElementById("opacitySlider");
  const fpCheckbox = document.getElementById("toggleFP");
  const fnCheckbox = document.getElementById("toggleFN");
  const jumpButton = document.getElementById("jumpError");

  const buttonElement = document.getElementById("submitButton");
  console.log("test")
  if (buttonElement) {
    buttonElement.addEventListener("click", async (e) => {
      const caseSelector = document.getElementById("caseSelector");
      currentCase = caseSelector.value;

      await loadCase(currentCase);
      await errorMaps(currentCase);
    });
  };


  if (opacitySlider) {
    opacitySlider.addEventListener("input", (e) => {
      setOpacity(parseFloat(e.target.value));
    });
  }

  if (fpCheckbox) {
    fpCheckbox.addEventListener("change", (e) => {
      toggleFP(e.target.checked);
    });
  }

  if (fnCheckbox) {
    fnCheckbox.addEventListener("change", (e) => {
      toggleFN(e.target.checked);
    });
  }

  if (jumpButton) {
    jumpButton.addEventListener("click", () => {
      jumpToLargestError(currentCase);
    });
  }

}  
async function loadCase(caseID){

  const res = await fetch(`/load/${caseID}`);
  const data = await res.json();

  console.log("Volume URLs:");
  console.log(data.image);
  console.log(data.gt);
  console.log(data.pred);

  await nv.loadVolumes([
    { url: data.image },
    { url: data.gt, colormap: "green", opacity: 0.4 },
    { url: data.pred, colormap: "yellow", opacity: 0.4 },
  ]);

  console.log("Volumes:", nv.volumes.length);
  //loadMetrics(caseID);
}

function setOpacity(value){
  nv.setOpacity(1,value)
  nv.setOpacity(2,value)
}

function toggleFP(enabled){
  nv.setOpacity(3, enabled ? 0.9 : 0)
}

function toggleFN(enabled){
  nv.setOpacity(4, enabled ? 0.9 : 0)
}



async function errorMaps(caseID){

  const res = await fetch(`/get_error_maps/${caseID}`);
  const data = await res.json();

  console.log("FP/FN URLs:");
  console.log(data.fp);
  console.log(data.fn);

  await nv.addVolume({ url: data.fp, colormap: "red", opacity: 0.7 });
  await nv.addVolume({ url: data.fn, colormap: "blue", opacity: 0.7 });
}

async function jumpToLargestError(caseID){

  const res = await fetch(`/largest_error_slice/${caseID}`)
  const data = await res.json()

  nv.setSlice(data.slice)
}

async function loadMetrics(caseID){

  const res = await fetch(`/metrics/${caseID}`)
  const metrics = await res.json()

  console.log("Dice:", metrics.dice)
}


window.addEventListener("DOMContentLoaded", init());