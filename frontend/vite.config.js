import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/load": "http://localhost:8000",
      "/data": "http://localhost:8000",
      "/dicom_files": "http://localhost:8000",
      "/segment": "http://localhost:8000",
      "/models": "http://localhost:8000",
      "/extract_radiomics": "http://localhost:8000",
      "/tace_plan": "http://localhost:8000",
      "/run_survival_analysis": "http://localhost:8000",
      "/run_outcome_prediction": "http://localhost:8000"
    }
  }
});
