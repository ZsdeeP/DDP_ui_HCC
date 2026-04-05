import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/load": "http://localhost:8000",
      "/data": "http://localhost:8000",
      "/segment": "http://localhost:8000",
      "/models": "http://localhost:8000",
    }
  }
});
