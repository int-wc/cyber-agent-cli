import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  clearScreen: false,
  build: {
    sourcemap: false,
  },
  server: {
    port: 1420,
    strictPort: true,
    watch: { ignored: ["**/dist/**"] },
  },
});
