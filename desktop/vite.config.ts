import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  base: "./",
  root: ".",
  build: {
    outDir: "dist/renderer",
    target: "esnext",
    emptyOutDir: true,
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src/renderer"),
    },
  },
  server: {
    port: 5173,
    strictPort: true,
  },
});
