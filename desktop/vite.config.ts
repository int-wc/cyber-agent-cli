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
    // Monaco 已延迟加载并拆分 worker，剩余基础编辑器包属于预期体积。
    chunkSizeWarningLimit: 3000,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes("node_modules")) return undefined;
          // Monaco 体积较大，单独拆包以免拖大首屏主 chunk。
          if (id.includes("monaco-editor") || id.includes("@monaco-editor")) {
            return "vendor-monaco";
          }
          if (id.includes("xterm")) {
            return "vendor-terminal";
          }
          if (id.includes("react") || id.includes("scheduler")) {
            return "vendor-react";
          }
          return "vendor";
        },
      },
    },
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
