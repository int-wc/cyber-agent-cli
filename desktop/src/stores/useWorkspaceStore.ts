import { create } from "zustand";
import type { FileEntry, OpenTab } from "@/types/file";
import type { SidebarView, BackendState } from "@/types/workspace";

const API = () => {
  const port = useWorkspaceStore.getState().backendPort;
  return port ? `http://127.0.0.1:${port}/api` : "";
};

interface WorkspaceStore {
  // Backend
  backendPort: number | null;
  backendStatus: BackendState["status"];
  setBackendPort: (port: number) => void;
  setBackendStatus: (s: BackendState["status"]) => void;

  // Sidebar
  sidebarView: SidebarView;
  setSidebarView: (v: SidebarView) => void;

  // File tree
  rootEntries: FileEntry[];
  expandedDirs: Set<string>;
  loadRootFiles: () => Promise<void>;
  loadDirChildren: (dirPath: string) => Promise<void>;
  toggleDir: (dirPath: string) => void;

  // Tabs
  openTabs: OpenTab[];
  activeTabPath: string | null;
  openFile: (filePath: string) => Promise<void>;
  closeTab: (filePath: string) => void;
  setActiveTab: (filePath: string) => void;
  updateTabContent: (filePath: string, content: string) => void;
  markTabDirty: (filePath: string, dirty: boolean) => void;
}

export const useWorkspaceStore = create<WorkspaceStore>((set, get) => ({
  backendPort: (() => {
    try {
      const saved = localStorage.getItem("cyber-agent-ide-backend-port");
      return saved ? parseInt(saved, 10) : null;
    } catch {
      return null;
    }
  })(),
  backendStatus: "disconnected",
  setBackendPort: (port) => {
    try { localStorage.setItem("cyber-agent-ide-backend-port", String(port)); } catch {}
    set({ backendPort: port, backendStatus: "connected" });
  },
  setBackendStatus: (s) => set({ backendStatus: s }),

  sidebarView: "files",
  setSidebarView: (v) => set({ sidebarView: v }),

  rootEntries: [],
  expandedDirs: new Set(),
  loadRootFiles: async () => {
    const base = API();
    if (!base) return;
    try {
      const resp = await fetch(`${base}/fs/list?path=.`);
      const data = await resp.json();
      if (data.entries) set({ rootEntries: data.entries });
    } catch {}
  },
  loadDirChildren: async (dirPath) => {
    const base = API();
    if (!base) return;
    try {
      const resp = await fetch(`${base}/fs/list?path=${encodeURIComponent(dirPath)}`);
      const data = await resp.json();
      if (data.entries) {
        set((s) => ({
          rootEntries: s.rootEntries.map((e) =>
            e.path === dirPath ? { ...e, children: data.entries, loaded: true } : e
          ),
        }));
      }
    } catch {}
  },
  toggleDir: (dirPath) => {
    set((s) => {
      const next = new Set(s.expandedDirs);
      if (next.has(dirPath)) next.delete(dirPath);
      else next.add(dirPath);
      return { expandedDirs: next };
    });
  },

  openTabs: [],
  activeTabPath: null,
  openFile: async (filePath) => {
    const existing = get().openTabs.find((t) => t.path === filePath);
    if (existing) {
      set({ activeTabPath: filePath });
      return;
    }
    const base = API();
    if (!base) return;
    try {
      const resp = await fetch(`${base}/fs/read?path=${encodeURIComponent(filePath)}`);
      const data = await resp.json();
      const name = filePath.split("/").pop() || filePath;
      const ext = name.split(".").pop()?.toLowerCase();
      const langMap: Record<string, string> = {
        ts: "typescript", tsx: "typescript", js: "javascript", jsx: "javascript",
        py: "python", rs: "rust", go: "go", java: "java", c: "c", cpp: "cpp",
        h: "c", hpp: "cpp", css: "css", scss: "scss", html: "html", xml: "xml",
        json: "json", yaml: "yaml", yml: "yaml", md: "markdown", sql: "sql",
        sh: "shell", bash: "shell", zsh: "shell", toml: "ini", lock: "text",
      };
      set((s) => ({
        openTabs: [...s.openTabs, { path: filePath, name, content: data.content || "", language: langMap[ext || ""] || "plaintext" }],
        activeTabPath: filePath,
      }));
    } catch {}
  },
  closeTab: (filePath) => {
    set((s) => {
      const idx = s.openTabs.findIndex((t) => t.path === filePath);
      const nextTabs = s.openTabs.filter((t) => t.path !== filePath);
      let nextActive = s.activeTabPath;
      if (s.activeTabPath === filePath) {
        if (nextTabs.length === 0) nextActive = null;
        else nextActive = nextTabs[Math.min(idx, nextTabs.length - 1)].path;
      }
      return { openTabs: nextTabs, activeTabPath: nextActive };
    });
  },
  setActiveTab: (filePath) => set({ activeTabPath: filePath }),
  updateTabContent: (filePath, content) => {
    set((s) => ({
      openTabs: s.openTabs.map((t) => (t.path === filePath ? { ...t, content, dirty: true } : t)),
    }));
  },
  markTabDirty: (filePath, dirty) => {
    set((s) => ({
      openTabs: s.openTabs.map((t) => (t.path === filePath ? { ...t, dirty } : t)),
    }));
  },
}));
