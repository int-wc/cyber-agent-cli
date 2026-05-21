import { create } from "zustand";

interface EditorTab {
  path: string;
  name: string;
  content: string;
  language: string;
  dirty: boolean;
}

interface EditorState {
  tabs: EditorTab[];
  activeTabPath: string | null;
  openFile: (path: string, content: string) => void;
  closeTab: (path: string) => void;
  setActiveTab: (path: string) => void;
  updateContent: (path: string, content: string) => void;
  markClean: (path: string) => void;
}

function detectLanguage(path: string): string {
  const ext = path.split(".").pop() || "";
  const map: Record<string, string> = {
    py: "python", ts: "typescript", tsx: "typescript", js: "javascript",
    jsx: "javascript", json: "json", html: "html", css: "css",
    md: "markdown", yaml: "yaml", yml: "yaml", toml: "toml",
    rs: "rust", go: "go", java: "java", c: "c", cpp: "cpp",
    h: "c", hpp: "cpp", sh: "shell", bash: "shell", zsh: "shell",
    xml: "xml", svg: "xml", sql: "sql", graphql: "graphql",
    dockerfile: "dockerfile", env: "plaintext", txt: "plaintext",
  };
  const name = path.split("/").pop()?.toLowerCase() || "";
  if (name === "dockerfile") return "dockerfile";
  if (name === "makefile") return "makefile";
  return map[ext] || "plaintext";
}

export const useEditorStore = create<EditorState>((set) => ({
  tabs: [],
  activeTabPath: null,

  openFile: (path, content) => {
    const name = path.split("/").pop() || path;
    set((s) => {
      const existing = s.tabs.find((t) => t.path === path);
      if (existing) {
        return { activeTabPath: path };
      }
      return {
        tabs: [...s.tabs, { path, name, content, language: detectLanguage(path), dirty: false }],
        activeTabPath: path,
      };
    });
  },

  closeTab: (path) => {
    set((s) => {
      const tabs = s.tabs.filter((t) => t.path !== path);
      const activeTabPath = s.activeTabPath === path
        ? (tabs[tabs.length - 1]?.path || null)
        : s.activeTabPath;
      return { tabs, activeTabPath };
    });
  },

  setActiveTab: (path) => set({ activeTabPath: path }),

  updateContent: (path, content) => {
    set((s) => ({
      tabs: s.tabs.map((t) => (t.path === path ? { ...t, content, dirty: true } : t)),
    }));
  },

  markClean: (path) => {
    set((s) => ({
      tabs: s.tabs.map((t) => (t.path === path ? { ...t, dirty: false } : t)),
    }));
  },
}));
