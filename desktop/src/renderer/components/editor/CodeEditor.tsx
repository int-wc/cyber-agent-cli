import { useCallback, useRef, useEffect, lazy, Suspense } from "react";
import { useEditorStore } from "../../stores/editorStore";
import { fsApi } from "../../services/api";
import { X } from "lucide-react";

const MonacoEditor = lazy(() => import("@monaco-editor/react"));

// Monaco dark theme matching Liquid Glass design
const MONACO_THEME = {
  base: "vs-dark" as const,
  inherit: true,
  rules: [
    { token: "comment", foreground: "6A9955", fontStyle: "italic" },
    { token: "keyword", foreground: "C586C0" },
    { token: "string", foreground: "CE9178" },
    { token: "number", foreground: "B5CEA8" },
    { token: "type", foreground: "4EC9B0" },
    { token: "function", foreground: "DCDCAA" },
    { token: "variable", foreground: "9CDCFE" },
  ],
  colors: {
    "editor.background": "#0a0a0f",
    "editor.foreground": "#d4d4d4",
    "editor.lineHighlightBackground": "#ffffff08",
    "editor.selectionBackground": "#6c5ce740",
    "editor.inactiveSelectionBackground": "#6c5ce720",
    "editorCursor.foreground": "#a29bfe",
    "editorLineNumber.foreground": "#ffffff30",
    "editorLineNumber.activeForeground": "#a29bfe",
    "editor.selectionHighlightBackground": "#ffffff08",
  },
};

const MONACO_OPTIONS = {
  fontSize: 13,
  fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
  lineHeight: 1.7,
  minimap: { enabled: false },
  scrollBeyondLastLine: false,
  wordWrap: "off" as const,
  tabSize: 4,
  automaticLayout: true,
  padding: { top: 8 },
  smoothScrolling: true,
  cursorBlinking: "smooth" as const,
  cursorSmoothCaretAnimation: "on" as const,
  bracketPairColorization: { enabled: true },
  guides: { indentation: false },
  renderLineHighlight: "line" as const,
};

// Inline textarea fallback when Monaco is not loaded
function FallbackEditor({ content, onChange }: { content: string; onChange: (v: string) => void }) {
  return (
    <textarea
      value={content}
      onChange={(e) => onChange(e.target.value)}
      spellCheck={false}
      style={{
        flex: 1, width: "100%", resize: "none", background: "transparent",
        color: "var(--text-primary)", border: "none", outline: "none",
        padding: 16, fontFamily: "monospace", fontSize: 13, lineHeight: 1.7,
      }}
    />
  );
}

export default function CodeEditor() {
  const { tabs, activeTabPath, closeTab, setActiveTab, updateContent, markClean } = useEditorStore();
  const saveTimerRef = useRef<ReturnType<typeof setTimeout>>();
  const editorRef = useRef<{ getValue: () => string } | null>(null);

  const activeTab = tabs.find((t) => t.path === activeTabPath);

  const handleContentChange = useCallback((value: string | undefined) => {
    if (!activeTabPath || value === undefined) return;
    updateContent(activeTabPath, value);
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    saveTimerRef.current = setTimeout(async () => {
      try {
        await fsApi.write(activeTabPath, value);
        markClean(activeTabPath);
      } catch { /* ignore */ }
    }, 2000);
  }, [activeTabPath, updateContent, markClean]);

  // Ctrl+S save
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "s") {
        e.preventDefault();
        if (activeTabPath && activeTab) {
          const content = editorRef.current?.getValue() ?? activeTab.content;
          fsApi.write(activeTabPath, content).then(() => markClean(activeTabPath));
        }
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [activeTabPath, activeTab, markClean]);

  // Empty state
  if (!activeTab) {
    return (
      <div style={{
        height: "100%", display: "flex", alignItems: "center", justifyContent: "center",
        color: "var(--text-tertiary)", fontSize: 14, flexDirection: "column", gap: 12,
      }}>
        <div style={{
          width: 80, height: 80, borderRadius: 20,
          background: "linear-gradient(135deg, rgba(108,92,231,0.2), rgba(79,195,247,0.2))",
          display: "flex", alignItems: "center", justifyContent: "center",
        }}>
          <span style={{ fontSize: 32 }}>⚡</span>
        </div>
        <span style={{ fontWeight: 500, color: "var(--text-secondary)" }}>打开文件开始编辑</span>
        <span style={{ fontSize: 12, color: "var(--text-tertiary)" }}>Ctrl+P 快速搜索文件</span>
      </div>
    );
  }

  const language = activeTab.language === "plaintext" ? "text" :
    activeTab.language === "shell" ? "bash" : activeTab.language;

  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column" }}>
      {/* Tabs bar */}
      <div className="glass-surface" style={{
        display: "flex", alignItems: "center",
        borderBottom: "1px solid rgba(255,255,255,0.06)",
        overflow: "auto", flexShrink: 0,
      }}>
        {tabs.map((tab) => (
          <div
            key={tab.path}
            onClick={() => setActiveTab(tab.path)}
            style={{
              display: "flex", alignItems: "center", gap: 6,
              padding: "6px 12px", fontSize: 12, cursor: "pointer",
              color: tab.path === activeTabPath ? "var(--text-primary)" : "var(--text-tertiary)",
              background: tab.path === activeTabPath ? "var(--glass-fill-active)" : "transparent",
              borderRight: "1px solid rgba(255,255,255,0.05)",
              whiteSpace: "nowrap", transition: "background 150ms ease",
            }}
          >
            {tab.dirty && (
              <span style={{ width: 7, height: 7, borderRadius: "50%", background: "var(--accent-light)" }} />
            )}
            <span>{tab.name}</span>
            <button
              onClick={(e) => { e.stopPropagation(); closeTab(tab.path); }}
              style={{ background: "none", border: "none", cursor: "pointer", padding: 1, borderRadius: 3 }}
            >
              <X size={12} color="var(--text-tertiary)" />
            </button>
          </div>
        ))}
      </div>
      {/* Editor */}
      <div style={{ flex: 1 }}>
        <Suspense fallback={
          <FallbackEditor content={activeTab.content} onChange={(v) => handleContentChange(v)} />
        }>
          <MonacoEditor
            key={activeTab.path}
            language={language}
            value={activeTab.content}
            theme="cyber-dark"
            options={MONACO_OPTIONS}
            onChange={handleContentChange}
            onMount={(editor, monaco) => {
              editorRef.current = editor;
              monaco.editor.defineTheme("cyber-dark", MONACO_THEME);
              monaco.editor.setTheme("cyber-dark");
            }}
            loading={<FallbackEditor content={activeTab.content} onChange={(v) => handleContentChange(v)} />}
          />
        </Suspense>
      </div>
    </div>
  );
}
