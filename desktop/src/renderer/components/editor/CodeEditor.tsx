import { useCallback, useRef, useEffect, lazy, Suspense, useState } from "react";
import { useEditorStore } from "../../stores/editorStore";
import { fsApi } from "../../services/api";
import { X } from "lucide-react";

const MonacoEditor = lazy(() => import("@monaco-editor/react"));

const MONACO_THEME = {
  base: "vs" as const,
  inherit: true,
  rules: [
    { token: "comment", foreground: "8c8c8c", fontStyle: "italic" },
    { token: "keyword", foreground: "7c6ff7" },
    { token: "string", foreground: "22c55e" },
    { token: "number", foreground: "f59e0b" },
    { token: "type", foreground: "3b82f6" },
    { token: "function", foreground: "7c6ff7" },
    { token: "variable", foreground: "1e293b" },
  ],
  colors: {
    "editor.background": "#f8f8fc99",
    "editor.foreground": "#1e293b",
    "editor.lineHighlightBackground": "#7c6ff712",
    "editor.selectionBackground": "#7c6ff725",
    "editor.inactiveSelectionBackground": "#7c6ff712",
    "editorCursor.foreground": "#7c6ff7",
    "editorLineNumber.foreground": "#94a3b8",
    "editorLineNumber.activeForeground": "#7c6ff7",
    "editor.selectionHighlightBackground": "#7c6ff710",
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
  smoothScrolling: true,
  cursorBlinking: "smooth" as const,
  cursorSmoothCaretAnimation: "on" as const,
  bracketPairColorization: { enabled: true },
  guides: { indentation: false },
  renderLineHighlight: "line" as const,
};

function FallbackEditor({ content, onChange, height }: { content: string; onChange: (v: string) => void; height: number }) {
  return (
    <textarea
      value={content}
      onChange={(e) => onChange(e.target.value)}
      spellCheck={false}
      style={{
        width: "100%", height, resize: "none", background: "transparent",
        color: "var(--text-primary)", border: "none", outline: "none",
        padding: "8px 16px", fontFamily: "monospace", fontSize: 13, lineHeight: 1.7,
      }}
    />
  );
}

let monacoReady = false;
let monacoPromise: Promise<void> | null = null;

function ensureMonacoLoaded(): Promise<void> {
  if (monacoReady) return Promise.resolve();
  if (!monacoPromise) {
    monacoPromise = Promise.all([
      import("@monaco-editor/react"),
      import("monaco-editor"),
    ]).then(([reactMonaco, monaco]) => {
      reactMonaco.loader.config({ monaco });
      monacoReady = true;
    });
  }
  return monacoPromise;
}

export default function CodeEditor() {
  const { tabs, activeTabPath, closeTab, setActiveTab, updateContent, markClean } = useEditorStore();
  const saveTimerRef = useRef<ReturnType<typeof setTimeout>>();
  const editorRef = useRef<{ getValue: () => string } | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerHeight, setContainerHeight] = useState<number | null>(null);
  const [monacoLoaded, setMonacoLoaded] = useState(false);

  const activeTab = tabs.find((t) => t.path === activeTabPath);

  // 编辑器首次出现时再动态加载 monaco-editor。
  // 静态导入会在应用启动时执行 monaco-editor，并影响 window.prompt()。
  useEffect(() => {
    if (!activeTab) return;
    let cancelled = false;
    ensureMonacoLoaded().then(() => {
      if (!cancelled) setMonacoLoaded(true);
    });
    return () => { cancelled = true; };
  }, [!!activeTab]);

  // 使用 ResizeObserver 测量容器像素高度，避免编辑器高度抖动。
  useEffect(() => {
    if (!activeTab) return;
    const el = containerRef.current;
    if (!el) return;
    const measure = () => {
      const h = el.getBoundingClientRect().height;
      if (h > 0) setContainerHeight(h);
    };
    measure();
    const ro = new ResizeObserver(() => measure());
    ro.observe(el);
    return () => ro.disconnect();
  }, [!!activeTab]);

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

  if (!activeTab) {
    return (
      <div style={{
        position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center",
        color: "var(--text-tertiary)", fontSize: 14, flexDirection: "column", gap: 12,
      }}>
        <div style={{
          width: 80, height: 80, borderRadius: 20,
          background: "linear-gradient(135deg, rgba(124,111,247,0.15), rgba(59,130,246,0.15))",
          display: "flex", alignItems: "center", justifyContent: "center",
        }}>
          <span style={{ fontSize: 32 }}>⚡</span>
        </div>
        <span style={{ fontWeight: 500, color: "var(--text-secondary)" }}>打开文件开始编辑</span>
        <span style={{ fontSize: 12, color: "var(--text-tertiary)" }}>Ctrl+P 快速搜索文件</span>
      </div>
    );
  }

  const language = activeTab.language === "plaintext" ? "text"
    : activeTab.language === "shell" ? "bash"
    : activeTab.language;

  const editorHeight = containerHeight ?? 300;

  return (
    <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", minHeight: 0 }}>
      {/* Tabs bar */}
      <div className="glass-surface" style={{
        display: "flex", alignItems: "center", flexShrink: 0,
        borderBottom: "1px solid rgba(0,0,0,0.05)", overflow: "auto",
      }}>
        {tabs.map((tab) => (
          <div
            key={tab.path}
            onClick={() => setActiveTab(tab.path)}
            style={{
              display: "flex", alignItems: "center", gap: 6,
              padding: "6px 12px", fontSize: 12, cursor: "pointer",
              color: tab.path === activeTabPath ? "var(--text-primary)" : "var(--text-tertiary)",
              background: tab.path === activeTabPath ? "rgba(124,111,247,0.08)" : "transparent",
              borderRight: "1px solid rgba(0,0,0,0.05)",
              whiteSpace: "nowrap", transition: "background 150ms ease",
            }}
          >
            {tab.dirty && (
              <span style={{ width: 7, height: 7, borderRadius: "50%", background: "var(--accent)" }} />
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

      {/* Editor container */}
      <div ref={containerRef} style={{ flex: 1, minHeight: 0 }}>
        {monacoLoaded ? (
          <Suspense fallback={
            <FallbackEditor content={activeTab.content} onChange={(v) => handleContentChange(v)} height={editorHeight} />
          }>
            <MonacoEditor
              key={activeTab.path}
              height={editorHeight}
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
              loading={<FallbackEditor content={activeTab.content} onChange={(v) => handleContentChange(v)} height={editorHeight} />}
            />
          </Suspense>
        ) : (
          <FallbackEditor content={activeTab.content} onChange={(v) => handleContentChange(v)} height={editorHeight} />
        )}
      </div>
    </div>
  );
}
