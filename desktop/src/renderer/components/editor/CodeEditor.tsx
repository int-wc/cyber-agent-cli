import { useEffect, useRef, useCallback } from "react";
import { useEditorStore } from "../../stores/editorStore";
import { fsApi } from "../../services/api";
import { X } from "lucide-react";

// Monaco will be lazy-loaded in Phase 4. This is a placeholder editor.
export default function CodeEditor() {
  const { tabs, activeTabPath, closeTab, setActiveTab, updateContent, markClean } = useEditorStore();
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const saveTimerRef = useRef<ReturnType<typeof setTimeout>>();

  const activeTab = tabs.find((t) => t.path === activeTabPath);

  const handleChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    if (!activeTabPath) return;
    const content = e.target.value;
    updateContent(activeTabPath, content);
    // Debounced auto-save
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    saveTimerRef.current = setTimeout(async () => {
      try {
        await fsApi.write(activeTabPath, content);
        markClean(activeTabPath);
      } catch { /* ignore */ }
    }, 2000);
  }, [activeTabPath, updateContent, markClean]);

  // Save on Ctrl+S
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "s") {
        e.preventDefault();
        if (activeTabPath && activeTab) {
          fsApi.write(activeTabPath, activeTab.content).then(() => markClean(activeTabPath));
        }
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [activeTabPath, activeTab, markClean]);

  if (!activeTab) {
    return (
      <div style={{
        height: "100%", display: "flex", alignItems: "center", justifyContent: "center",
        color: "var(--text-tertiary)", fontSize: 14, flexDirection: "column", gap: 12,
      }}>
        <div style={{
          width: 64, height: 64, borderRadius: 16,
          background: "linear-gradient(135deg, rgba(108,92,231,0.2), rgba(79,195,247,0.2))",
          display: "flex", alignItems: "center", justifyContent: "center",
        }}>
          <span style={{ fontSize: 28 }}>⚡</span>
        </div>
        <span>打开文件开始编辑</span>
        <span style={{ fontSize: 12, color: "var(--text-tertiary)" }}>
          Ctrl+P 快速搜索文件
        </span>
      </div>
    );
  }

  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column" }}>
      {/* Tabs */}
      <div
        className="glass-surface"
        style={{
          display: "flex", alignItems: "center",
          borderBottom: "1px solid rgba(255,255,255,0.06)",
          overflow: "auto",
        }}
      >
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
              whiteSpace: "nowrap",
            }}
          >
            {tab.dirty && (
              <span style={{ width: 7, height: 7, borderRadius: "50%", background: "var(--accent-light)" }} />
            )}
            <span>{tab.name}</span>
            <button
              onClick={(e) => { e.stopPropagation(); closeTab(tab.path); }}
              style={{
                background: "none", border: "none", cursor: "pointer",
                padding: 1, borderRadius: 3, color: "var(--text-tertiary)",
              }}
            >
              <X size={12} />
            </button>
          </div>
        ))}
      </div>
      {/* Editor area */}
      <textarea
        ref={textareaRef}
        value={activeTab.content}
        onChange={handleChange}
        spellCheck={false}
        style={{
          flex: 1, width: "100%", resize: "none",
          background: "var(--bg-base)", color: "var(--text-primary)",
          border: "none", outline: "none", padding: 16,
          fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
          fontSize: 13, lineHeight: 1.7,
          tabSize: 4,
        }}
      />
    </div>
  );
}
