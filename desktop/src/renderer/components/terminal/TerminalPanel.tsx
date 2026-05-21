import { useEffect, useRef, useCallback, useState } from "react";
import { Plus, X, Terminal as TerminalIcon } from "lucide-react";

interface TerminalTab {
  id: string;
  title: string;
}

// Terminal will be integrated with xterm.js in Phase 4. Placeholder for now.
export default function TerminalPanel() {
  const [tabs, setTabs] = useState<TerminalTab[]>([{ id: "term-1", title: "bash" }]);
  const [activeTab, setActiveTab] = useState("term-1");
  const outputRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [output, setOutput] = useState<string[]>(["$ Cyber Agent IDE Terminal\n"]);
  const [history, setHistory] = useState<string[]>([]);
  const [historyIdx, setHistoryIdx] = useState(-1);

  const addOutput = useCallback((text: string) => {
    setOutput((prev) => [...prev, text]);
    setTimeout(() => {
      outputRef.current?.scrollTo({ top: outputRef.current.scrollHeight, behavior: "smooth" });
    }, 10);
  }, []);

  const execute = useCallback((cmd: string) => {
    const trimmed = cmd.trim();
    if (!trimmed) return;
    addOutput(`$ ${trimmed}\n`);
    setHistory((prev) => [...prev, trimmed]);
    setHistoryIdx(-1);

    if (trimmed === "clear" || trimmed === "cls") {
      setOutput(["$ Cyber Agent IDE Terminal\n"]);
      return;
    }

    // In Phase 4, this will use node-pty via Electron IPC
    // For now, show placeholder
    addOutput(`[终端将通过 node-pty 在 Electron 中集成]\n`);
  }, [addOutput]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      const value = inputRef.current?.value || "";
      execute(value);
      if (inputRef.current) inputRef.current.value = "";
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      if (history.length > 0) {
        const idx = historyIdx === -1 ? history.length - 1 : Math.max(0, historyIdx - 1);
        setHistoryIdx(idx);
        if (inputRef.current) inputRef.current.value = history[idx];
      }
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      const idx = historyIdx === -1 ? -1 : Math.min(history.length - 1, historyIdx + 1);
      setHistoryIdx(idx);
      if (inputRef.current) inputRef.current.value = idx === -1 ? "" : history[idx];
    }
  }, [execute, history, historyIdx]);

  const addTab = () => {
    const id = `term-${tabs.length + 1}`;
    setTabs([...tabs, { id, title: "bash" }]);
    setActiveTab(id);
  };

  const removeTab = (id: string) => {
    setTabs(tabs.filter((t) => t.id !== id));
    if (activeTab === id) setActiveTab(tabs[0]?.id || "term-1");
  };

  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column", background: "var(--bg-base)" }}>
      {/* Tab bar */}
      <div
        className="glass-surface"
        style={{
          display: "flex", alignItems: "center",
          borderBottom: "1px solid rgba(255,255,255,0.06)",
        }}
      >
        {tabs.map((tab) => (
          <div
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              display: "flex", alignItems: "center", gap: 4,
              padding: "4px 12px", fontSize: 12, cursor: "pointer",
              color: tab.id === activeTab ? "var(--text-primary)" : "var(--text-tertiary)",
              background: tab.id === activeTab ? "var(--glass-fill-active)" : "transparent",
              borderRight: "1px solid rgba(255,255,255,0.05)",
            }}
          >
            <TerminalIcon size={11} />
            <span>{tab.title}</span>
            {tabs.length > 1 && (
              <X size={11} onClick={(e) => { e.stopPropagation(); removeTab(tab.id); }} />
            )}
          </div>
        ))}
        <button className="glass-btn" style={{ padding: "2px 8px", margin: "2px 4px" }} onClick={addTab}>
          <Plus size={12} />
        </button>
      </div>
      {/* Output */}
      <div ref={outputRef} style={{ flex: 1, overflow: "auto", padding: "8px 12px", fontFamily: "monospace", fontSize: 13, whiteSpace: "pre-wrap", color: "var(--text-secondary)" }}>
        {output.join("")}
      </div>
      {/* Input */}
      <div style={{ padding: "4px 8px", borderTop: "1px solid rgba(255,255,255,0.06)", display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{ color: "var(--accent-light)", fontFamily: "monospace", fontSize: 13 }}>$</span>
        <input
          ref={inputRef}
          onKeyDown={handleKeyDown}
          className="glass-input"
          style={{
            flex: 1, background: "transparent", border: "none",
            fontFamily: "monospace", fontSize: 13, padding: "4px 0",
          }}
          placeholder="输入命令..."
          autoFocus
        />
      </div>
    </div>
  );
}
