import { useEffect, useRef, useCallback, useState } from "react";
import { Plus, X, Terminal as TerminalIcon } from "lucide-react";

interface TerminalTab {
  id: string;
  title: string;
}

export default function TerminalPanel() {
  const [tabs, setTabs] = useState<TerminalTab[]>([{ id: "term-1", title: "bash" }]);
  const [activeTab, setActiveTab] = useState("term-1");
  const [xtermReady, setXtermReady] = useState(false);
  const terminalRef = useRef<HTMLDivElement>(null);
  const xtermRef = useRef<{
    term: { dispose: () => void; write: (d: string) => void; resize: (c: number, r: number) => void };
    fitAddon: { fit: () => void };
    sessionId: string | null;
  } | null>(null);

  useEffect(() => {
    let disposed = false;

    async function init() {
      try {
        const [{ Terminal }, { FitAddon }] = await Promise.all([
          import("xterm"),
          import("xterm-addon-fit"),
        ]);
        if (disposed || !terminalRef.current) return;

        const term = new Terminal({
          fontSize: 13,
          fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
          theme: {
            background: "#f8f8fc",
            foreground: "#1e293b",
            cursor: "#7c6ff7",
            selectionBackground: "#7c6ff720",
            black: "#e2e8f0",
            red: "#ef4444",
            green: "#22c55e",
            yellow: "#eab308",
            blue: "#3b82f6",
            magenta: "#a855f7",
            cyan: "#06b6d4",
            white: "#1e293b",
            brightBlack: "#94a3b8",
            brightRed: "#f87171",
            brightGreen: "#4ade80",
            brightYellow: "#facc15",
            brightBlue: "#60a5fa",
            brightMagenta: "#c084fc",
            brightCyan: "#22d3ee",
            brightWhite: "#0f172a",
          },
          cursorBlink: true,
          cursorStyle: "bar",
          allowProposedApi: true,
        });

        const fitAddon = new FitAddon();
        term.loadAddon(fitAddon);
        term.open(terminalRef.current);

        // Delay fit to let DOM settle
        setTimeout(() => {
          try { fitAddon.fit(); } catch {}
        }, 100);

        xtermRef.current = {
          term,
          fitAddon,
          sessionId: null,
        };

        // If Electron terminal is available, hook it up
        const electronAPI = window.electronAPI;
        if (electronAPI) {
          electronAPI.terminalCreate().then((sess) => {
            if (xtermRef.current && sess) {
              xtermRef.current.sessionId = sess.sessionId;
            }
          });
          electronAPI.onTerminalOutput((sessId, data) => {
            const ref = xtermRef.current;
            if (ref && sessId === ref.sessionId) {
              ref.term.write(data);
            }
          });
        }

        term.onData((data) => {
          const ref = xtermRef.current;
          if (ref?.sessionId && window.electronAPI) {
            window.electronAPI.terminalWrite(ref.sessionId, data);
          } else {
            // Fallback: handle locally
            if (data === "\r") {
              term.write("\r\n$ ");
            } else if (data === "") {
              // backspace handled by PTY in real mode
            } else {
              term.write(data);
            }
          }
        });

        // Resize observer
        const resizeObserver = new ResizeObserver(() => {
          try { fitAddon.fit(); } catch {}
          const ref = xtermRef.current;
          if (ref?.sessionId && window.electronAPI) {
            window.electronAPI.terminalResize(ref.sessionId, term.cols, term.rows);
          }
        });
        if (terminalRef.current) {
          resizeObserver.observe(terminalRef.current);
        }

        setXtermReady(true);

        term.writeln("Cyber Agent IDE Terminal");
        term.writeln("终端已就绪。");
        if (!window.electronAPI) {
          term.writeln("(Electron 环境未检测到 — 使用本地回显模式)");
        }
        term.write("$ ");

        return () => {
          resizeObserver.disconnect();
        };
      } catch (err) {
        console.error("xterm init failed:", err);
      }
    }

    init();

    return () => {
      disposed = true;
      xtermRef.current?.term.dispose();
      xtermRef.current = null;
    };
  }, []);

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
    <div style={{ height: "100%", display: "flex", flexDirection: "column", background: "transparent" }}>
      {/* Tab bar */}
      <div className="glass-surface" style={{
        display: "flex", alignItems: "center", flexShrink: 0,
        borderBottom: "1px solid rgba(255,255,255,0.06)",
      }}>
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
              transition: "background 150ms ease",
            }}
          >
            <TerminalIcon size={11} />
            <span>{tab.title}</span>
            {tabs.length > 1 && (
              <X size={11} style={{ cursor: "pointer" }} onClick={(e) => { e.stopPropagation(); removeTab(tab.id); }} />
            )}
          </div>
        ))}
        <button className="glass-btn" style={{ padding: "2px 8px", margin: "2px 4px" }} onClick={addTab}>
          <Plus size={12} />
        </button>
        {!xtermReady && (
          <span style={{ fontSize: 11, color: "var(--text-tertiary)", marginLeft: 8 }}>
            加载中...
          </span>
        )}
      </div>
      {/* Terminal container */}
      <div ref={terminalRef} style={{ flex: 1, padding: "4px" }} />
    </div>
  );
}
