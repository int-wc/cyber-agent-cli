import { useEffect, useRef, useState } from "react";
import { Terminal as XTerm } from "xterm";
import { FitAddon } from "xterm-addon-fit";
// @ts-ignore - xterm CSS import
import "xterm/css/xterm.css";
import { Plus, X, Terminal } from "lucide-react";
import { api } from "@/services/api";

interface TermTab {
  id: string;
  title: string;
  term: XTerm;
  fitAddon: FitAddon;
  cwd: string;
}

let tabCounter = 0;

export function TerminalPanel() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [tabs, setTabs] = useState<TermTab[]>([]);
  const [activeTab, setActiveTab] = useState<string | null>(null);
  const tabsRef = useRef<TermTab[]>([]);

  const createTerminal = () => {
    const id = `term-${++tabCounter}`;
    const term = new XTerm({
      fontSize: 13,
      fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
      theme: {
        background: "#0f172a",
        foreground: "#e2e8f0",
        cursor: "#14b8a6",
        selectionBackground: "#334155",
        black: "#1e293b",
        red: "#ef4444",
        green: "#22c55e",
        yellow: "#f59e0b",
        blue: "#3b82f6",
        magenta: "#a855f7",
        cyan: "#06b6d4",
        white: "#e2e8f0",
        brightBlack: "#475569",
        brightRed: "#f87171",
        brightGreen: "#4ade80",
        brightYellow: "#fbbf24",
        brightBlue: "#60a5fa",
        brightMagenta: "#c084fc",
        brightCyan: "#22d3ee",
        brightWhite: "#f8fafc",
      },
      cursorBlink: true,
      allowProposedApi: true,
    });
    const fitAddon = new FitAddon();
    term.loadAddon(fitAddon);

    term.onData((data) => {
      // Simple echo - in a real implementation this would go to a PTY
      if (data === "\r") {
        term.write("\r\n$ ");
      } else if (data === "\x7f") {
        // backspace
        term.write("\b \b");
      } else {
        term.write(data);
      }
    });

    term.writeln("Cyber Agent Terminal");
    term.writeln("输入命令或使用 API 后端执行");
    term.write("$ ");

    const tab: TermTab = { id, title: `终端 ${tabCounter}`, term, fitAddon, cwd: "." };
    tabsRef.current = [...tabsRef.current, tab];
    setTabs([...tabsRef.current]);
    setActiveTab(id);
  };

  useEffect(() => {
    if (activeTab && containerRef.current) {
      const tab = tabsRef.current.find((t) => t.id === activeTab);
      if (tab && containerRef.current) {
        containerRef.current.innerHTML = "";
        tab.term.open(containerRef.current);
        tab.fitAddon.fit();
      }
    }
  }, [activeTab]);

  useEffect(() => {
    if (tabs.length === 0) createTerminal();
    const handleResize = () => {
      tabsRef.current.forEach((t) => {
        try { t.fitAddon.fit(); } catch {}
      });
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const closeTab = (id: string) => {
    const tab = tabsRef.current.find((t) => t.id === id);
    tab?.term.dispose();
    tabsRef.current = tabsRef.current.filter((t) => t.id !== id);
    setTabs([...tabsRef.current]);
    if (activeTab === id) setActiveTab(tabsRef.current[0]?.id || null);
  };

  return (
    <div className="h-full flex flex-col glass-panel-heavy">
      <div className="flex items-center gap-0 px-2 text-xs text-muted no-select border-b border-border-glass">
        <Terminal size={12} className="ml-1" />
        {tabs.map((tab) => (
          <div
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-1 px-2 py-1 cursor-pointer ${tab.id === activeTab ? "text-primary border-b border-accent-teal" : "text-muted hover:text-soft"}`}
          >
            <span>{tab.title}</span>
            {tabs.length > 1 && (
              <button onClick={(e) => { e.stopPropagation(); closeTab(tab.id); }} className="hover:text-accent-red">
                <X size={10} />
              </button>
            )}
          </div>
        ))}
        <button onClick={createTerminal} className="ml-1 p-1 hover:bg-surface-light rounded">
          <Plus size={12} />
        </button>
      </div>
      <div ref={containerRef} className="flex-1 overflow-hidden p-1" />
    </div>
  );
}
