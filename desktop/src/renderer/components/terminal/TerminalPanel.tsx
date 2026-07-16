import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

interface Props {
  standalone?: boolean;
  sessionKey?: string;
}

interface TerminalOutputPayload {
  session_id: string;
  data: string;
}

export default function TerminalPanel({ standalone, sessionKey }: Props) {
  const [xtermReady, setXtermReady] = useState(false);
  const terminalRef = useRef<HTMLDivElement>(null);
  const xtermRef = useRef<{
    term: { dispose: () => void; write: (d: string) => void };
    fitAddon: { fit: () => void };
    sessionId: string | null;
  } | null>(null);
  const unlistenRef = useRef<UnlistenFn | null>(null);

  useEffect(() => {
    let disposed = false;
    let currentUnlisten: UnlistenFn | null = null;

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
            black: "#e2e8f0", red: "#ef4444", green: "#22c55e",
            yellow: "#eab308", blue: "#3b82f6", magenta: "#a855f7",
            cyan: "#06b6d4", white: "#1e293b",
            brightBlack: "#94a3b8", brightRed: "#f87171",
            brightGreen: "#4ade80", brightYellow: "#facc15",
            brightBlue: "#60a5fa", brightMagenta: "#c084fc",
            brightCyan: "#22d3ee", brightWhite: "#0f172a",
          },
          cursorBlink: true,
          cursorStyle: "bar",
          allowProposedApi: true,
        });

        const fitAddon = new FitAddon();
        term.loadAddon(fitAddon);
        term.open(terminalRef.current);

        setTimeout(() => {
          try { fitAddon.fit(); } catch {}
        }, 100);

        xtermRef.current = { term, fitAddon, sessionId: null };

        // 通过 Tauri PTY 桥接调用 Rust 命令创建终端会话。
        try {
          const sess = await invoke<{ sessionId: string; pid: number }>("terminal_create", {
            options: { shell: null },
          });
          if (xtermRef.current && sess) {
            xtermRef.current.sessionId = sess.sessionId;
          }

          // 监听 Rust 侧转发的 PTY 输出事件。
          currentUnlisten = await listen<TerminalOutputPayload>(
            "terminal:output",
            (event) => {
              const ref = xtermRef.current;
              if (ref && event.payload.session_id === ref.sessionId) {
                ref.term.write(event.payload.data);
              }
            },
          );
          unlistenRef.current = currentUnlisten;
        } catch (err) {
          console.error("Terminal PTY init failed:", err);
        }

        term.onData((data) => {
          const ref = xtermRef.current;
          if (ref?.sessionId) {
            invoke("terminal_write", {
              data: { sessionId: ref.sessionId, data },
            }).catch(() => {});
          } else {
            if (data === "\r") term.write("\r\n$ ");
            else term.write(data);
          }
        });

        // 监听容器尺寸变化，并同步终端行列数。
        const resizeObserver = new ResizeObserver(() => {
          try { fitAddon.fit(); } catch {}
          const ref = xtermRef.current;
          if (ref?.sessionId) {
            invoke("terminal_resize", {
              data: { sessionId: ref.sessionId, cols: term.cols, rows: term.rows },
            }).catch(() => {});
          }
        });
        if (terminalRef.current) {
          resizeObserver.observe(terminalRef.current);
        }

        setXtermReady(true);

        term.writeln("Cyber Agent IDE Terminal");
        term.writeln("");
        term.write("$ ");

        return () => { resizeObserver.disconnect(); };
      } catch (err) {
        console.error("xterm init failed:", err);
      }
    }

    init();

    return () => {
      disposed = true;
      if (currentUnlisten) currentUnlisten();
      if (xtermRef.current?.sessionId) {
        invoke("terminal_kill", {
          data: { sessionId: xtermRef.current.sessionId },
        }).catch(() => {});
      }
      xtermRef.current?.term.dispose();
      xtermRef.current = null;
    };
  }, [sessionKey]);

  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column", background: "transparent" }}>
      {!standalone && (
        <div className="glass-surface" style={{
          display: "flex", alignItems: "center", flexShrink: 0, padding: "2px 8px", gap: 4,
          borderBottom: "1px solid rgba(0,0,0,0.05)",
        }}>
          <span style={{ fontSize: 11, color: "var(--text-secondary)" }}>
            终端
          </span>
          {!xtermReady && (
            <span style={{ fontSize: 10, color: "var(--text-tertiary)" }}>加载中...</span>
          )}
        </div>
      )}
      <div ref={terminalRef} style={{ flex: 1, padding: standalone ? "6px" : "4px" }} />
    </div>
  );
}
