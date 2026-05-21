import { Minus, Square, X } from "lucide-react";
import { useSessionStore } from "../../stores/sessionStore";
import { useChatStore } from "../../stores/chatStore";

export default function TitleBar() {
  const { config, connected } = useSessionStore();
  const isStreaming = useChatStore((s) => s.isStreaming);

  return (
    <div
      className="titlebar-drag glass-surface"
      style={{
        height: 38,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "0 12px",
        borderBottom: "1px solid rgba(255,255,255,0.06)",
      }}
    >
      {/* Left: App icon + name */}
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <div style={{
          width: 18, height: 18, borderRadius: 6,
          background: "linear-gradient(135deg, #6c5ce7, #a29bfe)",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 10, fontWeight: 700, color: "#fff",
        }}>
          C
        </div>
        <span style={{ fontSize: 13, fontWeight: 500, color: "var(--text-primary)" }}>
          Cyber Agent IDE
        </span>
        {isStreaming && (
          <span style={{ fontSize: 11, color: "var(--accent-light)" }}>
            ● 生成中
          </span>
        )}
      </div>

      {/* Center: Connection status */}
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <div style={{
          width: 6, height: 6, borderRadius: "50%",
          background: connected ? "var(--green)" : "var(--red)",
          boxShadow: connected ? "0 0 6px var(--green)" : "none",
        }} />
        <span style={{ fontSize: 11, color: "var(--text-tertiary)" }}>
          {connected ? `${config.service} / ${config.model || "default"}` : "未连接"}
        </span>
        <span style={{
          fontSize: 10, padding: "1px 6px", borderRadius: 4,
          background: "var(--accent-soft)", color: "var(--accent-light)",
        }}>
          {config.mode === "authorized" ? "授权" : "标准"}
        </span>
      </div>

      {/* Right: Window controls */}
      <div className="titlebar-no-drag" style={{ display: "flex", gap: 4 }}>
        <button
          className="glass-btn"
          style={{ padding: "4px 8px", borderRadius: 4 }}
          onClick={() => window.electronAPI?.minimizeWindow()}
        >
          <Minus size={12} />
        </button>
        <button
          className="glass-btn"
          style={{ padding: "4px 8px", borderRadius: 4 }}
          onClick={() => window.electronAPI?.maximizeWindow()}
        >
          <Square size={10} />
        </button>
        <button
          className="glass-btn"
          style={{
            padding: "4px 8px", borderRadius: 4,
            background: "rgba(255,82,82,0.15)", borderColor: "rgba(255,82,82,0.3)",
          }}
          onClick={() => window.electronAPI?.closeWindow()}
        >
          <X size={12} color="var(--red)" />
        </button>
      </div>
    </div>
  );
}
