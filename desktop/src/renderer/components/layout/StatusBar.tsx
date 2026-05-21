import { useSessionStore } from "../../stores/sessionStore";
import { useChatStore } from "../../stores/chatStore";
import { useEditorStore } from "../../stores/editorStore";
import { useUIStore } from "../../stores/uiStore";
import { Sidebar, MessageSquare, Terminal } from "lucide-react";

export default function StatusBar() {
  const { config, connected } = useSessionStore();
  const usage = useChatStore((s) => s.usage);
  const activeTabPath = useEditorStore((s) => s.activeTabPath);
  const { sidebarVisible, chatPanelVisible, terminalVisible,
          toggleSidebar, toggleChatPanel, toggleTerminal } = useUIStore();

  return (
    <div
      className="glass-surface"
      style={{
        height: 28,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "0 10px",
        borderTop: "1px solid rgba(255,255,255,0.06)",
        fontSize: 11,
      }}
    >
      {/* Left: Toggle buttons + file path */}
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <button
          className="glass-btn"
          style={{
            padding: "2px 6px", borderRadius: 4, fontSize: 11,
            background: sidebarVisible ? "var(--accent-soft)" : undefined,
          }}
          onClick={toggleSidebar}
        >
          <Sidebar size={12} />
        </button>
        <button
          className="glass-btn"
          style={{
            padding: "2px 6px", borderRadius: 4, fontSize: 11,
            background: terminalVisible ? "var(--accent-soft)" : undefined,
          }}
          onClick={toggleTerminal}
        >
          <Terminal size={12} />
        </button>
        <button
          className="glass-btn"
          style={{
            padding: "2px 6px", borderRadius: 4, fontSize: 11,
            background: chatPanelVisible ? "var(--accent-soft)" : undefined,
          }}
          onClick={toggleChatPanel}
        >
          <MessageSquare size={12} />
        </button>
        {activeTabPath && (
          <span style={{ color: "var(--text-tertiary)", marginLeft: 8 }}>
            {activeTabPath}
          </span>
        )}
      </div>

      {/* Right: Session info */}
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        {usage && (
          <span style={{ color: "var(--text-tertiary)" }}>
            Tokens: {usage.total_tokens?.toLocaleString() || 0}
          </span>
        )}
        <span style={{ color: "var(--text-tertiary)" }}>
          {config.mode.toUpperCase()} · {config.service}
        </span>
        <span style={{
          width: 6, height: 6, borderRadius: "50%",
          background: connected ? "var(--green)" : "var(--red)",
        }} />
      </div>
    </div>
  );
}
