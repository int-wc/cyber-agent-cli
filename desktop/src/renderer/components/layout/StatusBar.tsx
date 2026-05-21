import { useSessionStore } from "../../stores/sessionStore";
import { useChatStore } from "../../stores/chatStore";
import { useEditorStore } from "../../stores/editorStore";
import { useUIStore } from "../../stores/uiStore";
import { Sidebar, MessageSquare, Terminal } from "lucide-react";

export default function StatusBar() {
  const { config, connected } = useSessionStore();
  const usage = useChatStore((s) => s.usage);
  const activeTabPath = useEditorStore((s) => s.activeTabPath);
  const {
    sidebarVisible, chatPanelVisible,
    toggleSidebar, toggleChatPanel,
    centerTab, setCenterTab, terminalTabs,
  } = useUIStore();

  const tabLabel = terminalTabs.includes(centerTab)
    ? `终端 ${terminalTabs.indexOf(centerTab) + 1}`
    : centerTab === "viewer" ? "阅览"
    : centerTab === "yakit" ? "Yakit 工具"
    : centerTab === "mitm" ? "MITM 浏览器"
    : centerTab;

  const jumpToTerminal = () => setCenterTab(terminalTabs[0] || "viewer");

  return (
    <div
      className="glass-surface"
      style={{
        height: 28, display: "flex", alignItems: "center",
        justifyContent: "space-between", padding: "0 10px",
        borderTop: "1px solid rgba(0,0,0,0.05)", fontSize: 11,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <button
          className="glass-btn"
          style={{
            padding: "2px 6px", borderRadius: 4, fontSize: 11,
            background: sidebarVisible ? "rgba(124,111,247,0.10)" : undefined,
          }}
          onClick={toggleSidebar}
        >
          <Sidebar size={12} />
        </button>
        <button
          className="glass-btn"
          style={{
            padding: "2px 6px", borderRadius: 4, fontSize: 11,
            background: terminalTabs.includes(centerTab) ? "rgba(124,111,247,0.10)" : undefined,
          }}
          onClick={jumpToTerminal}
        >
          <Terminal size={12} />
        </button>
        <button
          className="glass-btn"
          style={{
            padding: "2px 6px", borderRadius: 4, fontSize: 11,
            background: chatPanelVisible ? "rgba(124,111,247,0.10)" : undefined,
          }}
          onClick={toggleChatPanel}
        >
          <MessageSquare size={12} />
        </button>
        <span style={{ color: "var(--text-secondary)", fontWeight: 500 }}>
          {tabLabel}
        </span>
        {activeTabPath && (
          <span style={{ color: "var(--text-tertiary)" }}>
            · {activeTabPath}
          </span>
        )}
      </div>

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
