import { Globe, Wrench, FileCode, Terminal, Plus, X } from "lucide-react";
import { useUIStore, CenterTab } from "../../stores/uiStore";

const FIXED_TABS: { id: CenterTab; label: string; icon: React.ReactNode; hint: string }[] = [
  { id: "viewer", label: "编辑器", icon: <FileCode size={14} />, hint: "代码编辑 / 文件浏览" },
  { id: "yakit",  label: "Yakit", icon: <Wrench size={14} />, hint: "安全工具集" },
  { id: "mitm",   label: "MITM",  icon: <Globe size={14} />, hint: "代理浏览器" },
];

export default function NavTabs() {
  const { centerTab, setCenterTab, terminalTabs, addTerminal, removeTerminal } = useUIStore();

  return (
    <div
      className="glass-surface"
      style={{
        display: "flex",
        alignItems: "center",
        flexShrink: 0,
        padding: "0 4px",
        gap: 1,
        minHeight: 36,
        overflowX: "auto",
      }}
    >
      {/* Fixed tabs */}
      {FIXED_TABS.map((tab) => {
        const isActive = centerTab === tab.id;
        return (
          <button
            key={tab.id}
            onClick={() => setCenterTab(tab.id)}
            title={tab.hint}
            style={{
              display: "flex", alignItems: "center", gap: 5,
              padding: "5px 12px", borderRadius: 8, border: "none",
              cursor: "pointer", fontSize: 12, whiteSpace: "nowrap",
              fontWeight: isActive ? 600 : 400,
              background: isActive ? "rgba(124,111,247,0.12)" : "transparent",
              color: isActive ? "var(--accent)" : "var(--text-secondary)",
              transition: "all 150ms var(--ease-out-expo)",
            }}
          >
            {tab.icon}
            {tab.label}
          </button>
        );
      })}

      {/* Separator */}
      <div style={{ width: 1, height: 18, background: "rgba(0,0,0,0.08)", margin: "0 4px", flexShrink: 0 }} />

      {/* Dynamic terminal tabs */}
      {terminalTabs.map((termId, i) => {
        const isActive = centerTab === termId;
        return (
          <button
            key={termId}
            onClick={() => setCenterTab(termId)}
            style={{
              display: "flex", alignItems: "center", gap: 4,
              padding: "5px 8px 5px 10px", borderRadius: 8, border: "none",
              cursor: "pointer", fontSize: 12, whiteSpace: "nowrap",
              fontWeight: isActive ? 600 : 400,
              background: isActive ? "rgba(124,111,247,0.12)" : "transparent",
              color: isActive ? "var(--accent)" : "var(--text-secondary)",
              transition: "all 150ms var(--ease-out-expo)",
            }}
          >
            <Terminal size={12} />
            终端 {i + 1}
            {terminalTabs.length > 1 && (
              <span
                onClick={(e) => { e.stopPropagation(); removeTerminal(termId); }}
                style={{
                  display: "inline-flex", alignItems: "center", justifyContent: "center",
                  width: 16, height: 16, borderRadius: 4, marginLeft: 2,
                  color: "var(--text-tertiary)",
                }}
                title="关闭终端"
              >
                <X size={10} />
              </span>
            )}
          </button>
        );
      })}

      {/* Add terminal button */}
      <button
        onClick={addTerminal}
        className="glass-btn"
        style={{ padding: "3px 8px", flexShrink: 0, marginLeft: 2 }}
        title="新建终端"
      >
        <Plus size={12} />
      </button>
    </div>
  );
}
