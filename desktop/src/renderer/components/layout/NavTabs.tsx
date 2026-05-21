import { Globe, Wrench, FileCode, Shield, Bug } from "lucide-react";

export type CenterTab = "viewer" | "yakit" | "mitm";

interface NavTabsProps {
  active: CenterTab;
  onSelect: (tab: CenterTab) => void;
}

const TABS: { id: CenterTab; label: string; icon: React.ReactNode; hint: string }[] = [
  { id: "viewer", label: "阅览", icon: <FileCode size={14} />, hint: "文件 / 数据浏览" },
  { id: "yakit",   label: "Yakit",  icon: <Wrench size={14} />, hint: "安全工具集" },
  { id: "mitm",    label: "MITM",   icon: <Globe size={14} />, hint: "代理浏览器" },
];

export default function NavTabs({ active, onSelect }: NavTabsProps) {
  return (
    <div
      className="glass-surface"
      style={{
        display: "flex",
        alignItems: "center",
        flexShrink: 0,
        padding: "0 6px",
        gap: 2,
        minHeight: 36,
      }}
    >
      {TABS.map((tab) => {
        const isActive = active === tab.id;
        return (
          <button
            key={tab.id}
            onClick={() => onSelect(tab.id)}
            title={tab.hint}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
              padding: "5px 14px",
              borderRadius: 8,
              border: "none",
              cursor: "pointer",
              fontSize: 12,
              fontWeight: isActive ? 600 : 400,
              background: isActive
                ? "rgba(124,111,247,0.12)"
                : "transparent",
              color: isActive
                ? "var(--accent)"
                : "var(--text-secondary)",
              transition: "all 150ms var(--ease-out-expo)",
            }}
          >
            {tab.icon}
            {tab.label}
          </button>
        );
      })}
    </div>
  );
}
