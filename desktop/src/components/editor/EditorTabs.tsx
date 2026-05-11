import { X, Circle } from "lucide-react";
import { useWorkspaceStore } from "@/stores/useWorkspaceStore";

export function EditorTabs() {
  const tabs = useWorkspaceStore((s) => s.openTabs);
  const activeTabPath = useWorkspaceStore((s) => s.activeTabPath);
  const setActiveTab = useWorkspaceStore((s) => s.setActiveTab);
  const closeTab = useWorkspaceStore((s) => s.closeTab);

  return (
    <div className="flex items-center gap-0 overflow-x-auto glass-panel-heavy border-b border-border-glass no-select" style={{ minHeight: 35 }}>
      {tabs.map((tab) => {
        const isActive = tab.path === activeTabPath;
        return (
          <div
            key={tab.path}
            onClick={() => setActiveTab(tab.path)}
            className={`
              flex items-center gap-1.5 px-3 py-1.5 cursor-pointer text-xs border-r border-border-glass
              transition-colors duration-150
              ${isActive ? "bg-window text-primary border-t-2 border-t-accent-teal" : "text-muted hover:bg-surface-light"}
            `}
          >
            <span className="truncate max-w-[120px]">{tab.name}</span>
            {tab.dirty && <Circle size={8} fill="#f59e0b" color="#f59e0b" />}
            <button
              onClick={(e) => { e.stopPropagation(); closeTab(tab.path); }}
              className="p-0.5 rounded hover:bg-surface rounded-full opacity-60 hover:opacity-100"
            >
              <X size={12} />
            </button>
          </div>
        );
      })}
    </div>
  );
}
