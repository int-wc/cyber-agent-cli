import { Files, Search, GitBranch, Puzzle, Settings } from "lucide-react";
import { useWorkspaceStore } from "@/stores/useWorkspaceStore";
import type { SidebarView } from "@/types/workspace";

const ICONS: { view: SidebarView; icon: typeof Files; label: string }[] = [
  { view: "files", icon: Files, label: "文件浏览器" },
  { view: "search", icon: Search, label: "搜索" },
  { view: "git", icon: GitBranch, label: "版本控制" },
  { view: "extensions", icon: Puzzle, label: "扩展" },
  { view: "settings", icon: Settings, label: "设置" },
];

export function Sidebar() {
  const sidebarView = useWorkspaceStore((s) => s.sidebarView);
  const setSidebarView = useWorkspaceStore((s) => s.setSidebarView);

  return (
    <div className="h-full w-[48px] flex-shrink-0 flex flex-col items-center gap-1 pt-3 pb-3 glass-panel-heavy border-r border-b-0 border-l-0 border-t-0 border-border-glass no-select">
      {ICONS.slice(0, 3).map(({ view, icon: Icon, label }) => (
        <button
          key={view}
          title={label}
          onClick={() => setSidebarView(view)}
          className={`glass-icon-btn ${sidebarView === view ? "active" : ""}`}
        >
          <Icon size={20} />
        </button>
      ))}
      <div className="my-2 w-6 h-px bg-border" />
      {ICONS.slice(3).map(({ view, icon: Icon, label }) => (
        <button
          key={view}
          title={label}
          onClick={() => setSidebarView(view)}
          className={`glass-icon-btn ${sidebarView === view ? "active" : ""}`}
        >
          <Icon size={20} />
        </button>
      ))}
    </div>
  );
}
