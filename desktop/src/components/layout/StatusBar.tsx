import { useWorkspaceStore } from "@/stores/useWorkspaceStore";
import { useChatStore } from "@/stores/useChatStore";
import { Circle, GitBranch } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "@/services/api";

export function StatusBar({ chatWidth }: { chatWidth: number }) {
  const backendStatus = useWorkspaceStore((s) => s.backendStatus);
  const wsConnected = useChatStore((s) => s.wsConnected);
  const activeTab = useWorkspaceStore((s) => {
    const tab = s.openTabs.find((t) => t.path === s.activeTabPath);
    return tab;
  });
  const [branch, setBranch] = useState("main");
  const [cursorLine, setCursorLine] = useState<number | null>(null);

  useEffect(() => {
    api.gitStatus().then((s) => { if (s.branch) setBranch(s.branch); }).catch(() => {});
  }, []);

  const statusColor = wsConnected ? "#22c55e" : backendStatus === "connected" ? "#f59e0b" : "#64748b";
  const statusText = wsConnected ? "已连接" : backendStatus === "connected" ? "等待中" : "离线";

  return (
    <div
      className="h-7 flex-shrink-0 flex items-center gap-4 px-3 text-xs text-muted glass-panel-heavy border-t border-border-glass no-select"
      style={{ paddingRight: chatWidth + 16 }}
    >
      <div className="flex items-center gap-2">
        <GitBranch size={12} />
        <span>{branch}</span>
      </div>
      <span className="flex-1" />
      {activeTab && (
        <span>{activeTab.name}</span>
      )}
      {cursorLine !== null && (
        <span>行 {cursorLine}</span>
      )}
      <div className="flex items-center gap-1.5">
        <Circle size={8} fill={statusColor} color={statusColor} />
        <span>{statusText}</span>
      </div>
    </div>
  );
}
