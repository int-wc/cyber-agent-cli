import { useEffect } from "react";
import { FileTreeNode } from "./FileTreeNode";
import { useWorkspaceStore } from "@/stores/useWorkspaceStore";
import { RefreshCw, FolderOpen } from "lucide-react";

export function FileTree() {
  const rootEntries = useWorkspaceStore((s) => s.rootEntries);
  const loadRootFiles = useWorkspaceStore((s) => s.loadRootFiles);
  const backendPort = useWorkspaceStore((s) => s.backendPort);

  useEffect(() => {
    if (backendPort) loadRootFiles();
  }, [backendPort, loadRootFiles]);

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between px-3 py-2 text-xs text-muted font-medium no-select">
        <div className="flex items-center gap-1.5">
          <FolderOpen size={14} />
          <span>资源管理器</span>
        </div>
        <button onClick={loadRootFiles} className="p-0.5 rounded hover:bg-surface-light">
          <RefreshCw size={12} />
        </button>
      </div>
      <div className="flex-1 overflow-y-auto px-1">
        {rootEntries.length === 0 ? (
          <p className="text-xs text-muted p-3">加载中...</p>
        ) : (
          rootEntries.map((entry) => (
            <FileTreeNode key={entry.path} entry={entry} depth={0} />
          ))
        )}
      </div>
    </div>
  );
}
