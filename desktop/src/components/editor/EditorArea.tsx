import { MonacoEditor } from "./MonacoEditor";
import { EditorTabs } from "./EditorTabs";
import { useWorkspaceStore } from "@/stores/useWorkspaceStore";

export function EditorArea() {
  const tabs = useWorkspaceStore((s) => s.openTabs);
  const activeTabPath = useWorkspaceStore((s) => s.activeTabPath);
  const activeTab = tabs.find((t) => t.path === activeTabPath);

  if (tabs.length === 0) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <div className="text-6xl mb-4 opacity-30">🔷</div>
          <p className="text-muted text-sm">从左侧文件树打开文件开始编辑</p>
          <p className="text-muted text-xs mt-2 opacity-60">
            Ctrl+P 快速打开文件
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col overflow-hidden">
      <EditorTabs />
      <div className="flex-1 overflow-hidden">
        {activeTab ? (
          <MonacoEditor
            key={activeTab.path}
            path={activeTab.path}
            language={activeTab.language}
            value={activeTab.content || ""}
          />
        ) : (
          <div className="h-full flex items-center justify-center text-muted text-sm">
            选择一个标签页
          </div>
        )}
      </div>
    </div>
  );
}
