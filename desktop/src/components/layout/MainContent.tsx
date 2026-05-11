import { useState, useCallback } from "react";
import { EditorArea } from "../editor/EditorArea";
import { TerminalPanel } from "../terminal/TerminalPanel";
import { FileTree } from "../filetree/FileTree";
import { GitPanel } from "../git/GitPanel";
import { useWorkspaceStore } from "@/stores/useWorkspaceStore";

export function MainContent() {
  const [termHeight, setTermHeight] = useState(200);
  const sidebarView = useWorkspaceStore((s) => s.sidebarView);

  const onDragTerminal = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const startY = e.clientY;
    const startH = termHeight;
    const onMove = (ev: MouseEvent) => {
      setTermHeight(Math.max(80, Math.min(600, startH + (startY - ev.clientY))));
    };
    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }, [termHeight]);

  const sidePanel = () => {
    switch (sidebarView) {
      case "files": return <FileTree />;
      case "git": return <GitPanel />;
      case "search": return (
        <div className="p-4 text-muted text-sm">搜索功能开发中...</div>
      );
      case "extensions": return (
        <div className="p-4 text-muted text-sm">扩展功能开发中...</div>
      );
      case "settings": return (
        <div className="p-4 text-muted text-sm">设置功能开发中...</div>
      );
    }
  };

  return (
    <div className="flex flex-1 overflow-hidden">
      {/* Side panel */}
      <div className="w-[260px] flex-shrink-0 glass-panel-heavy border-r border-b-0 border-t-0 border-l-0 border-border-glass overflow-hidden">
        {sidePanel()}
      </div>
      {/* Editor + Terminal */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="flex-1 overflow-hidden">
          <EditorArea />
        </div>
        <div className="resize-handle resize-handle-v" onMouseDown={onDragTerminal} />
        <div style={{ height: termHeight }} className="flex-shrink-0 overflow-hidden">
          <TerminalPanel />
        </div>
      </div>
    </div>
  );
}
