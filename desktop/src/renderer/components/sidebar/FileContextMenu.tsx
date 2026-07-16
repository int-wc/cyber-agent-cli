import { useState, useRef, useEffect, useCallback } from "react";
import { FilePlus, FolderPlus, Trash2, Pencil, Copy, Scissors, ClipboardPaste } from "lucide-react";

export interface ContextMenuAction {
  label: string;
  icon: React.ReactNode;
  shortcut?: string;
  danger?: boolean;
  action: () => void;
}

interface Props {
  x: number;
  y: number;
  onClose: () => void;
  items: ContextMenuAction[];
}

export default function FileContextMenu({ x, y, onClose, items }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        onClose();
      }
    };
    const keyHandler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    // 延迟绑定事件，避免打开菜单的同一次点击立即触发关闭。
    const t = setTimeout(() => {
      document.addEventListener("mousedown", handler);
      document.addEventListener("keydown", keyHandler);
    }, 50);
    return () => {
      clearTimeout(t);
      document.removeEventListener("mousedown", handler);
      document.removeEventListener("keydown", keyHandler);
    };
  }, [onClose]);

  // 将菜单限制在视口内，避免右侧或底部被裁切。
  const [adjX, adjY] = (() => {
    const w = 220; const h = items.length * 34 + 8;
    let ax = x; let ay = y;
    if (ax + w > window.innerWidth) ax = window.innerWidth - w - 4;
    if (ay + h > window.innerHeight) ay = window.innerHeight - h - 4;
    return [Math.max(4, ax), Math.max(4, ay)];
  })();

  return (
    <div
      ref={ref}
      className="glass-panel"
      style={{
        position: "fixed",
        left: adjX, top: adjY,
        zIndex: 200,
        minWidth: 200,
        padding: "4px 0",
        fontSize: 12,
      }}
    >
      {items.map((item, i) => (
        <button
          key={i}
          onClick={() => { item.action(); onClose(); }}
          style={{
            display: "flex", alignItems: "center",
            width: "100%", padding: "7px 12px",
            border: "none", background: "transparent",
            cursor: "pointer", fontSize: 12,
            color: item.danger ? "var(--red)" : "var(--text-primary)",
            gap: 8,
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = item.danger
              ? "rgba(239,68,68,0.08)"
              : "rgba(124,111,247,0.06)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = "transparent";
          }}
        >
          <span style={{ display: "flex", alignItems: "center" }}>{item.icon}</span>
          <span style={{ flex: 1, textAlign: "left" }}>{item.label}</span>
          {item.shortcut && (
            <span style={{ color: "var(--text-tertiary)", fontSize: 10, marginLeft: 16 }}>
              {item.shortcut}
            </span>
          )}
        </button>
      ))}
    </div>
  );
}

// ── Pre-built menu builders ──

export function fileMenuItems(
  name: string, path: string,
  onNewFile: (parentDir: string) => void,
  onNewFolder: (parentDir: string) => void,
  onRename: (p: string) => void,
  onDelete: (p: string) => void,
): ContextMenuAction[] {
  return [
    // { label: "在新标签页打开", icon: <FilePlus size={14} />, shortcut: "⏎", action: () => {} },
    { label: "新建文件", icon: <FilePlus size={14} />, action: () => onNewFile(path) },
    { label: "新建文件夹", icon: <FolderPlus size={14} />, action: () => onNewFolder(path) },
    { label: "重命名", icon: <Pencil size={14} />, shortcut: "F2", action: () => onRename(path) },
    { label: "删除", icon: <Trash2 size={14} />, shortcut: "⌫", danger: true, action: () => onDelete(path) },
  ];
}

export function folderMenuItems(
  name: string, path: string,
  onNewFile: (parentDir: string) => void,
  onNewFolder: (parentDir: string) => void,
  onRename: (p: string) => void,
  onDelete: (p: string) => void,
): ContextMenuAction[] {
  return [
    { label: "新建文件", icon: <FilePlus size={14} />, action: () => onNewFile(path) },
    { label: "新建文件夹", icon: <FolderPlus size={14} />, action: () => onNewFolder(path) },
    { label: "重命名", icon: <Pencil size={14} />, shortcut: "F2", action: () => onRename(path) },
    { label: "删除", icon: <Trash2 size={14} />, shortcut: "⌫", danger: true, action: () => onDelete(path) },
  ];
}
