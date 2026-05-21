import { useCallback, useEffect, useState } from "react";
import { useEditorStore } from "../../stores/editorStore";
import { fsApi } from "../../services/api";
import type { FileEntry } from "../../types/agent";
import {
  ChevronRight, Folder, File, FolderOpen, RefreshCw,
  FolderOpen as FolderOpenIcon, FolderPlus,
} from "lucide-react";

// ── FileTreeNode ──

function FileTreeNode({ entry, depth, onSelect }: {
  entry: FileEntry;
  depth: number;
  onSelect: (path: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const [children, setChildren] = useState<FileEntry[] | null>(null);
  const [loading, setLoading] = useState(false);
  const activeTabPath = useEditorStore((s) => s.activeTabPath);

  const handleClick = useCallback(async () => {
    if (entry.is_dir) {
      if (!expanded && children === null) {
        setLoading(true);
        try {
          const res = await fsApi.list(entry.path);
          setChildren(res.entries.filter((e) => !e.name.startsWith(".")));
        } catch { /* ignore */ }
        setLoading(false);
      }
      setExpanded(!expanded);
    } else {
      try {
        const res = await fsApi.read(entry.path);
        onSelect(entry.path);
      } catch { /* ignore */ }
    }
  }, [entry, expanded, children, onSelect]);

  const isActive = activeTabPath === entry.path;

  return (
    <div>
      <div
        onClick={handleClick}
        style={{
          display: "flex", alignItems: "center", gap: 4,
          padding: "3px 8px", paddingLeft: 8 + depth * 16,
          cursor: "pointer", fontSize: 13,
          color: isActive ? "var(--accent)" : "var(--text-secondary)",
          background: isActive ? "rgba(124,111,247,0.08)" : "transparent",
          borderRadius: 4, margin: "1px 4px",
        }}
      >
        {entry.is_dir ? (
          <>
            <ChevronRight size={14} style={{
              transform: expanded ? "rotate(90deg)" : "rotate(0deg)",
              transition: "transform 150ms ease",
            }} />
            {expanded ? <FolderOpen size={14} /> : <Folder size={14} />}
          </>
        ) : (
          <>
            <span style={{ width: 14 }} />
            <File size={14} />
          </>
        )}
        <span className="truncate" style={{ flex: 1 }}>{entry.name}</span>
        {loading && <RefreshCw size={10} style={{ animation: "spin 1s linear infinite" }} />}
      </div>
      {expanded && children && (
        <div>
          {children.map((child) => (
            <FileTreeNode key={child.path} entry={child} depth={depth + 1} onSelect={onSelect} />
          ))}
        </div>
      )}
    </div>
  );
}

// ── empty state ──

function EmptyState({ onOpen }: { onOpen: () => void }) {
  return (
    <div style={{
      flex: 1, display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center",
      padding: 24, gap: 16, textAlign: "center",
    }}>
      <div style={{
        width: 56, height: 56, borderRadius: 14,
        background: "rgba(124,111,247,0.06)",
        display: "flex", alignItems: "center", justifyContent: "center",
      }}>
        <FolderOpenIcon size={26} color="var(--text-tertiary)" />
      </div>
      <div>
        <div style={{ fontSize: 13, fontWeight: 500, color: "var(--text-secondary)", marginBottom: 4 }}>
          未打开工作目录
        </div>
        <div style={{ fontSize: 11, color: "var(--text-tertiary)", lineHeight: 1.5 }}>
          打开一个文件夹以开始浏览文件
        </div>
      </div>
      <button
        className="glass-btn glass-btn-primary"
        style={{ fontSize: 12 }}
        onClick={onOpen}
      >
        <FolderPlus size={14} />
        打开文件夹
      </button>
    </div>
  );
}

// ── Sidebar ──

export default function Sidebar() {
  const [workspaceRoot, setWorkspaceRoot] = useState<string | null>(null);
  const [workspaceName, setWorkspaceName] = useState("");
  const [rootEntries, setRootEntries] = useState<FileEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const openFile = useEditorStore((s) => s.openFile);

  const loadRoot = useCallback(async (rootPath: string) => {
    setLoading(true);
    try {
      const res = await fsApi.list(rootPath);
      setRootEntries(res.entries.filter((e) => !e.name.startsWith(".")));
      setWorkspaceName(rootPath.split("/").pop() || rootPath.split("\\").pop() || rootPath);
    } catch {
      setRootEntries([]);
    }
    setLoading(false);
  }, []);

  const handleOpenFolder = useCallback(async () => {
    const api = window.electronAPI;
    if (api) {
      const folderPath = await api.openFileDialog({
        properties: ["openDirectory"],
      });
      if (folderPath) {
        setWorkspaceRoot(folderPath);
        loadRoot(folderPath);
      }
    }
  }, [loadRoot]);

  const handleChangeFolder = useCallback(() => {
    handleOpenFolder();
  }, [handleOpenFolder]);

  const handleSelect = useCallback(async (path: string) => {
    try {
      const res = await fsApi.read(path);
      openFile(path, res.content);
    } catch { /* ignore */ }
  }, [openFile]);

  // Try to auto-detect a reasonable default from env
  useEffect(() => {
    // Don't auto-select — wait for user to pick
  }, []);

  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column", background: "transparent" }}>
      {/* Header */}
      <div className="glass-surface" style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "8px 12px",
      }}>
        <span style={{
          fontSize: 12, fontWeight: 600, letterSpacing: "0.03em",
          color: "var(--text-secondary)",
        }}>
          {workspaceRoot ? workspaceName : "资源管理器"}
        </span>
        <div style={{ display: "flex", gap: 4 }}>
          {workspaceRoot && (
            <>
              <button className="glass-btn" style={{ padding: "2px 6px" }} onClick={() => loadRoot(workspaceRoot)}>
                <RefreshCw size={12} />
              </button>
              <button className="glass-btn" style={{ padding: "2px 6px", fontSize: 11 }} onClick={handleChangeFolder}>
                <FolderPlus size={12} />
              </button>
            </>
          )}
        </div>
      </div>

      {/* Content */}
      {!workspaceRoot ? (
        <EmptyState onOpen={handleOpenFolder} />
      ) : loading ? (
        <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center" }}>
          <span style={{ fontSize: 12, color: "var(--text-tertiary)" }}>加载中...</span>
        </div>
      ) : (
        <div style={{ flex: 1, overflow: "auto", padding: "4px 0" }}>
          {rootEntries.map((entry) => (
            <FileTreeNode key={entry.path} entry={entry} depth={0} onSelect={handleSelect} />
          ))}
        </div>
      )}
    </div>
  );
}
