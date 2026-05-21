import { useCallback, useEffect, useState } from "react";
import { useEditorStore } from "../../stores/editorStore";
import { fsApi } from "../../services/api";
import type { FileEntry } from "../../types/agent";
import { ChevronRight, Folder, File, FolderOpen, RefreshCw, Search } from "lucide-react";

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
          setChildren(res.entries);
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
          color: isActive ? "var(--accent-light)" : "var(--text-secondary)",
          background: isActive ? "var(--accent-soft)" : "transparent",
          borderRadius: 4,
          margin: "1px 4px",
        }}
      >
        {entry.is_dir ? (
          <>
            <ChevronRight
              size={14}
              style={{
                transform: expanded ? "rotate(90deg)" : "rotate(0deg)",
                transition: "transform 150ms ease",
              }}
            />
            {expanded ? <FolderOpen size={14} /> : <Folder size={14} />}
          </>
        ) : (
          <>
            <span style={{ width: 14 }} />
            <File size={14} />
          </>
        )}
        <span className="truncate" style={{ flex: 1 }}>{entry.name}</span>
        {loading && (
          <RefreshCw size={10} style={{ animation: "spin 1s linear infinite" }} />
        )}
      </div>
      {expanded && children && (
        <div>
          {children.map((child) => (
            <FileTreeNode
              key={child.path}
              entry={child}
              depth={depth + 1}
              onSelect={onSelect}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export default function Sidebar() {
  const [rootEntries, setRootEntries] = useState<FileEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const openFile = useEditorStore((s) => s.openFile);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fsApi.list(".");
      setRootEntries(
        res.entries.filter((e) => !e.name.startsWith(".") || e.name === ".env")
      );
    } catch {
      // Backend may not be ready
    }
    setLoading(false);
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const handleSelect = useCallback(async (path: string) => {
    try {
      const res = await fsApi.read(path);
      openFile(path, res.content);
    } catch { /* ignore */ }
  }, [openFile]);

  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column", background: "transparent" }}>
      {/* Header */}
      <div
        className="glass-surface"
        style={{
          display: "flex", alignItems: "center", justifyContent: "space-between",
          padding: "8px 12px", borderBottom: "1px solid rgba(255,255,255,0.06)",
        }}
      >
        <span style={{ fontSize: 12, fontWeight: 600, textTransform: "uppercase",
                       letterSpacing: "0.05em", color: "var(--text-secondary)" }}>
          资源管理器
        </span>
        <div style={{ display: "flex", gap: 4 }}>
          <button className="glass-btn" style={{ padding: "2px 6px" }} onClick={refresh}>
            <RefreshCw size={12} />
          </button>
          <button className="glass-btn" style={{ padding: "2px 6px" }}>
            <Search size={12} />
          </button>
        </div>
      </div>
      {/* Tree */}
      <div style={{ flex: 1, overflow: "auto", padding: "4px 0" }}>
        {loading ? (
          <div style={{ padding: 16, textAlign: "center", color: "var(--text-tertiary)", fontSize: 12 }}>
            加载中...
          </div>
        ) : (
          rootEntries.map((entry) => (
            <FileTreeNode key={entry.path} entry={entry} depth={0} onSelect={handleSelect} />
          ))
        )}
      </div>
    </div>
  );
}
