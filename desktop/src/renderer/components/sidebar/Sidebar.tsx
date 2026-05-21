import { useCallback, useEffect, useState } from "react";
import { useEditorStore } from "../../stores/editorStore";
import { fsApi } from "../../services/api";
import type { FileEntry } from "../../types/agent";
import { ChevronRight, Folder, File, FolderOpen, RefreshCw } from "lucide-react";

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

/** 判断路径是否为 OS 根目录（/ 或 Windows 盘符）。 */
function isOSRoot(p: string): boolean {
  return p === "/" || /^[A-Z]:\\?$/i.test(p);
}

export default function Sidebar() {
  const [roots, setRoots] = useState<FileEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const openFile = useEditorStore((s) => s.openFile);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const r = await fsApi.roots();
      // 只取 OS 根，过滤掉 cwd 等非系统根路径
      const osRoots = r.roots
        .filter((x) => isOSRoot(x.path))
        .map((x) => ({
          name: x.name,
          path: x.path,
          is_dir: true,
          size: 0,
          modified: 0,
        } as FileEntry));
      setRoots(osRoots);
    } catch { /* backend not ready */ }
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
      <div className="glass-surface" style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "8px 12px",
      }}>
        <span style={{ fontSize: 12, fontWeight: 600,
                       letterSpacing: "0.03em", color: "var(--text-secondary)" }}>
          磁盘浏览
        </span>
        <button className="glass-btn" style={{ padding: "2px 6px" }} onClick={refresh}>
          <RefreshCw size={12} />
        </button>
      </div>
      <div style={{ flex: 1, overflow: "auto", padding: "4px 0" }}>
        {loading ? (
          <div style={{ padding: 16, textAlign: "center", color: "var(--text-tertiary)", fontSize: 12 }}>
            检测磁盘...
          </div>
        ) : roots.length === 0 ? (
          <div style={{ padding: 16, textAlign: "center", color: "var(--text-tertiary)", fontSize: 12 }}>
            未检测到可用磁盘
          </div>
        ) : (
          roots.map((entry) => (
            <FileTreeNode key={entry.path} entry={entry} depth={0} onSelect={handleSelect} />
          ))
        )}
      </div>
    </div>
  );
}
