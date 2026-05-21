import { useCallback, useState } from "react";
import { useEditorStore } from "../../stores/editorStore";
import { fsApi } from "../../services/api";
import type { FileEntry } from "../../types/agent";
import {
  ChevronRight, Folder, File, FolderOpen, RefreshCw,
  FolderOpen as FolderOpenIcon, FolderPlus, Ellipsis,
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

// ── Recent folders (persisted in localStorage) ──

const RECENT_KEY = "cyber-ide-recent-folders";
const MAX_RECENT = 8;

function getRecentFolders(): string[] {
  try {
    return JSON.parse(localStorage.getItem(RECENT_KEY) || "[]");
  } catch { return []; }
}

function addRecentFolder(p: string) {
  const list = [p, ...getRecentFolders().filter((x) => x !== p)].slice(0, MAX_RECENT);
  localStorage.setItem(RECENT_KEY, JSON.stringify(list));
}

// ── Workspace header ──

function WorkspaceHeader({
  root, name, onSwitch, onRefresh, recent,
}: {
  root: string; name: string; onSwitch: (p: string) => void;
  onRefresh: () => void; recent: string[];
}) {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <div style={{ position: "relative" }}>
      <div className="glass-surface" style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "6px 8px",
      }}>
        {/* Path area — click to switch */}
        <div
          onClick={() => onSwitch(root)}
          style={{
            flex: 1, minWidth: 0, cursor: "pointer",
            borderRadius: 6, padding: "4px 8px",
            background: "transparent",
            transition: "background 120ms ease",
          }}
          onMouseEnter={(e) => (e.currentTarget.style.background = "rgba(124,111,247,0.06)")}
          onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
        >
          <div style={{
            fontSize: 12, fontWeight: 600, color: "var(--text-primary)",
            overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
          }}>
            {name}
          </div>
          <div style={{
            fontSize: 10, color: "var(--text-tertiary)",
            overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
            marginTop: 1,
          }}>
            {root}
          </div>
        </div>

        {/* Actions */}
        <div style={{ display: "flex", gap: 2, flexShrink: 0 }}>
          <button className="glass-btn" style={{ padding: "2px 6px" }} onClick={onRefresh} title="刷新">
            <RefreshCw size={12} />
          </button>
          <button
            className="glass-btn"
            style={{ padding: "2px 6px" }}
            onClick={() => setMenuOpen(!menuOpen)}
            title="更多"
          >
            <Ellipsis size={12} />
          </button>
        </div>
      </div>

      {/* Dropdown menu */}
      {menuOpen && (
        <>
          <div
            style={{ position: "fixed", inset: 0, zIndex: 49 }}
            onClick={() => setMenuOpen(false)}
          />
          <div className="glass-panel" style={{
            position: "absolute", top: "100%", right: 8, zIndex: 50,
            minWidth: 200, padding: "4px 0", marginTop: 4,
          }}>
            <button
              style={{
                display: "flex", alignItems: "center", gap: 8,
                width: "100%", padding: "7px 12px", border: "none",
                background: "transparent", cursor: "pointer",
                fontSize: 12, color: "var(--text-primary)",
              }}
              onClick={() => { setMenuOpen(false); onSwitch(root); }}
            >
              <FolderPlus size={14} />
              切换工作目录...
            </button>

            {recent.length > 0 && (
              <>
                <div style={{
                  padding: "4px 12px", fontSize: 10, color: "var(--text-tertiary)",
                  textTransform: "uppercase", letterSpacing: "0.05em",
                }}>
                  最近打开
                </div>
                {recent.filter((x) => x !== root).slice(0, 5).map((p) => (
                  <button
                    key={p}
                    style={{
                      display: "flex", alignItems: "center", gap: 8,
                      width: "100%", padding: "6px 12px", border: "none",
                      background: "transparent", cursor: "pointer",
                      fontSize: 12, color: "var(--text-secondary)",
                      overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                    }}
                    onClick={() => { setMenuOpen(false); onSwitch(p); }}
                  >
                    <Folder size={13} style={{ flexShrink: 0, color: "var(--text-tertiary)" }} />
                    {p.split("/").pop() || p.split("\\").pop() || p}
                    <span style={{ fontSize: 10, color: "var(--text-tertiary)", marginLeft: "auto" }}>
                      {p}
                    </span>
                  </button>
                ))}
              </>
            )}
          </div>
        </>
      )}
    </div>
  );
}

// ── Sidebar ──

export default function Sidebar() {
  const [workspaceRoot, setWorkspaceRoot] = useState<string | null>(null);
  const [workspaceName, setWorkspaceName] = useState("");
  const [rootEntries, setRootEntries] = useState<FileEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [recentFolders, setRecentFolders] = useState<string[]>(getRecentFolders);
  const openFile = useEditorStore((s) => s.openFile);

  const loadRoot = useCallback(async (rootPath: string) => {
    setLoading(true);
    try {
      const res = await fsApi.list(rootPath);
      setRootEntries(res.entries.filter((e) => !e.name.startsWith(".")));
      setWorkspaceName(rootPath.split("/").pop() || rootPath.split("\\").pop() || rootPath);
      addRecentFolder(rootPath);
      setRecentFolders(getRecentFolders());
    } catch {
      setRootEntries([]);
    }
    setLoading(false);
  }, []);

  const pickAndLoad = useCallback(async (currentRoot?: string) => {
    const api = window.electronAPI;
    if (!api) return;
    const folderPath = await api.openFileDialog({
      properties: ["openDirectory"],
      ...(currentRoot ? { defaultPath: currentRoot } : {}),
    });
    if (folderPath) {
      setWorkspaceRoot(folderPath);
      loadRoot(folderPath);
    }
  }, [loadRoot]);

  const handleSelect = useCallback(async (path: string) => {
    try {
      const res = await fsApi.read(path);
      openFile(path, res.content);
    } catch { /* ignore */ }
  }, [openFile]);

  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column", background: "transparent" }}>
      {/* Header */}
      {workspaceRoot ? (
        <WorkspaceHeader
          root={workspaceRoot}
          name={workspaceName}
          onSwitch={pickAndLoad}
          onRefresh={() => loadRoot(workspaceRoot)}
          recent={recentFolders}
        />
      ) : (
        <div className="glass-surface" style={{
          display: "flex", alignItems: "center", padding: "10px 12px",
        }}>
          <span style={{
            fontSize: 12, fontWeight: 600, letterSpacing: "0.03em",
            color: "var(--text-secondary)",
          }}>
            资源管理器
          </span>
        </div>
      )}

      {/* Content */}
      {!workspaceRoot ? (
        <EmptyState onOpen={() => pickAndLoad()} />
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
