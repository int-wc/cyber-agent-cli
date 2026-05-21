import { useCallback, useState } from "react";
import { useEditorStore } from "../../stores/editorStore";
import { fsApi } from "../../services/api";
import type { FileEntry } from "../../types/agent";
import {
  ChevronRight, RefreshCw, FolderOpen as FolderOpenIcon,
  FolderPlus, Ellipsis, Folder, FilePlus,
} from "lucide-react";
import { getFileIcon } from "./fileIcons";
import FileContextMenu, {
  ContextMenuAction, fileMenuItems, folderMenuItems,
} from "./FileContextMenu";

// ── FileTreeNode ──

function FileTreeNode({ entry, depth, onSelect, selectedPath, onSelectPath, onRefresh, onNewFile, onNewFolder, onRename, onDelete }: {
  entry: FileEntry;
  depth: number;
  onSelect: (path: string, content: string) => void;
  selectedPath: string | null;
  onSelectPath: (p: string) => void;
  onRefresh: () => void;
  onNewFile: (parentDir: string) => void;
  onNewFolder: (parentDir: string) => void;
  onRename: (path: string) => void;
  onDelete: (path: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const [children, setChildren] = useState<FileEntry[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number } | null>(null);
  const activeTabPath = useEditorStore((s) => s.activeTabPath);

  const loadChildren = useCallback(async () => {
    if (children !== null) return;
    setLoading(true);
    try {
      const res = await fsApi.list(entry.path);
      setChildren(res.entries.filter((e) => !e.name.startsWith(".")));
    } catch { /* ignore */ }
    setLoading(false);
  }, [entry.path, children]);

  const isSelected = selectedPath === entry.path;
  const isActive = activeTabPath === entry.path;

  const handleClick = useCallback(async () => {
    onSelectPath(entry.path);
    if (entry.is_dir) {
      if (!expanded) await loadChildren();
      setExpanded(!expanded);
    } else {
      try {
        const res = await fsApi.read(entry.path);
        onSelect(entry.path, res.content);
      } catch (err) {
        console.error("读取文件失败:", entry.path, err);
      }
    }
  }, [entry, expanded, loadChildren, onSelect, onSelectPath]);

  const handleContextMenu = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    onSelectPath(entry.path);
    setContextMenu({ x: e.clientX, y: e.clientY });
  }, [entry.path, onSelectPath]);

  const meta = getFileIcon(entry.name, entry.is_dir, expanded);

  const items: ContextMenuAction[] = entry.is_dir
    ? folderMenuItems(entry.name, entry.path, onNewFile, onNewFolder, onRename, onDelete)
    : fileMenuItems(entry.name, entry.path, onNewFile, onNewFolder, onRename, onDelete);

  const IconComp = meta.icon;

  return (
    <div>
      <div
        onClick={handleClick}
        onContextMenu={handleContextMenu}
        style={{
          display: "flex", alignItems: "center", gap: 4,
          padding: "3px 8px", paddingLeft: 8 + depth * 16,
          cursor: "pointer", fontSize: 13,
          color: isActive ? "var(--accent)" : "var(--text-secondary)",
          background: isSelected
            ? (isActive ? "rgba(124,111,247,0.12)" : "rgba(124,111,247,0.06)")
            : "transparent",
          outline: isSelected ? "1px solid rgba(124,111,247,0.18)" : "none",
          outlineOffset: -1,
          borderRadius: 4, margin: "1px 4px",
          userSelect: "none",
          fontWeight: isSelected ? 500 : 400,
        }}
      >
        {entry.is_dir ? (
          <ChevronRight size={14} style={{
            transform: expanded ? "rotate(90deg)" : "rotate(0deg)",
            transition: "transform 150ms ease", flexShrink: 0,
          }} />
        ) : (
          <span style={{ width: 14, flexShrink: 0 }} />
        )}
        <IconComp size={15} color={meta.color} style={{ flexShrink: 0 }} />
        <span className="truncate" style={{ flex: 1, marginLeft: 2 }}>{entry.name}</span>
        {loading && <RefreshCw size={10} style={{ animation: "spin 1s linear infinite" }} />}
      </div>

      {expanded && children && (
        <div>
          {children.map((child) => (
            <FileTreeNode
              key={child.path}
              entry={child}
              depth={depth + 1}
              onSelect={onSelect}
              selectedPath={selectedPath}
              onSelectPath={onSelectPath}
              onRefresh={onRefresh}
              onNewFile={onNewFile}
              onNewFolder={onNewFolder}
              onRename={onRename}
              onDelete={onDelete}
            />
          ))}
        </div>
      )}

      {contextMenu && (
        <FileContextMenu
          x={contextMenu.x}
          y={contextMenu.y}
          items={items}
          onClose={() => setContextMenu(null)}
        />
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
      <button className="glass-btn glass-btn-primary" style={{ fontSize: 12 }} onClick={onOpen}>
        <FolderPlus size={14} />
        打开文件夹
      </button>
    </div>
  );
}

// ── Workspace header ──

function WorkspaceHeader({
  root, name, onSwitch, onRefresh, onNewFile, onNewFolder, recent,
}: {
  root: string; name: string; onSwitch: (p: string) => void;
  onRefresh: () => void;
  onNewFile: (parentDir: string) => void;
  onNewFolder: (parentDir: string) => void;
  recent: string[];
}) {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <div style={{ position: "relative" }}>
      <div className="glass-surface" style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "6px 8px",
      }}>
        <div
          onClick={() => onSwitch(root)}
          style={{
            flex: 1, minWidth: 0, cursor: "pointer",
            borderRadius: 6, padding: "4px 8px",
            transition: "background 120ms ease",
          }}
          onMouseEnter={(e) => (e.currentTarget.style.background = "rgba(124,111,247,0.06)")}
          onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
        >
          <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text-primary)",
                        overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {name}
          </div>
          <div style={{ fontSize: 10, color: "var(--text-tertiary)",
                        overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", marginTop: 1 }}>
            {root}
          </div>
        </div>

        <div style={{ display: "flex", gap: 2, flexShrink: 0 }}>
          <button className="glass-btn" style={{ padding: "2px 6px" }} onClick={() => onNewFile(root)} title="新建文件">
            <FilePlus size={12} />
          </button>
          <button className="glass-btn" style={{ padding: "2px 6px" }} onClick={() => onNewFolder(root)} title="新建文件夹">
            <FolderPlus size={12} />
          </button>
          <button className="glass-btn" style={{ padding: "2px 6px" }} onClick={onRefresh} title="刷新">
            <RefreshCw size={12} />
          </button>
          <button className="glass-btn" style={{ padding: "2px 6px" }} onClick={() => setMenuOpen(!menuOpen)}>
            <Ellipsis size={12} />
          </button>
        </div>
      </div>

      {menuOpen && (
        <>
          <div style={{ position: "fixed", inset: 0, zIndex: 49 }} onClick={() => setMenuOpen(false)} />
          <div className="glass-panel" style={{ position: "absolute", top: "100%", right: 8, zIndex: 50,
                                                  minWidth: 200, padding: "4px 0", marginTop: 4 }}>
            <button style={{ display: "flex", alignItems: "center", gap: 8, width: "100%",
                              padding: "7px 12px", border: "none", background: "transparent",
                              cursor: "pointer", fontSize: 12, color: "var(--text-primary)" }}
                    onClick={() => { setMenuOpen(false); onSwitch(root); }}>
              <FolderPlus size={14} /> 切换工作目录...
            </button>
            {recent.filter((x) => x !== root).slice(0, 5).map((p) => (
              <button key={p}
                style={{ display: "flex", alignItems: "center", gap: 8, width: "100%",
                          padding: "6px 12px", border: "none", background: "transparent",
                          cursor: "pointer", fontSize: 12, color: "var(--text-secondary)",
                          overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
                onClick={() => { setMenuOpen(false); onSwitch(p); }}>
                <Folder size={13} style={{ flexShrink: 0, color: "var(--text-tertiary)" }} />
                {p.split("/").pop() || p.split("\\").pop() || p}
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

// ── Recent folders (localStorage) ──

const RECENT_KEY = "cyber-ide-recent-folders";
const MAX_RECENT = 8;

function getRecentFolders(): string[] {
  try { return JSON.parse(localStorage.getItem(RECENT_KEY) || "[]"); }
  catch { return []; }
}
function addRecentFolder(p: string) {
  const list = [p, ...getRecentFolders().filter((x) => x !== p)].slice(0, MAX_RECENT);
  localStorage.setItem(RECENT_KEY, JSON.stringify(list));
}

// ── Sidebar ──

export default function Sidebar() {
  const [workspaceRoot, setWorkspaceRoot] = useState<string | null>(null);
  const [workspaceName, setWorkspaceName] = useState("");
  const [rootEntries, setRootEntries] = useState<FileEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [recentFolders, setRecentFolders] = useState<string[]>(getRecentFolders);
  const openFile = useEditorStore((s) => s.openFile);

  const loadRoot = useCallback(async (rootPath: string) => {
    setLoading(true);
    setSelectedPath(null);
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
      setSelectedPath(null);
      loadRoot(folderPath);
    }
  }, [loadRoot]);

  const handleSelect = useCallback((path: string, content: string) => {
    openFile(path, content);
  }, [openFile]);

  // ── File operations ──

  const handleNewFile = useCallback(async (parentDir: string) => {
    const name = prompt("文件名:");
    if (!name) return;
    const filePath = parentDir.replace(/\/?$/, "/") + name;
    try {
      await fsApi.write(filePath, "");
      loadRoot(workspaceRoot!);
    } catch { /* ignore */ }
  }, [loadRoot, workspaceRoot]);

  const handleNewFolder = useCallback(async (parentDir: string) => {
    const name = prompt("文件夹名:");
    if (!name) return;
    const dirPath = parentDir.replace(/\/?$/, "/") + name;
    try {
      await fsApi.createDir(dirPath);
      loadRoot(workspaceRoot!);
    } catch { /* ignore */ }
  }, [loadRoot, workspaceRoot]);

  const handleRename = useCallback(async (path: string) => {
    const oldName = path.split("/").pop() || path;
    const newName = prompt("重命名为:", oldName);
    if (!newName || newName === oldName) return;
    const newPath = path.replace(/[^/]+$/, newName);
    try {
      await fsApi.rename(path, newPath);
      loadRoot(workspaceRoot!);
    } catch { /* ignore */ }
  }, [loadRoot, workspaceRoot]);

  const handleDelete = useCallback(async (path: string) => {
    const name = path.split("/").pop() || path;
    if (!confirm(`确认删除 "${name}"？`)) return;
    try {
      await fsApi.delete(path);
      loadRoot(workspaceRoot!);
    } catch { /* ignore */ }
  }, [loadRoot, workspaceRoot]);

  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column", background: "transparent" }}>
      {workspaceRoot ? (
        <WorkspaceHeader
          root={workspaceRoot}
          name={workspaceName}
          onSwitch={pickAndLoad}
          onRefresh={() => loadRoot(workspaceRoot)}
          onNewFile={handleNewFile}
          onNewFolder={handleNewFolder}
          recent={recentFolders}
        />
      ) : (
        <div className="glass-surface" style={{ display: "flex", alignItems: "center", padding: "10px 12px" }}>
          <span style={{ fontSize: 12, fontWeight: 600, letterSpacing: "0.03em", color: "var(--text-secondary)" }}>
            资源管理器
          </span>
        </div>
      )}

      {!workspaceRoot ? (
        <EmptyState onOpen={() => pickAndLoad()} />
      ) : loading ? (
        <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center" }}>
          <span style={{ fontSize: 12, color: "var(--text-tertiary)" }}>加载中...</span>
        </div>
      ) : (
        <div style={{ flex: 1, overflow: "auto", padding: "4px 0" }}>
          {rootEntries.map((entry) => (
            <FileTreeNode
              key={entry.path}
              entry={entry}
              depth={0}
              onSelect={handleSelect}
              selectedPath={selectedPath}
              onSelectPath={setSelectedPath}
              onRefresh={() => loadRoot(workspaceRoot!)}
              onNewFile={handleNewFile}
              onNewFolder={handleNewFolder}
              onRename={handleRename}
              onDelete={handleDelete}
            />
          ))}
        </div>
      )}
    </div>
  );
}
