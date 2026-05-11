import { ChevronRight, Folder, FolderOpen, File, FileCode } from "lucide-react";
import type { FileEntry } from "@/types/file";
import { useWorkspaceStore } from "@/stores/useWorkspaceStore";

const CODE_EXTS = new Set(["ts", "tsx", "js", "jsx", "py", "rs", "go", "java", "c", "cpp", "h", "css", "html", "json", "yaml", "yml", "toml", "sql", "sh", "bash", "xml"]);

function getIcon(entry: FileEntry) {
  if (entry.type === "dir") return null;
  const ext = entry.name.split(".").pop()?.toLowerCase();
  if (ext && CODE_EXTS.has(ext)) return <FileCode size={15} className="text-accent-teal" />;
  return <File size={15} className="text-muted" />;
}

export function FileTreeNode({ entry, depth }: { entry: FileEntry; depth: number }) {
  const expandedDirs = useWorkspaceStore((s) => s.expandedDirs);
  const toggleDir = useWorkspaceStore((s) => s.toggleDir);
  const loadDirChildren = useWorkspaceStore((s) => s.loadDirChildren);
  const openFile = useWorkspaceStore((s) => s.openFile);
  const activeTabPath = useWorkspaceStore((s) => s.activeTabPath);

  const isExpanded = expandedDirs.has(entry.path);
  const isActive = activeTabPath === entry.path;

  if (entry.type === "dir") {
    return (
      <div>
        <div
          className={`flex items-center gap-0.5 py-0.5 px-1 cursor-pointer rounded text-xs hover:bg-surface-light no-select ${isExpanded ? "text-primary" : "text-soft"}`}
          style={{ paddingLeft: depth * 16 + 4 }}
          onClick={() => {
            if (!entry.loaded) loadDirChildren(entry.path);
            toggleDir(entry.path);
          }}
        >
          <ChevronRight size={14} className={`transition-transform duration-150 ${isExpanded ? "rotate-90" : ""}`} />
          {isExpanded ? <FolderOpen size={15} className="text-accent-amber" /> : <Folder size={15} className="text-accent-amber" />}
          <span className="ml-1 truncate">{entry.name}</span>
        </div>
        {isExpanded && entry.children?.map((child) => (
          <FileTreeNode key={child.path} entry={child} depth={depth + 1} />
        ))}
      </div>
    );
  }

  return (
    <div
      className={`flex items-center gap-0.5 py-0.5 px-1 cursor-pointer rounded text-xs hover:bg-surface-light no-select ${isActive ? "bg-surface-light text-primary" : "text-soft"}`}
      style={{ paddingLeft: depth * 16 + 4 + 20 }}
      onClick={() => openFile(entry.path)}
    >
      {getIcon(entry)}
      <span className="ml-1 truncate">{entry.name}</span>
    </div>
  );
}
