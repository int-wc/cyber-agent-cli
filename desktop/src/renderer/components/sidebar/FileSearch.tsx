import { useState, useEffect, useRef, useCallback } from "react";
import { useEditorStore } from "../../stores/editorStore";
import { fsApi } from "../../services/api";
import { Search, File, CornerDownLeft } from "lucide-react";

interface FileSearchProps {
  onClose: () => void;
}

export default function FileSearch({ onClose }: FileSearchProps) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<{ path: string; name: string }[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedIdx, setSelectedIdx] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const openFile = useEditorStore((s) => s.openFile);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  useEffect(() => {
    if (!query.trim()) {
      setResults([]);
      return;
    }
    setLoading(true);
    const timer = setTimeout(async () => {
      try {
        const res = await fsApi.search(query);
        setResults(res.results);
        setSelectedIdx(0);
      } catch {}
      setLoading(false);
    }, 200);
    return () => clearTimeout(timer);
  }, [query]);

  const handleSelect = useCallback(async (path: string) => {
    try {
      const res = await fsApi.read(path);
      openFile(path, res.content);
      onClose();
    } catch {}
  }, [openFile, onClose]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedIdx((i) => Math.min(i + 1, results.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedIdx((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (results[selectedIdx]) {
        handleSelect(results[selectedIdx].path);
      }
    } else if (e.key === "Escape") {
      onClose();
    }
  }, [results, selectedIdx, handleSelect, onClose]);

  return (
    <div className="glass-dialog-overlay" onClick={onClose}>
      <div className="glass-dialog" onClick={(e) => e.stopPropagation()} style={{ padding: 16 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
          <Search size={16} color="var(--text-secondary)" />
          <input
            ref={inputRef}
            className="glass-input"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="搜索文件..."
            style={{ flex: 1, fontSize: 14 }}
          />
        </div>

        {loading && (
          <div style={{ textAlign: "center", padding: 20, color: "var(--text-tertiary)", fontSize: 12 }}>
            搜索中...
          </div>
        )}

        {results.length > 0 && (
          <div style={{ maxHeight: 300, overflow: "auto" }}>
            {results.map((r, i) => (
              <div
                key={r.path}
                onClick={() => handleSelect(r.path)}
                style={{
                  display: "flex", alignItems: "center", gap: 8,
                  padding: "6px 10px", borderRadius: 6, cursor: "pointer",
                  background: i === selectedIdx ? "var(--accent-soft)" : "transparent",
                  color: i === selectedIdx ? "var(--text-primary)" : "var(--text-secondary)",
                  fontSize: 13, transition: "background 100ms ease",
                }}
                onMouseEnter={() => setSelectedIdx(i)}
              >
                <File size={14} />
                <span className="truncate" style={{ flex: 1 }}>{r.name}</span>
                <span style={{ fontSize: 10, color: "var(--text-tertiary)" }}>{r.path}</span>
              </div>
            ))}
          </div>
        )}

        {query && !loading && results.length === 0 && (
          <div style={{ textAlign: "center", padding: 20, color: "var(--text-tertiary)", fontSize: 12 }}>
            未找到匹配文件
          </div>
        )}

        <div style={{
          marginTop: 12, padding: "6px 0 0 0",
          borderTop: "1px solid rgba(255,255,255,0.06)",
          display: "flex", alignItems: "center", gap: 12, fontSize: 10, color: "var(--text-tertiary)",
        }}>
          <span>
            <CornerDownLeft size={10} style={{ marginRight: 2 }} />
            选择
          </span>
          <span>↑↓ 导航</span>
          <span>Esc 关闭</span>
        </div>
      </div>
    </div>
  );
}
