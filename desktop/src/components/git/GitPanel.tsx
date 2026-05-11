import { useEffect, useState } from "react";
import { GitBranch, Plus, Minus, RefreshCw } from "lucide-react";
import { api } from "@/services/api";
import type { GitStatus, GitCommit } from "@/types/git";
import { GlassButton } from "@/components/ui/GlassButton";

export function GitPanel() {
  const [status, setStatus] = useState<GitStatus | null>(null);
  const [commits, setCommits] = useState<GitCommit[]>([]);
  const [message, setMessage] = useState("");
  const [selectedFiles, setSelectedFiles] = useState<Set<string>>(new Set());
  const [view, setView] = useState<"changes" | "history">("changes");
  const [loading, setLoading] = useState(false);

  const loadStatus = async () => {
    try {
      const s = await api.gitStatus();
      setStatus(s);
    } catch {}
  };

  const loadCommits = async () => {
    try {
      const log = await api.gitLog(20);
      setCommits(log.commits);
    } catch {}
  };

  useEffect(() => {
    loadStatus();
    loadCommits();
  }, []);

  const toggleFile = (path: string) => {
    setSelectedFiles((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  };

  const handleStage = async () => {
    if (selectedFiles.size === 0) return;
    setLoading(true);
    await api.gitStage([...selectedFiles]);
    setSelectedFiles(new Set());
    await loadStatus();
    setLoading(false);
  };

  const handleUnstage = async () => {
    if (selectedFiles.size === 0) return;
    setLoading(true);
    await api.gitUnstage([...selectedFiles]);
    setSelectedFiles(new Set());
    await loadStatus();
    setLoading(false);
  };

  const handleCommit = async () => {
    if (!message.trim()) return;
    setLoading(true);
    const result = await api.gitCommit(message);
    if (result.status === "ok") {
      setMessage("");
      await loadStatus();
      await loadCommits();
    }
    setLoading(false);
  };

  return (
    <div className="h-full flex flex-col text-xs">
      <div className="flex items-center justify-between px-3 py-2 text-xs text-muted font-medium no-select border-b border-border-glass">
        <div className="flex items-center gap-1.5">
          <GitBranch size={14} />
          <span>版本控制</span>
          {status?.branch && <span className="text-accent-teal ml-1">{status.branch}</span>}
        </div>
        <button onClick={loadStatus} className="p-0.5 rounded hover:bg-surface-light">
          <RefreshCw size={12} className={loading ? "animate-spin" : ""} />
        </button>
      </div>

      {/* View tabs */}
      <div className="flex gap-0 border-b border-border-glass no-select">
        <button
          onClick={() => setView("changes")}
          className={`flex-1 py-1.5 text-center text-xs ${view === "changes" ? "text-primary border-b border-accent-teal" : "text-muted hover:text-soft"}`}
        >
          更改
        </button>
        <button
          onClick={() => setView("history")}
          className={`flex-1 py-1.5 text-center text-xs ${view === "history" ? "text-primary border-b border-accent-teal" : "text-muted hover:text-soft"}`}
        >
          历史
        </button>
      </div>

      <div className="flex-1 overflow-y-auto">
        {view === "changes" && (
          <div className="p-2 space-y-2">
            {status?.staged.length === 0 && status?.unstaged.length === 0 && status?.untracked.length === 0 ? (
              <p className="text-muted text-xs p-3 text-center">没有更改</p>
            ) : (
              <>
                {status?.staged.map((f) => (
                  <div
                    key={f.path}
                    onClick={() => toggleFile(f.path)}
                    className={`flex items-center gap-1.5 p-1 rounded cursor-pointer hover:bg-surface-light ${selectedFiles.has(f.path) ? "bg-surface-light" : ""}`}
                  >
                    <span className="text-[10px] text-green-400 w-6">暂存</span>
                    <span className="text-soft truncate flex-1">{f.path}</span>
                  </div>
                ))}
                {status?.unstaged.map((f) => (
                  <div
                    key={f.path}
                    onClick={() => toggleFile(f.path)}
                    className={`flex items-center gap-1.5 p-1 rounded cursor-pointer hover:bg-surface-light ${selectedFiles.has(f.path) ? "bg-surface-light" : ""}`}
                  >
                    <span className="text-[10px] text-amber-400 w-6">修改</span>
                    <span className="text-soft truncate flex-1">{f.path}</span>
                  </div>
                ))}
                {status?.untracked.map((f) => (
                  <div
                    key={f}
                    onClick={() => toggleFile(f)}
                    className={`flex items-center gap-1.5 p-1 rounded cursor-pointer hover:bg-surface-light ${selectedFiles.has(f) ? "bg-surface-light" : ""}`}
                  >
                    <span className="text-[10px] text-red-400 w-6">新建</span>
                    <span className="text-soft truncate flex-1">{f}</span>
                  </div>
                ))}
                {selectedFiles.size > 0 && (
                  <div className="flex gap-2 pt-2">
                    <GlassButton small onClick={handleStage} disabled={loading}>
                      <Plus size={10} /> 暂存
                    </GlassButton>
                    <GlassButton small onClick={handleUnstage} disabled={loading}>
                      <Minus size={10} /> 取消暂存
                    </GlassButton>
                  </div>
                )}
                <div className="flex gap-2 pt-1">
                  <input
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    placeholder="提交信息..."
                    className="flex-1 glass-input text-xs py-1"
                  />
                  <GlassButton accent small onClick={handleCommit} disabled={!message.trim() || loading}>
                    提交
                  </GlassButton>
                </div>
              </>
            )}
          </div>
        )}

        {view === "history" && (
          <div className="p-2 space-y-1">
            {commits.length === 0 ? (
              <p className="text-muted text-xs p-3 text-center">无提交记录</p>
            ) : (
              commits.map((c) => (
                <div key={c.hash} className="p-2 rounded hover:bg-surface-light glass-card border-0">
                  <div className="flex items-center gap-2">
                    <span className="text-accent-teal font-mono text-[10px]">{c.hash}</span>
                    <span className="text-soft truncate flex-1">{c.message}</span>
                  </div>
                  <div className="flex gap-3 mt-1 text-[10px] text-muted">
                    <span>{c.author}</span>
                    <span>{c.date}</span>
                  </div>
                </div>
              ))
            )}
          </div>
        )}
      </div>
    </div>
  );
}
