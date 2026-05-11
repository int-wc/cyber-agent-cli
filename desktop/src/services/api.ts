const getBase = () => {
  const port = localStorage.getItem("cyber-agent-ide-backend-port");
  return port ? `http://127.0.0.1:${port}/api` : "";
};

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const base = getBase();
  if (!base) throw new Error("后端未连接");
  const resp = await fetch(`${base}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!resp.ok) throw new Error(`API error: ${resp.status}`);
  return resp.json();
}

export const api = {
  // FS
  fsList: (path = ".") => request<{ entries: import("@/types/file").FileEntry[] }>(`/fs/list?path=${encodeURIComponent(path)}`),
  fsRead: (path: string) => request<{ content: string; encoding: string }>(`/fs/read?path=${encodeURIComponent(path)}`),
  fsWrite: (path: string, content: string) => request<{ status: string }>("/fs/write", { method: "POST", body: JSON.stringify({ path, content }) }),
  fsDelete: (path: string, recursive = false) => request<{ status: string }>(`/fs/delete?path=${encodeURIComponent(path)}&recursive=${recursive}`, { method: "DELETE" }),
  fsRename: (oldPath: string, newPath: string) => request<{ status: string }>("/fs/rename", { method: "POST", body: JSON.stringify({ old_path: oldPath, new_path: newPath }) }),
  fsCreateDir: (path: string) => request<{ status: string }>("/fs/create-dir", { method: "POST", body: JSON.stringify({ path }) }),

  // Git
  gitStatus: () => request<import("@/types/git").GitStatus>("/git/status"),
  gitDiff: (path = "", staged = false) => request<import("@/types/git").GitDiff>(`/git/diff?path=${encodeURIComponent(path)}&staged=${staged}`),
  gitStage: (paths: string[]) => request<{ status: string }>("/git/stage", { method: "POST", body: JSON.stringify({ paths }) }),
  gitUnstage: (paths: string[]) => request<{ status: string }>("/git/unstage", { method: "POST", body: JSON.stringify({ paths }) }),
  gitCommit: (message: string) => request<{ status: string; commit_hash?: string; error?: string }>("/git/commit", { method: "POST", body: JSON.stringify({ message }) }),
  gitLog: (limit = 20) => request<{ commits: import("@/types/git").GitCommit[] }>(`/git/log?limit=${limit}`),
  gitBranches: () => request<{ current: string; branches: string[] }>("/git/branches"),

  // Session
  health: () => request<{ status: string }>("/health"),
  sessionStatus: () => request<{ mode: string; service: string; model: string }>("/session/status"),
  config: () => request<{ service: string; model: string; mode: string; cwd: string }>("/config"),
  providers: () => request<{ providers: string[]; models: Record<string, string> }>("/config/providers"),

  // Terminal
  terminalExec: (command: string, cwd = ".") => request<{ stdout: string; stderr: string; exit_code: number }>("/terminal/exec", { method: "POST", body: JSON.stringify({ command, cwd }) }),
};
