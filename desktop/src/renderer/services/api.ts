export let BACKEND_PORT = 0;

export function setBackendPort(port: number) {
  BACKEND_PORT = port;
}

export async function apiFetch<T = unknown>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const baseUrl = `http://127.0.0.1:${BACKEND_PORT}`;
  const url = `${baseUrl}${endpoint}`;
  const res = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`API error ${res.status}: ${err}`);
  }
  return res.json();
}

// File system APIs
export const fsApi = {
  list: (path = ".") => apiFetch<{ path: string; entries: import("../types/agent").FileEntry[] }>(`/api/fs/list?path=${encodeURIComponent(path)}`),
  read: (path: string) => apiFetch<{ path: string; content: string; size: number; truncated: boolean }>(`/api/fs/read?path=${encodeURIComponent(path)}`),
  write: (path: string, content: string) => apiFetch<{ path: string; written: boolean }>("/api/fs/write", { method: "POST", body: JSON.stringify({ path, content }) }),
  createDir: (path: string) => apiFetch<{ path: string; created: boolean }>("/api/fs/create-dir", { method: "POST", body: JSON.stringify({ path }) }),
  delete: (path: string) => apiFetch<{ path: string; deleted: boolean }>(`/api/fs/delete?path=${encodeURIComponent(path)}`, { method: "DELETE" }),
  rename: (oldPath: string, newPath: string) => apiFetch<{ old_path: string; new_path: string; renamed: boolean }>("/api/fs/rename", { method: "POST", body: JSON.stringify({ old_path: oldPath, new_path: newPath }) }),
  search: (query: string, path = ".") => apiFetch<{ query: string; results: { path: string; name: string; match: string }[] }>(`/api/fs/search?query=${encodeURIComponent(query)}&path=${encodeURIComponent(path)}`),
};

// Git APIs
export const gitApi = {
  status: () => apiFetch<{ stdout: string; stderr: string; returncode: number }>("/api/git/status"),
  diff: (staged = false) => apiFetch<{ stdout: string }>(`/api/git/diff?staged=${staged}`),
  stage: (files: string[]) => apiFetch("/api/git/stage", { method: "POST", body: JSON.stringify({ files }) }),
  unstage: (files: string[]) => apiFetch("/api/git/unstage", { method: "POST", body: JSON.stringify({ files }) }),
  commit: (message: string) => apiFetch("/api/git/commit", { method: "POST", body: JSON.stringify({ message }) }),
  log: (max = 50) => apiFetch<{ stdout: string }>(`/api/git/log?max_count=${max}`),
  branches: () => apiFetch<{ stdout: string }>("/api/git/branches"),
};

// Session APIs
export const sessionApi = {
  status: () => apiFetch<{ mode: string; service: string; model: string; allowed_roots: string[]; message_count: number }>("/api/session/status"),
  reset: () => apiFetch("/api/session/reset", { method: "POST" }),
  setMode: (mode: string) => apiFetch("/api/session/mode", { method: "POST", body: JSON.stringify({ mode }) }),
  setApprovalPolicy: (policy: string) => apiFetch("/api/session/approval-policy", { method: "POST", body: JSON.stringify({ policy }) }),
  switchModel: (service: string, model: string) => apiFetch("/api/session/switch-model", { method: "POST", body: JSON.stringify({ service, model }) }),
  stop: () => apiFetch("/api/session/stop", { method: "POST" }),
};
