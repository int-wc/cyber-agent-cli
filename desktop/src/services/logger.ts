// IDE 前端日志 — 写入本地文件，用于排查连接问题

const LOG_PREFIX = "[cyber-ide]";

function now(): string {
  return new Date().toISOString().slice(11, 23);
}

function format(level: string, msg: string, data?: unknown): string {
  const ts = now();
  const extra = data !== undefined ? " " + JSON.stringify(data) : "";
  return `${ts} ${level} ${msg}${extra}`;
}

export const logger = {
  debug(msg: string, data?: unknown) {
    const line = format("DEBUG", msg, data);
    console.debug(LOG_PREFIX, line);
    try {
      const existing = localStorage.getItem("cyber-ide-logs") || "";
      localStorage.setItem("cyber-ide-logs", (existing + line + "\n").slice(-8000));
    } catch {}
  },
  info(msg: string, data?: unknown) {
    const line = format("INFO ", msg, data);
    console.info(LOG_PREFIX, line);
    try {
      const existing = localStorage.getItem("cyber-ide-logs") || "";
      localStorage.setItem("cyber-ide-logs", (existing + line + "\n").slice(-8000));
    } catch {}
  },
  error(msg: string, data?: unknown) {
    const line = format("ERROR", msg, data);
    console.error(LOG_PREFIX, line);
    try {
      const existing = localStorage.getItem("cyber-ide-logs") || "";
      localStorage.setItem("cyber-ide-logs", (existing + line + "\n").slice(-8000));
    } catch {}
  },
  dump(): string {
    try {
      return localStorage.getItem("cyber-ide-logs") || "(空)";
    } catch {
      return "(不可用)";
    }
  },
};
