export type SidebarView = "files" | "search" | "git" | "extensions" | "settings";

export type ConnectionStatus = "connecting" | "connected" | "disconnected" | "error";

export interface BackendState {
  port: number | null;
  status: ConnectionStatus;
  error?: string;
}
