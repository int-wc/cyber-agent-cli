import { create } from "zustand";
import type { SessionConfig } from "../types/agent";

interface SessionState {
  config: SessionConfig;
  connected: boolean;
  backendPort: number;
  setConfig: (config: Partial<SessionConfig>) => void;
  setConnected: (connected: boolean) => void;
  setBackendPort: (port: number) => void;
}

export const useSessionStore = create<SessionState>((set) => ({
  config: {
    mode: "standard",
    service: "openai",
    model: "",
    approvalPolicy: "prompt",
  },
  connected: false,
  backendPort: 0,

  setConfig: (partial) => set((s) => ({ config: { ...s.config, ...partial } })),
  setConnected: (connected) => set({ connected }),
  setBackendPort: (port) => set({ backendPort: port }),
}));
