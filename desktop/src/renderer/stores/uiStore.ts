import { create } from "zustand";

interface UIState {
  sidebarWidth: number;
  chatPanelWidth: number;
  terminalHeight: number;
  sidebarVisible: boolean;
  chatPanelVisible: boolean;
  terminalVisible: boolean;
  setSidebarWidth: (w: number) => void;
  setChatPanelWidth: (w: number) => void;
  setTerminalHeight: (h: number) => void;
  toggleSidebar: () => void;
  toggleChatPanel: () => void;
  toggleTerminal: () => void;
}

export const useUIStore = create<UIState>((set) => ({
  sidebarWidth: 280,
  chatPanelWidth: 380,
  terminalHeight: 250,
  sidebarVisible: true,
  chatPanelVisible: true,
  terminalVisible: true,

  setSidebarWidth: (w) => set({ sidebarWidth: Math.max(180, Math.min(w, 500)) }),
  setChatPanelWidth: (w) => set({ chatPanelWidth: Math.max(260, Math.min(w, 600)) }),
  setTerminalHeight: (h) => set({ terminalHeight: Math.max(100, Math.min(h, 500)) }),
  toggleSidebar: () => set((s) => ({ sidebarVisible: !s.sidebarVisible })),
  toggleChatPanel: () => set((s) => ({ chatPanelVisible: !s.chatPanelVisible })),
  toggleTerminal: () => set((s) => ({ terminalVisible: !s.terminalVisible })),
}));
