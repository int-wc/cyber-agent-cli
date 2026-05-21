import { create } from "zustand";

export type CenterTab = "viewer" | "yakit" | "mitm" | string; // string = terminal-{n}

interface UIState {
  sidebarWidth: number;
  chatPanelWidth: number;
  sidebarVisible: boolean;
  chatPanelVisible: boolean;
  setSidebarWidth: (w: number) => void;
  setChatPanelWidth: (w: number) => void;
  toggleSidebar: () => void;
  toggleChatPanel: () => void;

  centerTab: CenterTab;
  setCenterTab: (t: CenterTab) => void;
  terminalTabs: string[];     // ["term-1", "term-2", ...]
  addTerminal: () => void;
  removeTerminal: (id: string) => void;
}

let termCounter = 0;

export const useUIStore = create<UIState>((set) => ({
  sidebarWidth: 260,
  chatPanelWidth: 380,
  sidebarVisible: true,
  chatPanelVisible: true,

  setSidebarWidth: (w) => set({ sidebarWidth: Math.max(180, Math.min(w, 500)) }),
  setChatPanelWidth: (w) => set({ chatPanelWidth: Math.max(260, Math.min(w, 600)) }),
  toggleSidebar: () => set((s) => ({ sidebarVisible: !s.sidebarVisible })),
  toggleChatPanel: () => set((s) => ({ chatPanelVisible: !s.chatPanelVisible })),

  centerTab: "viewer",
  setCenterTab: (t) => set({ centerTab: t }),
  terminalTabs: [`term-${++termCounter}`],
  addTerminal: () => set((s) => {
    const id = `term-${++termCounter}`;
    return { terminalTabs: [...s.terminalTabs, id], centerTab: id };
  }),
  removeTerminal: (id) => set((s) => {
    const remaining = s.terminalTabs.filter((t) => t !== id);
    return {
      terminalTabs: remaining.length > 0 ? remaining : [`term-${++termCounter}`],
      centerTab: s.centerTab === id
        ? (remaining[0] || "viewer")
        : s.centerTab,
    };
  }),
}));
