import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("electronAPI", {
  // Backend
  getBackendPort: () => ipcRenderer.invoke("backend:getPort"),
  onBackendStatus: (callback: (status: { ready: boolean; port: number }) => void) => {
    ipcRenderer.on("backend:status", (_event, status) => callback(status));
  },

  // Terminal
  terminalCreate: (shell?: string) => ipcRenderer.invoke("terminal:create", shell),
  terminalWrite: (sessionId: string, data: string) =>
    ipcRenderer.send("terminal:data", sessionId, data),
  terminalResize: (sessionId: string, cols: number, rows: number) =>
    ipcRenderer.send("terminal:resize", sessionId, cols, rows),
  terminalKill: (sessionId: string) => ipcRenderer.invoke("terminal:kill", sessionId),
  onTerminalOutput: (callback: (sessionId: string, data: string) => void) => {
    ipcRenderer.on("terminal:output", (_event, sessionId, data) =>
      callback(sessionId, data)
    );
  },

  // Dialog
  openFileDialog: (options?: object) => ipcRenderer.invoke("dialog:openFile", options),
  saveFileDialog: (options?: object) => ipcRenderer.invoke("dialog:saveFile", options),

  // Window
  minimizeWindow: () => ipcRenderer.send("window:minimize"),
  maximizeWindow: () => ipcRenderer.send("window:maximize"),
  closeWindow: () => ipcRenderer.send("window:close"),
  isMaximized: () => ipcRenderer.sendSync("window:isMaximized"),

  // App
  getVersion: () => ipcRenderer.invoke("app:version"),
});
