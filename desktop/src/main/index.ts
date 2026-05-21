import { app, BrowserWindow, ipcMain, dialog } from "electron";
import path from "path";
import { spawnBackend, shutdownBackend } from "./backend";
import { TerminalManager } from "./terminal";

const isDev = !app.isPackaged;
let mainWindow: BrowserWindow | null = null;
let terminalManager: TerminalManager | null = null;

function createWindow(backendPort: number) {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 900,
    minHeight: 600,
    title: "Cyber Agent IDE",
    titleBarStyle: "hidden",
    frame: false,
    backgroundColor: "#0a0a0f",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  if (isDev) {
    mainWindow.loadURL("http://localhost:5173");
    mainWindow.webContents.openDevTools({ mode: "detach" });
  } else {
    mainWindow.loadFile(path.join(__dirname, "../renderer/index.html"));
  }

  mainWindow.webContents.on("did-finish-load", () => {
    mainWindow?.webContents.send("backend:status", {
      ready: true,
      port: backendPort,
    });
  });

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

// ── IPC handlers ──

function registerIpcHandlers() {
  ipcMain.handle("dialog:openFile", async (_event, options) => {
    if (!mainWindow) return null;
    const result = await dialog.showOpenDialog(mainWindow, {
      properties: ["openFile"],
      ...options,
    });
    return result.canceled ? null : result.filePaths[0];
  });

  ipcMain.handle("dialog:saveFile", async (_event, options) => {
    if (!mainWindow) return null;
    const result = await dialog.showSaveDialog(mainWindow, options || {});
    return result.canceled ? null : result.filePath;
  });

  ipcMain.handle("app:version", () => app.getVersion());

  // Window controls
  ipcMain.on("window:minimize", () => mainWindow?.minimize());
  ipcMain.on("window:maximize", () => {
    if (mainWindow?.isMaximized()) {
      mainWindow.unmaximize();
    } else {
      mainWindow?.maximize();
    }
  });
  ipcMain.on("window:close", () => mainWindow?.close());
  ipcMain.on("window:isMaximized", (event) => {
    event.returnValue = mainWindow?.isMaximized() ?? false;
  });
}

app.whenReady().then(async () => {
  registerIpcHandlers();

  // Parse backend port from CLI args or env
  const backendPortArg = process.argv.find((a) => a.startsWith("--backend-port="));
  const envPort = parseInt(process.env.CYBER_AGENT_BACKEND_PORT || "0", 10);
  let backendPort = backendPortArg
    ? parseInt(backendPortArg.split("=")[1], 10)
    : (envPort || 0);

  // Only spawn backend if NO port was provided via any channel
  if (!backendPort) {
    const port = await spawnBackend();
    if (port) {
      backendPort = port;
    }
  }

  // Initialize terminal manager
  terminalManager = new TerminalManager();
  terminalManager.registerIpcHandlers();

  createWindow(backendPort);
});

app.on("window-all-closed", () => {
  shutdownBackend();
  terminalManager?.dispose();
  app.quit();
});

app.on("before-quit", () => {
  shutdownBackend();
  terminalManager?.dispose();
});
