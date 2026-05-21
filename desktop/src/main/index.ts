import { app, BrowserWindow, ipcMain, dialog } from "electron";
import path from "path";
import fs from "fs";
import http from "http";
import { spawnBackend, shutdownBackend } from "./backend";
import { TerminalManager } from "./terminal";

const isDev = !app.isPackaged;
let mainWindow: BrowserWindow | null = null;
let terminalManager: TerminalManager | null = null;

function checkViteRunning(): Promise<boolean> {
  return new Promise((resolve) => {
    const req = http.get("http://localhost:5173", (res) => {
      resolve(res.statusCode === 200);
    });
    req.on("error", () => resolve(false));
    req.setTimeout(1500, () => { req.destroy(); resolve(false); });
  });
}

async function createWindow(backendPort: number) {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 900,
    minHeight: 600,
    title: "Cyber Agent IDE",
    titleBarStyle: "hidden",
    frame: false,
    backgroundColor: "#f0f0f5",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  const distIndex = path.join(__dirname, "../renderer/index.html");

  if (isDev && await checkViteRunning()) {
    mainWindow.loadURL("http://localhost:5173");
    mainWindow.webContents.openDevTools({ mode: "detach" });
  } else if (fs.existsSync(distIndex)) {
    mainWindow.loadFile(distIndex);
  } else {
    // No Vite and no build — show a simple loading page
    mainWindow.loadURL(`data:text/html,
      <html>
      <body style="background:#0a0a0f;color:#fff;display:flex;align-items:center;justify-content:center;height:100vh;font-family:sans-serif">
        <div style="text-align:center">
          <h2>Cyber Agent IDE</h2>
          <p>前端未构建。请运行:</p>
          <pre>cd desktop && npm run build</pre>
          <p>后端: http://127.0.0.1:${backendPort}</p>
        </div>
      </body>
      </html>
    `);
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
