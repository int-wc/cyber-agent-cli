import { ipcMain } from "electron";
import * as pty from "node-pty";

interface TerminalSession {
  pty: pty.IPty;
  sessionId: string;
}

export class TerminalManager {
  private sessions: Map<string, TerminalSession> = new Map();
  private sessionCounter = 0;

  registerIpcHandlers() {
    ipcMain.handle("terminal:create", async (_event, shellType?: string) => {
      const sessionId = `term-${++this.sessionCounter}`;
      const shell = shellType || (process.platform === "win32" ? "powershell.exe" : process.env.SHELL || "/bin/bash");

      const ptyProcess = pty.spawn(shell, [], {
        name: "xterm-256color",
        cols: 120,
        rows: 30,
        cwd: process.cwd(),
        env: process.env as Record<string, string>,
      });

      ptyProcess.onData((data: string) => {
        const win = require("electron").BrowserWindow.getAllWindows()[0];
        if (win) {
          win.webContents.send("terminal:output", sessionId, data);
        }
      });

      ptyProcess.onExit(({ exitCode }) => {
        const win = require("electron").BrowserWindow.getAllWindows()[0];
        if (win) {
          win.webContents.send("terminal:output", sessionId, `\r\n[进程退出, code=${exitCode}]\r\n`);
        }
        this.sessions.delete(sessionId);
      });

      this.sessions.set(sessionId, { pty: ptyProcess, sessionId });
      return { sessionId, pid: ptyProcess.pid };
    });

    ipcMain.on("terminal:data", (_event, sessionId: string, data: string) => {
      const session = this.sessions.get(sessionId);
      if (session) {
        session.pty.write(data);
      }
    });

    ipcMain.on("terminal:resize", (_event, sessionId: string, cols: number, rows: number) => {
      const session = this.sessions.get(sessionId);
      if (session) {
        session.pty.resize(cols, rows);
      }
    });

    ipcMain.handle("terminal:kill", async (_event, sessionId: string) => {
      const session = this.sessions.get(sessionId);
      if (session) {
        session.pty.kill();
        this.sessions.delete(sessionId);
      }
      return true;
    });
  }

  dispose() {
    for (const [, session] of this.sessions) {
      try {
        session.pty.kill();
      } catch {}
    }
    this.sessions.clear();
  }
}
