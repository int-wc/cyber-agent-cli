import { ChildProcess, spawn } from "child_process";
import path from "path";
import { BrowserWindow } from "electron";

let backendProcess: ChildProcess | null = null;
let backendPort: number = 0;

export function getBackendPort(): number {
  return backendPort;
}

export async function spawnBackend(): Promise<number> {
  return new Promise((resolve, reject) => {
    const pythonCmd = process.platform === "win32" ? "python" : "python3";
    const backendScript = path.join(__dirname, "..", "..", "..", "src", "cyber_agent", "cli", "ide_launcher.py");

    // Run the IDE server as a module
    const args = ["-m", "cyber_agent.cli.ide_server", "--host", "127.0.0.1", "--port", "0"];

    backendProcess = spawn(pythonCmd, args, {
      cwd: path.join(__dirname, "..", "..", ".."),
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
      stdio: ["ignore", "pipe", "pipe"],
    });

    const timeout = setTimeout(() => {
      reject(new Error("Backend startup timed out after 30s"));
    }, 30000);

    backendProcess.stdout?.on("data", (data: Buffer) => {
      const line = data.toString().trim();
      console.log("[backend]", line);

      const match = line.match(/IDE_SERVER_PORT=(\d+)/);
      if (match) {
        backendPort = parseInt(match[1], 10);
        clearTimeout(timeout);
        resolve(backendPort);
      }
    });

    backendProcess.stderr?.on("data", (data: Buffer) => {
      console.error("[backend:err]", data.toString().trim());
    });

    backendProcess.on("error", (err) => {
      clearTimeout(timeout);
      reject(err);
    });

    backendProcess.on("exit", (code) => {
      clearTimeout(timeout);
      if (backendPort === 0) {
        reject(new Error(`Backend exited with code ${code} before reporting port`));
      }
      backendProcess = null;
    });
  });
}

export function shutdownBackend() {
  if (backendProcess) {
    backendProcess.kill("SIGTERM");
    setTimeout(() => {
      if (backendProcess && backendProcess.exitCode === null) {
        backendProcess.kill("SIGKILL");
      }
    }, 5000);
    backendProcess = null;
  }
}
