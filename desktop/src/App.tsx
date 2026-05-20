import { useEffect, useState } from "react";
import { IdeLayout } from "@/components/layout/IdeLayout";
import { useWorkspaceStore } from "@/stores/useWorkspaceStore";

function App() {
  const [ready, setReady] = useState(false);
  const backendPort = useWorkspaceStore((s) => s.backendPort);
  const setBackendPort = useWorkspaceStore((s) => s.setBackendPort);
  const setBackendStatus = useWorkspaceStore((s) => s.setBackendStatus);

  useEffect(() => {
    const init = async () => {
      // 1) 尝试 Tauri invoke 读取端口（从 Rust state）
      let port: number | null = null;
      try {
        const { invoke } = await import("@tauri-apps/api/core");
        port = await invoke<number>("get_server_port");
        if (port && port > 0) {
          setBackendPort(port);
          setReady(true);
          // 监听后续事件
          const { listen } = await import("@tauri-apps/api/event");
          return await listen<number>("backend-ready", (event) => {
            if (event.payload > 0) setBackendPort(event.payload);
          });
        }
      } catch {
        // 不在 Tauri 环境
      }

      // 2) 回退：从 localStorage 读取
      try {
        const saved = localStorage.getItem("cyber-agent-ide-backend-port");
        if (saved) {
          port = parseInt(saved, 10);
        }
      } catch {}

      // 3) 回退：探测常见端口
      setBackendStatus("connecting");
      const probePorts = port ? [port] : [9876, 9877, 9878, 9879, 9880];
      for (const p of probePorts) {
        try {
          const resp = await fetch(`http://127.0.0.1:${p}/api/health`);
          if (resp.ok) {
            setBackendPort(p);
            setReady(true);
            return;
          }
        } catch {
          continue;
        }
      }
    };

    init();
  }, []);

  if (ready || backendPort) {
    return <IdeLayout />;
  }

  return (
    <div className="h-full w-full flex items-center justify-center bg-window">
      <div className="glass-card p-12 text-center max-w-md animate-fade-in">
        <div className="text-5xl mb-6">🔷</div>
        <h1 className="text-2xl font-bold text-primary mb-3">
          Cyber Agent IDE
        </h1>
        <p className="text-muted mb-6">现代桌面 AI 编程助手</p>
        <div className="flex flex-col items-center gap-3">
          <div className="animate-spin w-6 h-6 border-2 border-accent-teal border-t-transparent rounded-full" />
          <p className="text-xs text-muted">
            正在连接后端服务...
          </p>
          <p className="text-[10px] text-muted opacity-60">
            请运行 cyber-agent ide-server 启动后端
          </p>
        </div>
      </div>
    </div>
  );
}

export default App;
