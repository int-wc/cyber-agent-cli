import { useEffect, useState } from "react";
import { IdeLayout } from "@/components/layout/IdeLayout";
import { useWorkspaceStore } from "@/stores/useWorkspaceStore";
import { logger } from "@/services/logger";

function App() {
  const [ready, setReady] = useState(false);
  const [showLogs, setShowLogs] = useState(false);
  const backendPort = useWorkspaceStore((s) => s.backendPort);
  const setBackendPort = useWorkspaceStore((s) => s.setBackendPort);
  const setBackendStatus = useWorkspaceStore((s) => s.setBackendStatus);

  logger.info("App 加载", { userAgent: navigator.userAgent, url: location.href });

  useEffect(() => {
    const init = async () => {
      // 1) 尝试 Tauri invoke 读取端口
      let port: number | null = null;
      try {
        const { invoke } = await import("@tauri-apps/api/core");
        port = await invoke<number>("get_server_port");
        logger.info("Tauri invoke get_server_port 返回", { port });
        if (port && port > 0) {
          setBackendPort(port);
          setReady(true);
          const { listen } = await import("@tauri-apps/api/event");
          const unlisten = await listen<number>("backend-ready", (event) => {
            logger.info("收到 backend-ready 事件", { port: event.payload });
            if (event.payload > 0) setBackendPort(event.payload);
          });
          return unlisten;
        } else {
          logger.info("端口为 0，等待 backend-ready 事件...");
          const { listen } = await import("@tauri-apps/api/event");
          const unlisten = await listen<number>("backend-ready", (event) => {
            logger.info("收到 backend-ready 事件", { port: event.payload });
            if (event.payload > 0) setBackendPort(event.payload);
          });
          return unlisten;
        }
      } catch (e) {
        logger.info("不在 Tauri 环境，回退探测", { error: String(e) });
      }

      // 2) 回退：localStorage
      try {
        const saved = localStorage.getItem("cyber-agent-ide-backend-port");
        logger.info("localStorage 端口", { saved });
        if (saved) port = parseInt(saved, 10);
      } catch (e) {
        logger.error("localStorage 读取失败", { error: String(e) });
      }

      // 3) 回退：探测端口
      setBackendStatus("connecting");
      const probePorts = port ? [port] : [9876, 9877, 9878, 9879, 9880];
      for (const p of probePorts) {
        const url = `http://127.0.0.1:${p}/api/health`;
        logger.info("探测", { url });
        try {
          const resp = await fetch(url);
          if (resp.ok) {
            logger.info("探测成功", { port: p });
            setBackendPort(p);
            setReady(true);
            return;
          }
        } catch (e) {
          logger.info("探测失败", { port: p, error: String(e) });
          continue;
        }
      }

      logger.error("所有端口探测失败");
    };

    init();
  }, []);

  if (ready || backendPort) {
    return <IdeLayout />;
  }

  return (
    <div className="h-full w-full flex items-center justify-center bg-window">
      <div className="glass-card p-8 text-center max-w-lg">
        <div className="text-5xl mb-4">🔷</div>
        <h1 className="text-xl font-bold text-primary mb-3">
          Cyber Agent IDE
        </h1>
        <div className="flex flex-col items-center gap-2 mb-4">
          <div className="animate-spin w-5 h-5 border-2 border-accent-teal border-t-transparent rounded-full" />
          <p className="text-xs text-muted">正在连接后端服务...</p>
        </div>
        <button
          onClick={() => setShowLogs(!showLogs)}
          className="glass-button text-xs py-1 px-3"
        >
          {showLogs ? "隐藏" : "显示"}调试日志
        </button>
        {showLogs && (
          <pre className="text-left mt-3 p-3 rounded glass-panel text-[10px] text-soft max-h-64 overflow-y-auto whitespace-pre-wrap">
            {logger.dump()}
          </pre>
        )}
      </div>
    </div>
  );
}

export default App;
