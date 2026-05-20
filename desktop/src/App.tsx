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

  logger.info("App 加载", { url: location.href });

  useEffect(() => {
    const init = async () => {
      // 1) 从 localStorage 读取上次使用的端口
      let port: number | null = null;
      try {
        const saved = localStorage.getItem("cyber-agent-ide-backend-port");
        logger.info("localStorage 端口", { saved });
        if (saved) port = parseInt(saved, 10);
      } catch {}

      // 2) 探测后端
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
        } catch {
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
        <h1 className="text-xl font-bold text-primary mb-3">Cyber Agent IDE</h1>
        <p className="text-muted mb-4 text-sm">现代桌面 AI 编程助手</p>
        <div className="flex flex-col items-center gap-2 mb-4">
          <div className="animate-spin w-5 h-5 border-2 border-accent-teal border-t-transparent rounded-full" />
          <p className="text-xs text-muted">正在连接后端服务...</p>
          <p className="text-[10px] text-muted opacity-60">
            请运行 cyber-agent ide 或 cyber-agent ide-server --port 9876
          </p>
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
