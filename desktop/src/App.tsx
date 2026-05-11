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
      // Try Tauri event first
      try {
        const { listen } = await import("@tauri-apps/api/event");
        const unlisten = await listen<number>("backend-ready", (event) => {
          const port = event.payload;
          setBackendPort(port);
        });
        // Also check if backend already started (stored port)
        const savedPort = backendPort;
        if (savedPort) {
          setBackendPort(savedPort);
        }
        return () => { unlisten.then((fn: unknown) => (fn as () => void)()); };
      } catch {
        // Not running in Tauri — probe for backend server
        setBackendStatus("connecting");
        for (const port of [9876, 9877, 9878, 9879, 9880]) {
          try {
            const resp = await fetch(`http://127.0.0.1:${port}/api/health`);
            if (resp.ok) {
              setBackendPort(port);
              setReady(true);
              return;
            }
          } catch {
            continue;
          }
        }
      }
    };

    init();

    // Listen for backend port changes
    const check = setInterval(() => {
      const port = useWorkspaceStore.getState().backendPort;
      if (port && !ready) {
        fetch(`http://127.0.0.1:${port}/api/health`)
          .then((r) => r.json())
          .then(() => setReady(true))
          .catch(() => {});
      }
    }, 1000);

    return () => clearInterval(check);
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
