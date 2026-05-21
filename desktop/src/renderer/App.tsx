import { useEffect } from "react";
import { useSessionStore } from "./stores/sessionStore";
import { wsClient } from "./services/ws";
import AppShell from "./components/layout/AppShell";

export default function App() {
  const { setBackendPort, setConnected, setConfig, backendPort } = useSessionStore();

  useEffect(() => {
    const electronAPI = window.electronAPI;

    if (electronAPI) {
      // Electron environment
      electronAPI.onBackendStatus((status) => {
        if (status.ready) {
          setBackendPort(status.port);
          wsClient.connect(status.port);
        }
      });
    } else {
      // Browser dev mode: read port from URL param or default
      const params = new URLSearchParams(window.location.search);
      const port = parseInt(params.get("port") || "8765", 10);
      setBackendPort(port);
      wsClient.connect(port);
    }

    const unsub = wsClient.onEvent((event) => {
      switch (event.type) {
        case "ws_connected":
          setConnected(true);
          break;
        case "ws_disconnected":
          setConnected(false);
          break;
        case "connected": {
          const p = event.payload as Record<string, unknown> | undefined;
          if (p) {
            setConfig({
              mode: (p.mode as "standard" | "authorized") || "standard",
              service: (p.service as string) || "",
              model: (p.model as string) || "",
            });
          }
          break;
        }
      }
    });

    return () => {
      unsub();
      wsClient.disconnect();
    };
  }, []);

  return <AppShell />;
}
