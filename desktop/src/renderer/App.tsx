import { useEffect } from "react";
import { useSessionStore } from "./stores/sessionStore";
import { wsClient } from "./services/ws";
import AppShell from "./components/layout/AppShell";
import LiquidBackground from "./components/glass/LiquidBackground";
import { listen } from "@tauri-apps/api/event";

export default function App() {
  const { setBackendPort, setConnected, setConfig, backendPort } = useSessionStore();

  useEffect(() => {
    let unlistenFn: (() => void) | null = null;

    // Tauri: listen for backend:status event from Rust backend manager
    listen<{ ready: boolean; port: number }>("backend:status", (event) => {
      if (event.payload.ready) {
        setBackendPort(event.payload.port);
        wsClient.connect(event.payload.port);
      }
    }).then((fn) => { unlistenFn = fn; });

    // Fallback: URL query param for browser dev mode
    const params = new URLSearchParams(window.location.search);
    const port = parseInt(params.get("port") || "0", 10);
    if (port) {
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
      if (unlistenFn) unlistenFn();
      unsub();
      wsClient.disconnect();
    };
  }, []);

  return (
    <div style={{ position: "relative", height: "100vh" }}>
      <LiquidBackground />
      <div style={{ position: "relative", zIndex: 1, height: "100%" }}>
        <AppShell />
      </div>
    </div>
  );
}
