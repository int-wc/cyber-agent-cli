import { useUIStore } from "../../stores/uiStore";
import NavTabs from "./NavTabs";
import CodeEditor from "../editor/CodeEditor";
import TerminalPanel from "../terminal/TerminalPanel";

function YakitPlaceholder() {
  return (
    <div style={{
      height: "100%", display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center", gap: 16,
      color: "var(--text-tertiary)",
    }}>
      <div style={{
        width: 72, height: 72, borderRadius: 18,
        background: "rgba(124,111,247,0.10)",
        display: "flex", alignItems: "center", justifyContent: "center",
      }}>
        <span style={{ fontSize: 32 }}>🔧</span>
      </div>
      <div style={{ textAlign: "center" }}>
        <div style={{ fontSize: 15, fontWeight: 600, color: "var(--text-secondary)", marginBottom: 4 }}>
          Yakit 工具
        </div>
        <div style={{ fontSize: 12, lineHeight: 1.6 }}>
          安全工具集成面板
        </div>
        <div style={{ fontSize: 11, marginTop: 8, color: "var(--text-tertiary)" }}>
          端口扫描 · 漏洞检测 · 数据包分析 · 插件管理
        </div>
      </div>
    </div>
  );
}

function MitmPlaceholder() {
  return (
    <div style={{
      height: "100%", display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center", gap: 16,
      color: "var(--text-tertiary)",
    }}>
      <div style={{
        width: 72, height: 72, borderRadius: 18,
        background: "rgba(59,130,246,0.10)",
        display: "flex", alignItems: "center", justifyContent: "center",
      }}>
        <span style={{ fontSize: 32 }}>🌐</span>
      </div>
      <div style={{ textAlign: "center" }}>
        <div style={{ fontSize: 15, fontWeight: 600, color: "var(--text-secondary)", marginBottom: 4 }}>
          MITM 代理浏览器
        </div>
        <div style={{ fontSize: 12, lineHeight: 1.6 }}>
          启用中间人代理的浏览器实例
        </div>
        <div style={{ fontSize: 11, marginTop: 8, color: "var(--text-tertiary)" }}>
          HTTPS 流量拦截 · 请求重放 · 自动化测试
        </div>
      </div>
      <button className="glass-btn glass-btn-primary" style={{ fontSize: 12 }}>
        启动 MITM 浏览器
      </button>
    </div>
  );
}

export default function CenterWorkspace() {
  const { terminalVisible, terminalHeight, setTerminalHeight, centerTab, setCenterTab } = useUIStore();

  const renderContent = () => {
    switch (centerTab) {
      case "viewer":
        return <CodeEditor />;
      case "yakit":
        return <YakitPlaceholder />;
      case "mitm":
        return <MitmPlaceholder />;
    }
  };

  return (
    <div style={{
      flex: 1, display: "flex", flexDirection: "column",
      overflow: "hidden", minWidth: 0,
    }}>
      {/* Navigation tabs */}
      <NavTabs active={centerTab} onSelect={setCenterTab} />

      {/* Content area */}
      <div style={{ flex: 1, overflow: "hidden" }}>
        {renderContent()}
      </div>

      {/* Terminal */}
      {terminalVisible && (
        <>
          <div
            className="resize-handle resize-handle-vertical"
            onMouseDown={(e) => {
              e.preventDefault();
              const startY = e.clientY;
              const startHeight = terminalHeight;
              const onMove = (ev: MouseEvent) => setTerminalHeight(startHeight + (startY - ev.clientY));
              const onUp = () => {
                document.removeEventListener("mousemove", onMove);
                document.removeEventListener("mouseup", onUp);
                document.body.style.cursor = "";
                document.body.style.userSelect = "";
              };
              document.body.style.cursor = "row-resize";
              document.body.style.userSelect = "none";
              document.addEventListener("mousemove", onMove);
              document.addEventListener("mouseup", onUp);
            }}
          />
          <div style={{ height: terminalHeight, flexShrink: 0, overflow: "hidden" }}>
            <TerminalPanel />
          </div>
        </>
      )}
    </div>
  );
}
