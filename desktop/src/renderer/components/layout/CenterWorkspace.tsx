import { useEffect, useState } from "react";
import { useUIStore } from "../../stores/uiStore";
import NavTabs from "./NavTabs";
import CodeEditor from "../editor/CodeEditor";
import TerminalPanel from "../terminal/TerminalPanel";

// ── Yakit 真实集成 ──

const YAKIT_DEFAULT_PORT = 8087;

function YakitPanel() {
  const [status, setStatus] = useState<"checking" | "running" | "stopped">("checking");

  useEffect(() => {
    checkYakit();
  }, []);

  async function checkYakit() {
    try {
      const resp = await fetch(`http://127.0.0.1:${YAKIT_DEFAULT_PORT}/api/info`, {
        signal: AbortSignal.timeout(2000),
      });
      if (resp.ok) {
        setStatus("running");
      } else {
        setStatus("stopped");
      }
    } catch {
      setStatus("stopped");
    }
  }

  if (status === "checking") {
    return (
      <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center" }}>
        <span style={{ color: "var(--text-tertiary)", fontSize: 13 }}>检测 Yakit 引擎 ...</span>
      </div>
    );
  }

  if (status === "running") {
    return (
      <iframe
        src={`http://127.0.0.1:${YAKIT_DEFAULT_PORT}`}
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%", border: "none" }}
        sandbox="allow-scripts allow-same-origin allow-forms allow-popups"
      />
    );
  }

  return (
    <div style={{
      position: "absolute", inset: 0, display: "flex", flexDirection: "column",
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
          Yakit 引擎未启动
        </div>
        <div style={{ fontSize: 12, lineHeight: 1.6, maxWidth: 320 }}>
          Yakit 是一个网络安全工具平台。
          <br />
          请先启动 Yakit 引擎，或下载安装：
        </div>
      </div>
      <div style={{ display: "flex", gap: 8 }}>
        <button className="glass-btn" style={{ fontSize: 12 }} onClick={checkYakit}>
          重新检测
        </button>
        <button
          className="glass-btn glass-btn-primary"
          style={{ fontSize: 12 }}
          onClick={() => {
            window.open(`http://127.0.0.1:${YAKIT_DEFAULT_PORT}`, "_blank");
          }}
        >
          在浏览器打开
        </button>
      </div>
      <div style={{ fontSize: 11, color: "var(--text-tertiary)" }}>
        端口: {YAKIT_DEFAULT_PORT} · 状态: 未连接
      </div>
    </div>
  );
}

// ── MITM 占位 ──

function MitmPanel() {
  return (
    <div style={{
      position: "absolute", inset: 0, display: "flex", flexDirection: "column",
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

// ── 主组件 ──

export default function CenterWorkspace() {
  const { centerTab, terminalTabs } = useUIStore();

  const isTerminalTab = terminalTabs.includes(centerTab);

  const renderContent = () => {
    if (isTerminalTab) {
      return <TerminalPanel key={centerTab} sessionKey={centerTab} standalone />;
    }
    switch (centerTab) {
      case "viewer": return <CodeEditor />;
      case "yakit":  return <YakitPanel />;
      case "mitm":   return <MitmPanel />;
      default:       return <CodeEditor />;
    }
  };

  return (
    <div style={{
      flex: 1, display: "flex", flexDirection: "column",
      overflow: "hidden", minWidth: 0,
    }}>
      <NavTabs />
      <div style={{ flex: 1, minHeight: 0, overflow: "hidden", position: "relative" }}>
        {renderContent()}
      </div>
    </div>
  );
}
