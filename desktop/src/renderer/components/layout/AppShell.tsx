import { useCallback, useRef, useState, useEffect } from "react";
import { useUIStore } from "../../stores/uiStore";
import TitleBar from "./TitleBar";
import StatusBar from "./StatusBar";
import Sidebar from "../sidebar/Sidebar";
import CenterWorkspace from "./CenterWorkspace";
import ChatPanel from "../chat/ChatPanel";
import FileSearch from "../sidebar/FileSearch";
import SettingsPanel from "../chat/SettingsPanel";

export default function AppShell() {
  const {
    sidebarWidth, chatPanelWidth,
    sidebarVisible, chatPanelVisible,
    setSidebarWidth, setChatPanelWidth,
  } = useUIStore();

  const [showFileSearch, setShowFileSearch] = useState(false);
  const [showSettings, setShowSettings] = useState(false);

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      const mod = e.ctrlKey || e.metaKey;
      if (mod && e.key === "p") {
        e.preventDefault();
        setShowFileSearch((v) => !v);
      } else if (mod && e.key === ",") {
        e.preventDefault();
        setShowSettings((v) => !v);
      } else if (e.key === "Escape") {
        if (showFileSearch) setShowFileSearch(false);
        if (showSettings) setShowSettings(false);
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [showFileSearch, showSettings]);

  const sidebarRef = useRef<HTMLDivElement>(null);
  const chatPanelRef = useRef<HTMLDivElement>(null);

  const handleSidebarResize = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const startX = e.clientX;
    const startW = sidebarWidth;
    const onMove = (ev: MouseEvent) => setSidebarWidth(startW + (ev.clientX - startX));
    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }, [sidebarWidth, setSidebarWidth]);

  const handleChatResize = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const startX = e.clientX;
    const startW = chatPanelWidth;
    const onMove = (ev: MouseEvent) => setChatPanelWidth(startW - (ev.clientX - startX));
    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }, [chatPanelWidth, setChatPanelWidth]);

  return (
    <div style={{ height: "100vh", display: "flex", flexDirection: "column", background: "transparent" }}>
      <TitleBar />
      <div ref={(el) => {
        if (el) {
          const r = el.getBoundingClientRect();
          console.log("[AppShell main row]", "w:", r.width, "h:", r.height);
        }
      }} style={{ flex: 1, display: "flex", overflow: "hidden" }}>
        {/* ── Left: 磁盘 / 文件浏览 ── */}
        {sidebarVisible && (
          <>
            <div ref={sidebarRef} style={{ width: sidebarWidth, flexShrink: 0, overflow: "hidden" }}>
              <Sidebar />
            </div>
            <div className="resize-handle resize-handle-horizontal" onMouseDown={handleSidebarResize} />
          </>
        )}

        {/* ── Center: 工具调用层 (导航标签 + 阅览 + 终端) ── */}
        <CenterWorkspace />

        {/* ── Right: AI Agent 辅助层 ── */}
        {chatPanelVisible && (
          <>
            <div className="resize-handle resize-handle-horizontal" onMouseDown={handleChatResize} />
            <div ref={chatPanelRef} style={{ width: chatPanelWidth, flexShrink: 0, overflow: "hidden" }}>
              <ChatPanel />
            </div>
          </>
        )}
      </div>
      <StatusBar />
      {showFileSearch && <FileSearch onClose={() => setShowFileSearch(false)} />}
      {showSettings && <SettingsPanel onClose={() => setShowSettings(false)} />}
    </div>
  );
}
