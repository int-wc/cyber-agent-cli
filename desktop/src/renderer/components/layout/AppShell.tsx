import { useCallback, useRef, useState, useEffect } from "react";
import { useUIStore } from "../../stores/uiStore";
import TitleBar from "./TitleBar";
import StatusBar from "./StatusBar";
import Sidebar from "../sidebar/Sidebar";
import CodeEditor from "../editor/CodeEditor";
import TerminalPanel from "../terminal/TerminalPanel";
import ChatPanel from "../chat/ChatPanel";
import FileSearch from "../sidebar/FileSearch";
import SettingsPanel from "../chat/SettingsPanel";

export default function AppShell() {
  const {
    sidebarWidth, chatPanelWidth, terminalHeight,
    sidebarVisible, chatPanelVisible, terminalVisible,
    setSidebarWidth, setChatPanelWidth, setTerminalHeight,
  } = useUIStore();

  const [showFileSearch, setShowFileSearch] = useState(false);
  const [showSettings, setShowSettings] = useState(false);

  // Global keyboard shortcuts
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
  const terminalRef = useRef<HTMLDivElement>(null);

  const handleSidebarResize = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const startX = e.clientX;
    const startWidth = sidebarWidth;
    const onMove = (ev: MouseEvent) => {
      setSidebarWidth(startWidth + (ev.clientX - startX));
    };
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
    const startWidth = chatPanelWidth;
    const onMove = (ev: MouseEvent) => {
      setChatPanelWidth(startWidth - (ev.clientX - startX));
    };
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

  const handleTerminalResize = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const startY = e.clientY;
    const startHeight = terminalHeight;
    const onMove = (ev: MouseEvent) => {
      setTerminalHeight(startHeight + (startY - ev.clientY));
    };
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
  }, [terminalHeight, setTerminalHeight]);

  return (
    <div style={{ height: "100vh", display: "flex", flexDirection: "column", background: "transparent" }}>
      <TitleBar />
      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
        {/* Sidebar */}
        {sidebarVisible && (
          <>
            <div ref={sidebarRef} style={{ width: sidebarWidth, flexShrink: 0, overflow: "hidden" }}>
              <Sidebar />
            </div>
            <div className="resize-handle resize-handle-horizontal" onMouseDown={handleSidebarResize} />
          </>
        )}

        {/* Main content */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", minWidth: 0 }}>
          <div style={{ flex: 1, overflow: "hidden" }}>
            <CodeEditor />
          </div>
          {terminalVisible && (
            <>
              <div className="resize-handle resize-handle-vertical" onMouseDown={handleTerminalResize} />
              <div ref={terminalRef} style={{ height: terminalHeight, flexShrink: 0, overflow: "hidden" }}>
                <TerminalPanel />
              </div>
            </>
          )}
        </div>

        {/* Chat Panel */}
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
