import { useState, useCallback, useRef } from "react";
import { Sidebar } from "./Sidebar";
import { MainContent } from "./MainContent";
import { ChatPanel } from "../chat/ChatPanel";
import { StatusBar } from "./StatusBar";
import { useWorkspaceStore } from "@/stores/useWorkspaceStore";

export function IdeLayout() {
  const [chatWidth, setChatWidth] = useState(380);
  const draggingRef = useRef(false);
  const backendPort = useWorkspaceStore((s) => s.backendPort);

  const onMouseDown = useCallback(() => {
    draggingRef.current = true;
    document.addEventListener("mousemove", onMouseMove);
    document.addEventListener("mouseup", onMouseUp);
  }, []);

  const onMouseMove = useCallback((e: MouseEvent) => {
    if (!draggingRef.current) return;
    const w = window.innerWidth - e.clientX;
    setChatWidth(Math.max(280, Math.min(700, w)));
  }, []);

  const onMouseUp = useCallback(() => {
    draggingRef.current = false;
    document.removeEventListener("mousemove", onMouseMove);
    document.removeEventListener("mouseup", onMouseUp);
  }, [onMouseMove]);

  return (
    <div className="h-full w-full flex flex-col bg-window overflow-hidden">
      <div className="flex flex-1 overflow-hidden">
        <Sidebar />
        <MainContent />
        <div className="resize-handle resize-handle-h" onMouseDown={onMouseDown} />
        <div style={{ width: chatWidth }} className="flex-shrink-0">
          <ChatPanel backendPort={backendPort} />
        </div>
      </div>
      <StatusBar chatWidth={chatWidth} />
    </div>
  );
}
