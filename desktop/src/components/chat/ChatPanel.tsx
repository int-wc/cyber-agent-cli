import { useEffect, useRef, useCallback, useState } from "react";
import { ChatMessage } from "./ChatMessage";
import { ChatInput } from "./ChatInput";
import { useChatStore } from "@/stores/useChatStore";
import { AgentWebSocket } from "@/services/websocket";
import { MessageSquare, AlertTriangle } from "lucide-react";
import { logger } from "@/services/logger";

let wsInstance: AgentWebSocket | null = null;

export function ChatPanel({ backendPort }: { backendPort: number | null }) {
  const messages = useChatStore((s) => s.messages);
  const streaming = useChatStore((s) => s.streaming);
  const streamContent = useChatStore((s) => s.streamContent);
  const reasoningContent = useChatStore((s) => s.reasoningContent);
  const wsConnected = useChatStore((s) => s.wsConnected);
  const pendingApproval = useChatStore((s) => s.pendingApproval);
  const clearPendingApproval = useChatStore((s) => s.clearPendingApproval);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (backendPort && !wsInstance) {
      logger.info("ChatPanel 创建 WebSocket", { port: backendPort });
      wsInstance = new AgentWebSocket(backendPort);
      wsInstance.connect();
    }
    return () => {
      logger.info("ChatPanel 清理 WebSocket");
      wsInstance?.disconnect();
      wsInstance = null;
    };
  }, [backendPort]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, streamContent, reasoningContent]);

  const onSend = useCallback((text: string) => {
    useChatStore.getState().addMessage({ id: "u-" + Date.now(), role: "user", content: text, timestamp: Date.now() });
    wsInstance?.sendMessage(text);
  }, []);

  const onStop = useCallback(() => {
    wsInstance?.sendStop();
  }, []);

  const onApprove = useCallback(() => {
    if (pendingApproval) {
      wsInstance?.approveTool(pendingApproval.tool_call_id);
      clearPendingApproval();
    }
  }, [pendingApproval, clearPendingApproval]);

  const onReject = useCallback(() => {
    if (pendingApproval) {
      wsInstance?.rejectTool(pendingApproval.tool_call_id);
      clearPendingApproval();
    }
  }, [pendingApproval, clearPendingApproval]);

  return (
    <div className="h-full flex flex-col glass-panel-heavy border-l border-b-0 border-t-0 border-r-0 border-border-glass">
      <div className="flex items-center gap-2 px-3 py-2 text-xs text-muted font-medium no-select border-b border-border-glass">
        <MessageSquare size={14} />
        <span className="flex-1">AI 对话</span>
        <span className={`w-2 h-2 rounded-full ${wsConnected ? "bg-green-500" : "bg-gray-500"}`} />
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto px-3 py-2 space-y-2">
        {messages.length === 0 && !reasoningContent && !streamContent && (
          <div className="text-center py-8 text-muted text-xs">
            <div className="text-3xl mb-2 opacity-50">💬</div>
            <p>开始与 AI 对话</p>
            <p className="mt-1 opacity-60">可使用 @文件 引用编辑器中的代码</p>
          </div>
        )}
        {messages.map((msg) => (
          <ChatMessage key={msg.id} message={msg} />
        ))}
        {reasoningContent && (
          <ChatMessage
            message={{ id: "reasoning", role: "reasoning", content: reasoningContent, streaming: true, timestamp: Date.now() }}
          />
        )}
        {streamContent && (
          <ChatMessage
            message={{ id: "streaming", role: "assistant", content: streamContent, streaming: true, timestamp: Date.now() }}
          />
        )}
      </div>

      {pendingApproval && (
        <div className="mx-3 my-1 p-3 glass-card border border-accent-amber/30 animate-slide-up">
          <div className="flex items-center gap-2 text-xs mb-2">
            <AlertTriangle size={14} className="text-accent-amber" />
            <span className="text-soft">审批请求</span>
          </div>
          <p className="text-xs text-primary mb-2">
            允许执行 <span className="text-accent-amber">{pendingApproval.tool_name}</span> ？
            <span className="text-muted ml-1">(风险: {pendingApproval.risk})</span>
          </p>
          <div className="flex gap-2">
            <button onClick={onApprove} className="glass-button glass-button-accent text-xs py-1">批准</button>
            <button onClick={onReject} className="glass-button text-xs py-1">拒绝</button>
          </div>
        </div>
      )}

      <ChatInput onSend={onSend} onStop={onStop} streaming={streaming} />
    </div>
  );
}
