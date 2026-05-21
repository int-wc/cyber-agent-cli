import { useCallback, useRef, useEffect, useState } from "react";
import { useChatStore } from "../../stores/chatStore";
import { wsClient } from "../../services/ws";
import { Send, Square, Zap, MessageSquare } from "lucide-react";

export default function ChatPanel() {
  const {
    messages, currentStream, currentReasoning, isStreaming,
    addUserMessage, clearStream, recentToolCalls, pendingApprovals,
    addApprovalRequest, clearApproval,
  } = useChatStore();

  const [input, setInput] = useState("");
  const [mode, setMode] = useState<"chat" | "builder">("chat");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, currentStream]);

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "l") {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, []);

  const handleSend = useCallback(() => {
    const text = input.trim();
    if (!text || isStreaming) return;
    addUserMessage(text);
    wsClient.send({ type: "user_message", content: text });
    setInput("");
  }, [input, isStreaming, addUserMessage]);

  const handleStop = useCallback(() => {
    wsClient.send({ type: "stop" });
    clearStream();
  }, [clearStream]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }, [handleSend]);

  const handleApprove = useCallback((toolCallId: string) => {
    wsClient.send({ type: "approve", tool_call_id: toolCallId });
    clearApproval();
  }, [clearApproval]);

  const handleReject = useCallback((toolCallId: string, reason: string) => {
    wsClient.send({ type: "reject", tool_call_id: toolCallId, reason });
    clearApproval();
  }, [clearApproval]);

  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column", background: "var(--bg-elevated)" }}>
      {/* Header */}
      <div className="glass-surface" style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "8px 12px", borderBottom: "1px solid rgba(255,255,255,0.06)",
      }}>
        <span style={{ fontSize: 12, fontWeight: 600, textTransform: "uppercase",
                       letterSpacing: "0.05em", color: "var(--text-secondary)" }}>
          AI 助手
        </span>
        {/* Mode toggle */}
        <div style={{ display: "flex", borderRadius: 8, overflow: "hidden", border: "1px solid rgba(255,255,255,0.08)" }}>
          <button
            onClick={() => setMode("chat")}
            style={{
              padding: "4px 10px", fontSize: 11, cursor: "pointer",
              background: mode === "chat" ? "var(--accent-soft)" : "transparent",
              color: mode === "chat" ? "var(--accent-light)" : "var(--text-tertiary)",
              border: "none",
            }}
          >
            <MessageSquare size={12} style={{ marginRight: 4 }} />
            Chat
          </button>
          <button
            onClick={() => setMode("builder")}
            style={{
              padding: "4px 10px", fontSize: 11, cursor: "pointer",
              background: mode === "builder" ? "var(--accent-soft)" : "transparent",
              color: mode === "builder" ? "var(--accent-light)" : "var(--text-tertiary)",
              border: "none",
              borderLeft: "1px solid rgba(255,255,255,0.08)",
            }}
          >
            <Zap size={12} style={{ marginRight: 4 }} />
            Builder
          </button>
        </div>
      </div>

      {/* Messages */}
      <div style={{ flex: 1, overflow: "auto", padding: "12px" }}>
        {messages.map((msg) => (
          <div key={msg.id} style={{ marginBottom: 16 }}>
            {/* Role label */}
            <div style={{
              fontSize: 11, fontWeight: 600, marginBottom: 4,
              color: msg.role === "user" ? "var(--accent-light)" :
                     msg.role === "assistant" ? "var(--green)" :
                     "var(--text-tertiary)",
            }}>
              {msg.role === "user" ? "你" :
               msg.role === "assistant" ? "AI" :
               msg.role}
            </div>
            {/* Reasoning */}
            {msg.reasoning && (
              <div style={{
                fontSize: 12, color: "var(--text-tertiary)", fontStyle: "italic",
                padding: "6px 10px", marginBottom: 6, borderLeft: "2px solid var(--accent-soft)",
                background: "rgba(255,255,255,0.02)", borderRadius: "0 6px 6px 0",
              }}>
                {msg.reasoning}
              </div>
            )}
            {/* Content */}
            <div className="glass-panel" style={{
              padding: "10px 14px", fontSize: 13, lineHeight: 1.7,
              whiteSpace: "pre-wrap", wordBreak: "break-word",
            }}>
              {msg.content || "(空)"}
            </div>
            {/* Tool calls */}
            {msg.toolCalls?.map((tc) => (
              <div key={tc.id} className="glass-panel glass-card-accent" style={{
                marginTop: 8, padding: "8px 12px", fontSize: 12,
              }}>
                <div style={{ color: "var(--blue)", fontWeight: 600 }}>
                  🔧 {tc.name}
                </div>
                {tc.result && (
                  <div style={{
                    marginTop: 6, padding: "6px 8px", background: "rgba(0,0,0,0.3)",
                    borderRadius: 6, maxHeight: 200, overflow: "auto",
                    fontFamily: "monospace", fontSize: 11, color: "var(--text-secondary)",
                  }}>
                    {tc.result.slice(0, 2000)}
                  </div>
                )}
              </div>
            ))}
          </div>
        ))}

        {/* Streaming content */}
        {isStreaming && (
          <div style={{ marginBottom: 16 }}>
            <div style={{ fontSize: 11, fontWeight: 600, marginBottom: 4, color: "var(--green)" }}>
              AI 正在生成...
            </div>
            {currentReasoning && (
              <div style={{
                fontSize: 12, color: "var(--text-tertiary)", fontStyle: "italic",
                padding: "6px 10px", marginBottom: 6, borderLeft: "2px solid var(--accent-soft)",
                background: "rgba(255,255,255,0.02)", borderRadius: "0 6px 6px 0",
              }}>
                {currentReasoning}
              </div>
            )}
            {currentStream && (
              <div className="glass-panel" style={{
                padding: "10px 14px", fontSize: 13, lineHeight: 1.7,
                whiteSpace: "pre-wrap", wordBreak: "break-word",
              }}>
                {currentStream}
                <span style={{ animation: "blink 1s step-end infinite", color: "var(--accent-light)" }}>▌</span>
              </div>
            )}
          </div>
        )}

        {/* Pending approvals */}
        {pendingApprovals.size > 0 && Array.from(pendingApprovals.entries()).map(([id, req]) => (
          <div key={id} className="glass-panel" style={{
            marginBottom: 12, padding: 12,
            borderColor: "rgba(255,171,64,0.4)",
            background: "rgba(255,171,64,0.05)",
          }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: "var(--orange)", marginBottom: 6 }}>
              ⚠️ 需要审批: {req.toolName} ({req.risk || "unknown"} 风险)
            </div>
            <div style={{ fontSize: 11, color: "var(--text-secondary)", marginBottom: 8,
                          fontFamily: "monospace", maxHeight: 100, overflow: "auto" }}>
              {JSON.stringify(req.toolCall, null, 2)}
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <button className="glass-btn glass-btn-primary" style={{ fontSize: 11 }} onClick={() => handleApprove(id)}>
                批准
              </button>
              <button className="glass-btn" style={{ fontSize: 11, color: "var(--red)" }} onClick={() => handleReject(id, "用户拒绝")}>
                拒绝
              </button>
            </div>
          </div>
        ))}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div style={{ padding: "8px 12px", borderTop: "1px solid rgba(255,255,255,0.06)" }}>
        <div style={{ display: "flex", gap: 8, alignItems: "flex-end" }}>
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={2}
            placeholder={mode === "builder" ? "描述你想构建的内容..." : "输入消息... (Ctrl+L 聚焦)"}
            className="glass-input"
            style={{
              flex: 1, resize: "none", minHeight: 40, maxHeight: 120,
              background: "rgba(255,255,255,0.03)", fontFamily: "inherit",
            }}
          />
          {isStreaming ? (
            <button className="glass-btn" onClick={handleStop}
              style={{ flexShrink: 0, height: 40, width: 40, padding: 0,
                       background: "rgba(255,82,82,0.15)", borderColor: "rgba(255,82,82,0.3)" }}>
              <Square size={14} fill="var(--red)" color="var(--red)" />
            </button>
          ) : (
            <button className="glass-btn glass-btn-primary"
              onClick={handleSend}
              disabled={!input.trim()}
              style={{ flexShrink: 0, height: 40, width: 40, padding: 0 }}>
              <Send size={16} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
