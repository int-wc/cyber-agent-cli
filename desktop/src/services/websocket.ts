import type { AgentEvent } from "@/types/agent";
import { useChatStore } from "@/stores/useChatStore";
import { useWorkspaceStore } from "@/stores/useWorkspaceStore";

export class AgentWebSocket {
  private ws: WebSocket | null = null;
  private port: number;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectDelay = 1000;
  private maxReconnectDelay = 30000;
  private shouldReconnect = true;

  constructor(port: number) {
    this.port = port;
  }

  connect() {
    if (this.ws?.readyState === WebSocket.OPEN) return;
    this.shouldReconnect = true;
    this.reconnectDelay = 1000;
    useChatStore.getState().setWsConnected(false);
    this._doConnect();
  }

  private _doConnect() {
    const url = `ws://127.0.0.1:${this.port}/ws/chat`;
    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      this.reconnectDelay = 1000;
      useChatStore.getState().setWsConnected(true);
    };

    this.ws.onmessage = (event) => {
      try {
        const msg: AgentEvent = JSON.parse(event.data);
        this._handleMessage(msg);
      } catch {}
    };

    this.ws.onclose = () => {
      useChatStore.getState().setWsConnected(false);
      if (this.shouldReconnect) {
        this.reconnectTimer = setTimeout(() => {
          this.reconnectDelay = Math.min(this.reconnectDelay * 2, this.maxReconnectDelay);
          this._doConnect();
        }, this.reconnectDelay);
      }
    };

    this.ws.onerror = () => {
      this.ws?.close();
    };
  }

  private _handleMessage(msg: AgentEvent) {
    const store = useChatStore.getState();
    const wsStore = useWorkspaceStore.getState();

    switch (msg.type) {
      case "connected":
        if (msg.session_id) store.setSessionId(msg.session_id);
        wsStore.setBackendStatus("connected");
        break;

      case "turn_start":
        store.flushReasoning();
        break;

      case "response_begin":
        store.setStreaming(true);
        store.flushReasoning();
        store.appendStreamToken("");
        break;

      case "reasoning_token":
        store.appendReasoning(msg.text || "");
        break;

      case "response_token":
        store.appendStreamToken(msg.text || "");
        store.setStreaming(true);
        break;

      case "response_end":
        store.flushStream(
          (msg as AgentEvent).content || "",
          (msg as AgentEvent).has_tool_calls || false
        );
        break;

      case "tool_call":
        store.addToolCallMessage((msg as AgentEvent).tool_calls);
        break;

      case "tool_result":
        store.addToolResult(
          (msg as AgentEvent).tool_name || "unknown",
          (msg as AgentEvent).content || ""
        );
        break;

      case "approval_request":
        store.setPendingApproval({
          tool_call_id: msg.tool_call_id || "",
          tool_name: msg.tool_name || "",
          risk: msg.risk || "unknown",
        });
        break;

      case "approval_result":
        store.clearPendingApproval();
        break;

      case "turn_end":
        store.setLastUsage({
          input_tokens: msg.input_tokens || 0,
          output_tokens: msg.output_tokens || 0,
          total_tokens: msg.total_tokens || 0,
        });
        store.setStreaming(false);
        break;

      case "error":
        store.addMessage({
          id: "err-" + Date.now(),
          role: "error",
          content: msg.message || "未知错误",
          timestamp: Date.now(),
        });
        store.setStreaming(false);
        break;

      case "stopped":
        store.setStreaming(false);
        break;
    }
  }

  sendMessage(content: string) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "user_message", content }));
    }
  }

  sendStop() {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "stop" }));
    }
  }

  approveTool(toolCallId: string) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "approve", tool_call_id: toolCallId }));
    }
  }

  rejectTool(toolCallId: string, reason = "拒绝") {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "reject", tool_call_id: toolCallId, reason }));
    }
  }

  disconnect() {
    this.shouldReconnect = false;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.ws?.close();
    this.ws = null;
  }
}
