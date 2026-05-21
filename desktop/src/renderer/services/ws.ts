import { setBackendPort } from "./api";

type EventHandler = (event: { type: string; payload?: Record<string, unknown> }) => void;

class WebSocketClient {
  private ws: WebSocket | null = null;
  private handlers: Set<EventHandler> = new Set();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectDelay = 1000;
  private pingInterval: ReturnType<typeof setInterval> | null = null;
  private port: number = 0;

  connect(port: number) {
    this.port = port;
    setBackendPort(port);
    this._doConnect();
  }

  private _doConnect() {
    const url = `ws://127.0.0.1:${this.port}/ws/chat`;

    try {
      this.ws = new WebSocket(url);
    } catch {
      this._scheduleReconnect();
      return;
    }

    this.ws.onopen = () => {
      this.reconnectAttempts = 0;
      this.reconnectDelay = 1000;
      this._startPing();
      this._emit({ type: "ws_connected" });
    };

    this.ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data);
        this._emit(data);
      } catch {
        // ignore malformed messages
      }
    };

    this.ws.onclose = () => {
      this._stopPing();
      this._emit({ type: "ws_disconnected" });
      this._scheduleReconnect();
    };

    this.ws.onerror = () => {
      // onclose will fire after onerror
    };
  }

  private _scheduleReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) return;
    this.reconnectAttempts++;
    setTimeout(() => this._doConnect(), this.reconnectDelay);
    this.reconnectDelay = Math.min(this.reconnectDelay * 1.5, 30000);
  }

  private _startPing() {
    this._stopPing();
    this.pingInterval = setInterval(() => {
      this.send({ type: "ping" });
    }, 30000);
  }

  private _stopPing() {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  onEvent(handler: EventHandler) {
    this.handlers.add(handler);
    return () => this.handlers.delete(handler);
  }

  private _emit(event: { type: string; payload?: Record<string, unknown> }) {
    for (const h of this.handlers) {
      try {
        h(event);
      } catch {}
    }
  }

  send(msg: Record<string, unknown>) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    }
  }

  disconnect() {
    this._stopPing();
    this.ws?.close();
    this.ws = null;
  }
}

export const wsClient = new WebSocketClient();
