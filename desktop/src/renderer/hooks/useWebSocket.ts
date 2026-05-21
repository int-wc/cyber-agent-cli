import { useEffect, useRef, useCallback } from "react";
import { wsClient } from "../services/ws";

type AgentEventHandler = (event: {
  type: string;
  payload?: Record<string, unknown>;
}) => void;

export function useWebSocket(port: number) {
  const handlerRef = useRef<AgentEventHandler | null>(null);

  useEffect(() => {
    wsClient.connect(port);
    const unsub = wsClient.onEvent((event) => {
      handlerRef.current?.(event);
    });
    return () => {
      unsub();
    };
  }, [port]);

  const onEvent = useCallback((handler: AgentEventHandler) => {
    handlerRef.current = handler;
  }, []);

  const send = useCallback((msg: Record<string, unknown>) => {
    wsClient.send(msg);
  }, []);

  return { onEvent, send };
}
