import { useEffect, useRef } from "react";
import { wsClient } from "../services/ws";
import { useChatStore } from "../stores/chatStore";

export function useAgentEvents() {
  const store = useRef(useChatStore.getState());
  store.current = useChatStore.getState();

  useEffect(() => {
    const unsubState = useChatStore.subscribe((state) => {
      store.current = state;
    });

    const unsubWs = wsClient.onEvent((event) => {
      const s = store.current;
      const { type, payload } = event;

      switch (type) {
        case "response_token":
          s.appendToStream((payload?.token as string) || "");
          break;
        case "reasoning_token":
          s.appendToReasoning((payload?.token as string) || "");
          break;
        case "response_end":
          s.finalizeMessage(
            (payload?.has_tool_calls as boolean) || false,
            payload?.usage as Record<string, number> | undefined as never,
          );
          break;
        case "tool_call":
          s.addToolCall((payload?.tool_calls as Record<string, unknown>[]) || []);
          break;
        case "tool_result": {
          const name = (payload?.tool_name as string) || "";
          const content = (payload?.content as string) || "";
          s.addToolResult(name, content);
          break;
        }
        case "approval_request": {
          const req = payload as Record<string, unknown> | undefined;
          if (req) {
            s.addApprovalRequest({
              toolName: (req.tool_name as string) || "",
              toolCall: (req.tool_call as Record<string, unknown>) || {},
              risk: (req.risk as string) || "unknown",
            });
          }
          break;
        }
        case "turn_end":
          s.finalizeMessage(false, payload?.usage as Record<string, number> | undefined as never);
          break;
        case "error":
          s.clearStream();
          break;
      }
    });

    return () => {
      unsubState();
      unsubWs();
    };
  }, []);
}
