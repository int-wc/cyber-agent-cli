import { create } from "zustand";
import type { ChatMessage, AgentEvent } from "@/types/agent";

interface ChatStore {
  messages: ChatMessage[];
  streaming: boolean;
  streamContent: string;
  reasoningContent: string;
  pendingApproval: { tool_call_id: string; tool_name: string; risk: string } | null;
  wsConnected: boolean;
  sessionId: string;

  addMessage: (msg: ChatMessage) => void;
  updateLastMessage: (content: string) => void;
  setStreaming: (v: boolean) => void;
  appendStreamToken: (text: string) => void;
  appendReasoning: (text: string) => void;
  flushStream: (content: string, hasToolCalls: boolean) => void;
  flushReasoning: () => void;
  setPendingApproval: (a: ChatStore["pendingApproval"]) => void;
  clearPendingApproval: () => void;
  setLastUsage: (usage: { input_tokens: number; output_tokens: number; total_tokens: number }) => void;
  addToolCallMessage: (toolCalls: AgentEvent["tool_calls"]) => void;
  addToolResult: (name: string, content: string) => void;
  clearMessages: () => void;
  setWsConnected: (v: boolean) => void;
  setSessionId: (id: string) => void;
}

let msgId = 0;
const nextId = () => `msg-${++msgId}`;

export const useChatStore = create<ChatStore>((set, get) => ({
  messages: [],
  streaming: false,
  streamContent: "",
  reasoningContent: "",
  pendingApproval: null,
  wsConnected: false,
  sessionId: "",

  addMessage: (msg) => set((s) => ({ messages: [...s.messages, msg] })),
  updateLastMessage: (content) =>
    set((s) => {
      const msgs = [...s.messages];
      if (msgs.length > 0) msgs[msgs.length - 1] = { ...msgs[msgs.length - 1], content };
      return { messages: msgs };
    }),
  setStreaming: (v) => set({ streaming: v }),
  appendStreamToken: (text) => set((s) => ({ streamContent: s.streamContent + text })),
  appendReasoning: (text) => set((s) => ({ reasoningContent: s.reasoningContent + text })),
  flushStream: (content, hasToolCalls) => {
    if (!hasToolCalls && content) {
      set((s) => ({
        messages: [...s.messages, { id: nextId(), role: "assistant", content, timestamp: Date.now() }],
        streamContent: "",
        streaming: false,
      }));
    } else {
      set({ streamContent: "", streaming: false });
    }
  },
  flushReasoning: () => {
    const content = get().reasoningContent;
    if (content) {
      set((s) => ({
        messages: [...s.messages, { id: nextId(), role: "reasoning", content, timestamp: Date.now() }],
        reasoningContent: "",
      }));
    }
  },
  setPendingApproval: (a) => set({ pendingApproval: a }),
  clearPendingApproval: () => set({ pendingApproval: null }),
  setLastUsage: (usage) =>
    set((s) => {
      const msgs = [...s.messages];
      if (msgs.length > 0) msgs[msgs.length - 1] = { ...msgs[msgs.length - 1], usage };
      return { messages: msgs };
    }),
  addToolCallMessage: (toolCalls) => {
    if (!toolCalls) return;
    const names = toolCalls.map((tc) => tc.name).join(", ");
    set((s) => ({
      messages: [
        ...s.messages,
        { id: nextId(), role: "system", content: `工具调用: ${names}`, toolCalls, timestamp: Date.now() },
      ],
    }));
  },
  addToolResult: (name, content) =>
    set((s) => ({
      messages: [
        ...s.messages,
        { id: nextId(), role: "system", content: `工具结果 [${name}]: ${content.slice(0, 500)}`, toolResults: [{ name, content }], timestamp: Date.now() },
      ],
    })),
  clearMessages: () => set({ messages: [], streamContent: "", reasoningContent: "" }),
  setWsConnected: (v) => set({ wsConnected: v }),
  setSessionId: (id) => set({ sessionId: id }),
}));
