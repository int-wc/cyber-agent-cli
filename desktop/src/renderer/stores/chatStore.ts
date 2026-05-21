import { create } from "zustand";
import type { ChatMessage, TokenUsage } from "../types/agent";

interface ChatState {
  messages: ChatMessage[];
  currentStream: string;
  currentReasoning: string;
  isStreaming: boolean;
  pendingApprovals: Map<string, { toolName: string; toolCall: Record<string, unknown>; risk: string }>;
  recentToolCalls: { id: string; name: string; args: Record<string, unknown>; result?: string }[];
  usage: TokenUsage | null;

  addUserMessage: (content: string) => void;
  appendToStream: (token: string) => void;
  appendToReasoning: (token: string) => void;
  finalizeMessage: (hasToolCalls: boolean, usage?: TokenUsage) => void;
  addToolCall: (toolCalls: Record<string, unknown>[]) => void;
  addToolResult: (toolName: string, content: string) => void;
  addApprovalRequest: (req: { toolName: string; toolCall: Record<string, unknown>; risk: string }) => void;
  clearApproval: () => void;
  clearStream: () => void;
}

let msgCounter = 0;

export const useChatStore = create<ChatState>((set, get) => ({
  messages: [],
  currentStream: "",
  currentReasoning: "",
  isStreaming: false,
  pendingApprovals: new Map(),
  recentToolCalls: [],
  usage: null,

  addUserMessage: (content) => {
    const msg: ChatMessage = {
      id: `msg-${++msgCounter}`,
      role: "user",
      content,
      timestamp: Date.now(),
    };
    set((s) => ({ messages: [...s.messages, msg], currentStream: "", currentReasoning: "" }));
  },

  appendToStream: (token) => {
    set((s) => ({ currentStream: s.currentStream + token, isStreaming: true }));
  },

  appendToReasoning: (token) => {
    set((s) => ({ currentReasoning: s.currentReasoning + token }));
  },

  finalizeMessage: (hasToolCalls, usage) => {
    const { currentStream, currentReasoning, messages } = get();
    if (currentStream || currentReasoning) {
      const msg: ChatMessage = {
        id: `msg-${++msgCounter}`,
        role: "assistant",
        content: currentStream,
        reasoning: currentReasoning || undefined,
        timestamp: Date.now(),
        toolCalls: hasToolCalls ? [...get().recentToolCalls] : undefined,
        usage: usage || undefined,
      };
      set({
        messages: [...messages, msg],
        currentStream: "",
        currentReasoning: "",
        isStreaming: false,
        recentToolCalls: [],
        usage: usage || null,
      });
    }
  },

  addToolCall: (toolCalls) => {
    const entries = (Array.isArray(toolCalls) ? toolCalls : []).map((tc: Record<string, unknown>) => ({
      id: (tc.id as string) || `tc-${++msgCounter}`,
      name: (tc.name as string) || "unknown",
      args: (tc.args as Record<string, unknown>) || {},
    }));
    set((s) => ({ recentToolCalls: [...s.recentToolCalls, ...entries] }));
  },

  addToolResult: (toolName, content) => {
    set((s) => ({
      recentToolCalls: s.recentToolCalls.map((tc) =>
        tc.name === toolName && !tc.result ? { ...tc, result: content } : tc
      ),
    }));
  },

  addApprovalRequest: (req) => {
    set((s) => {
      const m = new Map(s.pendingApprovals);
      m.set(req.toolCall.id as string, req);
      return { pendingApprovals: m };
    });
  },

  clearApproval: () => {
    set({ pendingApprovals: new Map() });
  },

  clearStream: () => {
    set({ currentStream: "", currentReasoning: "", isStreaming: false });
  },
}));
