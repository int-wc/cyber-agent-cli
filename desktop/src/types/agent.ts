export type AgentEventType =
  | "connected"
  | "turn_start"
  | "response_begin"
  | "reasoning_token"
  | "response_token"
  | "response_end"
  | "tool_call"
  | "tool_result"
  | "approval_request"
  | "approval_result"
  | "turn_end"
  | "error"
  | "stopped"
  | "ping"
  | "pong";

export interface AgentEvent {
  type: AgentEventType;
  payload?: unknown;
  session_id?: string;
  input?: string;
  content?: string;
  tool_calls?: ToolCall[];
  tool_name?: string;
  tool_call_id?: string;
  risk?: string;
  approved?: boolean;
  reason?: string;
  message?: string;
  text?: string;
  has_tool_calls?: boolean;
  input_tokens?: number;
  output_tokens?: number;
  total_tokens?: number;
}

export interface ToolCall {
  name: string;
  args: Record<string, unknown>;
  id: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "reasoning" | "system" | "error";
  content: string;
  toolCalls?: ToolCall[];
  toolResults?: { name: string; content: string }[];
  streaming?: boolean;
  usage?: { input_tokens: number; output_tokens: number; total_tokens: number };
  timestamp: number;
}
