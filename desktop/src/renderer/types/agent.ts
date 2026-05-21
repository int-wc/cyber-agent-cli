export interface AgentEvent {
  type: string;
  payload?: Record<string, unknown>;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system" | "tool";
  content: string;
  timestamp: number;
  toolCalls?: ToolCall[];
  reasoning?: string;
  usage?: TokenUsage;
}

export interface ToolCall {
  id: string;
  name: string;
  args: Record<string, unknown>;
  result?: string;
  error?: string;
  risk?: "read" | "write" | "execute";
}

export interface TokenUsage {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
}

export interface FileEntry {
  name: string;
  path: string;
  is_dir: boolean;
  size: number;
  modified: number;
}

export interface SessionConfig {
  mode: "standard" | "authorized";
  service: string;
  model: string;
  approvalPolicy: "prompt" | "auto" | "never";
}
