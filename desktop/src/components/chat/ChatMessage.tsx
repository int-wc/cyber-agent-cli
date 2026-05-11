import type { ChatMessage as ChatMessageType } from "@/types/agent";
import { Wrench, User, Bot, Brain, AlertCircle } from "lucide-react";
import { useState } from "react";

const ROLE_CONFIG = {
  user: { icon: User, border: "border-amber-500/30", bg: "bg-amber-500/5", label: "你" },
  assistant: { icon: Bot, border: "border-teal-500/30", bg: "bg-teal-500/5", label: "AI" },
  reasoning: { icon: Brain, border: "border-indigo-500/20", bg: "bg-indigo-500/5", label: "思考" },
  system: { icon: Wrench, border: "border-slate-500/30", bg: "bg-slate-500/5", label: "系统" },
  error: { icon: AlertCircle, border: "border-red-500/30", bg: "bg-red-500/5", label: "错误" },
};

export function ChatMessage({ message }: { message: ChatMessageType }) {
  const config = ROLE_CONFIG[message.role] || ROLE_CONFIG.system;
  const Icon = config.icon;
  const [showTools, setShowTools] = useState(false);

  return (
    <div className={`p-2.5 rounded-glass-sm border ${config.border} ${config.bg} text-xs animate-fade-in`}>
      <div className="flex items-center gap-1.5 mb-1 no-select">
        <Icon size={12} />
        <span className="font-medium text-muted">{config.label}</span>
        {message.usage && (
          <span className="ml-auto text-[10px] text-muted">
            ↑{message.usage.input_tokens} ↓{message.usage.output_tokens}
          </span>
        )}
      </div>
      <div className={`whitespace-pre-wrap break-words leading-relaxed ${message.role === "reasoning" ? "text-indigo-300/70 italic text-[11px]" : "text-primary"}`}>
        {message.content || (message.streaming ? "..." : "")}
        {message.streaming && <span className="animate-pulse text-accent-teal">│</span>}
      </div>
      {message.toolCalls && (
        <div className="mt-1.5">
          <button
            onClick={() => setShowTools(!showTools)}
            className="text-[10px] text-accent-amber hover:underline"
          >
            {showTools ? "收起" : "查看"}工具调用 ({message.toolCalls.length})
          </button>
          {showTools && message.toolCalls.map((tc, i) => (
            <pre key={i} className="mt-1 p-1.5 rounded bg-window/80 text-[10px] text-soft overflow-x-auto">
              {tc.name}({JSON.stringify(tc.args, null, 2)})
            </pre>
          ))}
        </div>
      )}
      {message.toolResults && message.toolResults.map((tr, i) => (
        <details key={i} className="mt-1">
          <summary className="text-[10px] text-accent-teal cursor-pointer hover:underline">
            {tr.name} 结果
          </summary>
          <pre className="mt-1 p-1.5 rounded bg-window/80 text-[10px] text-soft overflow-x-auto max-h-32 overflow-y-auto">
            {tr.content}
          </pre>
        </details>
      ))}
    </div>
  );
}
