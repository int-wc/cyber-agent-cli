import { useState } from "react";
import type { ToolCall } from "../../types/agent";
import { ChevronDown, ChevronRight, Wrench, CheckCircle, XCircle } from "lucide-react";

interface ToolCallCardProps {
  toolCall: ToolCall;
}

export default function ToolCallCard({ toolCall }: ToolCallCardProps) {
  const [expanded, setExpanded] = useState(false);

  const hasResult = toolCall.result !== undefined;
  const hasError = toolCall.error !== undefined;
  const riskColor = toolCall.risk === "execute" ? "var(--orange)" :
    toolCall.risk === "write" ? "var(--yellow)" : "var(--blue)";

  return (
    <div
      className="glass-panel glass-card-accent"
      style={{
        marginTop: 8, fontSize: 12,
        borderColor: hasError ? "rgba(255,82,82,0.3)" : undefined,
        background: hasError ? "rgba(255,82,82,0.04)" : undefined,
      }}
    >
      {/* Header */}
      <div
        onClick={() => setExpanded(!expanded)}
        style={{
          display: "flex", alignItems: "center", gap: 8,
          padding: "8px 12px", cursor: "pointer",
          userSelect: "none",
        }}
      >
        {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        <Wrench size={14} color={riskColor} />
        <span style={{ fontWeight: 600, color: "var(--text-primary)" }}>
          {toolCall.name}
        </span>
        {toolCall.risk && (
          <span style={{
            fontSize: 10, padding: "1px 5px", borderRadius: 4,
            background: `${riskColor}20`, color: riskColor,
          }}>
            {toolCall.risk}
          </span>
        )}
        {hasResult && <CheckCircle size={12} color="var(--green)" />}
        {hasError && <XCircle size={12} color="var(--red)" />}
      </div>

      {/* Expanded content */}
      {expanded && (
        <div style={{ padding: "0 12px 10px 12px" }}>
          {/* Args */}
          {Object.keys(toolCall.args).length > 0 && (
            <div style={{ marginBottom: 8 }}>
              <div style={{ fontSize: 10, color: "var(--text-tertiary)", marginBottom: 4 }}>
                参数
              </div>
              <pre style={{
                fontSize: 11, fontFamily: "monospace",
                background: "rgba(0,0,0,0.3)", borderRadius: 6,
                padding: "6px 10px", overflow: "auto", maxHeight: 150,
                color: "var(--text-secondary)",
              }}>
                {JSON.stringify(toolCall.args, null, 2)}
              </pre>
            </div>
          )}

          {/* Result */}
          {hasResult && (
            <div>
              <div style={{ fontSize: 10, color: "var(--text-tertiary)", marginBottom: 4 }}>
                结果
              </div>
              <pre style={{
                fontSize: 11, fontFamily: "monospace",
                background: "rgba(0,0,0,0.3)", borderRadius: 6,
                padding: "6px 10px", overflow: "auto", maxHeight: 300,
                color: "var(--text-secondary)", whiteSpace: "pre-wrap",
                wordBreak: "break-all",
              }}>
                {toolCall.result.length > 2000
                  ? toolCall.result.slice(0, 2000) + "\n... (截断)"
                  : toolCall.result}
              </pre>
            </div>
          )}

          {/* Error */}
          {hasError && (
            <div style={{
              marginTop: 8, padding: "6px 10px",
              background: "rgba(255,82,82,0.1)", borderRadius: 6,
              color: "var(--red)", fontSize: 11,
            }}>
              {toolCall.error}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
