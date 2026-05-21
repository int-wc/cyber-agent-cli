import { useEffect, useState } from "react";
import { AlertTriangle, Clock } from "lucide-react";

interface ApprovalRequestProps {
  toolName: string;
  toolCall: Record<string, unknown>;
  risk: string;
  onApprove: () => void;
  onReject: (reason: string) => void;
}

export default function ApprovalRequest({
  toolName, toolCall, risk, onApprove, onReject,
}: ApprovalRequestProps) {
  const [countdown, setCountdown] = useState(30);
  const [rejectReason, setRejectReason] = useState("");

  useEffect(() => {
    if (countdown <= 0) {
      onReject("审批超时自动拒绝");
      return;
    }
    const timer = setTimeout(() => setCountdown(countdown - 1), 1000);
    return () => clearTimeout(timer);
  }, [countdown, onReject]);

  const riskColor = risk === "execute" ? "var(--orange)" :
    risk === "write" ? "var(--yellow)" : "var(--blue)";

  return (
    <div className="glass-panel" style={{
      marginBottom: 12, padding: 14,
      borderColor: "rgba(255,171,64,0.4)",
      background: "rgba(255,171,64,0.06)",
    }}>
      {/* Header */}
      <div style={{
        display: "flex", alignItems: "center", gap: 8,
        marginBottom: 8,
      }}>
        <AlertTriangle size={16} color="var(--orange)" />
        <span style={{ fontSize: 13, fontWeight: 600, color: "var(--orange)" }}>
          需要审批
        </span>
        <span style={{
          fontSize: 10, padding: "2px 6px", borderRadius: 4,
          background: `${riskColor}20`, color: riskColor,
        }}>
          {risk.toUpperCase()} 风险
        </span>
        <span style={{
          marginLeft: "auto", display: "flex", alignItems: "center", gap: 4,
          fontSize: 11, color: countdown < 10 ? "var(--red)" : "var(--text-tertiary)",
        }}>
          <Clock size={12} />
          {countdown}s
        </span>
      </div>

      {/* Tool info */}
      <div style={{ marginBottom: 8 }}>
        <div style={{ fontSize: 11, color: "var(--text-tertiary)", marginBottom: 2 }}>
          工具
        </div>
        <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text-primary)" }}>
          {toolName}
        </div>
      </div>

      {/* Args */}
      <div style={{ marginBottom: 12 }}>
        <div style={{ fontSize: 11, color: "var(--text-tertiary)", marginBottom: 2 }}>
          参数
        </div>
        <pre style={{
          fontSize: 11, fontFamily: "monospace",
          background: "rgba(0,0,0,0.3)", borderRadius: 6,
          padding: "6px 10px", maxHeight: 120, overflow: "auto",
          color: "var(--text-secondary)", whiteSpace: "pre-wrap",
          wordBreak: "break-all",
        }}>
          {JSON.stringify(toolCall.args || toolCall, null, 2)}
        </pre>
      </div>

      {/* Actions */}
      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <button
          className="glass-btn glass-btn-primary"
          onClick={onApprove}
          style={{ fontSize: 12, padding: "6px 16px" }}
        >
          批准执行
        </button>
        <input
          className="glass-input"
          value={rejectReason}
          onChange={(e) => setRejectReason(e.target.value)}
          placeholder="拒绝原因（可选）"
          style={{ flex: 1, fontSize: 12, padding: "4px 10px",
                   background: "rgba(255,255,255,0.03)" }}
          onKeyDown={(e) => {
            if (e.key === "Enter") onReject(rejectReason || "用户拒绝");
          }}
        />
        <button
          className="glass-btn"
          onClick={() => onReject(rejectReason || "用户拒绝")}
          style={{ fontSize: 12, padding: "6px 12px", color: "var(--red)",
                   borderColor: "rgba(255,82,82,0.3)" }}
        >
          拒绝
        </button>
      </div>
    </div>
  );
}
