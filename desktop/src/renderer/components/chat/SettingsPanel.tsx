import { useState } from "react";
import { useSessionStore } from "../../stores/sessionStore";
import { sessionApi } from "../../services/api";
import { X, Settings, Globe, Shield, Cpu } from "lucide-react";

interface SettingsPanelProps {
  onClose: () => void;
}

const PROVIDERS = ["openai", "deepseek", "claude", "mimo"];

export default function SettingsPanel({ onClose }: SettingsPanelProps) {
  const { config, setConfig } = useSessionStore();
  const [localService, setLocalService] = useState(config.service);
  const [localModel, setLocalModel] = useState(config.model);
  const [localMode, setLocalMode] = useState(config.mode);
  const [localPolicy, setLocalPolicy] = useState(config.approvalPolicy);

  const handleApply = async () => {
    setConfig({
      service: localService,
      model: localModel,
      mode: localMode,
      approvalPolicy: localPolicy,
    });
    try {
      await sessionApi.setMode(localMode);
      await sessionApi.setApprovalPolicy(localPolicy);
      await sessionApi.switchModel(localService, localModel);
    } catch { /* backend may not be ready */ }
    onClose();
  };

  return (
    <div className="glass-dialog-overlay" onClick={onClose}>
      <div className="glass-dialog" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 20 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <Settings size={18} color="var(--accent-light)" />
            <span style={{ fontSize: 15, fontWeight: 600, color: "var(--text-primary)" }}>
              设置
            </span>
          </div>
          <button
            className="glass-btn"
            style={{ padding: "4px 8px" }}
            onClick={onClose}
          >
            <X size={14} />
          </button>
        </div>

        {/* Provider */}
        <div style={{ marginBottom: 16 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
            <Globe size={14} color="var(--text-secondary)" />
            <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>服务商</span>
          </div>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            {PROVIDERS.map((p) => (
              <button
                key={p}
                className={`glass-btn ${localService === p ? "glass-btn-primary" : ""}`}
                style={{ fontSize: 12 }}
                onClick={() => setLocalService(p)}
              >
                {p.charAt(0).toUpperCase() + p.slice(1)}
              </button>
            ))}
          </div>
        </div>

        {/* Model */}
        <div style={{ marginBottom: 16 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
            <Cpu size={14} color="var(--text-secondary)" />
            <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>模型</span>
          </div>
          <input
            className="glass-input"
            value={localModel}
            onChange={(e) => setLocalModel(e.target.value)}
            placeholder="输入模型名称 (留空使用默认)"
            style={{ width: "100%" }}
          />
        </div>

        {/* Mode */}
        <div style={{ marginBottom: 16 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
            <Shield size={14} color="var(--text-secondary)" />
            <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>运行模式</span>
          </div>
          <div style={{ display: "flex", gap: 6 }}>
            {(["standard", "authorized"] as const).map((m) => (
              <button
                key={m}
                className={`glass-btn ${localMode === m ? "glass-btn-primary" : ""}`}
                style={{ fontSize: 12 }}
                onClick={() => setLocalMode(m)}
              >
                {m === "standard" ? "标准" : "授权"}
              </button>
            ))}
          </div>
        </div>

        {/* Approval Policy */}
        <div style={{ marginBottom: 24 }}>
          <div style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 6 }}>
            审批策略
          </div>
          <div style={{ display: "flex", gap: 6 }}>
            {(["prompt", "auto", "never"] as const).map((p) => (
              <button
                key={p}
                className={`glass-btn ${localPolicy === p ? "glass-btn-primary" : ""}`}
                style={{ fontSize: 12 }}
                onClick={() => setLocalPolicy(p)}
              >
                {p === "prompt" ? "交互审批" : p === "auto" ? "自动批准" : "全部拒绝"}
              </button>
            ))}
          </div>
        </div>

        {/* Actions */}
        <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
          <button className="glass-btn" onClick={onClose} style={{ fontSize: 12 }}>
            取消
          </button>
          <button className="glass-btn glass-btn-primary" onClick={handleApply} style={{ fontSize: 12 }}>
            应用
          </button>
        </div>
      </div>
    </div>
  );
}
