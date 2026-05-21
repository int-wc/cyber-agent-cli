import type { ReactNode, CSSProperties } from "react";

interface GlassPanelProps {
  children: ReactNode;
  className?: string;
  focused?: boolean;
  hasGlow?: boolean;
  intensity?: "light" | "medium" | "heavy";
  style?: CSSProperties;
  onClick?: () => void;
}

const intensityClasses = {
  light: "glass-surface-light",
  medium: "glass-surface",
  heavy: "glass-surface-heavy",
};

export default function GlassPanel({
  children,
  className = "",
  focused = false,
  hasGlow = false,
  intensity = "medium",
  style,
  onClick,
}: GlassPanelProps) {
  const base = intensityClasses[intensity];
  const glow = hasGlow ? "glass-border-glow" : "";
  const focus = focused ? "focused" : "";

  return (
    <div
      className={`glass-panel ${base} ${glow} ${focus} ${className}`}
      style={{ borderRadius: "var(--radius-md)", ...style }}
      onClick={onClick}
    >
      {children}
    </div>
  );
}
