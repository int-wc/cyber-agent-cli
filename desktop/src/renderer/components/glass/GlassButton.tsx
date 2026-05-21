import type { ReactNode, CSSProperties } from "react";

interface GlassButtonProps {
  children: ReactNode;
  primary?: boolean;
  disabled?: boolean;
  size?: "sm" | "md";
  style?: CSSProperties;
  onClick?: () => void;
}

export default function GlassButton({
  children, primary, disabled, size = "md", style, onClick,
}: GlassButtonProps) {
  const sizeStyle = size === "sm"
    ? { padding: "3px 10px", fontSize: 11 }
    : { padding: "6px 14px", fontSize: 13 };

  return (
    <button
      className={`glass-btn ${primary ? "glass-btn-primary" : ""}`}
      disabled={disabled}
      style={{
        ...sizeStyle,
        opacity: disabled ? 0.4 : 1,
        cursor: disabled ? "not-allowed" : "pointer",
        ...style,
      }}
      onClick={onClick}
    >
      {children}
    </button>
  );
}
