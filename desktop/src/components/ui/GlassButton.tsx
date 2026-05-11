import { cn } from "@/utils/cn";

interface GlassButtonProps {
  children: React.ReactNode;
  onClick?: () => void;
  accent?: boolean;
  small?: boolean;
  className?: string;
  disabled?: boolean;
  title?: string;
}

export function GlassButton({ children, onClick, accent, small, className, disabled, title }: GlassButtonProps) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={cn(
        "glass-button",
        accent && "glass-button-accent",
        small && "p-1.5 text-xs",
        disabled && "opacity-50 cursor-not-allowed",
        className
      )}
    >
      {children}
    </button>
  );
}
