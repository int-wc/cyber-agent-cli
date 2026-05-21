import { forwardRef } from "react";

interface GlassInputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  fullWidth?: boolean;
}

const GlassInput = forwardRef<HTMLInputElement, GlassInputProps>(
  ({ fullWidth, style, className = "", ...rest }, ref) => {
    return (
      <input
        ref={ref}
        className={`glass-input ${className}`}
        style={{ width: fullWidth ? "100%" : undefined, ...style }}
        {...rest}
      />
    );
  }
);

GlassInput.displayName = "GlassInput";
export default GlassInput;
