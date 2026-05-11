/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        window: "#0f172a",
        surface: "#111827",
        "surface-light": "#1e293b",
        border: "#334155",
        "border-glass": "rgba(255,255,255,0.08)",
        primary: "#e2e8f0",
        muted: "#94a3b8",
        soft: "#cbd5e1",
        accent: {
          amber: "#f59e0b",
          teal: "#14b8a6",
          red: "#ef4444",
          indigo: "#6366f1",
          cyan: "#67e8f9",
          purple: "#c4b5fd",
        },
      },
      borderRadius: {
        glass: "12px",
        "glass-lg": "16px",
        "glass-sm": "8px",
      },
      backdropBlur: {
        glass: "16px",
        "glass-heavy": "24px",
      },
      animation: {
        "fade-in": "fadeIn 0.3s ease-out",
        "slide-up": "slideUp 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
        "slide-right": "slideRight 0.2s ease-out",
        pulse: "pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        slideUp: {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        slideRight: {
          "0%": { opacity: "0", transform: "translateX(-8px)" },
          "100%": { opacity: "1", transform: "translateX(0)" },
        },
        pulse: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.5" },
        },
      },
    },
  },
  plugins: [],
};
