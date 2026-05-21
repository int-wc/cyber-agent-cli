/** Animated gradient orbs that sit behind glass panels to show the blur effect. */
export default function LiquidBackground() {
  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 0,
        overflow: "hidden",
        pointerEvents: "none",
      }}
    >
      {/* Deep gradient base */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          background: `
            radial-gradient(ellipse 80% 60% at 20% 30%, rgba(108,92,231,0.15) 0%, transparent 60%),
            radial-gradient(ellipse 60% 80% at 80% 70%, rgba(79,195,247,0.12) 0%, transparent 55%),
            radial-gradient(ellipse 50% 50% at 50% 50%, rgba(105,240,174,0.06) 0%, transparent 50%),
            radial-gradient(ellipse 70% 40% at 60% 15%, rgba(255,171,64,0.08) 0%, transparent 50%),
            #0a0a0f
          `,
        }}
      />

      {/* Floating orb 1 — purple */}
      <div
        className="liquid-orb"
        style={{
          position: "absolute",
          width: 420,
          height: 420,
          borderRadius: "50%",
          background: "radial-gradient(circle, rgba(108,92,231,0.25) 0%, rgba(108,92,231,0.05) 40%, transparent 70%)",
          top: "-10%",
          left: "-8%",
          filter: "blur(40px)",
          animation: "orbFloat1 18s ease-in-out infinite",
        }}
      />

      {/* Floating orb 2 — cyan */}
      <div
        className="liquid-orb"
        style={{
          position: "absolute",
          width: 350,
          height: 350,
          borderRadius: "50%",
          background: "radial-gradient(circle, rgba(79,195,247,0.20) 0%, rgba(79,195,247,0.04) 40%, transparent 70%)",
          bottom: "-12%",
          right: "-5%",
          filter: "blur(36px)",
          animation: "orbFloat2 22s ease-in-out infinite",
        }}
      />

      {/* Floating orb 3 — green accent */}
      <div
        className="liquid-orb"
        style={{
          position: "absolute",
          width: 240,
          height: 240,
          borderRadius: "50%",
          background: "radial-gradient(circle, rgba(105,240,174,0.18) 0%, rgba(105,240,174,0.03) 40%, transparent 70%)",
          top: "55%",
          left: "40%",
          filter: "blur(30px)",
          animation: "orbFloat3 14s ease-in-out infinite",
        }}
      />

      {/* Floating orb 4 — warm orange */}
      <div
        className="liquid-orb"
        style={{
          position: "absolute",
          width: 180,
          height: 180,
          borderRadius: "50%",
          background: "radial-gradient(circle, rgba(255,171,64,0.15) 0%, rgba(255,171,64,0.03) 40%, transparent 70%)",
          top: "20%",
          right: "25%",
          filter: "blur(28px)",
          animation: "orbFloat4 16s ease-in-out infinite",
        }}
      />

      {/* Subtle grain/noise texture overlay */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          opacity: 0.03,
          backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='1'/%3E%3C/svg%3E")`,
        }}
      />
    </div>
  );
}
