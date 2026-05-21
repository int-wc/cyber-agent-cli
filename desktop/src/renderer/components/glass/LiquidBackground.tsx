/** Apple Liquid Glass background — white base with soft colorful orbs. */
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
      {/* Soft white gradient base */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          background: `
            radial-gradient(ellipse 80% 60% at 20% 20%, rgba(124,111,247,0.08) 0%, transparent 55%),
            radial-gradient(ellipse 60% 80% at 80% 80%, rgba(59,130,246,0.06) 0%, transparent 55%),
            radial-gradient(ellipse 50% 50% at 50% 50%, rgba(34,197,94,0.04) 0%, transparent 50%),
            radial-gradient(ellipse 70% 30% at 60% 10%, rgba(245,158,11,0.05) 0%, transparent 50%),
            linear-gradient(180deg, #f5f5fa 0%, #fafafe 50%, #f0f0f5 100%)
          `,
        }}
      />

      {/* Orb 1 — soft purple */}
      <div
        style={{
          position: "absolute",
          width: 460,
          height: 460,
          borderRadius: "50%",
          background: "radial-gradient(circle, rgba(124,111,247,0.18) 0%, rgba(124,111,247,0.04) 35%, transparent 70%)",
          top: "-12%",
          left: "-8%",
          filter: "blur(44px)",
          animation: "orbFloat1 20s ease-in-out infinite",
        }}
      />

      {/* Orb 2 — soft blue */}
      <div
        style={{
          position: "absolute",
          width: 380,
          height: 380,
          borderRadius: "50%",
          background: "radial-gradient(circle, rgba(59,130,246,0.14) 0%, rgba(59,130,246,0.03) 35%, transparent 70%)",
          bottom: "-15%",
          right: "-5%",
          filter: "blur(40px)",
          animation: "orbFloat2 24s ease-in-out infinite",
        }}
      />

      {/* Orb 3 — soft green */}
      <div
        style={{
          position: "absolute",
          width: 260,
          height: 260,
          borderRadius: "50%",
          background: "radial-gradient(circle, rgba(34,197,94,0.12) 0%, rgba(34,197,94,0.03) 35%, transparent 70%)",
          top: "50%",
          left: "45%",
          filter: "blur(34px)",
          animation: "orbFloat3 16s ease-in-out infinite",
        }}
      />

      {/* Orb 4 — soft warm */}
      <div
        style={{
          position: "absolute",
          width: 200,
          height: 200,
          borderRadius: "50%",
          background: "radial-gradient(circle, rgba(245,158,11,0.10) 0%, rgba(245,158,11,0.02) 35%, transparent 70%)",
          top: "15%",
          right: "20%",
          filter: "blur(30px)",
          animation: "orbFloat4 18s ease-in-out infinite",
        }}
      />

      {/* Very subtle grain texture */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          opacity: 0.015,
          backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='1'/%3E%3C/svg%3E")`,
        }}
      />
    </div>
  );
}
