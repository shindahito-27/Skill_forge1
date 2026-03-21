function BrandLogo({ compact = false }) {
  return (
    <div className={`brand-lockup ${compact ? 'compact' : ''}`} aria-label="Skillforge">
      <div className="brand-glow brand-glow-left" />
      <div className="brand-glow brand-glow-right" />
      <svg
        className="brand-circuit"
        viewBox="0 0 260 90"
        role="img"
        aria-hidden="true"
      >
        <defs>
          <linearGradient id="circuitGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#59d6ff" />
            <stop offset="55%" stopColor="#7d57ff" />
            <stop offset="100%" stopColor="#ff7ac8" />
          </linearGradient>
        </defs>
        <path d="M20 56 H82 C98 56 110 48 110 34 V26" />
        <path d="M20 67 H98 C114 67 124 58 124 45 V24" />
        <path d="M35 78 H128 C144 78 156 69 156 56 V30" />
        <path d="M90 84 H184 C198 84 208 72 208 58 V38" />
        <path d="M120 66 H230" />
        <path d="M74 45 H156" />
        <circle cx="20" cy="56" r="3.6" />
        <circle cx="20" cy="67" r="3.6" />
        <circle cx="35" cy="78" r="3.6" />
        <circle cx="90" cy="84" r="3.6" />
        <circle cx="74" cy="45" r="3.6" />
        <circle cx="156" cy="56" r="3.6" />
        <circle cx="230" cy="66" r="3.6" />
      </svg>
      <div className="brand-wordmark">Skillforge</div>
    </div>
  )
}

export default BrandLogo
