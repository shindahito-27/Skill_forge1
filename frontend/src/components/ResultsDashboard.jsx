function scoreToPercent(value) {
  const numeric = Number(value || 0)
  if (numeric <= 1) return Math.max(0, Math.min(100, Math.round(numeric * 100)))
  return Math.max(0, Math.min(100, Math.round(numeric)))
}

function formatNumber(value) {
  return Number(value || 0).toFixed(2).replace(/\.00$/, '')
}

function splitReason(reason) {
  if (!reason) return []
  return String(reason)
    .split(';')
    .map((item) => item.trim())
    .filter(Boolean)
}

function SkillTable({ title, items }) {
  return (
    <div className="panel">
      <h3>{title}</h3>
      <div className="skill-table">
        {items.length === 0 && <span className="chip muted">No data</span>}
        {items.map((item) => (
          <div className="skill-row" key={`${title}-${item.name}`}>
            <div className="skill-main">
              <strong>{item.name}</strong>
              {'gapScore' in item ? (
                <div className="skill-details">
                  <span className="chip subtle">{item.level}</span>
                  <span className="chip subtle">{item.status}</span>
                  <span className="chip subtle">{item.action}</span>
                  <span className="chip subtle">JD {formatNumber(item.jdScore)}</span>
                  <span className="chip subtle">Resume {formatNumber(item.resumeScore)}</span>
                </div>
              ) : (
                <div className="skill-details">
                  <span className="chip subtle">{item.taxonomy}</span>
                  <span className="chip subtle">Confidence {formatNumber(item.confidence)}</span>
                </div>
              )}
            </div>
            <span className="badge">{formatNumber(item.gapScore ?? item.score)}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

const STATUS_COLORS = {
  known: '#22c55e',
  next_step: '#facc15',
  missing: '#ef4444',
  prerequisite: '#f97316',
  context: '#94a3b8',
}

const STATUS_LABELS = {
  known: 'Known skill',
  next_step: 'Next step',
  missing: 'Missing skill',
  prerequisite: 'Prerequisite',
  context: 'Context node',
}

function CircularMatch({ value, label, accent = 'var(--accent-2)' }) {
  const percent = scoreToPercent(value)
  const style = {
    background: `conic-gradient(${accent} 0 ${percent}%, rgba(255,255,255,0.12) ${percent}% 100%)`,
  }

  return (
    <div className="match-ring-wrap">
      <div className="match-ring" style={style}>
        <div className="match-ring-inner">
          <strong>{percent}%</strong>
          <span>{label}</span>
        </div>
      </div>
    </div>
  )
}

function RoleCard({ role, featured = false }) {
  const reasonLines = splitReason(role.reason).slice(0, 3)
  return (
    <article className={`role-card ${featured ? 'featured' : ''}`}>
      <div className="role-card-top">
        <div>
          <span className="eyebrow">{featured ? 'Best fit' : 'Role fit'}</span>
          <h3>{role.role}</h3>
        </div>
        <CircularMatch value={role.score} label="match" accent={featured ? 'var(--accent-3)' : 'var(--accent-2)'} />
      </div>
      <div className="role-chip-row">
        {(role.missingCoreSkills || []).slice(0, 3).map((skill) => (
          <span key={`${role.role}-${skill}`} className="chip subtle">
            Missing: {skill}
          </span>
        ))}
      </div>
      <ul className="reason-list">
        {reasonLines.length === 0 && <li>Role fit derived from overlap and scoring signals.</li>}
        {reasonLines.map((line) => (
          <li key={`${role.role}-${line}`}>{line}</li>
        ))}
      </ul>
    </article>
  )
}

function GraphLegend() {
  return (
    <div className="graph-legend-grid">
      {Object.entries(STATUS_COLORS).map(([status, color]) => (
        <div className="legend-item" key={status}>
          <span className="legend-dot" style={{ background: color }} />
          <span>{STATUS_LABELS[status] || status}</span>
        </div>
      ))}
      <div className="legend-item wide">
        <span className="legend-size legend-size-sm" />
        <span>Small node = easier skill</span>
      </div>
      <div className="legend-item wide">
        <span className="legend-size legend-size-lg" />
        <span>Large node = higher difficulty</span>
      </div>
    </div>
  )
}

function RoadmapGraph({ graph }) {
  const nodes = graph?.nodes || []
  const edges = graph?.edges || []
  if (!nodes.length) {
    return <div className="chip muted">No graph data available.</div>
  }

  const centerX = 50
  const centerY = 50
  const radius = 34
  const nodeMap = new Map(
    nodes.map((node, index) => {
      const angle = (2 * Math.PI * index) / Math.max(nodes.length, 1)
      return [
        node.id,
        {
          ...node,
          x: centerX + radius * Math.cos(angle),
          y: centerY + radius * Math.sin(angle),
        },
      ]
    }),
  )

  return (
    <div className="roadmap-graph-wrap">
      <svg className="roadmap-graph" viewBox="0 0 100 100" role="img" aria-label={graph.title}>
        {edges.map((edge, index) => {
          const source = nodeMap.get(edge.source)
          const target = nodeMap.get(edge.target)
          if (!source || !target) return null
          return (
            <line
              key={`edge-${index}`}
              x1={source.x}
              y1={source.y}
              x2={target.x}
              y2={target.y}
              stroke="rgba(206, 210, 255, 0.42)"
              strokeWidth="0.7"
            />
          )
        })}
        {Array.from(nodeMap.values()).map((node) => (
          <g key={node.id}>
            <circle
              cx={node.x}
              cy={node.y}
              r={Math.max(2.5, Math.min(5.4, (node.size || 18) / 6))}
              fill={STATUS_COLORS[node.status] || '#94a3b8'}
              stroke="#0f1025"
              strokeWidth="0.3"
            />
            <text x={node.x} y={node.y - 4.2} textAnchor="middle" className="graph-node-label">
              {node.label}
            </text>
          </g>
        ))}
      </svg>
      <div className="roadmap-meta-row">
        <span className="chip">Nodes: {graph.meta?.node_count ?? nodes.length}</span>
        <span className="chip">Edges: {graph.meta?.edge_count ?? edges.length}</span>
      </div>
      <GraphLegend />
    </div>
  )
}

function ResultsDashboard({ data }) {
  const { overview, skills, insights } = data
  const topRoles = insights.recommendedRoles || []

  return (
    <section className="results">
      <div className="hero-metrics">
        <div className="metric-card hero-card">
          <div>
            <span className="eyebrow">Best-fit role</span>
            <h2>{overview.bestFitRole}</h2>
            <p className="metric-support">Top recommendation from profession mapping.</p>
          </div>
          <CircularMatch value={overview.bestFitScore} label="match" />
        </div>
        <div className="metric-card">
          <h4>Next Steps</h4>
          <div className="chips">
            {(overview.nextSteps || []).slice(0, 4).map((step) => (
              <span className="chip" key={step}>
                {step}
              </span>
            ))}
          </div>
        </div>
      </div>

      <div className="panel">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">Top 3 results</span>
            <h3>Best-Fit Roles</h3>
          </div>
        </div>
        <div className="role-grid">
          {topRoles.map((role, index) => (
            <RoleCard key={role.role} role={role} featured={index === 0} />
          ))}
        </div>
      </div>

      <div className="results-grid">
        <SkillTable title="Hard Skill Gaps" items={skills.hard} />
        <SkillTable title="Soft Skill Gaps" items={skills.soft} />
        <SkillTable title="Resume Hard Skills" items={skills.resumeHard} />
        <SkillTable title="Resume Soft Skills" items={skills.resumeSoft} />
      </div>

      <div className="panel">
        <h3>Critical Gaps</h3>
        <div className="chips">
          {insights.criticalGaps.map((item) => (
            <span key={`gap-${item.name}`} className="chip warning">
              {item.name} ({item.gapScore})
            </span>
          ))}
        </div>
      </div>

      <div className="panel">
        <h3>Roadmap Recommendations</h3>
        <div className="roadmap-list">
          {insights.roadmap.map((item) => (
            <article className="roadmap-item" key={`roadmap-${item.skill}`}>
              <div className="roadmap-item-top">
                <div>
                  <span className="eyebrow">Roadmap node</span>
                  <strong>{item.skill}</strong>
                </div>
                <div className="roadmap-item-meta">
                  <span className="chip subtle">{item.phase}</span>
                  <span className="chip subtle">Priority score {formatNumber(item.priority)}</span>
                </div>
              </div>
              <ul className="reason-list compact">
                {splitReason(item.reason).map((line) => (
                  <li key={`${item.skill}-${line}`}>{line}</li>
                ))}
              </ul>
              {!!item.resources?.length && (
                <div className="resource-row">
                  {item.resources.slice(0, 3).map((resource) => (
                    <span className="chip" key={`${item.skill}-${resource}`}>
                      {resource}
                    </span>
                  ))}
                </div>
              )}
            </article>
          ))}
        </div>
      </div>

      <div className="panel">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">Graph view</span>
            <h3>JD Roadmap Graph</h3>
          </div>
          <p className="panel-caption">Colors show node state. Larger circles indicate harder skills.</p>
        </div>
        <RoadmapGraph graph={insights.roadmapGraph} />
      </div>
    </section>
  )
}

export default ResultsDashboard
