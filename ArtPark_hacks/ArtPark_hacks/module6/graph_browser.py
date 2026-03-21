from __future__ import annotations

import html
import json
from pathlib import Path

from pyvis.network import Network


MODULE6_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE6_DIR.parent
REPO_ROOT = PROJECT_ROOT.parents[1]

INPUT_JSON = REPO_ROOT / "output" / "module_6" / "adaptive_path_output.json"

NODE_COLOR_PALETTE = {
    "missing": "#c2717e",
    "next_step": "#dfcc7f",
    "known": "#77ba85",
    "prerequisite": "#84d9e8",
    "default": "#94aed4",
}

RAW_COLOR_ALIASES = {
    "red": "missing",
    "yellow": "next_step",
    "green": "known",
    "blue": "prerequisite",
}


def _slugify(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or ""))
    return "_".join(part for part in cleaned.split("_") if part) or "roadmap"


def _node_color(payload: dict) -> str:
    status = str(payload.get("status") or "").strip().lower()
    raw_color = str(payload.get("color") or "").strip().lower()

    if raw_color.startswith("#"):
        return raw_color
    if status in NODE_COLOR_PALETTE:
        return NODE_COLOR_PALETTE[status]
    if raw_color in RAW_COLOR_ALIASES:
        return NODE_COLOR_PALETTE[RAW_COLOR_ALIASES[raw_color]]
    return NODE_COLOR_PALETTE["default"]


def _write_graph(track_payload: dict, output_html: Path) -> None:
    graph = track_payload.get("graph", {})
    nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
    edges = graph.get("edges", []) if isinstance(graph, dict) else []

    net = Network(height="750px", width="100%", directed=True)
    net.force_atlas_2based()

    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        payload = node.get("data", {}) if isinstance(node.get("data"), dict) else {}
        if not node_id:
            continue

        net.add_node(
            node_id,
            label=str(payload.get("label") or node_id),
            color=_node_color(payload),
            size=int(payload.get("size") or 20),
            title=str(payload.get("title") or "No details"),
        )

    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source = edge.get("source")
        target = edge.get("target")
        if not source or not target:
            continue
        weight = edge.get("weight")
        net.add_edge(source, target, value=weight if isinstance(weight, (int, float)) else 1)

    net.write_html(str(output_html))
    html_payload = output_html.read_text(encoding="utf-8")
    html_payload = html_payload.replace("</body>", f"{_overlay_html(track_payload)}</body>")
    output_html.write_text(html_payload, encoding="utf-8")
    print(f"Roadmap HTML written to: {output_html}")


def _track_context(track_payload: dict) -> str:
    track_type = str(track_payload.get("track_type") or "")
    candidate_role = track_payload.get("candidate_best_fit_role")
    jd_role = track_payload.get("target_jd_role")
    target_role = track_payload.get("target_role") or track_payload.get("role")

    if track_type == "jd_requirement":
        return (
            f"This is the target-JD gap view for {jd_role or 'the target role'}. "
            f"The candidate's current best-fit role remains {candidate_role or 'unknown'}."
        )
    if track_type == "profession_mapping":
        return (
            f"This is the candidate-fit profession view for {target_role or 'the mapped role'}. "
            f"The target JD remains {jd_role or 'separate'}."
        )
    return "Roadmap view generated from graph dependencies and gap priorities."


def _phase_rows(track_payload: dict) -> str:
    phase_rows = []
    for phase in track_payload.get("roadmap_phases", [])[:4]:
        if not isinstance(phase, dict):
            continue
        skills = phase.get("skills", [])
        if not isinstance(skills, list):
            continue
        skill_names = ", ".join(
            str(item.get("skill"))
            for item in skills[:5]
            if isinstance(item, dict) and item.get("skill")
        )
        if not skill_names:
            continue
        phase_rows.append(
            "<div class='phase-row'>"
            f"<span class='phase-name'>{html.escape(str(phase.get('phase') or 'Phase'))}</span>"
            f"<span class='phase-skills'>{html.escape(skill_names)}</span>"
            "</div>"
        )
    return "".join(phase_rows) or "<div class='phase-row'><span class='phase-skills'>No phase details available.</span></div>"


def _overlay_html(track_payload: dict) -> str:
    graph = track_payload.get("graph", {})
    nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
    edges = graph.get("edges", []) if isinstance(graph, dict) else []
    next_steps = track_payload.get("next_steps", [])
    next_steps_text = ", ".join(str(skill) for skill in next_steps[:4]) or "No immediate next step"
    candidate_role = str(track_payload.get("candidate_best_fit_role") or "Unknown")
    jd_role = str(track_payload.get("target_jd_role") or "Unknown")
    view_label = str(track_payload.get("view_label") or track_payload.get("track_type") or "Roadmap")
    view_purpose = str(track_payload.get("view_purpose") or "")
    deferred_targets = track_payload.get("deferred_targets", [])
    deferred_text = ", ".join(
        str(item.get("skill"))
        for item in deferred_targets[:4]
        if isinstance(item, dict) and item.get("skill")
    )

    return f"""
<style>
.roadmap-panel {{
  position: fixed;
  top: 18px;
  left: 18px;
  z-index: 9999;
  width: 340px;
  max-height: calc(100vh - 36px);
  overflow: auto;
  padding: 18px 18px 14px;
  border-radius: 18px;
  background: rgba(255, 252, 244, 0.94);
  box-shadow: 0 18px 40px rgba(33, 37, 41, 0.18);
  border: 1px solid rgba(181, 148, 84, 0.22);
  font-family: Georgia, 'Times New Roman', serif;
}}
.roadmap-panel h1 {{
  margin: 0 0 8px;
  font-size: 24px;
  line-height: 1.2;
  color: #1f2a37;
}}
.roadmap-panel p {{
  margin: 0 0 12px;
  font-size: 14px;
  line-height: 1.5;
  color: #425466;
}}
.legend-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin: 12px 0;
}}
.legend-item {{
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  color: #243447;
}}
.legend-dot {{
  width: 12px;
  height: 12px;
  border-radius: 999px;
  flex: 0 0 12px;
}}
.roadmap-meta {{
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin: 12px 0;
  font-size: 12px;
  color: #4b5563;
}}
.roadmap-chip {{
  padding: 5px 9px;
  border-radius: 999px;
  background: #f4ead7;
}}
.role-lines {{
  display: grid;
  gap: 6px;
  margin: 12px 0;
}}
.role-line {{
  font-size: 13px;
  color: #243447;
}}
.role-line strong {{
  color: #8b5e34;
}}
.phase-block {{
  margin-top: 14px;
  padding-top: 12px;
  border-top: 1px solid rgba(148, 163, 184, 0.28);
}}
.phase-row {{
  display: flex;
  flex-direction: column;
  gap: 2px;
  margin-bottom: 10px;
}}
.phase-name {{
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: #8b5e34;
}}
.phase-skills {{
  font-size: 13px;
  line-height: 1.45;
  color: #243447;
}}
</style>
<div class="roadmap-panel">
  <h1>{html.escape(str(track_payload.get("title") or "Roadmap"))}</h1>
  <p><strong>{html.escape(view_label)}</strong></p>
  <p>{html.escape(_track_context(track_payload))}</p>
  <p>{html.escape(view_purpose)}</p>
  <div class="role-lines">
    <div class="role-line"><strong>Candidate Best Fit:</strong> {html.escape(candidate_role)}</div>
    <div class="role-line"><strong>Target JD Role:</strong> {html.escape(jd_role)}</div>
  </div>
  <div class="legend-grid">
    <div class="legend-item"><span class="legend-dot" style="background:{NODE_COLOR_PALETTE["missing"]}"></span>Missing skill</div>
    <div class="legend-item"><span class="legend-dot" style="background:{NODE_COLOR_PALETTE["next_step"]}"></span>Next step</div>
    <div class="legend-item"><span class="legend-dot" style="background:{NODE_COLOR_PALETTE["known"]}"></span>Known skill</div>
    <div class="legend-item"><span class="legend-dot" style="background:{NODE_COLOR_PALETTE["prerequisite"]}"></span>Prerequisite</div>
  </div>
  <div class="roadmap-meta">
    <span class="roadmap-chip">Nodes: {len(nodes)}</span>
    <span class="roadmap-chip">Edges: {len(edges)}</span>
    <span class="roadmap-chip">Next: {html.escape(next_steps_text)}</span>
  </div>
  {"<p><strong>Deferred but important:</strong> " + html.escape(deferred_text) + "</p>" if deferred_text else ""}
  <div class="phase-block">
    {_phase_rows(track_payload)}
  </div>
</div>
"""


def main() -> None:
    if not INPUT_JSON.exists():
        raise FileNotFoundError(f"Adaptive path output not found: {INPUT_JSON}")

    data = json.loads(INPUT_JSON.read_text(encoding="utf-8"))

    primary_track = {
        "graph": data.get("graph", {}),
        "title": data.get("__meta__", {}).get("primary_track_type", "primary_roadmap"),
    }
    if isinstance(primary_track.get("graph"), dict) and primary_track.get("graph"):
        _write_graph(primary_track, MODULE6_DIR / "roadmap.html")

    jd_track = data.get("jd_requirement_roadmap", {})
    if isinstance(jd_track, dict) and jd_track:
        _write_graph(jd_track, MODULE6_DIR / "roadmap_jd.html")

    profession_tracks = data.get("profession_roadmaps", [])
    if isinstance(profession_tracks, list):
        if profession_tracks and isinstance(profession_tracks[0], dict):
            _write_graph(profession_tracks[0], MODULE6_DIR / "roadmap_profession.html")
        for index, track in enumerate(profession_tracks, start=1):
            if not isinstance(track, dict):
                continue
            top_role = track.get("top_role", {})
            role_name = top_role.get("role") if isinstance(top_role, dict) else None
            suffix = _slugify(role_name or f"profession_{index}")
            _write_graph(track, MODULE6_DIR / f"roadmap_profession_{index}_{suffix}.html")


if __name__ == "__main__":
    main()
