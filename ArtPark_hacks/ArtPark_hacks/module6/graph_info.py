from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

import networkx as nx


MODULE6_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE6_DIR.parent
REPO_ROOT = PROJECT_ROOT.parents[1]

DEFAULT_GAP_JSON = REPO_ROOT / "output" / "module_4" / "gapengine_output.json"
DEFAULT_PROFESSION_JSON = REPO_ROOT / "output" / "module_5" / "profession_mapping_output.json"
DEFAULT_JD_PARSED_JSON = REPO_ROOT / "output" / "jd" / "module_3" / "jd_parsed_output.json"
DEFAULT_DATASET_JSON = PROJECT_ROOT / "module5" / "profession_mapping_engine_dataset_v7.json"
DEFAULT_STATIC_RESOURCES_JSON = PROJECT_ROOT / "module7" / "resources.json"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "output" / "module_6" / "adaptive_path_output.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an adaptive learning roadmap from gap output and profession mapping."
    )
    parser.add_argument("gap_json_pos", nargs="?", type=Path, default=None)
    parser.add_argument("profession_json_pos", nargs="?", type=Path, default=None)
    parser.add_argument("dataset_json_pos", nargs="?", type=Path, default=None)
    parser.add_argument("output_json_pos", nargs="?", type=Path, default=None)
    parser.add_argument("--gap-json", type=Path, default=None)
    parser.add_argument("--profession-json", type=Path, default=None)
    parser.add_argument("--jd-parsed-json", type=Path, default=None)
    parser.add_argument("--dataset-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a top-level JSON object in {path}")
    return data


def _norm_skill(value: object) -> str:
    return str(value or "").strip().lower()


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _dedupe(values: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _display_label(value: object) -> Optional[str]:
    normalized = _norm_skill(value)
    if not normalized:
        return None

    tokens = []
    for token in normalized.split():
        if token in {"ai", "ml"}:
            tokens.append(token.upper())
        else:
            tokens.append(token.capitalize())
    return " ".join(tokens)


class GraphEngine:
    def __init__(
        self,
        dataset: Mapping[str, Any],
        gap_data: Mapping[str, Any],
        profession_data: Mapping[str, Any],
        jd_data: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.data = dataset
        self.gap_data = gap_data
        self.profession_data = profession_data
        self.jd_data = jd_data if isinstance(jd_data, Mapping) else {}
        self.G = nx.DiGraph()
        self.mapping_policy = dataset.get("mapping_policy", {}) if isinstance(dataset.get("mapping_policy"), dict) else {}
        self.alias_map = dataset.get("alias_map", {}) if isinstance(dataset.get("alias_map"), dict) else {}
        self.resources = dataset.get("resources", {}) if isinstance(dataset.get("resources"), dict) else {}
        if DEFAULT_STATIC_RESOURCES_JSON.exists():
            static_resource_payload = _load_json(DEFAULT_STATIC_RESOURCES_JSON)
            self.static_resources = static_resource_payload if isinstance(static_resource_payload, dict) else {}
        else:
            self.static_resources = {}
        self.generic_skill_penalties = (
            self.mapping_policy.get("generic_skill_penalties", {})
            if isinstance(self.mapping_policy.get("generic_skill_penalties"), dict)
            else {}
        )
        self.generic_taxonomy_penalties = (
            self.mapping_policy.get("generic_taxonomy_penalties", {})
            if isinstance(self.mapping_policy.get("generic_taxonomy_penalties"), dict)
            else {}
        )
        self.resource_fallbacks = (
            dataset.get("resource_fallbacks", {})
            if isinstance(dataset.get("resource_fallbacks"), dict)
            else {}
        )
        self.known_skill_threshold = max(
            _as_float(self.mapping_policy.get("adaptive_known_skill_threshold", 0.3), default=0.3),
            0.0,
        )
        self.role_matched_skill_floor_ratio = _clamp(
            _as_float(
                self.mapping_policy.get("adaptive_role_matched_skill_floor_ratio", 1.0),
                default=1.0,
            ),
            lower=0.0,
            upper=1.0,
        )
        self.known_skill_priority_discount = _clamp(
            _as_float(
                self.mapping_policy.get("adaptive_known_skill_priority_discount", 0.72),
                default=0.72,
            ),
            lower=0.0,
            upper=1.0,
        )
        self.strong_support_threshold = _clamp(
            _as_float(
                self.mapping_policy.get("adaptive_strong_support_threshold", 0.5),
                default=0.5,
            ),
            lower=0.0,
            upper=1.0,
        )
        self.min_target_gap = max(
            _as_float(self.mapping_policy.get("adaptive_min_target_gap", 0.6), default=0.6),
            0.0,
        )
        self.jd_known_skill_gap_discount = _clamp(
            _as_float(
                self.mapping_policy.get("adaptive_jd_known_skill_gap_discount", 0.35),
                default=0.35,
            ),
            lower=0.0,
            upper=1.0,
        )
        self.jd_strong_skill_gap_discount = _clamp(
            _as_float(
                self.mapping_policy.get("adaptive_jd_strong_skill_gap_discount", 0.15),
                default=0.15,
            ),
            lower=0.0,
            upper=1.0,
        )
        self.minimum_named_resources = max(
            int(_as_float(self.mapping_policy.get("minimum_named_resources_per_skill", 2), default=2)),
            1,
        )
        self.maximum_named_resources = max(
            int(_as_float(self.mapping_policy.get("maximum_named_resources_per_skill", 3), default=3)),
            self.minimum_named_resources,
        )
        self.prerequisite_sequence_boost = max(
            _as_float(
                self.mapping_policy.get("adaptive_prerequisite_sequence_boost_per_target", 0.08),
                default=0.08,
            ),
            0.0,
        )
        self.deferred_target_limit = max(
            int(_as_float(self.mapping_policy.get("adaptive_deferred_target_limit", 5), default=5)),
            0,
        )

    def _canonical_skill(self, value: object) -> str:
        current = _norm_skill(value)
        if not current:
            return ""

        visited: Set[str] = set()
        while current and current not in visited:
            visited.add(current)
            mapped = _norm_skill(self.alias_map.get(current))
            if not mapped or mapped == current:
                break
            current = mapped
        return current

    def build_base_graph(self) -> None:
        skill_graph = self.data.get("skill_graph", {})
        if not isinstance(skill_graph, dict):
            return

        for skill, neighbors in skill_graph.items():
            source = self._canonical_skill(skill)
            if not source:
                continue
            self.G.add_node(source)

            if not isinstance(neighbors, list):
                continue
            for item in neighbors:
                if not isinstance(item, dict):
                    continue
                target = self._canonical_skill(item.get("skill"))
                if not target:
                    continue
                weight = max(_as_float(item.get("weight"), default=1.0), 1.0)
                self.G.add_edge(source, target, weight=weight)

    def compute_difficulty(self, skill: str) -> float:
        roles = self.data.get("roles", {})
        if not isinstance(roles, dict):
            return 1.0

        role_count = 0
        total_weight = 0.0

        for role in roles.values():
            if not isinstance(role, dict):
                continue
            weights = role.get("weights", {})
            if not isinstance(weights, dict):
                continue
            if skill not in weights:
                continue
            role_count += 1
            total_weight += _as_float(weights.get(skill), default=0.0)

        if role_count == 0:
            return 1.0

        avg_weight = total_weight / role_count
        return round((1.0 / role_count) + avg_weight, 4)

    def enrich_graph(self) -> None:
        for node in self.G.nodes():
            self.G.nodes[node]["difficulty"] = self.compute_difficulty(node)
            self.G.nodes[node]["resources"] = self._get_resources(node)

    def _resource_title(self, payload: object) -> str:
        if isinstance(payload, dict):
            return str(payload.get("title") or "").strip()
        return ""

    def _merge_resources(self, *resource_groups: object) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for group in resource_groups:
            if not isinstance(group, list):
                continue
            for item in group:
                if not isinstance(item, dict):
                    continue
                title = self._resource_title(item)
                if not title:
                    continue
                key = _norm_skill(title)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(item)
        return merged

    def _curate_resources(self, resources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(resources) <= self.maximum_named_resources:
            return resources

        level_order = ("beginner", "intermediate", "advanced")
        selected: List[Dict[str, Any]] = []
        selected_titles: Set[str] = set()

        for level in level_order:
            for item in resources:
                title = self._resource_title(item)
                if not title:
                    continue
                key = _norm_skill(title)
                if key in selected_titles:
                    continue
                if _norm_skill(item.get("level")) != level:
                    continue
                selected.append(item)
                selected_titles.add(key)
                break
            if len(selected) >= self.maximum_named_resources:
                return selected[: self.maximum_named_resources]

        for item in resources:
            title = self._resource_title(item)
            if not title:
                continue
            key = _norm_skill(title)
            if key in selected_titles:
                continue
            selected.append(item)
            selected_titles.add(key)
            if len(selected) >= self.maximum_named_resources:
                break

        return selected[: self.maximum_named_resources]

    def _fallback_bucket_for_skill(self, skill: str) -> str:
        normalized = _norm_skill(skill)

        bucket_rules = {
            "cloud": {"aws", "azure", "gcp", "cloud", "terraform"},
            "devops": {"docker", "kubernetes", "ci/cd", "devops", "airflow", "monitoring", "mlops"},
            "data": {
                "sql",
                "excel",
                "pandas",
                "numpy",
                "statistics",
                "analytics",
                "analysis",
                "tableau",
                "power bi",
                "dashboard",
                "reporting",
            },
            "backend": {"api", "backend", "django", "flask", "fastapi", "database"},
            "frontend": {"html", "css", "javascript", "react", "frontend", "ui"},
            "security": {"security", "owasp", "siem", "threat"},
            "management": {"agile", "stakeholder", "communication", "documentation", "planning", "roadmap"},
        }

        for bucket, keywords in bucket_rules.items():
            if any(keyword in normalized for keyword in keywords):
                return bucket

        return "technical"

    def _fallback_resources(self, skill: str) -> List[Dict[str, Any]]:
        bucket = self._fallback_bucket_for_skill(skill)
        resources = self.resource_fallbacks.get(bucket, [])
        if not isinstance(resources, list) or not resources:
            resources = self.resource_fallbacks.get("technical", [])
        return resources if isinstance(resources, list) else []

    def _get_resources(self, skill: str) -> List[Dict[str, Any]]:
        skill = self._canonical_skill(skill)
        source_bundles = [
            self.static_resources.get(skill),
            self.resources.get(skill),
        ]

        selected: List[Dict[str, Any]] = []
        for bundle in source_bundles:
            curated = self._curate_resources(self._merge_resources(bundle))
            if curated:
                selected = curated
                if len(selected) >= self.minimum_named_resources:
                    return selected
                break

        for bundle in source_bundles:
            curated = self._curate_resources(self._merge_resources(bundle))
            if not curated:
                continue
            selected = self._curate_resources(self._merge_resources(selected, curated))
            if len(selected) >= self.minimum_named_resources:
                return selected

        return self._curate_resources(self._merge_resources(selected, self._fallback_resources(skill)))

    def _skill_priority_penalty(self, skill: str, payload: Optional[Mapping[str, Any]] = None) -> float:
        payload = payload if isinstance(payload, Mapping) else {}
        taxonomy_key = _norm_skill(payload.get("taxonomy_category"))
        category_key = _norm_skill(payload.get("category"))

        skill_penalty = _as_float(self.generic_skill_penalties.get(skill, 1.0), default=1.0)
        taxonomy_penalty = _as_float(self.generic_taxonomy_penalties.get(taxonomy_key, 1.0), default=1.0)
        category_penalty = 0.78 if category_key == "soft_skill" else 1.0

        return max(0.05, skill_penalty * taxonomy_penalty * category_penalty)

    def _is_role_like_skill(self, skill: str) -> bool:
        words = skill.split()
        if len(words) < 2:
            return False

        role_suffixes = {
            "engineer",
            "developer",
            "analyst",
            "scientist",
            "architect",
            "manager",
            "administrator",
            "consultant",
            "specialist",
        }
        if words[-1] not in role_suffixes:
            return False

        if skill in self.G or skill in self.resources:
            return False

        return True

    def _graph_degree(self, skill: str) -> int:
        if skill not in self.G:
            return 0
        return int(self.G.in_degree(skill) + self.G.out_degree(skill))

    def _support_skills_from_role(
        self,
        role_payload: Optional[Mapping[str, Any]],
        candidate_scores: Mapping[str, float],
        include_strong_matches: bool = False,
    ) -> List[str]:
        if not isinstance(role_payload, Mapping):
            return []

        ordered: List[str] = []
        for key in ("core_skills_found",):
            values = role_payload.get(key, [])
            if not isinstance(values, list):
                continue
            for value in values:
                skill = self._canonical_skill(value)
                if not skill:
                    continue
                if _as_float(candidate_scores.get(skill), default=0.0) <= 0.0:
                    continue
                ordered.append(skill)

        if include_strong_matches:
            values = role_payload.get("matched_skills", [])
            if isinstance(values, list):
                for value in values:
                    skill = self._canonical_skill(value)
                    if not skill:
                        continue
                    if _as_float(candidate_scores.get(skill), default=0.0) < self.strong_support_threshold:
                        continue
                    ordered.append(skill)

        return _dedupe(ordered)

    def _target_is_connected_to_support(self, skill: str, support_skills: Iterable[str]) -> bool:
        if skill not in self.G:
            return False

        for support in support_skills:
            if support == skill:
                return True
            if support not in self.G:
                continue
            if nx.has_path(self.G, support, skill):
                return True
        return False

    def _filter_graphable_targets(
        self,
        direct_targets: Mapping[str, Dict[str, Any]],
        support_skills: Iterable[str],
        min_keep: int = 4,
    ) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
        support_skill_list = [skill for skill in _dedupe(support_skills) if skill]
        connected_targets: Dict[str, Dict[str, Any]] = {}
        suppressed_targets: List[Dict[str, Any]] = []

        ranked_targets = sorted(
            direct_targets.values(),
            key=lambda item: (-(item["gap"] * item["importance"]), -item["gap"], item["skill"]),
        )

        for item in ranked_targets:
            skill = item["skill"]
            graph_degree = self._graph_degree(skill)
            is_connected = graph_degree > 0 and (
                not support_skill_list or self._target_is_connected_to_support(skill, support_skill_list)
            )
            if is_connected:
                connected_targets[skill] = item
            else:
                suppressed_payload = dict(item)
                suppressed_payload["suppressed_reason"] = "not connected to the candidate support graph"
                suppressed_targets.append(suppressed_payload)

        if len(connected_targets) >= min_keep or not suppressed_targets:
            return connected_targets, suppressed_targets

        for item in suppressed_targets:
            connected_targets[item["skill"]] = {
                key: value for key, value in item.items() if key != "suppressed_reason"
            }
            if len(connected_targets) >= min_keep:
                break

        remaining_suppressed = [item for item in suppressed_targets if item["skill"] not in connected_targets]
        return connected_targets, remaining_suppressed

    def _candidate_skill_scores(self) -> Dict[str, float]:
        candidate_profile = self.profession_data.get("candidate_profile", {})
        if not isinstance(candidate_profile, dict):
            return {}

        normalized_skills = candidate_profile.get("normalized_skills", [])
        if not isinstance(normalized_skills, list):
            return {}

        scores: Dict[str, float] = {}
        for item in normalized_skills:
            if not isinstance(item, dict):
                continue
            skill = self._canonical_skill(item.get("skill"))
            if not skill:
                continue
            signal = _as_float(
                item.get("roadmap_signal"),
                default=_as_float(
                    item.get("signal"),
                    default=_as_float(item.get("confidence"), default=0.0),
                ),
            )
            scores[skill] = max(scores.get(skill, 0.0), signal)
        return scores

    def _infer_target_jd_role(self) -> Optional[str]:
        role_suffixes = (
            "engineer",
            "developer",
            "analyst",
            "scientist",
            "architect",
            "manager",
            "specialist",
            "consultant",
            "administrator",
        )
        skip_lines = {
            "job description",
            "position specifications",
            "knowledge, skills & abilities",
            "education",
            "work experience",
            "general summary",
            "essential duties and responsibilities",
        }

        for field_name in ("raw_text", "resulting_text"):
            text = self.jd_data.get(field_name)
            if not isinstance(text, str) or not text.strip():
                continue

            lines = [" ".join(line.split()) for line in text.splitlines()]
            for raw_line in lines[:60]:
                line = raw_line.strip()
                normalized = _norm_skill(line)
                if (
                    not normalized
                    or normalized in skip_lines
                    or normalized.startswith("page ")
                    or ":" in normalized
                    or len(normalized.split()) < 2
                    or len(normalized.split()) > 8
                ):
                    continue
                if any(normalized.endswith(f" {suffix}") or normalized == suffix for suffix in role_suffixes):
                    return _display_label(normalized)

            match = re.search(
                r"(?im)^\s*([A-Za-z][A-Za-z/&+\- ]{2,80}?(?:Engineer|Developer|Analyst|Scientist|Architect|Manager|Specialist|Consultant|Administrator))\s*$",
                text,
            )
            if match:
                return _display_label(match.group(1))

        return None

    def _build_direct_gap_targets(self, candidate_scores: Mapping[str, float]) -> Dict[str, Dict[str, Any]]:
        targets: Dict[str, Dict[str, Any]] = {}

        for raw_skill, payload in self.gap_data.items():
            skill = self._canonical_skill(raw_skill)
            if not skill.startswith("__") and isinstance(payload, dict):
                if self._is_role_like_skill(skill):
                    continue

                resume_score = _as_float(payload.get("resume_score"), default=0.0)
                jd_score = _as_float(payload.get("jd_score"), default=0.0)
                gap = max(0.0, jd_score - resume_score)
                if gap <= 0.0 or jd_score <= 0.0:
                    continue

                candidate_signal = _as_float(candidate_scores.get(skill), default=0.0)
                if candidate_signal >= self.strong_support_threshold:
                    gap *= self.jd_strong_skill_gap_discount
                elif candidate_signal >= self.known_skill_threshold:
                    gap *= self.jd_known_skill_gap_discount

                priority_penalty = self._skill_priority_penalty(skill, payload)
                adjusted_importance = jd_score * priority_penalty
                if adjusted_importance <= 0.0 or gap < self.min_target_gap:
                    continue

                targets[skill] = {
                    "skill": skill,
                    "gap": round(gap, 4),
                    "importance": round(adjusted_importance, 4),
                    "resume_score": round(resume_score, 4),
                    "jd_score": round(jd_score, 4),
                    "raw_importance": round(jd_score, 4),
                    "priority_penalty": round(priority_penalty, 4),
                    "level": payload.get("level"),
                    "action": payload.get("action"),
                    "category": payload.get("category"),
                    "taxonomy_category": payload.get("taxonomy_category"),
                    "candidate_signal": round(candidate_signal, 4),
                    "gap_reason": (
                        "JD still expects deeper strength than the current evidence"
                        if candidate_signal >= self.known_skill_threshold
                        else "direct JD gap is high"
                    ),
                }

        return targets

    def _build_profession_targets(
        self,
        role_payload: Mapping[str, Any],
        candidate_scores: Mapping[str, float],
    ) -> Dict[str, Dict[str, Any]]:
        role_name = self._canonical_skill(role_payload.get("role"))
        roles = self.data.get("roles", {})
        dataset_role = roles.get(role_name.title()) if isinstance(roles, dict) else None
        if not isinstance(dataset_role, dict):
            dataset_role = roles.get(role_payload.get("role")) if isinstance(roles, dict) else None
        if not isinstance(dataset_role, dict):
            return {}

        weights = dataset_role.get("weights", {})
        if not isinstance(weights, dict):
            return {}

        missing_skills = {
            self._canonical_skill(skill)
            for skill in role_payload.get("missing_skills", [])
            if self._canonical_skill(skill)
        }
        missing_core = {
            self._canonical_skill(skill)
            for skill in role_payload.get("missing_core_skills", [])
            if self._canonical_skill(skill)
        }
        matched_skills = {
            self._canonical_skill(skill)
            for skill in role_payload.get("matched_skills", [])
            if self._canonical_skill(skill)
        }
        allowed_targets = missing_core | missing_skills

        targets: Dict[str, Dict[str, Any]] = {}
        for raw_skill, raw_weight in weights.items():
            skill = self._canonical_skill(raw_skill)
            if not skill:
                continue
            if self._is_role_like_skill(skill):
                continue
            if allowed_targets and skill not in allowed_targets:
                continue

            base_importance = max(_as_float(raw_weight, default=0.0) * 10.0, 0.0)
            if base_importance <= 0.0:
                continue

            candidate_strength = _as_float(candidate_scores.get(skill), default=0.0) * 10.0
            if skill in matched_skills:
                candidate_strength = max(
                    candidate_strength,
                    base_importance * self.role_matched_skill_floor_ratio,
                )

            importance = base_importance
            priority_penalty = self._skill_priority_penalty(skill)

            if skill in missing_core:
                importance *= 1.15
            elif skill in missing_skills:
                importance *= 1.05

            importance *= priority_penalty
            gap = max(0.0, base_importance - candidate_strength)
            if gap < self.min_target_gap or importance <= 0.0:
                continue

            targets[skill] = {
                "skill": skill,
                "gap": round(gap, 4),
                "importance": round(importance, 4),
                "resume_score": round(candidate_strength, 4),
                "jd_score": round(base_importance, 4),
                "raw_importance": round(base_importance, 4),
                "priority_penalty": round(priority_penalty, 4),
                "level": role_payload.get("level"),
                "action": "profession priority",
                "category": "profession_mapping",
                "gap_reason": "direct role gap is high",
            }

        return targets

    def _select_focus_skills(self, direct_targets: Mapping[str, Dict[str, Any]], limit: int = 12) -> List[str]:
        ranked = sorted(
            direct_targets.values(),
            key=lambda item: (
                -(item["gap"] * item["importance"]),
                -item["gap"],
                -item["importance"],
                item["skill"],
            ),
        )
        return [item["skill"] for item in ranked[:limit]]

    def _build_relevant_subgraph(self, focus_skills: Iterable[str], support_skills: Iterable[str] = ()) -> nx.DiGraph:
        relevant: Set[str] = set()
        focus_skill_list = [skill for skill in _dedupe(focus_skills) if skill]
        support_skill_list = [skill for skill in _dedupe(support_skills) if skill]

        for skill in focus_skill_list:
            if skill not in self.G:
                self.G.add_node(skill)
                self.G.nodes[skill]["difficulty"] = self.compute_difficulty(skill)
                self.G.nodes[skill]["resources"] = self._get_resources(skill)
            relevant.add(skill)

        if support_skill_list:
            path_backed_focus: Set[str] = set()
            for support in support_skill_list:
                if support not in self.G:
                    continue
                relevant.add(support)
                for focus_skill in focus_skill_list:
                    if focus_skill not in self.G:
                        continue
                    if not nx.has_path(self.G, support, focus_skill):
                        continue
                    relevant.update(nx.shortest_path(self.G, support, focus_skill))
                    path_backed_focus.add(focus_skill)

            for focus_skill in focus_skill_list:
                if focus_skill in path_backed_focus or focus_skill not in self.G:
                    continue
                relevant.update(nx.ancestors(self.G, focus_skill))
        else:
            for skill in focus_skill_list:
                if skill in self.G:
                    relevant.update(nx.ancestors(self.G, skill))

        subgraph = self.G.subgraph(relevant).copy()

        for skill in focus_skill_list:
            if skill not in subgraph:
                subgraph.add_node(skill)
                subgraph.nodes[skill]["difficulty"] = self.compute_difficulty(skill)
                subgraph.nodes[skill]["resources"] = self._get_resources(skill)

        return subgraph

    def _dependency_weight(self, subgraph: nx.DiGraph, skill: str, focus_skills: Set[str]) -> float:
        incoming_weight = max(
            (subgraph[u][skill].get("weight", 1.0) for u in subgraph.predecessors(skill)),
            default=1.0,
        )
        outgoing_weight = max(
            (subgraph[skill][v].get("weight", 1.0) for v in subgraph.successors(skill)),
            default=1.0,
        )

        unlocks = [desc for desc in nx.descendants(subgraph, skill) if desc in focus_skills]
        unlock_multiplier = 1.0 + (0.15 * len(unlocks))
        return round(max(incoming_weight, outgoing_weight, unlock_multiplier), 4)

    def _propagated_signal(
        self,
        subgraph: nx.DiGraph,
        skill: str,
        direct_targets: Mapping[str, Dict[str, Any]],
    ) -> Tuple[float, float, List[str]]:
        direct_gap = _as_float(direct_targets.get(skill, {}).get("gap"), default=0.0)
        direct_importance = _as_float(direct_targets.get(skill, {}).get("importance"), default=0.0)

        inherited_gap = 0.0
        inherited_importance = 0.0
        blocking_targets: List[Tuple[int, str]] = []

        for target_skill, target_payload in direct_targets.items():
            if target_skill == skill or target_skill not in subgraph or skill not in subgraph:
                continue
            if not nx.has_path(subgraph, skill, target_skill):
                continue

            distance = nx.shortest_path_length(subgraph, skill, target_skill)
            attenuation = 0.85 ** max(0, distance - 1)
            inherited_gap = max(inherited_gap, _as_float(target_payload.get("gap"), default=0.0) * attenuation)
            inherited_importance = max(
                inherited_importance,
                _as_float(target_payload.get("importance"), default=0.0) * attenuation,
            )
            blocking_targets.append((distance, target_skill))

        blocking_targets.sort(key=lambda item: (item[0], item[1]))
        effective_gap = max(direct_gap, inherited_gap)
        effective_importance = max(direct_importance, inherited_importance)

        return round(effective_gap, 4), round(effective_importance, 4), [skill for _, skill in blocking_targets[:3]]

    def _priority_item(
        self,
        subgraph: nx.DiGraph,
        skill: str,
        direct_targets: Mapping[str, Dict[str, Any]],
        candidate_scores: Mapping[str, float],
        focus_skills: Set[str],
    ) -> Optional[Dict[str, Any]]:
        effective_gap, effective_importance, blocking_targets = self._propagated_signal(
            subgraph,
            skill,
            direct_targets,
        )
        if effective_gap <= 0.0 or effective_importance <= 0.0:
            return None

        direct_gap = _as_float(direct_targets.get(skill, {}).get("gap"), default=0.0)
        candidate_signal = _as_float(candidate_scores.get(skill), default=0.0)
        if direct_gap <= 0.0 and candidate_signal >= self.known_skill_threshold:
            return None

        difficulty = _as_float(subgraph.nodes[skill].get("difficulty"), default=1.0)
        dependency_weight = self._dependency_weight(subgraph, skill, focus_skills)
        priority = (effective_gap * effective_importance * dependency_weight) / (difficulty + 0.1)
        if blocking_targets:
            priority *= 1.0 + (self.prerequisite_sequence_boost * len(blocking_targets))
        if direct_gap <= 0.0:
            priority *= 0.9
        if candidate_signal >= self.known_skill_threshold:
            priority *= self.known_skill_priority_discount

        unlocks = [node for node in subgraph.successors(skill) if node in focus_skills]
        reasons: List[str] = []
        if direct_gap > 0.0:
            reasons.append(str(direct_targets.get(skill, {}).get("gap_reason") or "direct gap is high"))
        if blocking_targets:
            reasons.append(f"prerequisite for {', '.join(blocking_targets[:2])}")
            reasons.append("sequenced earlier because it unlocks downstream skills")
        if dependency_weight > 1.0:
            reasons.append("foundational dependency weight boosts its priority")
        if effective_importance >= 4.0:
            reasons.append("JD importance is strong")
        if candidate_signal >= self.known_skill_threshold:
            reasons.append("candidate already shows baseline evidence")

        return {
            "skill": skill,
            "priority": round(priority, 4),
            "effective_gap": effective_gap,
            "direct_gap": round(direct_gap, 4),
            "jd_importance": effective_importance,
            "difficulty": round(difficulty, 4),
            "dependency_weight": dependency_weight,
            "candidate_signal": round(candidate_signal, 4),
            "blocking_targets": blocking_targets,
            "unlocks": unlocks,
            "resources": subgraph.nodes[skill].get("resources", []),
            "reason": "; ".join(reasons) if reasons else "recommended from gap and dependency signals",
        }

    def _topological_depths(self, subgraph: nx.DiGraph) -> Dict[str, int]:
        if not nx.is_directed_acyclic_graph(subgraph):
            return {node: 0 for node in subgraph.nodes()}

        depths: Dict[str, int] = {}
        for node in nx.topological_sort(subgraph):
            predecessors = list(subgraph.predecessors(node))
            if not predecessors:
                depths[node] = 0
                continue
            depths[node] = max(depths.get(pred, 0) + 1 for pred in predecessors)
        return depths

    def _roadmap_depths(
        self,
        subgraph: nx.DiGraph,
        roadmap_skills: Iterable[str],
    ) -> Dict[str, int]:
        roadmap_nodes = [skill for skill in _dedupe(roadmap_skills) if skill in subgraph]
        if not roadmap_nodes:
            return {}
        return self._topological_depths(subgraph.subgraph(roadmap_nodes).copy())

    def _phase_groups(
        self,
        items: List[Dict[str, Any]],
        depths: Mapping[str, int],
    ) -> List[Dict[str, Any]]:
        grouped: Dict[int, List[Dict[str, Any]]] = {}
        for item in items:
            depth = depths.get(item["skill"], 0)
            grouped.setdefault(depth, []).append(item)

        phases: List[Dict[str, Any]] = []
        for index, depth in enumerate(sorted(grouped.keys()), start=1):
            skills = sorted(grouped[depth], key=lambda item: (-item["priority"], item["skill"]))
            for skill in skills:
                skill["phase"] = f"Phase {index}"
            phases.append({"phase": f"Phase {index}", "skills": skills})
        return phases

    def _next_steps(
        self,
        subgraph: nx.DiGraph,
        candidate_scores: Mapping[str, float],
        roadmap_lookup: Mapping[str, Dict[str, Any]],
    ) -> List[str]:
        next_candidates: List[str] = []
        current_nodes = [
            skill for skill, score in candidate_scores.items()
            if score >= self.known_skill_threshold and skill in subgraph
        ]

        for node in current_nodes:
            for successor in subgraph.successors(node):
                if successor in roadmap_lookup:
                    next_candidates.append(successor)

        if not next_candidates:
            return [item["skill"] for item in sorted(roadmap_lookup.values(), key=lambda item: -item["priority"])[:5]]

        deduped = _dedupe(next_candidates)
        deduped.sort(key=lambda skill: (-roadmap_lookup[skill]["priority"], skill))
        return deduped[:5]

    def _graph_payload(
        self,
        subgraph: nx.DiGraph,
        roadmap_lookup: Mapping[str, Dict[str, Any]],
        next_steps: Set[str],
        candidate_scores: Mapping[str, float],
        direct_targets: Mapping[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []

        for node in subgraph.nodes():
            candidate_signal = _as_float(candidate_scores.get(node), default=0.0)
            direct_gap = _as_float(direct_targets.get(node, {}).get("gap"), default=0.0)

            if node in next_steps:
                status = "next_step"
                color = "yellow"
                size = 30
            elif direct_gap > 0.0:
                status = "missing"
                color = "red"
                size = 34
            elif candidate_signal >= self.known_skill_threshold:
                status = "known"
                color = "green"
                size = 24
            elif node in roadmap_lookup:
                status = "prerequisite"
                color = "orange"
                size = 22
            else:
                status = "context"
                color = "#97C2FC"
                size = 18

            roadmap_item = roadmap_lookup.get(node, {})
            resources = subgraph.nodes[node].get("resources", [])
            resource_titles = [str(item.get("title")) for item in resources if isinstance(item, dict) and item.get("title")]
            title = resource_titles[:3] or ["Curated learning resources are attached in module 7"]

            nodes.append(
                {
                    "id": node,
                    "data": {
                        "label": node,
                        "status": status,
                        "color": color,
                        "size": size,
                        "difficulty": _as_float(subgraph.nodes[node].get("difficulty"), default=1.0),
                        "priority": _as_float(roadmap_item.get("priority"), default=0.0),
                        "title": "<br>".join(title),
                    },
                }
            )

        for source, target in subgraph.edges():
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "weight": _as_float(subgraph[source][target].get("weight"), default=1.0),
                }
            )

        return {
            "nodes": nodes,
            "edges": edges,
            "meta": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "roadmap_node_count": len(roadmap_lookup),
            },
        }

    def _build_track(
        self,
        track_type: str,
        direct_targets: Mapping[str, Dict[str, Any]],
        candidate_scores: Mapping[str, float],
        focus_limit: int,
        title: str,
        target_role: Optional[str] = None,
        candidate_best_fit_role: Optional[str] = None,
        target_jd_role: Optional[str] = None,
        top_role_payload: Optional[Mapping[str, Any]] = None,
        support_skills: Iterable[str] = (),
        suppressed_targets: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        focus_skills = set(self._select_focus_skills(direct_targets, limit=focus_limit))
        support_skill_list = _dedupe(
            [self._canonical_skill(skill) for skill in support_skills if self._canonical_skill(skill)]
        )
        subgraph = self._build_relevant_subgraph(focus_skills, support_skill_list)

        roadmap_items: List[Dict[str, Any]] = []
        for skill in subgraph.nodes():
            item = self._priority_item(subgraph, skill, direct_targets, candidate_scores, focus_skills)
            if item:
                roadmap_items.append(item)

        roadmap_depths = self._roadmap_depths(subgraph, [item["skill"] for item in roadmap_items])
        roadmap_items.sort(
            key=lambda item: (
                roadmap_depths.get(item["skill"], 0),
                -item["priority"],
                item["skill"],
            )
        )
        roadmap_lookup = {item["skill"]: item for item in roadmap_items}
        phases = self._phase_groups(roadmap_items, roadmap_depths)
        next_steps = self._next_steps(subgraph, candidate_scores, roadmap_lookup)
        visible_suppressed_targets: List[Dict[str, Any]] = []
        deferred_targets: List[Dict[str, Any]] = []
        if suppressed_targets:
            visible_suppressed_targets = sorted(
                suppressed_targets,
                key=lambda item: (-(item.get("gap", 0.0) * item.get("importance", 0.0)), item.get("skill", "")),
            )[: self.deferred_target_limit]
            deferred_targets = list(visible_suppressed_targets)

        if track_type == "jd_requirement":
            view_label = "Target JD Gap View"
            view_purpose = (
                "This roadmap shows the most connected prerequisites for closing the target JD gaps."
            )
        elif track_type == "profession_mapping":
            view_label = "Candidate Best-Fit Role View"
            view_purpose = (
                "This roadmap shows the most relevant next skills for the candidate's best-fit profession."
            )
        else:
            view_label = "Roadmap View"
            view_purpose = "This roadmap shows dependency-aware skill progression."

        return {
            "track_type": track_type,
            "title": title,
            "view_label": view_label,
            "view_purpose": view_purpose,
            "role": target_role,
            "target_role": target_role,
            "candidate_best_fit_role": candidate_best_fit_role,
            "target_jd_role": target_jd_role,
            "top_role": dict(top_role_payload) if isinstance(top_role_payload, Mapping) else None,
            "__meta__": {
                "gap_skill_count": len(direct_targets),
                "focus_skill_count": len(focus_skills),
                "support_skill_count": len(support_skill_list),
                "relevant_graph_nodes": subgraph.number_of_nodes(),
                "relevant_graph_edges": subgraph.number_of_edges(),
                "known_skill_threshold": self.known_skill_threshold,
                "min_target_gap": self.min_target_gap,
                "maximum_named_resources": self.maximum_named_resources,
                "suppressed_direct_target_count": len(suppressed_targets or []),
            },
            "roadmap": [item["skill"] for item in roadmap_items],
            "roadmap_details": roadmap_items,
            "roadmap_phases": phases,
            "next_steps": next_steps,
            "direct_targets": sorted(
                direct_targets.values(),
                key=lambda item: (-(item["gap"] * item["importance"]), item["skill"]),
            ),
            "suppressed_direct_targets": visible_suppressed_targets,
            "deferred_targets": deferred_targets,
            "graph": self._graph_payload(
                subgraph,
                roadmap_lookup,
                set(next_steps),
                candidate_scores,
                direct_targets,
            ),
        }

    def run(self) -> Dict[str, Any]:
        self.build_base_graph()
        self.enrich_graph()

        candidate_scores = self._candidate_skill_scores()
        top_role_payload = None
        top_roles = self.profession_data.get("top_roles", [])
        if isinstance(top_roles, list) and top_roles:
            top_role_payload = top_roles[0]
        candidate_best_fit_role = top_role_payload.get("role") if isinstance(top_role_payload, dict) else None
        target_jd_role = self._infer_target_jd_role()
        jd_support_skills = self._support_skills_from_role(
            top_role_payload,
            candidate_scores,
            include_strong_matches=True,
        )
        raw_jd_targets = self._build_direct_gap_targets(candidate_scores)
        filtered_jd_targets, suppressed_jd_targets = self._filter_graphable_targets(
            raw_jd_targets,
            jd_support_skills,
            min_keep=3,
        )

        jd_track = self._build_track(
            track_type="jd_requirement",
            direct_targets=filtered_jd_targets or raw_jd_targets,
            candidate_scores=candidate_scores,
            focus_limit=8,
            title=f"JD Gap Roadmap: {target_jd_role}" if target_jd_role else "JD Gap Roadmap",
            target_role=target_jd_role,
            candidate_best_fit_role=candidate_best_fit_role,
            target_jd_role=target_jd_role,
            top_role_payload=top_role_payload,
            support_skills=jd_support_skills,
            suppressed_targets=suppressed_jd_targets,
        )

        profession_tracks: List[Dict[str, Any]] = []
        if isinstance(top_roles, list):
            for role_payload in top_roles:
                if not isinstance(role_payload, dict):
                    continue
                profession_targets = self._build_profession_targets(role_payload, candidate_scores)
                if not profession_targets:
                    continue
                profession_tracks.append(
                    self._build_track(
                        track_type="profession_mapping",
                        direct_targets=profession_targets,
                        candidate_scores=candidate_scores,
                        focus_limit=10,
                        title=f"Candidate Best-Fit Profession Roadmap: {role_payload.get('role', 'Unknown Role')}",
                        target_role=role_payload.get("role"),
                        candidate_best_fit_role=candidate_best_fit_role,
                        target_jd_role=target_jd_role,
                        top_role_payload=role_payload,
                        support_skills=self._support_skills_from_role(role_payload, candidate_scores),
                    )
                )

        primary_track = profession_tracks[0] if profession_tracks else jd_track

        return {
            "__meta__": {
                "known_skill_threshold": self.known_skill_threshold,
                "min_target_gap": self.min_target_gap,
                "top_role": top_role_payload.get("role") if isinstance(top_role_payload, dict) else None,
                "candidate_best_fit_role": candidate_best_fit_role,
                "target_jd_role": target_jd_role,
                "profession_roadmap_count": len(profession_tracks),
                "primary_track_type": primary_track.get("track_type"),
                "recommended_track_type": primary_track.get("track_type"),
            },
            "candidate_best_fit_role": candidate_best_fit_role,
            "target_jd_role": target_jd_role,
            "recommended_track_type": primary_track.get("track_type"),
            "top_role": top_role_payload,
            "jd_requirement_roadmap": jd_track,
            "profession_roadmaps": profession_tracks,
            "roadmap": primary_track["roadmap"],
            "roadmap_details": primary_track["roadmap_details"],
            "roadmap_phases": primary_track["roadmap_phases"],
            "next_steps": primary_track["next_steps"],
            "direct_targets": primary_track["direct_targets"],
            "graph": primary_track["graph"],
        }


def main() -> None:
    args = _parse_args()
    gap_json = args.gap_json or args.gap_json_pos or DEFAULT_GAP_JSON
    profession_json = args.profession_json or args.profession_json_pos or DEFAULT_PROFESSION_JSON
    jd_parsed_json = args.jd_parsed_json or DEFAULT_JD_PARSED_JSON
    dataset_json = args.dataset_json or args.dataset_json_pos or DEFAULT_DATASET_JSON
    output_json = args.output_json or args.output_json_pos or DEFAULT_OUTPUT_JSON

    gap_data = _load_json(gap_json)
    profession_data = _load_json(profession_json)
    dataset = _load_json(dataset_json)
    jd_data = _load_json(jd_parsed_json) if jd_parsed_json.exists() else {}

    engine = GraphEngine(dataset=dataset, gap_data=gap_data, profession_data=profession_data, jd_data=jd_data)
    output = engine.run()

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    jd_output_json = output_json.parent / "roadmap_jd.json"
    profession_output_json = output_json.parent / "roadmap_profession.json"

    jd_output_json.write_text(
        json.dumps(output.get("jd_requirement_roadmap", {}), indent=2),
        encoding="utf-8",
    )
    profession_track = {}
    profession_tracks = output.get("profession_roadmaps", [])
    if isinstance(profession_tracks, list) and profession_tracks:
        first_track = profession_tracks[0]
        if isinstance(first_track, dict):
            profession_track = first_track
    profession_output_json.write_text(
        json.dumps(profession_track, indent=2),
        encoding="utf-8",
    )

    print(f"Adaptive path JSON written to: {output_json}")
    print(f"JD graph JSON written to: {jd_output_json}")
    print(f"Profession graph JSON written to: {profession_output_json}")
    for phase in output.get("roadmap_phases", [])[:3]:
        skills = ", ".join(skill["skill"] for skill in phase.get("skills", [])[:5])
        print(f"{phase['phase']}: {skills}")


if __name__ == "__main__":
    main()
