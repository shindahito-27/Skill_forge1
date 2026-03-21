from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


MODULE7_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE7_DIR.parent
REPO_ROOT = PROJECT_ROOT.parents[1]

DEFAULT_ADAPTIVE_JSON = REPO_ROOT / "output" / "module_6" / "adaptive_path_output.json"
DEFAULT_DATASET_JSON = PROJECT_ROOT / "module5" / "profession_mapping_engine_dataset_v7.json"
DEFAULT_RESOURCES_JSON = MODULE7_DIR / "resources.json"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "output" / "module_7" / "learning_resources_output.json"

GENERIC_RESOURCE_TITLES = {
    "official documentation",
    "official tool documentation",
    "freecodecamp / youtube beginner tutorials",
    "hands-on labs",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach static learning resources to JD and profession roadmaps."
    )
    parser.add_argument("adaptive_json_pos", nargs="?", type=Path, default=None)
    parser.add_argument("dataset_json_pos", nargs="?", type=Path, default=None)
    parser.add_argument("resources_json_pos", nargs="?", type=Path, default=None)
    parser.add_argument("output_json_pos", nargs="?", type=Path, default=None)
    parser.add_argument("--adaptive-json", type=Path, default=None)
    parser.add_argument("--dataset-json", type=Path, default=None)
    parser.add_argument("--resources-json", type=Path, default=None)
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


class LearningResourceLayer:
    def __init__(
        self,
        adaptive_data: Mapping[str, Any],
        dataset_data: Mapping[str, Any],
        resource_data: Mapping[str, Any],
    ) -> None:
        self.adaptive_data = adaptive_data
        self.dataset_data = dataset_data
        self.resource_data = resource_data
        self.mapping_policy = (
            dataset_data.get("mapping_policy", {})
            if isinstance(dataset_data.get("mapping_policy"), dict)
            else {}
        )
        self.alias_map = dataset_data.get("alias_map", {}) if isinstance(dataset_data.get("alias_map"), dict) else {}
        self.dataset_resources = (
            dataset_data.get("resources", {})
            if isinstance(dataset_data.get("resources"), dict)
            else {}
        )
        self.resource_fallbacks = (
            dataset_data.get("resource_fallbacks", {})
            if isinstance(dataset_data.get("resource_fallbacks"), dict)
            else {}
        )
        self.minimum_named_resources = max(
            int(_as_float(self.mapping_policy.get("minimum_named_resources_per_skill", 2), default=2)),
            1,
        )
        self.maximum_named_resources = max(
            int(_as_float(self.mapping_policy.get("maximum_named_resources_per_skill", 3), default=3)),
            self.minimum_named_resources,
        )

    def _canonical_skill(self, value: object) -> str:
        current = _norm_skill(value)
        if not current:
            return ""

        visited = set()
        while current and current not in visited:
            visited.add(current)
            mapped = _norm_skill(self.alias_map.get(current))
            if not mapped or mapped == current:
                break
            current = mapped
        return current

    def _fallback_bucket_for_skill(self, skill: str) -> str:
        normalized = _norm_skill(skill)

        bucket_rules = {
            "cloud": {"aws", "azure", "gcp", "cloud", "terraform"},
            "devops": {"docker", "kubernetes", "ci/cd", "devops", "airflow", "monitoring"},
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

    def _resource_title(self, payload: object) -> str:
        if isinstance(payload, dict):
            return str(payload.get("title") or "").strip()
        return ""

    def _is_generic_placeholder_group(self, resources: List[Dict[str, Any]]) -> bool:
        titled = [item for item in resources if self._resource_title(item)]
        if not titled:
            return False
        return all(_norm_skill(self._resource_title(item)) in GENERIC_RESOURCE_TITLES for item in titled)

    def _merge_resources(self, *resource_groups: object) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        seen = set()

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

    def _resources_for_skill(self, skill: str, existing_resources: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        normalized = self._canonical_skill(skill)
        ordered_sources = [
            ("module7_static_json", self.resource_data.get(normalized)),
            ("module5_dataset_resources", self.dataset_resources.get(normalized)),
            ("module6_embedded_resources", existing_resources),
        ]
        resource_sources: List[str] = []
        merged_resources: List[Dict[str, Any]] = []

        for source_name, resource_group in ordered_sources:
            curated_group = self._curate_resources(self._merge_resources(resource_group))
            if source_name == "module6_embedded_resources" and self._is_generic_placeholder_group(curated_group):
                curated_group = []
            if not curated_group:
                continue
            if not merged_resources:
                merged_resources = curated_group
                resource_sources.append(source_name)
                if len(merged_resources) >= self.minimum_named_resources:
                    break
                continue

            if len(merged_resources) >= self.minimum_named_resources:
                break

            merged_resources = self._curate_resources(self._merge_resources(merged_resources, curated_group))
            resource_sources.append(source_name)
            if len(merged_resources) >= self.minimum_named_resources:
                break

        fallback_bucket = self._fallback_bucket_for_skill(normalized)
        fallback_resources = self.resource_fallbacks.get(fallback_bucket, [])
        if not isinstance(fallback_resources, list) or not fallback_resources:
            fallback_bucket = "technical"
            fallback_resources = self.resource_fallbacks.get("technical", [])
        fallback_list = fallback_resources if isinstance(fallback_resources, list) else []
        fallback_used = False
        if len(merged_resources) < self.minimum_named_resources and fallback_list:
            merged_resources = self._merge_resources(merged_resources, fallback_list)
            resource_sources.append(f"dataset_fallback_resources:{fallback_bucket}")
            fallback_used = True

        if not resource_sources:
            resource_sources.append(f"dataset_fallback_resources:{fallback_bucket}")
            fallback_used = True

        return {
            "resources": self._curate_resources(merged_resources),
            "resource_source": " + ".join(resource_sources),
            "resource_sources": resource_sources,
            "fallback_used": fallback_used,
        }

    def _curate_resources(self, resources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(resources) <= self.maximum_named_resources:
            return resources

        level_order = ("beginner", "intermediate", "advanced")
        selected: List[Dict[str, Any]] = []
        selected_titles = set()

        for level in level_order:
            for item in resources:
                title = self._resource_title(item)
                if not title or _norm_skill(title) in selected_titles:
                    continue
                if _norm_skill(item.get("level")) != level:
                    continue
                selected.append(item)
                selected_titles.add(_norm_skill(title))
                break
            if len(selected) >= self.maximum_named_resources:
                return selected[: self.maximum_named_resources]

        for item in resources:
            title = self._resource_title(item)
            if not title or _norm_skill(title) in selected_titles:
                continue
            selected.append(item)
            selected_titles.add(_norm_skill(title))
            if len(selected) >= self.maximum_named_resources:
                break

        return selected[: self.maximum_named_resources]

    def _format_track_items(self, track_payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
        roadmap_details = track_payload.get("roadmap_details", [])
        if not isinstance(roadmap_details, list):
            return []

        items: List[Dict[str, Any]] = []
        for detail in roadmap_details:
            if not isinstance(detail, dict):
                continue

            skill = _norm_skill(detail.get("skill"))
            skill = self._canonical_skill(skill)
            if not skill:
                continue

            resource_payload = self._resources_for_skill(skill, detail.get("resources"))
            items.append(
                {
                    "skill": skill,
                    "phase": detail.get("phase"),
                    "priority": round(_as_float(detail.get("priority"), default=0.0), 4),
                    "reason": detail.get("reason"),
                    "difficulty": round(_as_float(detail.get("difficulty"), default=0.0), 4),
                    "dependency_weight": round(_as_float(detail.get("dependency_weight"), default=0.0), 4),
                    "resources": resource_payload["resources"],
                    "resource_source": resource_payload["resource_source"],
                    "resource_sources": resource_payload.get("resource_sources", [resource_payload["resource_source"]]),
                    "fallback_used": resource_payload["fallback_used"],
                }
            )
        return items

    def _format_jd_resources(self) -> Dict[str, Any]:
        track = self.adaptive_data.get("jd_requirement_roadmap", {})
        if not isinstance(track, dict):
            return {"title": "JD Requirement Resources", "items": [], "next_steps": []}

        return {
            "title": track.get("title", "JD Requirement Resources"),
            "track_type": track.get("track_type"),
            "target_role": track.get("target_role"),
            "candidate_best_fit_role": track.get("candidate_best_fit_role"),
            "target_jd_role": track.get("target_jd_role"),
            "top_role": track.get("top_role"),
            "next_steps": track.get("next_steps", []),
            "items": self._format_track_items(track),
        }

    def _format_profession_resources(self) -> List[Dict[str, Any]]:
        tracks = self.adaptive_data.get("profession_roadmaps", [])
        if not isinstance(tracks, list):
            return []

        output: List[Dict[str, Any]] = []
        for track in tracks:
            if not isinstance(track, dict):
                continue
            top_role = track.get("top_role", {})
            role_name = top_role.get("role") if isinstance(top_role, dict) else None
            output.append(
                {
                    "role": role_name,
                    "title": track.get("title", f"Profession Resources: {role_name or 'Unknown'}"),
                    "track_type": track.get("track_type"),
                    "target_role": track.get("target_role"),
                    "candidate_best_fit_role": track.get("candidate_best_fit_role"),
                    "target_jd_role": track.get("target_jd_role"),
                    "next_steps": track.get("next_steps", []),
                    "items": self._format_track_items(track),
                }
            )
        return output

    def run(self) -> Dict[str, Any]:
        jd_resources = self._format_jd_resources()
        profession_resources = self._format_profession_resources()

        return {
            "__meta__": {
                "jd_item_count": len(jd_resources.get("items", [])),
                "profession_track_count": len(profession_resources),
                "resource_override_count": len(self.resource_data),
                "minimum_named_resources_per_skill": self.minimum_named_resources,
                "maximum_named_resources_per_skill": self.maximum_named_resources,
            },
            "candidate_best_fit_role": self.adaptive_data.get("candidate_best_fit_role"),
            "target_jd_role": self.adaptive_data.get("target_jd_role"),
            "recommended_track_type": self.adaptive_data.get("recommended_track_type"),
            "jd_requirement_resources": jd_resources,
            "profession_resources": profession_resources,
        }


def main() -> None:
    args = _parse_args()
    adaptive_json = args.adaptive_json or args.adaptive_json_pos or DEFAULT_ADAPTIVE_JSON
    dataset_json = args.dataset_json or args.dataset_json_pos or DEFAULT_DATASET_JSON
    resources_json = args.resources_json or args.resources_json_pos or DEFAULT_RESOURCES_JSON
    output_json = args.output_json or args.output_json_pos or DEFAULT_OUTPUT_JSON

    adaptive_data = _load_json(adaptive_json)
    dataset_data = _load_json(dataset_json)
    resource_data = _load_json(resources_json)

    layer = LearningResourceLayer(
        adaptive_data=adaptive_data,
        dataset_data=dataset_data,
        resource_data=resource_data,
    )
    output = layer.run()

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"Learning resource JSON written to: {output_json}")
    print(f"JD items: {len(output.get('jd_requirement_resources', {}).get('items', []))}")
    print(f"Profession tracks: {len(output.get('profession_resources', []))}")


if __name__ == "__main__":
    main()
