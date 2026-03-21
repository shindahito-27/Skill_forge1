from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_FIRST_JSON = PROJECT_DIR / "module2" / "layer_a_combined_scored.json"
DEFAULT_SECOND_JSON = PROJECT_DIR / "module_3_jd" / "jd_req" / "layer_a_combined_scored.json"
DEFAULT_OUTPUT_JSON = PROJECT_DIR / "module4" / "gapengine_output.json"
PRESERVED_TEXT_FIELDS = (
    "category",
    "taxonomy_category",
    "taxonomical_category",
    "sub_category",
    "subcategory",
    "taxonomy_subcategory",
)
FIRST_SCORE_FIELD = "resulting_score"
SECOND_SCORE_FIELD = "weight"
LEVEL_RANKS = {"entry": 0, "mid": 1, "senior": 2}
LEVEL_BASE_FACTORS = {
    0: 1.0,
    1: 0.9,
    2: 0.8,
    3: 0.72,
}
MIN_LEVEL_NORMALIZATION_FACTOR = 0.65


def _normalize_score(weight: float) -> float:
    return min(10.0, weight * 1.2)


def _classify_gap(gap: float) -> Dict[str, str]:
    if gap >= 6:
        return {"level": "Critical Gap", "action": "top priority"}
    if gap >= 3:
        return {"level": "Moderate Gap", "action": "important"}
    if gap > 0:
        return {"level": "Slight Gap", "action": "low priority"}
    return {"level": "Good Match", "action": "keep"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare resume scores against JD scores with level-aware normalization."
    )
    parser.add_argument("first_json", nargs="?", type=Path, default=DEFAULT_FIRST_JSON)
    parser.add_argument("second_json", nargs="?", type=Path, default=DEFAULT_SECOND_JSON)
    parser.add_argument(
        "-o",
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Path for the generated gap JSON file.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a top-level JSON object in {path}")
    return data


def _pick_text_field(field: str, first_item: Dict[str, Any], second_item: Dict[str, Any]) -> Any:
    if field in first_item and first_item[field] not in (None, ""):
        return first_item[field]
    return second_item.get(field)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_level(value: Any, default: str = "entry") -> str:
    normalized = str(value or "").strip().lower()
    if normalized in LEVEL_RANKS:
        return normalized
    return default


def _extract_candidate_profile(first_data: Dict[str, Any]) -> Dict[str, Any]:
    meta = first_data.get("__meta__", {})
    profile = meta.get("candidate_level_profile", {}) if isinstance(meta, dict) else {}
    if not isinstance(profile, dict):
        profile = {}

    candidate_level = _normalize_level(profile.get("candidate_level"), default="entry")
    extracted = dict(profile)
    extracted["candidate_level"] = candidate_level
    extracted["candidate_level_rank"] = LEVEL_RANKS.get(candidate_level, 0)
    return extracted


def _extract_jd_profile(second_data: Dict[str, Any]) -> Dict[str, Any]:
    meta = second_data.get("__meta__", {})
    profile = meta.get("jd_level_profile", {}) if isinstance(meta, dict) else {}
    if not isinstance(profile, dict):
        profile = {}

    jd_level = _normalize_level(profile.get("jd_level"), default="entry")
    extracted = dict(profile)
    extracted["jd_level"] = jd_level
    extracted["jd_level_rank"] = LEVEL_RANKS.get(jd_level, 0)
    return extracted


def _level_gap_steps(candidate_profile: Dict[str, Any], jd_profile: Dict[str, Any]) -> int:
    candidate_rank = _as_int(candidate_profile.get("candidate_level_rank"), default=0)
    jd_rank = _as_int(jd_profile.get("jd_level_rank"), default=0)
    return max(0, jd_rank - candidate_rank)


def _level_normalization_factor(
    candidate_profile: Dict[str, Any],
    jd_profile: Dict[str, Any],
    skill_years: Any,
) -> float:
    gap_steps = _level_gap_steps(candidate_profile, jd_profile)
    if gap_steps <= 0:
        return 1.0

    factor = LEVEL_BASE_FACTORS.get(gap_steps, LEVEL_BASE_FACTORS[max(LEVEL_BASE_FACTORS)])
    years = _as_int(skill_years, default=0)
    if years >= 4:
        factor -= 0.08
    elif years >= 2:
        factor -= 0.04

    return round(max(MIN_LEVEL_NORMALIZATION_FACTOR, factor), 4)


def _level_normalization_reason(
    candidate_profile: Dict[str, Any],
    jd_profile: Dict[str, Any],
    factor: float,
    skill_years: Any,
) -> str:
    candidate_level = _normalize_level(candidate_profile.get("candidate_level"), default="entry")
    jd_level = _normalize_level(jd_profile.get("jd_level"), default="entry")
    years = _as_int(skill_years, default=0)

    if factor >= 1.0:
        return "No level normalization needed because candidate and JD levels are aligned."

    reason = f"Candidate level '{candidate_level}' is below JD level '{jd_level}', so JD pressure was scaled to {factor:.2f}x."
    if years >= 4:
        reason += f" Skill-specific experience signal ({years}+ years) added extra seniority discount."
    elif years >= 2:
        reason += f" Skill-specific experience signal ({years}+ years) added a small seniority discount."
    return reason


def _build_gap_entry(
    first_item: Any,
    second_item: Any,
    candidate_profile: Dict[str, Any],
    jd_profile: Dict[str, Any],
) -> Dict[str, Any]:
    left = first_item if isinstance(first_item, dict) else {}
    right = second_item if isinstance(second_item, dict) else {}

    result: Dict[str, Any] = {}

    for field in PRESERVED_TEXT_FIELDS:
        value = _pick_text_field(field, left, right)
        if value not in (None, ""):
            result[field] = value

    if "sub_category" not in result:
        for alias in ("subcategory", "taxonomy_subcategory"):
            alias_value = result.get(alias)
            if alias_value not in (None, ""):
                result["sub_category"] = alias_value
                break

    first_score_raw = left.get(FIRST_SCORE_FIELD)
    second_score_raw = _as_float(right.get(SECOND_SCORE_FIELD), default=0.0)
    raw_resume_score = _normalize_score(_as_float(first_score_raw, default=0.0)) if first_score_raw is not None else 0.0
    raw_jd_score = _normalize_score(second_score_raw)

    skill_years = right.get("experience_years")
    normalization_factor = _level_normalization_factor(candidate_profile, jd_profile, skill_years)
    adjusted_jd_score = round(raw_jd_score * normalization_factor, 4)

    raw_gap = round(max(0.0, raw_jd_score - raw_resume_score), 4)
    adjusted_gap = round(max(0.0, adjusted_jd_score - raw_resume_score), 4)
    surplus_score = round(max(0.0, raw_resume_score - adjusted_jd_score), 4)

    result["resume_score"] = round(raw_resume_score, 4)
    result["jd_score"] = adjusted_jd_score
    result["jd_score_before_level_normalization"] = round(raw_jd_score, 4)
    result["raw_gap_score"] = raw_gap
    result["gap_score"] = adjusted_gap
    result["surplus_score"] = surplus_score
    if adjusted_gap <= 0.0:
        result["status"] = "matched"
    elif first_score_raw is not None:
        result["status"] = "partial_match"
    else:
        result["status"] = "missing"
    result["candidate_level"] = candidate_profile.get("candidate_level", "entry")
    result["jd_level"] = jd_profile.get("jd_level", "entry")
    result["level_gap_steps"] = _level_gap_steps(candidate_profile, jd_profile)
    result["jd_skill_experience_years"] = skill_years
    result["level_normalization_factor"] = normalization_factor
    result["level_normalization_applied"] = normalization_factor < 1.0
    result["level_normalization_reason"] = _level_normalization_reason(
        candidate_profile,
        jd_profile,
        normalization_factor,
        skill_years,
    )

    if right.get("phrase") not in (None, ""):
        result["jd_phrase"] = right.get("phrase")

    classification = _classify_gap(adjusted_gap)
    result.update(classification)
    return result


def build_gap_json(first_data: Dict[str, Any], second_data: Dict[str, Any]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    common_skills = set(first_data) & set(second_data)
    jd_only_skills = set(second_data) - set(first_data)
    candidate_profile = _extract_candidate_profile(first_data)
    jd_profile = _extract_jd_profile(second_data)

    for skill_name in sorted(common_skills | jd_only_skills):
        if str(skill_name).startswith("__"):
            continue
        output[skill_name] = _build_gap_entry(
            first_data.get(skill_name),
            second_data.get(skill_name),
            candidate_profile,
            jd_profile,
        )

    output["__meta__"] = {
        "candidate_level_profile": candidate_profile,
        "jd_level_profile": jd_profile,
        "level_normalization_policy": {
            "applies_when_candidate_below_jd": True,
            "base_factors_by_level_gap": LEVEL_BASE_FACTORS,
            "skill_year_adjustment": {
                "2_to_3_years": -0.04,
                "4_plus_years": -0.08,
            },
            "minimum_factor": MIN_LEVEL_NORMALIZATION_FACTOR,
            "note": "Gap scores are stored as non-negative deficit magnitudes using max(0, adjusted_jd_score - resume_score).",
        },
    }
    return output


def main() -> None:
    args = _parse_args()
    first_data = _load_json(args.first_json)
    second_data = _load_json(args.second_json)
    output = build_gap_json(first_data, second_data)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Gap JSON written to: {args.output_json}")


if __name__ == "__main__":
    main()
