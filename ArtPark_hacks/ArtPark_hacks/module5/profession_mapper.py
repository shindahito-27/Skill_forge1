from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set


MODULE5_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE5_DIR.parent
REPO_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_RESUME_JSON = REPO_ROOT / "output" / "resume" / "module_2" / "Module_2_combined.json"
DEFAULT_DATASET_JSON = MODULE5_DIR / "profession_mapping_engine_dataset_v7.json"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "output" / "module_5" / "profession_mapping_output.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict best-fit professions from resume skill scores using cosine similarity."
    )
    parser.add_argument("resume_json_pos", nargs="?", type=Path, default=None)
    parser.add_argument("dataset_json_pos", nargs="?", type=Path, default=None)
    parser.add_argument("output_json_pos", nargs="?", type=Path, default=None)
    parser.add_argument("--resume-json", type=Path, default=None)
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


def _norm_text(value: object) -> str:
    cleaned = str(value or "").strip().lower()
    return re.sub(r"\s+", " ", cleaned)


def _compact_text(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", _norm_text(value))


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _unique_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _format_template(template: str, replacements: Mapping[str, str]) -> str:
    formatted = str(template or "")
    for key, value in replacements.items():
        formatted = formatted.replace(f"{{{key}}}", value)
    return formatted.strip()


def _normalized_contexts(payload: Mapping[str, Any]) -> List[str]:
    raw_contexts = payload.get("contexts", [])
    if not isinstance(raw_contexts, list):
        return []
    return sorted({_norm_text(context) for context in raw_contexts if _norm_text(context)})


def _context_multiplier(contexts: List[str], mapping_policy: Mapping[str, Any]) -> float:
    context_policy = mapping_policy.get("context_multipliers", {})
    if not isinstance(context_policy, dict):
        context_policy = {}

    if not contexts:
        return _as_float(context_policy.get("unknown"), default=0.9)

    if set(contexts) == {"general"}:
        return _as_float(context_policy.get("general_only"), default=0.78)

    multiplier = 1.0
    for context in contexts:
        if context == "general":
            continue
        multiplier = max(multiplier, _as_float(context_policy.get(context), default=1.0))
    return multiplier


def _technical_taxonomies(mapping_policy: Mapping[str, Any]) -> Set[str]:
    roadmap_taxonomies = mapping_policy.get(
        "roadmap_technical_taxonomies",
        [
            "programming",
            "data",
            "cloud",
            "machine learning",
            "backend",
            "frontend",
            "devops",
            "security",
        ],
    )
    if not isinstance(roadmap_taxonomies, list):
        return set()
    return {_norm_text(value) for value in roadmap_taxonomies if _norm_text(value)}


def _roadmap_presence_signal(
    payload: Mapping[str, Any],
    confidence: float,
    mentions: int,
    contexts: List[str],
    signal: float,
    mapping_policy: Mapping[str, Any],
) -> float:
    category = _norm_text(payload.get("category"))
    taxonomy_key = _norm_text(payload.get("taxonomy_category"))
    roadmap_taxonomies = _technical_taxonomies(mapping_policy)

    is_technical = category == "hard_skill" or taxonomy_key in roadmap_taxonomies
    if not is_technical or confidence <= 0.0:
        return signal

    confidence_multiplier = max(
        _as_float(mapping_policy.get("roadmap_presence_confidence_multiplier", 0.42)),
        0.0,
    )
    general_only_multiplier = max(
        _as_float(mapping_policy.get("roadmap_presence_general_only_multiplier", 0.85)),
        0.0,
    )
    grounded_multiplier = max(
        _as_float(mapping_policy.get("roadmap_presence_grounded_multiplier", 1.0)),
        0.0,
    )
    mention_bonus = max(_as_float(mapping_policy.get("roadmap_presence_mention_bonus", 0.02)), 0.0)
    max_presence = _clamp(
        _as_float(mapping_policy.get("roadmap_presence_max", 0.45), default=0.45),
        lower=0.0,
        upper=1.0,
    )

    presence_floor = confidence * confidence_multiplier
    if contexts and set(contexts) == {"general"}:
        presence_floor *= general_only_multiplier
    else:
        presence_floor *= grounded_multiplier

    presence_floor += max(0, mentions - 1) * mention_bonus
    presence_floor = min(max_presence, presence_floor)
    return max(signal, _clamp(presence_floor))


def _candidate_signal(
    raw_skill: str,
    canonical_skill: str,
    payload: Mapping[str, Any],
    mapping_policy: Mapping[str, Any],
) -> Dict[str, Any]:
    confidence_field = str(mapping_policy.get("candidate_confidence_field", "confidence"))
    strength_field = str(mapping_policy.get("candidate_strength_field", "resulting_score"))
    strength_scale = max(_as_float(mapping_policy.get("candidate_strength_scale", 10.0)), 1.0)

    confidence = _clamp(_as_float(payload.get(confidence_field), default=0.0))
    raw_strength = _as_float(payload.get(strength_field), default=0.0)
    normalized_strength = _clamp(raw_strength / strength_scale) if raw_strength > 0.0 else confidence

    mentions = max(1, _as_int(payload.get("mentions"), default=1))
    mention_boost_per_extra = max(_as_float(mapping_policy.get("mention_boost_per_extra", 0.04)), 0.0)
    max_mention_multiplier = max(_as_float(mapping_policy.get("max_mention_multiplier", 1.15)), 1.0)
    mention_multiplier = min(max_mention_multiplier, 1.0 + (max(0, mentions - 1) * mention_boost_per_extra))

    contexts = _normalized_contexts(payload)
    context_multiplier = _context_multiplier(contexts, mapping_policy)

    generic_skill_penalties = mapping_policy.get("generic_skill_penalties", {})
    if not isinstance(generic_skill_penalties, dict):
        generic_skill_penalties = {}
    generic_taxonomy_penalties = mapping_policy.get("generic_taxonomy_penalties", {})
    if not isinstance(generic_taxonomy_penalties, dict):
        generic_taxonomy_penalties = {}
    generic_skill_signal_caps = mapping_policy.get("generic_skill_signal_caps", {})
    if not isinstance(generic_skill_signal_caps, dict):
        generic_skill_signal_caps = {}

    taxonomy_key = _norm_text(payload.get("taxonomy_category"))
    category_key = _norm_text(payload.get("category"))
    roadmap_taxonomies = _technical_taxonomies(mapping_policy)
    is_technical = category_key == "hard_skill" or taxonomy_key in roadmap_taxonomies
    skill_penalty = _as_float(
        generic_skill_penalties.get(canonical_skill, generic_skill_penalties.get(raw_skill, 1.0)),
        default=1.0,
    )
    taxonomy_penalty = _as_float(generic_taxonomy_penalties.get(taxonomy_key, 1.0), default=1.0)

    confidence_multiplier = 0.65 + (0.35 * confidence)
    signal = _clamp(
        normalized_strength
        * confidence_multiplier
        * mention_multiplier
        * context_multiplier
        * skill_penalty
        * taxonomy_penalty
    )
    roadmap_signal = _roadmap_presence_signal(
        payload=payload,
        confidence=confidence,
        mentions=mentions,
        contexts=contexts,
        signal=signal,
        mapping_policy=mapping_policy,
    )

    grounded_floor = _clamp(
        _as_float(mapping_policy.get("grounded_technical_signal_floor", 0.0), default=0.0),
        lower=0.0,
        upper=1.0,
    )
    grounded_floor_confidence = _clamp(
        _as_float(mapping_policy.get("grounded_technical_confidence_threshold", 0.7), default=0.7),
        lower=0.0,
        upper=1.0,
    )
    grounded_min_mentions = max(_as_int(mapping_policy.get("grounded_technical_min_mentions", 2), default=2), 1)
    has_grounded_context = bool(set(contexts) - {"general"}) or mentions >= grounded_min_mentions
    if is_technical and has_grounded_context and confidence >= grounded_floor_confidence and grounded_floor > 0.0:
        signal = max(signal, grounded_floor)
        roadmap_signal = max(roadmap_signal, grounded_floor)

    signal_cap = _as_float(
        generic_skill_signal_caps.get(canonical_skill, generic_skill_signal_caps.get(raw_skill, 1.0)),
        default=1.0,
    )
    if signal_cap > 0.0:
        signal = min(signal, signal_cap)
        roadmap_signal = min(roadmap_signal, signal_cap)

    return {
        "signal": signal,
        "roadmap_signal": roadmap_signal,
        "confidence": confidence,
        "strength": normalized_strength,
        "mentions": mentions,
        "contexts": contexts,
    }


def _drop_generic_candidate_skill(
    raw_skill: str,
    canonical_skill: str,
    payload: Mapping[str, Any],
    signal_payload: Mapping[str, Any],
    mapping_policy: Mapping[str, Any],
) -> bool:
    generic_drop_threshold = _clamp(
        _as_float(mapping_policy.get("generic_skill_drop_threshold", 0.0), default=0.0),
        lower=0.0,
        upper=1.0,
    )
    if generic_drop_threshold <= 0.0:
        return False

    generic_skill_penalties = mapping_policy.get("generic_skill_penalties", {})
    if not isinstance(generic_skill_penalties, dict):
        generic_skill_penalties = {}
    generic_skill_signal_caps = mapping_policy.get("generic_skill_signal_caps", {})
    if not isinstance(generic_skill_signal_caps, dict):
        generic_skill_signal_caps = {}
    generic_taxonomy_penalties = mapping_policy.get("generic_taxonomy_penalties", {})
    if not isinstance(generic_taxonomy_penalties, dict):
        generic_taxonomy_penalties = {}

    taxonomy_key = _norm_text(payload.get("taxonomy_category"))
    is_generic = any(
        key in generic_skill_penalties or key in generic_skill_signal_caps
        for key in (canonical_skill, raw_skill)
        if key
    ) or taxonomy_key in generic_taxonomy_penalties
    if not is_generic:
        return False

    max_signal = max(
        _as_float(signal_payload.get("signal"), default=0.0),
        _as_float(signal_payload.get("roadmap_signal"), default=0.0),
    )
    return max_signal <= generic_drop_threshold


@dataclass(frozen=True)
class SkillResolver:
    skill_universe: List[str]
    exact_map: Dict[str, str]
    compact_map: Dict[str, str]
    alias_map: Dict[str, str]

    def resolve(self, skill_name: object) -> Optional[str]:
        normalized = _norm_text(skill_name)
        if not normalized:
            return None

        alias_target = self.alias_map.get(normalized)
        if alias_target:
            return alias_target

        exact_target = self.exact_map.get(normalized)
        if exact_target:
            return exact_target

        compact = _compact_text(normalized)
        compact_target = self.compact_map.get(compact)
        if compact_target:
            return compact_target

        if compact.endswith("s"):
            singular_target = self.compact_map.get(compact[:-1])
            if singular_target:
                return singular_target

        return None


def _build_skill_resolver(dataset: Mapping[str, Any]) -> SkillResolver:
    raw_universe = dataset.get("skill_universe", [])
    if not isinstance(raw_universe, list) or not raw_universe:
        raise ValueError("Dataset must include a non-empty skill_universe list.")

    skill_universe = [_norm_text(skill) for skill in raw_universe if _norm_text(skill)]
    skill_universe = _unique_preserve_order(skill_universe)

    exact_map = {skill: skill for skill in skill_universe}
    compact_map = {_compact_text(skill): skill for skill in skill_universe}

    alias_map: Dict[str, str] = {}
    raw_aliases = dataset.get("alias_map", {})
    if isinstance(raw_aliases, dict):
        for alias, canonical in raw_aliases.items():
            alias_key = _norm_text(alias)
            canonical_value = _norm_text(canonical)
            if not alias_key or not canonical_value:
                continue

            resolved_canonical = exact_map.get(canonical_value)
            if not resolved_canonical:
                resolved_canonical = compact_map.get(_compact_text(canonical_value))
            if resolved_canonical:
                alias_map[alias_key] = resolved_canonical

    return SkillResolver(
        skill_universe=skill_universe,
        exact_map=exact_map,
        compact_map=compact_map,
        alias_map=alias_map,
    )


def validate_dataset(dataset: Mapping[str, Any], resolver: SkillResolver) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    roles = dataset.get("roles", {})
    if not isinstance(roles, dict) or not roles:
        errors.append("Dataset must include a non-empty roles object.")
        return {"is_valid": False, "errors": errors, "warnings": warnings}

    validation_rules = dataset.get("validation_rules", {})
    levels = set(validation_rules.get("levels", ["entry", "mid", "senior"]))
    universe_set = set(resolver.skill_universe)

    for role_name, role_payload in roles.items():
        if not isinstance(role_payload, dict):
            errors.append(f"Role '{role_name}' must be a JSON object.")
            continue

        weights = role_payload.get("weights", {})
        if not isinstance(weights, dict) or not weights:
            errors.append(f"Role '{role_name}' must define a non-empty weights object.")
            continue

        for skill_name, raw_weight in weights.items():
            normalized_skill = _norm_text(skill_name)
            if normalized_skill not in universe_set:
                errors.append(
                    f"Role '{role_name}' uses weight skill outside skill_universe: '{skill_name}'."
                )
            weight = _as_float(raw_weight, default=-1.0)
            if weight < 0.0 or weight > 1.0:
                errors.append(f"Role '{role_name}' has out-of-range weight for '{skill_name}': {raw_weight}")

        prior = _as_float(role_payload.get("prior"), default=-1.0)
        if prior < 0.0 or prior > 1.0:
            errors.append(f"Role '{role_name}' has prior outside 0-1 range.")

        level = _norm_text(role_payload.get("level"))
        if level and levels and level not in levels:
            errors.append(f"Role '{role_name}' uses unsupported level '{level}'.")

        core_skills = role_payload.get("core_skills", [])
        if isinstance(core_skills, list):
            for skill_name in core_skills:
                normalized_skill = _norm_text(skill_name)
                if normalized_skill not in weights:
                    errors.append(
                        f"Role '{role_name}' core skill '{skill_name}' is missing from weights."
                    )
                if normalized_skill not in universe_set:
                    errors.append(
                        f"Role '{role_name}' core skill '{skill_name}' is outside skill_universe."
                    )

    if len(resolver.skill_universe) != len(universe_set):
        warnings.append("Duplicate skills were found in skill_universe after normalization.")

    return {"is_valid": not errors, "errors": errors, "warnings": warnings}


def normalize_resume_skills(
    resume_data: Mapping[str, Any],
    resolver: SkillResolver,
    mapping_policy: Mapping[str, Any],
) -> Dict[str, Any]:
    confidence_field = str(mapping_policy.get("candidate_confidence_field", "confidence"))
    meta_prefix = str(mapping_policy.get("candidate_meta_prefix", "__"))

    normalized_scores: Dict[str, float] = {}
    evidence: Dict[str, Dict[str, Any]] = {}
    unmatched: List[Dict[str, Any]] = []

    for raw_skill, payload in resume_data.items():
        raw_key = str(raw_skill)
        if raw_key.startswith(meta_prefix):
            continue
        if not isinstance(payload, dict):
            continue

        signal_payload = _candidate_signal(raw_key, "", payload, mapping_policy)
        confidence = signal_payload["confidence"]
        if confidence <= 0.0:
            continue

        canonical_skill = resolver.resolve(raw_key)
        if not canonical_skill:
            unmatched.append(
                {
                    "raw_skill": raw_key,
                    "confidence": round(confidence, 4),
                }
            )
            continue

        signal_payload = _candidate_signal(raw_key, canonical_skill, payload, mapping_policy)
        if _drop_generic_candidate_skill(raw_key, canonical_skill, payload, signal_payload, mapping_policy):
            continue
        normalized_scores[canonical_skill] = max(normalized_scores.get(canonical_skill, 0.0), signal_payload["signal"])
        skill_evidence = evidence.setdefault(
            canonical_skill,
            {
                "skill": canonical_skill,
                "signal": 0.0,
                "roadmap_signal": 0.0,
                "confidence": 0.0,
                "strength": 0.0,
                "mentions": 0,
                "contexts": [],
                "raw_skills": [],
            },
        )
        skill_evidence["signal"] = max(skill_evidence["signal"], signal_payload["signal"])
        skill_evidence["roadmap_signal"] = max(skill_evidence["roadmap_signal"], signal_payload["roadmap_signal"])
        skill_evidence["confidence"] = max(skill_evidence["confidence"], signal_payload["confidence"])
        skill_evidence["strength"] = max(skill_evidence["strength"], signal_payload["strength"])
        skill_evidence["mentions"] = max(skill_evidence["mentions"], signal_payload["mentions"])
        skill_evidence["contexts"] = _unique_preserve_order(skill_evidence["contexts"] + signal_payload["contexts"])
        skill_evidence["raw_skills"] = _unique_preserve_order(skill_evidence["raw_skills"] + [raw_key])

    normalized_skill_list = sorted(
        (
            {
                "skill": skill_name,
                "signal": round(skill_payload["signal"], 4),
                "roadmap_signal": round(skill_payload["roadmap_signal"], 4),
                "confidence": round(skill_payload["confidence"], 4),
                "strength": round(skill_payload["strength"], 4),
                "mentions": skill_payload["mentions"],
                "contexts": skill_payload["contexts"],
                "raw_skills": skill_payload["raw_skills"],
            }
            for skill_name, skill_payload in evidence.items()
        ),
        key=lambda item: (-item["signal"], -item["confidence"], item["skill"]),
    )

    unmatched = sorted(unmatched, key=lambda item: (-item["confidence"], item["raw_skill"]))

    return {
        "scores": normalized_scores,
        "skills": normalized_skill_list,
        "unmatched": unmatched,
    }


def _build_skill_idf(roles: Mapping[str, Any], resolver: SkillResolver) -> Dict[str, float]:
    total_roles = 0
    document_frequency: Dict[str, int] = {}

    for role_payload in roles.values():
        if not isinstance(role_payload, dict):
            continue
        total_roles += 1
        role_skills = set(_role_weights(role_payload, resolver))
        role_skills.update(_normalized_skill_list(role_payload, "core_skills", resolver))
        for skill_name in role_skills:
            document_frequency[skill_name] = document_frequency.get(skill_name, 0) + 1

    skill_idf: Dict[str, float] = {}
    for skill_name in resolver.skill_universe:
        df = document_frequency.get(skill_name, 0)
        skill_idf[skill_name] = 1.0 + math.log((1.0 + total_roles) / (1.0 + df))
    return skill_idf


def _build_dense_vector(
    skill_universe: List[str],
    skill_scores: Mapping[str, float],
    skill_weights: Optional[Mapping[str, float]] = None,
) -> List[float]:
    vector: List[float] = []
    for skill_name in skill_universe:
        value = float(skill_scores.get(skill_name, 0.0))
        if skill_weights:
            value *= float(skill_weights.get(skill_name, 1.0))
        vector.append(value)
    return vector


def _vector_norm(vector: List[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _cosine_similarity(
    left_vector: List[float],
    right_vector: List[float],
    left_norm: float,
    right_norm: float,
    epsilon: float,
) -> float:
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0

    dot_product = sum(left * right for left, right in zip(left_vector, right_vector))
    return dot_product / max(left_norm * right_norm, epsilon)


def _similarity_skill_weights(
    resolver: SkillResolver,
    skill_metadata: Mapping[str, Any],
    role_payload: Mapping[str, Any],
    similarity_policy: Mapping[str, Any],
) -> Dict[str, float]:
    role_category = _norm_text(role_payload.get("category"))
    technical_soft_cap = _clamp(
        _as_float(similarity_policy.get("soft_skill_cap_technical", 0.0), default=0.0),
        lower=0.0,
        upper=1.0,
    )
    management_soft_cap = _clamp(
        _as_float(similarity_policy.get("soft_skill_cap_management", 0.35), default=0.35),
        lower=0.0,
        upper=1.0,
    )
    soft_cap = management_soft_cap if role_category == "management" else technical_soft_cap

    weights: Dict[str, float] = {}
    for skill_name in resolver.skill_universe:
        payload = skill_metadata.get(skill_name, {}) if isinstance(skill_metadata, Mapping) else {}
        skill_type = _norm_text(payload.get("type")) if isinstance(payload, Mapping) else ""
        if skill_type == "soft":
            weights[skill_name] = soft_cap
        else:
            weights[skill_name] = 1.0
    return weights


def _normalized_skill_list(role_payload: Mapping[str, Any], key: str, resolver: SkillResolver) -> List[str]:
    raw_values = role_payload.get(key, [])
    if not isinstance(raw_values, list):
        return []

    normalized: List[str] = []
    for value in raw_values:
        resolved = resolver.resolve(value)
        if resolved:
            normalized.append(resolved)
    return _unique_preserve_order(normalized)


def _role_weights(role_payload: Mapping[str, Any], resolver: SkillResolver) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    weights = role_payload.get("weights", {})
    if not isinstance(weights, dict):
        return normalized

    for skill_name, raw_weight in weights.items():
        canonical_skill = resolver.resolve(skill_name)
        if not canonical_skill:
            continue
        weight = _clamp(_as_float(raw_weight, default=0.0))
        normalized[canonical_skill] = max(normalized.get(canonical_skill, 0.0), weight)
    return normalized


def _matched_skills(
    candidate_scores: Mapping[str, float],
    role_weights: Mapping[str, float],
    limit: int,
) -> List[str]:
    contributions = []
    for skill_name, weight in role_weights.items():
        candidate_score = candidate_scores.get(skill_name, 0.0)
        if candidate_score <= 0.0:
            continue
        contributions.append((skill_name, candidate_score * weight, candidate_score, weight))

    contributions.sort(key=lambda item: (-item[1], -item[2], -item[3], item[0]))
    return [skill_name for skill_name, _, _, _ in contributions[:limit]]


def _missing_skills(
    candidate_scores: Mapping[str, float],
    role_payload: Mapping[str, Any],
    role_weights: Mapping[str, float],
    resolver: SkillResolver,
    limit: int,
) -> Dict[str, List[str]]:
    core_skills = _normalized_skill_list(role_payload, "core_skills", resolver)
    missing_core_skills = [skill for skill in core_skills if candidate_scores.get(skill, 0.0) <= 0.0]

    weighted_missing = [
        (skill_name, weight)
        for skill_name, weight in role_weights.items()
        if candidate_scores.get(skill_name, 0.0) <= 0.0
    ]
    weighted_missing.sort(key=lambda item: (-item[1], item[0]))

    combined_missing = missing_core_skills + [skill_name for skill_name, _ in weighted_missing]
    combined_missing = _unique_preserve_order(combined_missing)

    return {
        "core_found": [skill for skill in core_skills if candidate_scores.get(skill, 0.0) > 0.0],
        "missing_core": missing_core_skills,
        "missing": combined_missing[:limit],
    }


def _level_bias_for_role(
    role_payload: Mapping[str, Any],
    mapping_policy: Mapping[str, Any],
    role_bias_policy: Mapping[str, Any],
) -> float:
    role_level = _norm_text(role_payload.get("level"))
    default_bias = 1.0

    named_bias_keys = {
        "entry": _as_float(role_bias_policy.get("entry_boost"), default=0.0),
        "mid": _as_float(role_bias_policy.get("mid_boost"), default=0.0),
        "senior": _as_float(role_bias_policy.get("senior_boost"), default=0.0),
    }
    if named_bias_keys.get(role_level, 0.0) > 0.0:
        return named_bias_keys[role_level]

    role_biases = role_bias_policy.get("level_bias", {})
    if isinstance(role_biases, dict) and role_level in role_biases:
        return _as_float(role_biases.get(role_level), default=default_bias)

    student_biases = mapping_policy.get("student_level_bias", {})
    if isinstance(student_biases, dict) and role_level in student_biases:
        return _as_float(student_biases.get(role_level), default=default_bias)

    return default_bias


def _build_reason(
    role_name: str,
    role_payload: Mapping[str, Any],
    explanation_templates: Mapping[str, Any],
    matched_skills: List[str],
    core_skills_found: List[str],
    missing_core_skills: List[str],
) -> str:
    reason_parts: List[str] = []

    why_this_role = role_payload.get("why_this_role", [])
    if isinstance(why_this_role, list):
        why_points = [str(point).strip() for point in why_this_role if str(point).strip()]
        if why_points:
            reason_parts.append("; ".join(why_points))

    category = _norm_text(role_payload.get("category"))
    template_key = "management" if category == "management" else "technical"
    primary_template = str(explanation_templates.get(template_key, "")).strip()

    matched_summary = ", ".join((core_skills_found or matched_skills)[:3]) or "relevant skills"
    missing_summary = ", ".join(missing_core_skills[:3]) or "fewer missing skills"

    if primary_template:
        reason_parts.append(
            _format_template(
                primary_template,
                {
                    "role": role_name,
                    "skills": matched_summary,
                    "matched_skills": matched_summary,
                    "missing_skills": missing_summary,
                },
            )
        )

    if missing_core_skills:
        gap_template = str(explanation_templates.get("skill_gap", "")).strip()
        if gap_template:
            reason_parts.append(
                _format_template(
                    gap_template,
                    {
                        "role": role_name,
                        "skills": matched_summary,
                        "matched_skills": matched_summary,
                        "missing_skills": missing_summary,
                    },
                )
            )

    if not reason_parts:
        combined_template = str(explanation_templates.get("combined", "")).strip()
        if combined_template:
            reason_parts.append(
                _format_template(
                    combined_template,
                    {
                        "role": role_name,
                        "skills": matched_summary,
                        "matched_skills": matched_summary,
                        "missing_skills": missing_summary,
                    },
                )
            )

    return " ".join(part for part in reason_parts if part).strip()


def build_profession_mapping(resume_data: Mapping[str, Any], dataset: Mapping[str, Any]) -> Dict[str, Any]:
    resolver = _build_skill_resolver(dataset)
    validation = validate_dataset(dataset, resolver)
    if not validation["is_valid"]:
        raise ValueError("Profession mapping dataset validation failed: " + " | ".join(validation["errors"]))

    mapping_policy = dataset.get("mapping_policy", {})
    scoring_policy = dataset.get("scoring", {})
    role_bias_policy = dataset.get("role_bias_policy", {})
    explanation_templates = dataset.get("explanation_templates", {})
    similarity_policy = scoring_policy.get("similarity_policy", {}) if isinstance(scoring_policy, dict) else {}
    skill_metadata = dataset.get("skill_metadata", {}) if isinstance(dataset.get("skill_metadata"), dict) else {}

    normalized_candidate = normalize_resume_skills(resume_data, resolver, mapping_policy)
    candidate_scores = normalized_candidate["scores"]
    roles = dataset.get("roles", {})
    use_skill_idf = bool(mapping_policy.get("use_skill_idf", True))
    skill_idf = _build_skill_idf(roles, resolver) if use_skill_idf else {}

    epsilon = max(_as_float(mapping_policy.get("epsilon"), default=1e-9), 1e-12)
    top_k = max(int(_as_float(mapping_policy.get("top_k"), default=3)), 1)
    threshold = _clamp(_as_float(mapping_policy.get("minimum_similarity_threshold"), default=0.0))
    similarity_weight = _clamp(_as_float(mapping_policy.get("similarity_weight"), default=1.0))
    prior_weight = _clamp(_as_float(mapping_policy.get("prior_weight"), default=0.0))
    use_prior = bool(mapping_policy.get("use_prior", True))
    use_level_bias = bool(mapping_policy.get("use_level_bias", True))
    use_core_overlap_boost = bool(mapping_policy.get("use_core_overlap_boost", True))
    min_overlap = max(int(_as_float(mapping_policy.get("min_overlap"), default=2)), 0)
    core_overlap_boost_per_skill = max(
        _as_float(mapping_policy.get("core_overlap_boost_per_skill"), default=0.0),
        0.0,
    )
    core_overlap_boost_max = max(_as_float(mapping_policy.get("core_overlap_boost_max"), default=0.0), 0.0)
    matched_skills_limit = max(int(_as_float(mapping_policy.get("matched_skills_limit"), default=8)), 1)
    missing_skills_limit = max(int(_as_float(mapping_policy.get("missing_skills_limit"), default=6)), 1)
    zero_core_overlap_penalty = max(_as_float(mapping_policy.get("zero_core_overlap_penalty", 0.08)), 0.0)
    missing_core_penalty_per_skill = max(
        _as_float(mapping_policy.get("missing_core_penalty_per_skill", 0.025), default=0.025),
        0.0,
    )
    missing_core_penalty_ratio_weight = max(
        _as_float(mapping_policy.get("missing_core_penalty_ratio_weight", 0.03), default=0.03),
        0.0,
    )
    missing_core_penalty_max = max(
        _as_float(mapping_policy.get("missing_core_penalty_max", 0.14), default=0.14),
        0.0,
    )
    zero_core_overlap_extra_penalty = max(
        _as_float(mapping_policy.get("zero_core_overlap_extra_penalty", 0.04), default=0.04),
        0.0,
    )
    prior_similarity_floor = _clamp(
        _as_float(mapping_policy.get("prior_similarity_floor", 0.7), default=0.7),
        lower=0.0,
        upper=1.0,
    )
    prior_similarity_gain = _clamp(
        _as_float(mapping_policy.get("prior_similarity_gain", 0.3), default=0.3),
        lower=0.0,
        upper=1.0,
    )
    missing_core_score_scale_per_skill = max(
        _as_float(mapping_policy.get("missing_core_score_scale_per_skill", 0.05), default=0.05),
        0.0,
    )
    missing_core_score_scale_ratio_weight = max(
        _as_float(mapping_policy.get("missing_core_score_scale_ratio_weight", 0.1), default=0.1),
        0.0,
    )
    missing_core_score_scale_floor = _clamp(
        _as_float(mapping_policy.get("missing_core_score_scale_floor", 0.8), default=0.8),
        lower=0.0,
        upper=1.0,
    )

    ranked_roles: List[Dict[str, Any]] = []

    for role_name, role_payload in roles.items():
        if not isinstance(role_payload, dict):
            continue

        normalized_weights = _role_weights(role_payload, resolver)
        similarity_skill_weights = _similarity_skill_weights(
            resolver,
            skill_metadata,
            role_payload,
            similarity_policy,
        )
        candidate_vector = _build_dense_vector(
            resolver.skill_universe,
            candidate_scores,
            {skill: skill_idf.get(skill, 1.0) * similarity_skill_weights.get(skill, 1.0) for skill in resolver.skill_universe},
        )
        candidate_norm = _vector_norm(candidate_vector)
        role_vector = _build_dense_vector(
            resolver.skill_universe,
            normalized_weights,
            {skill: skill_idf.get(skill, 1.0) * similarity_skill_weights.get(skill, 1.0) for skill in resolver.skill_universe},
        )
        role_norm = _vector_norm(role_vector)
        base_similarity = _cosine_similarity(candidate_vector, role_vector, candidate_norm, role_norm, epsilon)

        level_bias = _level_bias_for_role(role_payload, mapping_policy, role_bias_policy) if use_level_bias else 1.0
        prior = _clamp(_as_float(role_payload.get("prior"), default=0.0))

        prior_component = prior_weight * prior if use_prior else 0.0
        prior_scale = 1.0
        if use_prior:
            scaled_prior = prior
            if use_level_bias:
                scaled_prior *= level_bias
            prior_scale = prior_similarity_floor + (prior_similarity_gain * scaled_prior)

        base_score = (similarity_weight * base_similarity * prior_scale) + prior_component

        missing_payload = _missing_skills(
            candidate_scores,
            role_payload,
            normalized_weights,
            resolver,
            missing_skills_limit,
        )
        core_skills_found = missing_payload["core_found"]
        missing_core_skills = missing_payload["missing_core"]
        overlap_count = len(core_skills_found)
        core_skill_count = len(_normalized_skill_list(role_payload, "core_skills", resolver))

        core_overlap_boost = 0.0
        if use_core_overlap_boost and overlap_count >= min_overlap:
            core_overlap_boost = min(core_overlap_boost_max, overlap_count * core_overlap_boost_per_skill)

        matched_skills = _matched_skills(candidate_scores, normalized_weights, matched_skills_limit)
        missing_core_count = len(missing_core_skills)
        missing_core_ratio = (missing_core_count / core_skill_count) if core_skill_count > 0 else 0.0
        missing_core_penalty = 0.0
        if core_skill_count > 0 and missing_core_count > 0:
            missing_core_penalty = min(
                missing_core_penalty_max,
                (missing_core_count * missing_core_penalty_per_skill)
                + (missing_core_ratio * missing_core_penalty_ratio_weight)
                + (zero_core_overlap_extra_penalty if overlap_count == 0 else 0.0),
            )
        if core_skill_count > 0 and overlap_count == 0:
            missing_core_penalty = max(missing_core_penalty, zero_core_overlap_penalty)
        missing_core_score_scale = 1.0
        if core_skill_count > 0 and missing_core_count > 0:
            missing_core_score_scale = max(
                missing_core_score_scale_floor,
                1.0
                - (missing_core_count * missing_core_score_scale_per_skill)
                - (missing_core_ratio * missing_core_score_scale_ratio_weight),
            )
        final_score = _clamp((base_score + core_overlap_boost - missing_core_penalty) * missing_core_score_scale)

        ranked_roles.append(
            {
                "role": role_name,
                "score": round(final_score, 4),
                "base_similarity": round(base_similarity, 4),
                "prior": round(prior, 4),
                "prior_scale": round(prior_scale, 4),
                "level": _norm_text(role_payload.get("level")) or "unknown",
                "level_bias": round(level_bias, 4),
                "core_overlap_boost": round(core_overlap_boost, 4),
                "missing_core_penalty": round(missing_core_penalty, 4),
                "missing_core_score_scale": round(missing_core_score_scale, 4),
                "overlap_count": overlap_count,
                "matched_skills": matched_skills,
                "core_skills_found": core_skills_found,
                "missing_core_skills": missing_core_skills,
                "missing_skills": missing_payload["missing"],
                "why_this_role": role_payload.get("why_this_role", []),
                "reason": _build_reason(
                    role_name,
                    role_payload,
                    explanation_templates,
                    matched_skills,
                    core_skills_found,
                    missing_core_skills,
                ),
            }
        )

    ranked_roles.sort(
        key=lambda item: (
            -item["score"],
            -item["base_similarity"],
            -item["overlap_count"],
            item["role"],
        )
    )

    filtered_roles = [role for role in ranked_roles if role["base_similarity"] >= threshold]
    if not filtered_roles:
        filtered_roles = ranked_roles

    return {
        "dataset_name": dataset.get("dataset_name", "profession_mapping_engine_dataset"),
        "dataset_version": dataset.get("version", "unknown"),
        "candidate_profile": {
            "normalized_skill_count": len(normalized_candidate["skills"]),
            "normalized_skills": normalized_candidate["skills"],
            "unmatched_skills": normalized_candidate["unmatched"],
        },
        "validation": validation,
        "top_roles": filtered_roles[:top_k],
        "all_roles_evaluated": len(ranked_roles),
    }


def main() -> None:
    args = _parse_args()
    resume_json = args.resume_json or args.resume_json_pos or DEFAULT_RESUME_JSON
    dataset_json = args.dataset_json or args.dataset_json_pos or DEFAULT_DATASET_JSON
    output_json = args.output_json or args.output_json_pos or DEFAULT_OUTPUT_JSON

    resume_data = _load_json(resume_json)
    dataset = _load_json(dataset_json)
    output = build_profession_mapping(resume_data, dataset)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"Profession mapping JSON written to: {output_json}")
    for rank, role_payload in enumerate(output.get("top_roles", []), start=1):
        print(
            f"{rank}. {role_payload['role']}: "
            f"{role_payload['score'] * 100:.1f}% "
            f"(cosine {role_payload['base_similarity'] * 100:.1f}%)"
        )


if __name__ == "__main__":
    main()
