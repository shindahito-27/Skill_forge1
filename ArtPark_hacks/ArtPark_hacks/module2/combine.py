from __future__ import annotations

import argparse
import json
import math
import re
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


MODULE2_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE2_DIR.parent
REPO_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_KEYWORD_JSON = REPO_ROOT / "output" / "resume" / "module_2" / "A" / "layer_a_keywords.json"
DEFAULT_SEMANTIC_JSON = REPO_ROOT / "output" / "resume" / "module_2" / "B" / "layer_a_semantic_resume.json"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "output" / "resume" / "module_2" / "layer_a_combined_scored.json"


import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.aliases import normalize_skill_name  # noqa: E402


HARD_CONTEXT_WEIGHTS = {
    "skills": 2.0,
    "project": 5.5,
    "experience": 7.0,
    "education": 1.0,
    "general": 1.0,
    "other": 1.0,
    "unknown": 1.0,
}

SOFT_CONTEXT_WEIGHTS = {
    "skills": 1.0,
    "project": 2.0,
    "experience": 2.5,
    "education": 0.5,
    "general": 0.5,
    "other": 0.5,
    "unknown": 0.5,
}

# Skill-type preference multipliers.
# Hard skills get more preference, soft skills get less preference.
HARD_SKILL_MULTIPLIER = 1.2
SOFT_SKILL_MULTIPLIER = 0.7
SOFT_SKILL_NORMALIZED_STRENGTH_CAP = 4.0

# Education contribution from CGPA.
# Rule requested: 10 CGPA should contribute 5 points.
CGPA_TO_EDUCATION_SCORE_MULTIPLIER = 0.5

# Small context bonus on top of weighted mentions.
HARD_CONTEXT_BONUS = {
    "skills": 0.5,
    "project": 1.0,
    "experience": 2.0,
}

SOFT_CONTEXT_BONUS = {
    "skills": 0.25,
    "project": 0.5,
    "experience": 0.75,
}

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

MONTH_NAME_TO_NUMBER = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

MONTH_PATTERN = (
    r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
EXPERIENCE_SECTION_RE = re.compile(
    r"(?ms)^\s*EXPERIENCE\s*$\s*(?P<body>.*?)(?=^\s*[A-Z][A-Z\s&/\-]{2,}\s*$|\Z)"
)
DATE_RANGE_RE = re.compile(
    rf"(?i)\b{MONTH_PATTERN}\s+(\d{{4}})\s*[-–]\s*(?:{MONTH_PATTERN}\s+(\d{{4}})|Present)\b"
)
YEAR_DIGIT_PLUS_RE = re.compile(r"(?i)\b(\d{1,2})\s*\+\s*(?:years?|yrs?)(?:['’]s?)?\b")
YEAR_DIGIT_PLAIN_RE = re.compile(r"(?i)\b(\d{1,2})\s*(?:years?|yrs?)(?:['’]s?)?\b")
YEAR_WORD_PLAIN_RE = re.compile(
    r"(?i)\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:years?|yrs?)(?:['’]s?)?\b"
)

LEVEL_RANKS = {"entry": 0, "mid": 1, "senior": 2}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine module2 keyword + semantic JSON and score each skill."
    )
    parser.add_argument("keyword_json_pos", nargs="?", type=Path, default=None)
    parser.add_argument("semantic_json_pos", nargs="?", type=Path, default=None)
    parser.add_argument("output_json_pos", nargs="?", type=Path, default=None)
    parser.add_argument("--keyword-json", type=Path, default=None)
    parser.add_argument("--semantic-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


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


def _normalize_context(context: str) -> str:
    c = str(context or "").strip().lower()
    if c in {"projects", "project"}:
        return "project"
    if c in {"experience", "internship", "job", "work", "employment"}:
        return "experience"
    if c in {"skills", "skill"}:
        return "skills"
    if c in {"education"}:
        return "education"
    if c in {"other"}:
        return "other"
    if c in {"general"}:
        return "general"
    return "unknown"


def _clean_contexts(payload: Dict[str, object]) -> List[str]:
    raw = payload.get("contexts")
    if not isinstance(raw, list):
        raw = payload.get("sections", [])
    if not isinstance(raw, list):
        return []
    cleaned = [_normalize_context(item) for item in raw if str(item).strip()]
    return sorted(set(cleaned))


def _dominant_context_for_frequency(contexts: Set[str]) -> str:
    # To reflect repeated project evidence (as in the example), prefer project first.
    if "project" in contexts:
        return "project"
    if "experience" in contexts:
        return "experience"
    if "skills" in contexts:
        return "skills"
    if "education" in contexts:
        return "education"
    if "general" in contexts:
        return "general"
    if "other" in contexts:
        return "other"
    return "unknown"


def _cgpa_to_10_scale(cgpa_payload: object) -> float:
    if not isinstance(cgpa_payload, dict):
        return 0.0

    raw_value = _as_float(cgpa_payload.get("value"), default=0.0)
    raw_scale = str(cgpa_payload.get("scale", "")).strip()

    if raw_value <= 0:
        return 0.0

    if raw_scale == "4":
        return min(10.0, raw_value * 2.5)
    # Default to /10 when unknown or already /10.
    return min(10.0, raw_value)


def _resolve_resume_text_path(semantic_data: Dict[str, object]) -> Optional[Path]:
    meta = semantic_data.get("__meta__")
    if not isinstance(meta, dict):
        return None

    input_path = str(meta.get("input_path", "")).strip()
    if not input_path:
        return None

    candidate = Path(input_path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / input_path
    if candidate.exists():
        return candidate
    return None


def _year_values_from_text(text: str) -> List[int]:
    values: List[int] = []
    for match in YEAR_DIGIT_PLUS_RE.finditer(text):
        values.append(int(match.group(1)))
    for match in YEAR_DIGIT_PLAIN_RE.finditer(text):
        values.append(int(match.group(1)))
    for match in YEAR_WORD_PLAIN_RE.finditer(text):
        word = str(match.group(1)).strip().lower()
        if word in NUMBER_WORDS:
            values.append(NUMBER_WORDS[word])
    return values


def _extract_experience_section(text: str) -> str:
    match = EXPERIENCE_SECTION_RE.search(text or "")
    if not match:
        return ""
    return str(match.group("body") or "").strip()


def _month_name_to_number(value: str) -> Optional[int]:
    if not value:
        return None
    return MONTH_NAME_TO_NUMBER.get(str(value).strip().lower())


def _estimate_experience_months(text: str) -> Tuple[int, List[Tuple[str, str]]]:
    experience_text = _extract_experience_section(text)
    if not experience_text:
        return 0, []

    today = date.today()
    seen_ranges: Set[Tuple[int, int, int, int]] = set()
    labels: List[Tuple[str, str]] = []
    total_months = 0

    for match in DATE_RANGE_RE.finditer(experience_text):
        start_month = _month_name_to_number(match.group(1))
        start_year = _as_int(match.group(2), default=0)
        end_month_text = match.group(3)
        end_year_text = match.group(4)

        if not start_month or start_year <= 0:
            continue

        if end_month_text and end_year_text:
            end_month = _month_name_to_number(end_month_text)
            end_year = _as_int(end_year_text, default=0)
        else:
            end_month = today.month
            end_year = today.year

        if not end_month or end_year <= 0:
            continue

        key = (start_year, start_month, end_year, end_month)
        if key in seen_ranges:
            continue
        seen_ranges.add(key)

        months = ((end_year - start_year) * 12) + (end_month - start_month) + 1
        if months <= 0:
            continue

        total_months += months
        labels.append(
            (
                f"{match.group(1)} {start_year}",
                f"{end_month_text} {end_year}" if end_month_text and end_year_text else "Present",
            )
        )

    return total_months, labels


def _infer_candidate_level_profile(cgpa_payload: object, semantic_data: Dict[str, object]) -> Dict[str, object]:
    resume_text_path = _resolve_resume_text_path(semantic_data)
    raw_text = ""
    if resume_text_path:
        raw_text = resume_text_path.read_text(encoding="utf-8", errors="ignore")

    experience_months, date_ranges = _estimate_experience_months(raw_text)
    year_mentions = _year_values_from_text(raw_text)
    explicit_years = max(year_mentions) if year_mentions else None
    inferred_years = max(float(explicit_years or 0), round(experience_months / 12.0, 2))

    lowered_text = raw_text.lower()
    student_signal = bool(cgpa_payload) or ("bachelor" in lowered_text and "present" in lowered_text)
    internship_signal = "intern" in lowered_text

    if inferred_years >= 5.0:
        candidate_level = "senior"
    elif inferred_years >= 2.0:
        candidate_level = "mid"
    else:
        candidate_level = "entry"

    if student_signal and inferred_years < 2.0:
        candidate_level = "entry"

    reason_parts: List[str] = []
    if student_signal:
        reason_parts.append("education/CGPA signal suggests student or early-career profile")
    if internship_signal:
        reason_parts.append("resume contains internship experience")
    if experience_months > 0:
        reason_parts.append(f"estimated hands-on experience is about {experience_months} months")
    elif explicit_years is not None:
        reason_parts.append(f"resume mentions about {explicit_years} years of experience")
    else:
        reason_parts.append("no explicit multi-year experience signal detected in resume")

    return {
        "resume_text_path": str(resume_text_path) if resume_text_path else None,
        "candidate_level": candidate_level,
        "candidate_level_rank": LEVEL_RANKS.get(candidate_level, 0),
        "explicit_years": explicit_years,
        "estimated_experience_months": experience_months,
        "estimated_experience_years": round(inferred_years, 2),
        "student_signal": student_signal,
        "internship_signal": internship_signal,
        "experience_date_ranges": [
            {"start": start_label, "end": end_label} for start_label, end_label in date_ranges
        ],
        "reason": "; ".join(reason_parts),
    }


def _experience_strength(
    mentions: int,
    contexts: Iterable[str],
    cgpa_payload: object = None,
    *,
    is_soft_skill: bool = False,
) -> Dict[str, float]:
    context_set: Set[str] = set(contexts)
    section_weights = SOFT_CONTEXT_WEIGHTS if is_soft_skill else HARD_CONTEXT_WEIGHTS
    context_bonus_map = SOFT_CONTEXT_BONUS if is_soft_skill else HARD_CONTEXT_BONUS
    section_weight = sum(section_weights.get(ctx, 1.0) for ctx in context_set)

    # Requested formula: frequency_weight = min(5, log2(mentions+1)*2)
    frequency_weight = min(5.0, math.log2(max(0, mentions) + 1) * 2.0)

    context_weight = sum(context_bonus_map.get(ctx, 0.0) for ctx in context_set)

    # Apply CGPA only to hard-skill evidence in education contexts.
    cgpa_10 = _cgpa_to_10_scale(cgpa_payload)
    education_score = 0.0
    if (not is_soft_skill) and "education" in context_set and cgpa_10 > 0:
        education_score = min(5.0, cgpa_10 * CGPA_TO_EDUCATION_SCORE_MULTIPLIER)

    raw_score = section_weight + frequency_weight + context_weight + education_score
    normalized_score = min(10.0, raw_score / 2.0)  # 20 -> 10/10
    soft_skill_cap_applied = False
    if is_soft_skill and normalized_score > SOFT_SKILL_NORMALIZED_STRENGTH_CAP:
        normalized_score = SOFT_SKILL_NORMALIZED_STRENGTH_CAP
        soft_skill_cap_applied = True

    return {
        "section_weight": round(section_weight, 4),
        "frequency_weight": round(frequency_weight, 4),
        "context_weight": round(context_weight, 4),
        "education_score": round(education_score, 4),
        "raw_strength_score": round(raw_score, 4),
        "normalized_strength_score": round(normalized_score, 4),
        "weight_profile": "soft_skill" if is_soft_skill else "hard_skill",
        "soft_skill_cap_applied": soft_skill_cap_applied,
    }


def _merge_skill_payloads(keyword_data: Dict[str, object], semantic_data: Dict[str, object]) -> Dict[str, dict]:
    merged: Dict[str, dict] = {}

    def ingest(source_name: str, payload: Dict[str, object]) -> None:
        for skill, info in payload.items():
            if str(skill).startswith("__"):
                continue
            if not isinstance(info, dict):
                continue

            canonical = normalize_skill_name(str(skill))
            entry = merged.setdefault(
                canonical,
                {
                    "skill": canonical,
                    "mentions": 0,
                    "contexts": set(),
                    "source": set(),
                    "keyword_confidence": 0.0,
                    "semantic_confidence": 0.0,
                    "category": "",
                    "taxonomy_category": "",
                    "sub_category": "",
                },
            )

            entry["mentions"] = max(entry["mentions"], _as_int(info.get("mentions", 0), default=0))
            entry["contexts"].update(_clean_contexts(info))
            entry["source"].add(source_name)
            if source_name == "keyword":
                entry["keyword_confidence"] = max(
                    entry["keyword_confidence"], _as_float(info.get("confidence", 0.0), default=0.0)
                )
            elif source_name == "semantic":
                entry["semantic_confidence"] = max(
                    entry["semantic_confidence"], _as_float(info.get("confidence", 0.0), default=0.0)
                )

            if not entry["category"] and info.get("category"):
                entry["category"] = str(info.get("category"))
            if not entry["taxonomy_category"] and info.get("taxonomy_category"):
                entry["taxonomy_category"] = str(info.get("taxonomy_category"))
            if not entry["sub_category"] and info.get("sub_category"):
                entry["sub_category"] = str(info.get("sub_category"))

    ingest("keyword", keyword_data)
    ingest("semantic", semantic_data)
    return merged


def _build_final_scores(merged: Dict[str, dict], cgpa_payload: object = None) -> Dict[str, object]:
    output: Dict[str, object] = {}
    for skill in sorted(merged.keys()):
        item = merged[skill]
        contexts = sorted(set(item["contexts"]))
        mentions = max(1, int(item["mentions"]))
        keyword_confidence = float(item["keyword_confidence"])
        semantic_confidence = float(item["semantic_confidence"])

        # Confidence fusion requested by user.
        final_confidence = (0.6 * keyword_confidence) + (0.4 * semantic_confidence)
        final_confidence = max(0.0, min(1.0, final_confidence))

        # If present in both -> boost. If not -> lower marks.
        in_both_sources = {"keyword", "semantic"}.issubset(set(item["source"]))
        source_multiplier = 1.15 if in_both_sources else 0.75

        category_raw = str(item.get("category", "") or "").strip().lower()
        is_soft_skill = "soft" in category_raw
        skill_type_multiplier = SOFT_SKILL_MULTIPLIER if is_soft_skill else HARD_SKILL_MULTIPLIER

        strength = _experience_strength(
            mentions=mentions,
            contexts=contexts,
            cgpa_payload=cgpa_payload,
            is_soft_skill=is_soft_skill,
        )
        resulting_score = (
            strength["normalized_strength_score"]
            * source_multiplier
            * skill_type_multiplier
            * (0.5 + (0.5 * final_confidence))
        )
        resulting_score = max(0.0, min(10.0, resulting_score))

        output[skill] = {
            "resulting_score": round(resulting_score, 4),
            "confidence": round(final_confidence, 4),
            "keyword_confidence": round(keyword_confidence, 4),
            "semantic_confidence": round(semantic_confidence, 4),
            "source": sorted(item["source"]),
            "mentions": mentions,
            "contexts": contexts,
            "category": item["category"] or "hard_skill",
            "taxonomy_category": item["taxonomy_category"] or "uncategorized",
            "sub_category": item["sub_category"] or "General",
            "strength_breakdown": strength,
            "verification_multiplier": source_multiplier,
            "skill_type_multiplier": skill_type_multiplier,
        }

    return output


def main() -> None:
    args = _parse_args()
    args.keyword_json = args.keyword_json or args.keyword_json_pos or DEFAULT_KEYWORD_JSON
    args.semantic_json = args.semantic_json or args.semantic_json_pos or DEFAULT_SEMANTIC_JSON
    args.output_json = args.output_json or args.output_json_pos or DEFAULT_OUTPUT_JSON

    keyword_data = _load_json(args.keyword_json)
    semantic_data = _load_json(args.semantic_json)

    merged = _merge_skill_payloads(keyword_data, semantic_data)
    cgpa_payload = semantic_data.get("__cgpa__") or keyword_data.get("__cgpa__")
    combined_scores = _build_final_scores(merged, cgpa_payload=cgpa_payload)
    candidate_level_profile = _infer_candidate_level_profile(cgpa_payload, semantic_data)

    combined_scores["__cgpa__"] = cgpa_payload
    combined_scores["__meta__"] = {
        "keyword_json": str(args.keyword_json),
        "semantic_json": str(args.semantic_json),
        "skills_count": len([k for k in combined_scores.keys() if not k.startswith("__")]),
        "candidate_level_profile": candidate_level_profile,
        "scoring_formula": {
            "confidence_fusion": "0.6*keyword_confidence + 0.4*semantic_confidence",
            "strength_engine": "SkillScore = SectionWeight + FrequencyWeight + ContextWeight",
            "education_engine": "EducationScore = min(5, CGPA_on_10_scale * 0.5) for hard skills when context includes education",
            "normalized_strength": "min(10, raw_strength/2), with a soft-skill normalized-strength cap applied afterward",
            "both_sources_multiplier": 1.15,
            "single_source_multiplier": 0.75,
            "hard_skill_multiplier": HARD_SKILL_MULTIPLIER,
            "soft_skill_multiplier": SOFT_SKILL_MULTIPLIER,
            "section_weights": HARD_CONTEXT_WEIGHTS,
            "soft_skill_section_weights": SOFT_CONTEXT_WEIGHTS,
            "hard_context_bonus": HARD_CONTEXT_BONUS,
            "soft_skill_context_bonus": SOFT_CONTEXT_BONUS,
            "soft_skill_normalized_strength_cap": SOFT_SKILL_NORMALIZED_STRENGTH_CAP,
            "cgpa_application_policy": "CGPA boost applies only to hard skills that appear in education context.",
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(combined_scores, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote combined score JSON: {args.output_json}")
    print(f"Skills scored: {combined_scores['__meta__']['skills_count']}")
    if "python" in combined_scores:
        print(f"python -> {combined_scores['python']}")


if __name__ == "__main__":
    main()
