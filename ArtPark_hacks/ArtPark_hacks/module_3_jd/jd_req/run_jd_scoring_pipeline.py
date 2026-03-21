from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


JD_REQ_DIR = Path(__file__).resolve().parent
JD_MODULE_DIR = JD_REQ_DIR.parent
PROJECT_ROOT = JD_MODULE_DIR.parent

for path in (PROJECT_ROOT, JD_REQ_DIR, JD_REQ_DIR / "module2_Keyword"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from module2_Keyword.lay1 import LayerAExtractor  # noqa: E402
from module2_semantic.embedding_utils import get_model_device  # noqa: E402
from module2_semantic.personB_pipeline import run_semantic_pipeline  # noqa: E402
from shared.aliases import normalize_skill_name  # noqa: E402


DEFAULT_INPUT_TEXT = JD_MODULE_DIR / "jd_resulting_text.txt"
DEFAULT_KEYWORD_JSON = JD_REQ_DIR / "module2_Keyword" / "layer_a_keywords.json"
DEFAULT_SEMANTIC_JSON = JD_REQ_DIR / "module2_semantic" / "layer_a_semantic_resume.json"
DEFAULT_COMBINED_JSON = JD_REQ_DIR / "layer_a_combined_scored.json"
DEFAULT_TAXONOMY_JSON = JD_REQ_DIR / "skill_taxonomy_500plus(1).json"

PHRASE_WEIGHTS = {
    "mandatory": 10.0,
    "must have": 10.0,
    "required": 9.0,
    "strong experience in": 8.0,
    "hands-on experience": 8.0,
    "preferred": 6.0,
    "good to have": 5.0,
    "exposure to": 4.0,
    "familiarity with": 3.0,
}
DEFAULT_PHRASE_LABEL = "unspecified"
DEFAULT_PHRASE_WEIGHT = 4.0

CORE_CATEGORY_HINTS = {
    "programming",
    "machine learning",
    "ml",
    "backend",
    "data",
    "cloud",
    "database",
    "ai",
}
SOFT_CATEGORY_HINTS = {
    "soft",
    "leadership",
    "communication",
    "collaboration",
    "management",
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

YEARS_RANGE_RE = re.compile(r"(?i)\b(\d{1,2})\s*[-–]\s*(\d{1,2})\s*(?:\+?\s*)?(?:years?|yrs?)\b")
YEARS_PLUS_RE = re.compile(r"(?i)\b(\d{1,2})\s*\+\s*(?:years?|yrs?)\b")
YEARS_PLAIN_RE = re.compile(r"(?i)\b(\d{1,2})\s*(?:years?|yrs?)\b")
YEARS_WORD_RE = re.compile(
    r"(?i)\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:years?|yrs?)(?:['’]s?)?\b"
)
LEVEL_RANKS = {"entry": 0, "mid": 1, "senior": 2}


@dataclass
class MentionSignal:
    start: int
    end: int
    phrase_label: str
    phrase_weight: float
    position_bonus: float
    local_years: Optional[int]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate JD keyword + semantic JSON and weighted combined scoring.")
    parser.add_argument("--jd-text", type=Path, default=DEFAULT_INPUT_TEXT)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY_JSON)
    parser.add_argument("--keyword-json", type=Path, default=DEFAULT_KEYWORD_JSON)
    parser.add_argument("--semantic-json", type=Path, default=DEFAULT_SEMANTIC_JSON)
    parser.add_argument("--combined-json", type=Path, default=DEFAULT_COMBINED_JSON)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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
    if c in {"experience", "internship", "employment", "job", "work"}:
        return "experience"
    if c in {"skills", "skill", "technical skills"}:
        return "skills"
    if c in {"education"}:
        return "education"
    if c in {"other"}:
        return "other"
    return "general"


def _compile_term_pattern(term: str) -> re.Pattern:
    escaped = re.escape(term)
    if re.fullmatch(r"[a-z0-9]+", term):
        return re.compile(rf"\b{escaped}\b", re.IGNORECASE)
    return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)


def _load_taxonomy_maps(taxonomy_path: Path) -> Tuple[Dict[str, Set[str]], Dict[str, str], Dict[str, str]]:
    payload = _load_json(taxonomy_path)
    canonical_to_terms: Dict[str, Set[str]] = {}
    category_map: Dict[str, str] = {}
    sub_category_map: Dict[str, str] = {}

    root_aliases = payload.get("aliases", {})
    if isinstance(root_aliases, dict):
        for alias, canonical in root_aliases.items():
            canonical_norm = normalize_skill_name(str(canonical))
            alias_norm = normalize_skill_name(str(alias))
            if not canonical_norm:
                continue
            canonical_to_terms.setdefault(canonical_norm, {canonical_norm})
            if alias_norm:
                canonical_to_terms[canonical_norm].add(alias_norm)

    for skill_entry in payload.get("skills", []):
        if not isinstance(skill_entry, dict):
            continue
        canonical = normalize_skill_name(skill_entry.get("canonical_skill", ""))
        if not canonical:
            continue
        canonical_to_terms.setdefault(canonical, {canonical})
        category_map[canonical] = str(skill_entry.get("category", "uncategorized"))
        sub_category_map[canonical] = str(
            skill_entry.get("sub_category") or skill_entry.get("subcategory") or "General"
        )
        for alias in skill_entry.get("aliases", []) or []:
            alias_norm = normalize_skill_name(str(alias))
            if alias_norm:
                canonical_to_terms[canonical].add(alias_norm)

    return canonical_to_terms, category_map, sub_category_map


def _extract_keyword_json(jd_text: str, taxonomy_path: Path) -> Dict[str, object]:
    extractor = LayerAExtractor(str(taxonomy_path))
    return extractor.run(jd_text)


def _extract_semantic_json(
    jd_text: str,
    keyword_json: Dict[str, object],
    category_map: Dict[str, str],
    sub_category_map: Dict[str, str],
    input_path: Path,
) -> Dict[str, object]:
    keyword_for_semantic: Dict[str, dict] = {}
    for skill, payload in keyword_json.items():
        if str(skill).startswith("__") or not isinstance(payload, dict):
            continue
        canonical = normalize_skill_name(skill)
        contexts = payload.get("contexts", [])
        if not isinstance(contexts, list):
            contexts = []
        keyword_for_semantic[canonical] = {
            "keyword_score": _as_float(payload.get("confidence", 0.0), default=0.0),
            "mentions": max(1, _as_int(payload.get("mentions", 1), default=1)),
            "contexts": [_normalize_context(c) for c in contexts],
        }

    fused_output = run_semantic_pipeline(jd_text, keyword_for_semantic)

    semantic_json: Dict[str, object] = {}
    all_skills = sorted(set(keyword_for_semantic.keys()) | set(fused_output.keys()))
    for skill in all_skills:
        canonical = normalize_skill_name(skill)
        fused = fused_output.get(canonical, fused_output.get(skill, {}))
        kw = keyword_for_semantic.get(canonical, {})

        contexts = kw.get("contexts", [])
        if not contexts:
            contexts = ["general"]
        mentions = max(1, _as_int(kw.get("mentions", 1), default=1))

        category_name = category_map.get(canonical, "uncategorized")
        sub_category_name = sub_category_map.get(canonical, "General")
        is_soft = "soft" in category_name.lower()
        semantic_json[canonical] = {
            "confidence": _as_float(fused.get("confidence", kw.get("keyword_score", 0.0)), default=0.0),
            "source": list(fused.get("source", ["keyword"] if kw else ["semantic"])),
            "mentions": mentions,
            "contexts": contexts,
            "category": "soft_skill" if is_soft else "hard_skill",
            "taxonomy_category": category_name,
            "sub_category": sub_category_name,
            "sections": contexts,
        }

    semantic_json["__meta__"] = {
        "input_path": str(input_path),
        "semantic_device": get_model_device(),
        "skills_count": len(all_skills),
    }
    return semantic_json


def _find_global_years(text_lower: str) -> Optional[int]:
    values: List[int] = []
    for match in YEARS_RANGE_RE.finditer(text_lower):
        values.append(max(int(match.group(1)), int(match.group(2))))
    for match in YEARS_PLUS_RE.finditer(text_lower):
        values.append(int(match.group(1)))
    for match in YEARS_PLAIN_RE.finditer(text_lower):
        values.append(int(match.group(1)))
    for match in YEARS_WORD_RE.finditer(text_lower):
        word = str(match.group(1)).strip().lower()
        if word in NUMBER_WORDS:
            values.append(NUMBER_WORDS[word])
    return max(values) if values else None


def _years_to_multiplier(years: Optional[int]) -> float:
    if years is None:
        return 1.0
    if years <= 1:
        return 1.0
    if years <= 3:
        return 1.2
    if years <= 5:
        return 1.4
    return 1.6


def _years_to_level(years: Optional[int]) -> str:
    if years is None or years <= 1:
        return "entry"
    if years <= 3:
        return "mid"
    return "senior"


def _skill_type_multiplier(category: str, skill_class: str) -> float:
    category_low = str(category or "").lower()
    skill_class_low = str(skill_class or "").lower()
    if "soft" in skill_class_low or any(hint in category_low for hint in SOFT_CATEGORY_HINTS):
        return 0.7
    if any(hint in category_low for hint in CORE_CATEGORY_HINTS):
        return 1.2
    return 1.0


def _position_bonus(start: int, total_len: int) -> float:
    if total_len <= 0:
        return 0.0
    ratio = start / total_len
    if ratio <= 0.25:
        return 2.0
    if ratio <= 0.75:
        return 1.0
    return 0.0


def _detect_phrase_near_span(text_lower: str, start: int, end: int) -> Tuple[str, float]:
    window_start = max(0, start - 120)
    window_end = min(len(text_lower), end + 120)
    snippet = text_lower[window_start:window_end]

    best_label = DEFAULT_PHRASE_LABEL
    best_weight = DEFAULT_PHRASE_WEIGHT
    best_distance = 10**9

    for phrase, weight in PHRASE_WEIGHTS.items():
        for match in re.finditer(re.escape(phrase), snippet):
            abs_pos = window_start + match.start()
            distance = abs(abs_pos - start)
            if distance < best_distance or (distance == best_distance and weight > best_weight):
                best_distance = distance
                best_label = phrase
                best_weight = weight

    return best_label, best_weight


def _extract_local_years(text_lower: str, start: int, end: int) -> Optional[int]:
    window_start = max(0, start - 160)
    window_end = min(len(text_lower), end + 160)
    snippet = text_lower[window_start:window_end]
    target_position = start - window_start
    candidates: List[Tuple[int, int]] = []

    for match in YEARS_RANGE_RE.finditer(snippet):
        years_value = max(int(match.group(1)), int(match.group(2)))
        distance = abs(match.start() - target_position)
        candidates.append((distance, years_value))
    for match in YEARS_PLUS_RE.finditer(snippet):
        years_value = int(match.group(1))
        distance = abs(match.start() - target_position)
        candidates.append((distance, years_value))
    for match in YEARS_PLAIN_RE.finditer(snippet):
        years_value = int(match.group(1))
        distance = abs(match.start() - target_position)
        candidates.append((distance, years_value))
    for match in YEARS_WORD_RE.finditer(snippet):
        word = str(match.group(1)).strip().lower()
        if word not in NUMBER_WORDS:
            continue
        years_value = NUMBER_WORDS[word]
        distance = abs(match.start() - target_position)
        candidates.append((distance, years_value))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][1]


def _collect_mentions_for_skill(text: str, terms: Iterable[str]) -> List[Tuple[int, int]]:
    spans: Set[Tuple[int, int]] = set()
    for term in sorted(set(terms)):
        if len(term) < 2:
            continue
        pattern = _compile_term_pattern(term)
        for match in pattern.finditer(text):
            spans.add((match.start(), match.end()))
    return sorted(spans)


def _detect_mention_signals(text: str, spans: List[Tuple[int, int]]) -> List[MentionSignal]:
    text_lower = text.lower()
    total_len = len(text)
    signals: List[MentionSignal] = []
    for start, end in spans:
        phrase_label, phrase_weight = _detect_phrase_near_span(text_lower, start, end)
        signals.append(
            MentionSignal(
                start=start,
                end=end,
                phrase_label=phrase_label,
                phrase_weight=phrase_weight,
                position_bonus=_position_bonus(start, total_len),
                local_years=_extract_local_years(text_lower, start, end),
            )
        )
    return signals


def _choose_phrase(signals: List[MentionSignal]) -> Tuple[str, float]:
    if not signals:
        return DEFAULT_PHRASE_LABEL, DEFAULT_PHRASE_WEIGHT
    ordered = sorted(signals, key=lambda s: (s.phrase_weight, s.position_bonus), reverse=True)
    return ordered[0].phrase_label, ordered[0].phrase_weight


def _combine_keyword_semantic_for_jd(
    jd_text: str,
    keyword_json: Dict[str, object],
    semantic_json: Dict[str, object],
    canonical_to_terms: Dict[str, Set[str]],
    category_map: Dict[str, str],
    sub_category_map: Dict[str, str],
    input_text_path: Path,
) -> Dict[str, object]:
    merged: Dict[str, dict] = {}

    def ingest(source_name: str, payload: Dict[str, object]) -> None:
        for skill, info in payload.items():
            if str(skill).startswith("__") or not isinstance(info, dict):
                continue
            canonical = normalize_skill_name(skill)
            entry = merged.setdefault(
                canonical,
                {
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
            contexts = info.get("contexts", info.get("sections", []))
            if isinstance(contexts, list):
                entry["contexts"].update(_normalize_context(c) for c in contexts)
            confidence = _as_float(info.get("confidence", 0.0), default=0.0)
            if source_name == "keyword":
                entry["source"].add("keyword")
                entry["keyword_confidence"] = max(entry["keyword_confidence"], confidence)
            else:
                declared_sources = info.get("source", [])
                if not isinstance(declared_sources, list):
                    declared_sources = []
                declared = {str(s).strip().lower() for s in declared_sources}

                if "semantic" in declared:
                    entry["source"].add("semantic")
                    entry["semantic_confidence"] = max(entry["semantic_confidence"], confidence)

                if "keyword" in declared and "keyword" not in entry["source"]:
                    entry["source"].add("keyword")
                    entry["keyword_confidence"] = max(entry["keyword_confidence"], confidence)
            if not entry["category"] and info.get("category"):
                entry["category"] = str(info.get("category"))
            if not entry["taxonomy_category"] and info.get("taxonomy_category"):
                entry["taxonomy_category"] = str(info.get("taxonomy_category"))
            if not entry["sub_category"] and info.get("sub_category"):
                entry["sub_category"] = str(info.get("sub_category"))

    ingest("keyword", keyword_json)
    ingest("semantic", semantic_json)

    text_lower = jd_text.lower()
    global_years = _find_global_years(text_lower)
    jd_level = _years_to_level(global_years)
    output: Dict[str, object] = {}

    for skill in sorted(merged.keys()):
        item = merged[skill]
        terms = canonical_to_terms.get(skill, {skill})
        mention_spans = _collect_mentions_for_skill(jd_text, terms)
        mention_signals = _detect_mention_signals(jd_text, mention_spans)

        mention_count = max(item["mentions"], len(mention_spans), 1)
        phrase_label, phrase_weight = _choose_phrase(mention_signals)
        position_bonus = max((signal.position_bonus for signal in mention_signals), default=0.0)
        frequency_bonus = min(3.0, math.log2(mention_count + 1))

        local_years = max((signal.local_years or 0 for signal in mention_signals), default=0) or None
        effective_years = local_years if local_years is not None else global_years
        experience_modifier = _years_to_multiplier(effective_years)

        keyword_confidence = float(item["keyword_confidence"])
        semantic_confidence = float(item["semantic_confidence"])
        confidence = max(0.0, min(1.0, (0.6 * keyword_confidence) + (0.4 * semantic_confidence)))

        skill_category = item["taxonomy_category"] or category_map.get(skill, "uncategorized")
        skill_sub_category = item["sub_category"] or sub_category_map.get(skill, "General")
        skill_class = item["category"] or "hard_skill"
        type_multiplier = _skill_type_multiplier(skill_category, skill_class)

        both_sources = {"keyword", "semantic"}.issubset(set(item["source"]))
        source_multiplier = 1.15 if both_sources else 0.8

        raw_weight = (phrase_weight + position_bonus + frequency_bonus) * experience_modifier * type_multiplier
        verified_weight = raw_weight * source_multiplier * (0.6 + 0.4 * confidence)
        normalized_weight = 10.0 * math.tanh(verified_weight / 10.0)

        output[skill] = {
            "weight": round(normalized_weight, 4),
            "raw_weight": round(verified_weight, 4),
            "phrase": phrase_label,
            "phrase_weight": round(phrase_weight, 4),
            "position_bonus": round(position_bonus, 4),
            "frequency_bonus": round(frequency_bonus, 4),
            "mentions": mention_count,
            "experience_years": effective_years,
            "experience_modifier": round(experience_modifier, 4),
            "confidence": round(confidence, 4),
            "source": sorted(item["source"]),
            "contexts": sorted(set(item["contexts"])) or ["general"],
            "category": skill_class,
            "taxonomy_category": skill_category,
            "sub_category": skill_sub_category,
            "skill_type_multiplier": round(type_multiplier, 4),
            "source_multiplier": round(source_multiplier, 4),
        }

    output["__meta__"] = {
        "input_text": str(input_text_path),
        "skills_count": len([k for k in output.keys() if not k.startswith("__")]),
        "formula": "((phrase_weight + position_bonus + frequency_bonus) * experience_modifier * skill_type_multiplier) * source_multiplier * confidence_factor",
        "confidence_fusion": "0.6*keyword_confidence + 0.4*semantic_confidence",
        "normalization": "10 * tanh(weight/10)",
        "phrase_weights": PHRASE_WEIGHTS,
        "jd_level_profile": {
            "global_experience_years": global_years,
            "jd_level": jd_level,
            "jd_level_rank": LEVEL_RANKS.get(jd_level, 0),
            "reason": (
                f"Detected about {global_years} years of experience in JD text."
                if global_years is not None
                else "No explicit years-of-experience signal detected in JD text."
            ),
        },
    }
    return output


def main() -> None:
    args = _parse_args()
    if not args.jd_text.exists():
        raise FileNotFoundError(f"JD text file not found: {args.jd_text}")
    if not args.taxonomy.exists():
        raise FileNotFoundError(f"Taxonomy file not found: {args.taxonomy}")

    if args.device in {"cpu", "cuda"}:
        import os

        os.environ["MODULE2_SEMANTIC_DEVICE"] = args.device

    jd_text = args.jd_text.read_text(encoding="utf-8", errors="ignore")
    canonical_to_terms, category_map, sub_category_map = _load_taxonomy_maps(args.taxonomy)

    keyword_json = _extract_keyword_json(jd_text, args.taxonomy)
    _write_json(args.keyword_json, keyword_json)

    semantic_json = _extract_semantic_json(jd_text, keyword_json, category_map, sub_category_map, args.jd_text)
    _write_json(args.semantic_json, semantic_json)

    combined_json = _combine_keyword_semantic_for_jd(
        jd_text=jd_text,
        keyword_json=keyword_json,
        semantic_json=semantic_json,
        canonical_to_terms=canonical_to_terms,
        category_map=category_map,
        sub_category_map=sub_category_map,
        input_text_path=args.jd_text,
    )
    _write_json(args.combined_json, combined_json)

    print(f"Keyword JSON: {args.keyword_json}")
    print(f"Semantic JSON: {args.semantic_json}")
    print(f"Combined JSON: {args.combined_json}")
    print(f"Semantic device: {get_model_device()}")
    if "python" in combined_json:
        print(f"python => {combined_json['python']}")


if __name__ == "__main__":
    main()
