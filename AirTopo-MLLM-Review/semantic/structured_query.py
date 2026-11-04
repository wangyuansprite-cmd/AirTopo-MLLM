#!/usr/bin/env python3
"""Deterministic free-form anatomical query parser for frozen ATM22 targets.

The valid target registry and canonical prompts are loaded from the generated
``language_queries/prompt_templates.csv`` artifact.  This module creates no
embedding and does not load a model; it only maps traceable phrases to the
four attributes that exist in the frozen data schema.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parent
QUERY_RESOURCES = REPOSITORY_ROOT / "resources" / "language_queries"
DEFAULT_TEMPLATE_REGISTRY = QUERY_RESOURCES / "prompt_templates.csv"
DEFAULT_VOCABULARY = QUERY_RESOURCES / "prompt_vocabulary.json"
SLOTS = ("target", "laterality", "extent", "generation_range")


class QueryParserError(RuntimeError):
    """Raised when the authoritative parser schema is unavailable or malformed."""


@dataclass(frozen=True)
class PhraseMatch:
    phrase: str
    normalized_phrase: str
    slot: str
    value: str
    start: int
    end: int
    source: str


@dataclass
class ParsedAnatomicalQuery:
    raw_text: str
    normalized_text: str
    explicit_attributes: Dict[str, Optional[str]]
    attributes: Dict[str, Optional[str]]
    matched_phrases: List[Dict[str, Any]]
    target_id: Optional[str]
    canonical_prompt: Optional[str]
    canonical_template_id: Optional[str]
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    schema_source: str = ""
    vocabulary_source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_text(text: str) -> str:
    """Apply only deterministic orthographic normalization."""
    if not isinstance(text, str):
        raise TypeError("query text must be a string")
    value = unicodedata.normalize("NFKC", text).lower().strip()
    value = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212]", "-", value)
    # Treat hyphen/underscore/slash variants as token separators.  No semantic
    # terms are added or removed here.
    value = re.sub(r"[-_/]+", " ", value)
    value = re.sub(r"[^a-z0-9\s']+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise QueryParserError(f"vocabulary JSON not found: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise QueryParserError(f"expected JSON object: {path}")
    return value


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    tokens = normalize_text(phrase).split()
    if not tokens:
        raise QueryParserError("empty phrase in synonym lexicon")
    body = r"\s+".join(re.escape(token) for token in tokens)
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])")


class AnatomicalQueryParser:
    """Rule-based parser whose validity is defined by the frozen registry."""

    def __init__(
        self,
        template_registry: Path = DEFAULT_TEMPLATE_REGISTRY,
        vocabulary_path: Path = DEFAULT_VOCABULARY,
    ) -> None:
        self.template_registry = Path(template_registry).resolve()
        self.vocabulary_path = Path(vocabulary_path).resolve()
        self._load_registry()
        self._load_lexicon()

    def _load_registry(self) -> None:
        if not self.template_registry.is_file():
            raise QueryParserError(
                f"prompt template registry not found: {self.template_registry}"
            )
        with self.template_registry.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "target_id", "template_id", "prompt_text", "target",
                "laterality", "extent", "generation_range",
            }
            missing = sorted(required - set(reader.fieldnames or []))
            if missing:
                raise QueryParserError(f"template registry missing fields: {missing}")
            rows = list(reader)
        if not rows:
            raise QueryParserError("empty prompt template registry")

        grouped: Dict[str, List[Dict[str, str]]] = {}
        for row in rows:
            grouped.setdefault(row["target_id"], []).append(row)
        schema: Dict[str, Dict[str, str]] = {}
        canonical: Dict[str, Tuple[str, str]] = {}
        normalized_templates: Dict[str, Dict[str, str]] = {}
        tuple_to_target: Dict[Tuple[str, str, str, str], str] = {}
        for target_id, target_rows in grouped.items():
            attribute_records = {
                tuple(row[slot] for slot in SLOTS) for row in target_rows
            }
            if len(attribute_records) != 1:
                raise QueryParserError(
                    f"inconsistent attributes for target_id={target_id}"
                )
            attributes = dict(zip(SLOTS, next(iter(attribute_records))))
            key = tuple(attributes[slot] for slot in SLOTS)
            if key in tuple_to_target:
                raise QueryParserError(
                    f"duplicate schema tuple for {target_id} and {tuple_to_target[key]}"
                )
            tuple_to_target[key] = target_id
            schema[target_id] = attributes
            first = sorted(target_rows, key=lambda row: row["template_id"])[0]
            canonical[target_id] = (first["template_id"], first["prompt_text"])
            for row in target_rows:
                normalized = normalize_text(row["prompt_text"])
                if normalized in normalized_templates:
                    raise QueryParserError(
                        f"normalized prompt collision: {row['prompt_text']}"
                    )
                normalized_templates[normalized] = {
                    "target_id": target_id,
                    "template_id": row["template_id"],
                    "prompt_text": row["prompt_text"],
                }

        if len(schema) != 28 or len(rows) != 168:
            raise QueryParserError(
                f"expected frozen 28-target/168-template registry, got "
                f"{len(schema)} targets/{len(rows)} templates"
            )
        self.template_rows = rows
        self.schema_by_target = schema
        self.target_by_attributes = tuple_to_target
        self.canonical_by_target = canonical
        self.template_by_normalized_text = normalized_templates

    def _load_lexicon(self) -> None:
        source = _read_json(self.vocabulary_path)
        for slot in SLOTS:
            if slot not in source or not isinstance(source[slot], dict):
                raise QueryParserError(f"vocabulary missing slot={slot}")

        lexicon: Dict[str, Dict[str, set[str]]] = {
            slot: {
                value: set(map(str, phrases))
                for value, phrases in source[slot].items()
                if value != "none"
            }
            for slot in SLOTS
        }
        # Safe, explicit surface-form extensions.  They map only to values that
        # already exist in the frozen schema and always canonicalize back to a
        # registered t01 prompt.
        additions: Dict[str, Dict[str, Sequence[str]]] = {
            "target": {
                "airway": (
                    "airway tree", "entire airway tree", "complete airway tree",
                    "whole airway", "full airway tree", "all airways",
                    "complete airway lumen", "bronchial tree", "bronchial branches",
                    "small distal branches", "distal branches",
                ),
                "trachea": ("windpipe", "tracheal lumen"),
            },
            "laterality": {
                "left": ("left side", "left sided", "left lung"),
                "right": ("right side", "right sided", "right lung"),
            },
            "extent": {
                "central": (
                    "proximal airway", "proximal airways", "proximal branches",
                    "main airway", "main airways", "large airway", "large airways",
                ),
                "peripheral": (
                    "distal airway", "distal airways", "distal airway branches",
                    "small distal branches", "distal branches", "outer airway",
                    "outer airways", "small airway", "small airways",
                ),
            },
            "generation_range": {
                "low": ("early generation", "early generations", "low generation", "low generations", "proximal generation"),
                "mid": ("mid generation", "mid generations", "intermediate generation", "intermediate generations"),
                "high": ("high generation", "high generations", "late generation", "late generations", "distal generation", "distal generations"),
            },
        }
        for slot, values in additions.items():
            for value, phrases in values.items():
                if value not in lexicon[slot]:
                    raise QueryParserError(
                        f"parser extension references unknown {slot}={value}"
                    )
                lexicon[slot][value].update(phrases)

        # Normalize all variants once and compile longest phrases first.
        entries: Dict[str, List[Tuple[str, str, str, re.Pattern[str]]]] = {}
        for slot, values in lexicon.items():
            slot_entries: List[Tuple[str, str, str, re.Pattern[str]]] = []
            for value, phrases in values.items():
                for phrase in phrases:
                    normalized = normalize_text(phrase)
                    slot_entries.append(
                        (value, str(phrase), normalized, _phrase_pattern(normalized))
                    )
            slot_entries.sort(key=lambda item: (-len(item[2]), item[2], item[0]))
            entries[slot] = slot_entries
        self.lexicon = lexicon
        self.lexicon_entries = entries
        self.generation_range_definition = source.get(
            "generation_range_definition", {}
        )

    def _match_slot(self, normalized_text: str, slot: str) -> List[PhraseMatch]:
        candidates: List[PhraseMatch] = []
        for value, source_phrase, normalized_phrase, pattern in self.lexicon_entries[slot]:
            for match in pattern.finditer(normalized_text):
                candidates.append(
                    PhraseMatch(
                        phrase=normalized_text[match.start():match.end()],
                        normalized_phrase=normalized_phrase,
                        slot=slot,
                        value=value,
                        start=match.start(),
                        end=match.end(),
                        source="synonym_lexicon",
                    )
                )
        # Longest-phrase-first suppression occurs only within a slot.  Cross-slot
        # overlap is meaningful (e.g. "left peripheral airway").
        candidates.sort(
            key=lambda item: (-(item.end - item.start), item.start, item.value)
        )
        selected: List[PhraseMatch] = []
        for candidate in candidates:
            if any(
                candidate.start < existing.end and existing.start < candidate.end
                for existing in selected
            ):
                continue
            selected.append(candidate)
        return sorted(selected, key=lambda item: (item.start, item.end, item.value))

    def parse(self, raw_text: str) -> ParsedAnatomicalQuery:
        normalized = normalize_text(raw_text)
        empty_attributes = {slot: None for slot in SLOTS}
        if not normalized:
            return ParsedAnatomicalQuery(
                raw_text=raw_text,
                normalized_text=normalized,
                explicit_attributes=empty_attributes.copy(),
                attributes=empty_attributes.copy(),
                matched_phrases=[],
                target_id=None,
                canonical_prompt=None,
                canonical_template_id=None,
                valid=False,
                errors=["empty query"],
                schema_source=str(self.template_registry),
                vocabulary_source=str(self.vocabulary_path),
            )

        # An exact registered prompt is authoritative and avoids trying to infer
        # intent from fragments of wording already validated by the data builder.
        exact = self.template_by_normalized_text.get(normalized)
        if exact is not None:
            target_id = exact["target_id"]
            attributes = self.schema_by_target[target_id].copy()
            return ParsedAnatomicalQuery(
                raw_text=raw_text,
                normalized_text=normalized,
                explicit_attributes=attributes.copy(),
                attributes=attributes,
                matched_phrases=[
                    {
                        "phrase": normalized,
                        "normalized_phrase": normalized,
                        "slot": "canonical_template",
                        "value": target_id,
                        "start": 0,
                        "end": len(normalized),
                        "source": "prompt_templates.csv",
                    }
                ],
                target_id=target_id,
                canonical_prompt=self.canonical_by_target[target_id][1],
                canonical_template_id=self.canonical_by_target[target_id][0],
                valid=True,
                schema_source=str(self.template_registry),
                vocabulary_source=str(self.vocabulary_path),
            )

        matches: List[PhraseMatch] = []
        explicit: Dict[str, Optional[str]] = {}
        errors: List[str] = []
        warnings: List[str] = []
        for slot in SLOTS:
            slot_matches = self._match_slot(normalized, slot)
            matches.extend(slot_matches)
            values = sorted({item.value for item in slot_matches})
            if len(values) > 1:
                explicit[slot] = None
                errors.append(
                    f"conflicting values for {slot}: {', '.join(values)}"
                )
            elif values:
                explicit[slot] = values[0]
            else:
                explicit[slot] = None

        resolved = explicit.copy()
        target_value = explicit["target"]
        if target_value is None and not any(
            message.startswith("conflicting values for target") for message in errors
        ):
            errors.append("missing target concept (airway or trachea)")

        if target_value == "trachea":
            trachea_schema = self.schema_by_target.get("trachea")
            if trachea_schema is None:
                errors.append("trachea is absent from the target registry")
            else:
                for slot in ("laterality", "extent", "generation_range"):
                    expected = trachea_schema[slot]
                    if explicit[slot] is None:
                        resolved[slot] = expected
                    elif explicit[slot] != expected:
                        # Keep the explicit value so normal tuple validation
                        # rejects the unsupported composition traceably.
                        resolved[slot] = explicit[slot]
                warnings.append(
                    "trachea implies laterality=none, extent=central, and "
                    "generation_range=low according to the frozen registry"
                )
        elif target_value == "airway":
            for slot in ("laterality", "extent", "generation_range"):
                if resolved[slot] is None:
                    resolved[slot] = "none"

        target_id: Optional[str] = None
        canonical_prompt: Optional[str] = None
        canonical_template_id: Optional[str] = None
        if not errors and all(resolved[slot] is not None for slot in SLOTS):
            key = tuple(str(resolved[slot]) for slot in SLOTS)
            target_id = self.target_by_attributes.get(key)
            if target_id is None:
                errors.append(
                    "unsupported anatomical composition: "
                    + ", ".join(f"{slot}={resolved[slot]}" for slot in SLOTS)
                )
            else:
                canonical_template_id, canonical_prompt = self.canonical_by_target[target_id]

        return ParsedAnatomicalQuery(
            raw_text=raw_text,
            normalized_text=normalized,
            explicit_attributes=explicit,
            attributes=resolved,
            matched_phrases=[asdict(item) for item in sorted(matches, key=lambda x: (x.start, x.slot, x.end))],
            target_id=target_id,
            canonical_prompt=canonical_prompt,
            canonical_template_id=canonical_template_id,
            valid=not errors and target_id is not None,
            errors=errors,
            warnings=warnings,
            schema_source=str(self.template_registry),
            vocabulary_source=str(self.vocabulary_path),
        )

    def canonical_prompt(self, target_id: str) -> str:
        if target_id not in self.canonical_by_target:
            raise QueryParserError(f"unknown target_id={target_id!r}")
        return self.canonical_by_target[target_id][1]

    def target_schema(self) -> Dict[str, Dict[str, str]]:
        return {key: value.copy() for key, value in self.schema_by_target.items()}


def extract_attributes(
    text: str, parser: Optional[AnatomicalQueryParser] = None
) -> Dict[str, Optional[str]]:
    return (parser or AnatomicalQueryParser()).parse(text).explicit_attributes


def validate_attributes(
    attributes: Mapping[str, str], parser: Optional[AnatomicalQueryParser] = None
) -> Tuple[bool, Optional[str]]:
    engine = parser or AnatomicalQueryParser()
    if set(attributes) != set(SLOTS):
        return False, "attributes must contain exactly: " + ", ".join(SLOTS)
    key = tuple(str(attributes[slot]) for slot in SLOTS)
    target_id = engine.target_by_attributes.get(key)
    return target_id is not None, target_id


def resolve_target(
    attributes: Mapping[str, str], parser: Optional[AnatomicalQueryParser] = None
) -> Optional[str]:
    valid, target_id = validate_attributes(attributes, parser)
    return target_id if valid else None


def build_canonical_query(
    target_id: str, parser: Optional[AnatomicalQueryParser] = None
) -> str:
    return (parser or AnatomicalQueryParser()).canonical_prompt(target_id)


def parse_free_text_query(
    text: str,
    template_registry: Path = DEFAULT_TEMPLATE_REGISTRY,
    vocabulary_path: Path = DEFAULT_VOCABULARY,
) -> ParsedAnatomicalQuery:
    return AnatomicalQueryParser(template_registry, vocabulary_path).parse(text)


def format_inference_trace(result: ParsedAnatomicalQuery) -> str:
    return "\n".join(
        [
            f"Raw query: {result.raw_text}",
            f"Normalized query: {result.normalized_text}",
            "Parsed attributes: " + json.dumps(result.attributes, sort_keys=True),
            f"Resolved target: {result.target_id}",
            f"Canonical prompt: {result.canonical_prompt}",
            f"Validation: {'PASS' if result.valid else 'FAIL'}",
            "Matched phrases: " + json.dumps(result.matched_phrases, ensure_ascii=False),
            "Errors: " + json.dumps(result.errors, ensure_ascii=False),
            "Warnings: " + json.dumps(result.warnings, ensure_ascii=False),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse an ATM22 free-form anatomical query without loading a model."
    )
    parser.add_argument("--text", required=True)
    parser.add_argument("--template-registry", type=Path, default=DEFAULT_TEMPLATE_REGISTRY)
    parser.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCABULARY)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = parse_free_text_query(args.text, args.template_registry, args.vocabulary)
    payload = result.to_dict()
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(args.output)
    return 0 if result.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
