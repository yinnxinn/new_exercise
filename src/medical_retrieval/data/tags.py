from __future__ import annotations

import re
from dataclasses import dataclass

from ..nlp.text import tokenize


TAG_TYPE_ALIASES = {
    "department": "dept",
    "category": "cat",
    "source": "src",
    "entity": "ent",
    "keyword": "kw",
    "quality": "qlt",
}


@dataclass(frozen=True)
class DocumentTag:
    code: str
    name: str
    tag_type: str
    source: str = "auto"
    confidence: float = 1.0


def normalize_tag_name(value: str) -> str:
    value = re.sub(r"\s+", "", str(value).strip().lower())
    value = value.replace("，", ",").replace("、", ",")
    return value.strip(",;；|")


def make_tag_code(tag_type: str, name: str) -> str:
    tag_type = TAG_TYPE_ALIASES.get(tag_type, tag_type)
    normalized = normalize_tag_name(name)
    normalized = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff_+-]+", "_", normalized)
    normalized = normalized.strip("_")
    if not normalized:
        normalized = "unknown"
    return f"{tag_type}:{normalized}"


def unique_tags(tags: list[DocumentTag]) -> list[DocumentTag]:
    seen: set[str] = set()
    kept: list[DocumentTag] = []
    for tag in tags:
        if tag.code in seen:
            continue
        seen.add(tag.code)
        kept.append(tag)
    return kept


def extract_keyword_tags(text: str, limit: int = 12) -> list[DocumentTag]:
    counts: dict[str, int] = {}
    for token in tokenize(text):
        if len(token) < 2:
            continue
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))[:limit]
    return [
        DocumentTag(
            code=make_tag_code("keyword", token),
            name=token,
            tag_type="keyword",
            source="tokenizer",
            confidence=min(1.0, 0.4 + count * 0.1),
        )
        for token, count in ranked
    ]


def build_dialogue_tags(
    *,
    department: str,
    source_category: str,
    title: str,
    question: str,
    answer: str,
    extra_tags: list[str] | None = None,
    keyword_limit: int = 12,
) -> list[DocumentTag]:
    tags: list[DocumentTag] = []
    if department:
        tags.append(DocumentTag(make_tag_code("department", department), department, "department", "source"))
    if source_category:
        tags.append(DocumentTag(make_tag_code("category", source_category), source_category, "category", "source"))

    for value in extra_tags or []:
        value = value.strip()
        if value:
            tags.append(DocumentTag(make_tag_code("entity", value), value, "entity", "manual"))

    keyword_text = " ".join(part for part in [title, question, answer] if part)
    tags.extend(extract_keyword_tags(keyword_text, limit=keyword_limit))
    return unique_tags(tags)
