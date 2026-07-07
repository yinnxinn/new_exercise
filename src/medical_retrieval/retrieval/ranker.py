from __future__ import annotations

from ..core.schema import Entity, MedicalDocument
from ..nlp.ner import ENTITY_WEIGHTS


def field_bonus(query_tokens: list[str], doc: MedicalDocument, config: dict) -> float:
    title = doc.title.lower()
    department = doc.department.lower()
    tags = " ".join(doc.tags).lower()
    score = 0.0

    for token in set(query_tokens):
        if len(token) < 2:
            continue
        if token in title:
            score += config.get("title_hit_bonus", 0.12)
        if token in tags:
            score += config.get("tag_hit_bonus", 0.08)
        if token in department:
            score += config.get("department_hit_bonus", 0.12)

    return min(score, config.get("max_field_bonus", 0.32))


def entity_bonus(query_entities: list[Entity], doc_entities: list[Entity], config: dict) -> tuple[float, list[str]]:
    doc_pairs = {(item.text, item.type) for item in doc_entities}
    score = 0.0
    matched: list[str] = []

    for entity in query_entities:
        pair = (entity.text, entity.type)
        if pair in doc_pairs:
            score += ENTITY_WEIGHTS.get(entity.type, 0.08)
            matched.append(entity.text)

    return min(score, config.get("max_entity_bonus", 0.45)), matched
