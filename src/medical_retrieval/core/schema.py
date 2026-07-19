from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MedicalDocument:
    id: str
    title: str
    department: str
    tags: list[str]
    content: str
    question: str = ""
    answer: str = ""
    source: str = ""
    tag_codes: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return f"{self.title} {self.department} {' '.join(self.tags)} {self.content}"


@dataclass(frozen=True)
class Entity:
    text: str
    type: str
    start: int
    end: int


@dataclass
class SearchResult:
    doc: MedicalDocument
    final_score: float
    vector_score: float
    bm25_score: float
    field_bonus: float
    entity_bonus: float
    matched_entities: list[str] = field(default_factory=list)
