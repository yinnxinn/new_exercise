from __future__ import annotations

from dataclasses import dataclass, field

from ..core.schema import Entity
from ..nlp.ner import RuleMedicalNER
from ..nlp.text import tokenize

INTENT_KEYWORDS = {
    "department": ["挂什么科", "哪个科", "科室", "就诊"],
    "check": ["检查", "查什么", "化验", "指标"],
    "drug": ["吃什么药", "用药", "药", "治疗"],
    "disease": ["什么病", "是不是", "可能是", "诊断"],
    "symptom": ["症状", "表现"],
}

EXPANSION_RULES = {
    "甲亢": ["甲状腺功能亢进症", "TSH", "FT3", "FT4", "内分泌科"],
    "肺部感染": ["肺炎", "咳嗽", "发热", "胸部影像"],
    "尿路感染": ["尿频", "尿急", "尿痛", "尿常规", "泌尿外科"],
    "类风湿": ["类风湿关节炎", "晨僵", "抗CCP抗体", "风湿免疫科"],
}


@dataclass(frozen=True)
class QueryAnalysis:
    raw_query: str
    normalized_query: str
    tokens: list[str]
    entities: list[Entity]
    intents: list[str]
    expanded_terms: list[str] = field(default_factory=list)

    @property
    def expanded_query(self) -> str:
        if not self.expanded_terms:
            return self.raw_query
        return f"{self.raw_query} {' '.join(self.expanded_terms)}"


def detect_intents(query: str) -> list[str]:
    intents: list[str] = []
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(keyword in query for keyword in keywords):
            intents.append(intent)
    return intents or ["general"]


def expand_query(query: str, entities: list[Entity]) -> list[str]:
    terms: set[str] = set()
    candidates = [query] + [entity.text for entity in entities]
    for candidate in candidates:
        for key, expansions in EXPANSION_RULES.items():
            if key in candidate:
                terms.update(expansions)
    return sorted(terms)


def analyze_query(query: str, ner: RuleMedicalNER | None = None) -> QueryAnalysis:
    ner = ner or RuleMedicalNER()
    normalized = query.strip().lower()
    entities = ner.extract(query)
    expanded_terms = expand_query(query, entities)
    return QueryAnalysis(
        raw_query=query,
        normalized_query=normalized,
        tokens=tokenize(query),
        entities=entities,
        intents=detect_intents(query),
        expanded_terms=expanded_terms,
    )
