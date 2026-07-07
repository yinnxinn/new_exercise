from __future__ import annotations

from dataclasses import dataclass

from ..core.schema import Entity, MedicalDocument
from ..embeddings.base import EmbeddingModel
from ..nlp.ner import RuleMedicalNER
from ..nlp.text import tokenize
from ..query.understanding import QueryAnalysis, analyze_query
from .bm25 import BM25Index
from .vector_store import InMemoryVectorStore


@dataclass(frozen=True)
class RecallResult:
    doc_id: str
    score: float
    source: str
    rank: int


@dataclass(frozen=True)
class HybridCandidate:
    doc: MedicalDocument
    score: float
    sources: list[str]
    matched_entities: list[str]


class BM25Recall:
    def __init__(self, documents: list[MedicalDocument]):
        self.documents = documents
        self.index = BM25Index(documents)

    def search(self, query: str, top_k: int) -> list[RecallResult]:
        scores = self.index.scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
        return [RecallResult(self.documents[i].id, score, "bm25", rank) for rank, (i, score) in enumerate(ranked, 1)]


class VectorRecall:
    def __init__(self, documents: list[MedicalDocument], embedding_model: EmbeddingModel):
        self.documents = documents
        self.embedding_model = embedding_model
        self.store = InMemoryVectorStore()
        vectors = embedding_model.encode_batch([doc.full_text for doc in documents])
        self.store.build(documents, vectors)

    def search(self, query: str, top_k: int) -> list[RecallResult]:
        query_vector = self.embedding_model.encode(query)
        return [RecallResult(item.doc_id, item.score, "vector", item.rank) for item in self.store.search(query_vector, top_k)]


class EntityRecall:
    def __init__(self, documents: list[MedicalDocument], ner: RuleMedicalNER):
        self.documents = documents
        self.ner = ner
        self.doc_entities = {doc.id: ner.extract(doc.full_text) for doc in documents}

    def search(self, entities: list[Entity], top_k: int) -> list[RecallResult]:
        if not entities:
            return []
        query_pairs = {(entity.text, entity.type) for entity in entities}
        scored: list[tuple[str, float]] = []
        for doc in self.documents:
            doc_pairs = {(entity.text, entity.type) for entity in self.doc_entities[doc.id]}
            overlap = query_pairs & doc_pairs
            if overlap:
                scored.append((doc.id, float(len(overlap))))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [RecallResult(doc_id, score, "entity", rank) for rank, (doc_id, score) in enumerate(scored[:top_k], 1)]

    def matched_entities(self, doc_id: str, entities: list[Entity]) -> list[str]:
        query_pairs = {(entity.text, entity.type) for entity in entities}
        doc_pairs = {(entity.text, entity.type) for entity in self.doc_entities.get(doc_id, [])}
        return sorted(text for text, _ in query_pairs & doc_pairs)


def reciprocal_rank_fusion(results: list[RecallResult], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for item in results:
        scores[item.doc_id] = scores.get(item.doc_id, 0.0) + 1.0 / (k + item.rank)
    return scores


class HybridRecallPipeline:
    def __init__(self, documents: list[MedicalDocument], embedding_model: EmbeddingModel, ner: RuleMedicalNER | None = None):
        self.documents = documents
        self.docs_by_id = {doc.id: doc for doc in documents}
        self.ner = ner or RuleMedicalNER()
        self.bm25 = BM25Recall(documents)
        self.vector = VectorRecall(documents, embedding_model)
        self.entity = EntityRecall(documents, self.ner)

    def search(self, query: str, top_k: int = 5, recall_k: int = 20) -> tuple[list[HybridCandidate], QueryAnalysis]:
        analysis = analyze_query(query, self.ner)
        bm25_results = self.bm25.search(analysis.expanded_query, recall_k)
        vector_results = self.vector.search(analysis.expanded_query, recall_k)
        entity_results = self.entity.search(analysis.entities, recall_k)
        all_results = bm25_results + vector_results + entity_results
        fused = reciprocal_rank_fusion(all_results)

        source_map: dict[str, set[str]] = {}
        for item in all_results:
            source_map.setdefault(item.doc_id, set()).add(item.source)

        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_k]
        candidates = [
            HybridCandidate(
                doc=self.docs_by_id[doc_id],
                score=score,
                sources=sorted(source_map.get(doc_id, [])),
                matched_entities=self.entity.matched_entities(doc_id, analysis.entities),
            )
            for doc_id, score in ranked
        ]
        return candidates, analysis
