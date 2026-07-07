from __future__ import annotations

from pathlib import Path

from ..core.schema import MedicalDocument, SearchResult
from ..data.io import load_documents, load_json, load_lexicon
from ..nlp.ner import RuleMedicalNER
from ..nlp.text import tokenize
from .bm25 import BM25Index
from .ranker import entity_bonus, field_bonus
from .vector import HashVectorizer, cosine


class MedicalSearchEngine:
    def __init__(self, documents: list[MedicalDocument], config: dict):
        self.documents = documents
        self.config = config
        self.alpha = float(config.get("alpha", 0.52))
        self.ner = RuleMedicalNER(lexicon=config.get("lexicon"))
        bm25_cfg = config.get("bm25", {})
        vector_cfg = config.get("vector", {})
        self.bm25 = BM25Index(
            documents,
            k1=float(bm25_cfg.get("k1", 1.5)),
            b=float(bm25_cfg.get("b", 0.75)),
        )
        self.vectorizer = HashVectorizer(dim=int(vector_cfg.get("dim", 256)))
        self.doc_vectors = [self.vectorizer.encode(doc.full_text) for doc in documents]
        self.doc_entities = [self.ner.extract(doc.full_text) for doc in documents]

    @classmethod
    def from_config(cls, config_path: str | Path) -> "MedicalSearchEngine":
        config_path = Path(config_path)
        config = load_json(config_path)
        data_path = Path(config["data_path"])
        if not data_path.is_absolute():
            data_path = config_path.parent.parent / data_path
        lexicon_path = config.get("lexicon_path")
        if lexicon_path:
            lexicon_path = Path(lexicon_path)
            if not lexicon_path.is_absolute():
                lexicon_path = config_path.parent.parent / lexicon_path
            config["lexicon"] = load_lexicon(lexicon_path)
        return cls(load_documents(data_path), config)

    def search(self, query: str, top_k: int | None = None) -> tuple[list[SearchResult], list]:
        top_k = top_k or int(self.config.get("top_k", 5))
        query_tokens = tokenize(query)
        query_entities = self.ner.extract(query)
        query_vector = self.vectorizer.encode(query)
        bm25_scores = self.bm25.scores(query_tokens)
        ranking_cfg = self.config.get("ranking", {})

        results: list[SearchResult] = []
        for i, doc in enumerate(self.documents):
            ## 向量数据库，milvus, elasticsearch, postsql
            vector_score = max(0.0, cosine(query_vector, self.doc_vectors[i]))
            bm25_score = bm25_scores[i]
            f_bonus = field_bonus(query_tokens, doc, ranking_cfg)
            e_bonus, matched = entity_bonus(query_entities, self.doc_entities[i], ranking_cfg)
            final_score = self.alpha * vector_score + (1 - self.alpha) * bm25_score + f_bonus + e_bonus
            results.append(
                SearchResult(
                    doc=doc,
                    final_score=final_score,
                    vector_score=vector_score,
                    bm25_score=bm25_score,
                    field_bonus=f_bonus,
                    entity_bonus=e_bonus,
                    matched_entities=matched,
                )
            )

        results.sort(key=lambda item: item.final_score, reverse=True)
        return results[:top_k], query_entities

