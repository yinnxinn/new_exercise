from __future__ import annotations

import math
from collections import Counter, defaultdict

from ..core.schema import MedicalDocument
from ..nlp.text import tokenize


class BM25Index:
    def __init__(self, documents: list[MedicalDocument], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(doc.full_text) for doc in documents]
        self.doc_len = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / max(1, len(self.doc_len))
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_freq: dict[str, int] = defaultdict(int)

        for tokens in self.doc_tokens:
            for token in set(tokens):
                self.doc_freq[token] += 1

    def idf(self, term: str) -> float:
        n_docs = len(self.documents)
        df = self.doc_freq.get(term, 0)
        return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))

    def score_one(self, query_tokens: list[str], doc_index: int) -> float:
        score = 0.0
        freqs = self.term_freqs[doc_index]
        dl = self.doc_len[doc_index] or 1
        for term in query_tokens:
            tf = freqs.get(term, 0)
            if tf == 0:
                continue
            denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += self.idf(term) * tf * (self.k1 + 1) / denom
        return score

    def scores(self, query_tokens: list[str]) -> list[float]:
        raw = [self.score_one(query_tokens, i) for i in range(len(self.documents))]
        max_score = max(raw, default=0.0)
        if max_score <= 0:
            return raw
        return [score / max_score for score in raw]
