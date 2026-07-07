from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..core.schema import MedicalDocument
from .vector import cosine


@dataclass(frozen=True)
class VectorMatch:
    doc_id: str
    score: float
    rank: int


class InMemoryVectorStore:
    """Simple vector store used for teaching, tests, and small datasets."""

    def __init__(self):
        self.doc_ids: list[str] = []
        self.vectors: list[list[float]] = []

    def build(self, documents: list[MedicalDocument], vectors: list[list[float]]) -> None:
        if len(documents) != len(vectors):
            raise ValueError("documents and vectors must have the same length")
        self.doc_ids = [doc.id for doc in documents]
        self.vectors = vectors

    def search(self, query_vector: list[float], top_k: int = 5) -> list[VectorMatch]:
        scored = [VectorMatch(doc_id, cosine(query_vector, vector), 0) for doc_id, vector in zip(self.doc_ids, self.vectors)]
        scored.sort(key=lambda item: item.score, reverse=True)
        return [VectorMatch(item.doc_id, item.score, rank) for rank, item in enumerate(scored[:top_k], 1)]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"backend": "memory", "doc_ids": self.doc_ids, "vectors": self.vectors}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "InMemoryVectorStore":
        with open(path, "r", encoding="utf-8-sig") as f:
            payload = json.load(f)
        store = cls()
        store.doc_ids = [str(item) for item in payload["doc_ids"]]
        store.vectors = [[float(value) for value in vector] for vector in payload["vectors"]]
        return store


class FaissVectorStore:
    """FAISS inner-product vector store. Vectors should be normalized."""

    def __init__(self):
        try:
            import faiss  # type: ignore
            import numpy as np  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("FAISS backend requires `pip install faiss-cpu numpy`.") from exc
        self.faiss = faiss
        self.np = np
        self.doc_ids: list[str] = []
        self.index = None

    def build(self, documents: list[MedicalDocument], vectors: list[list[float]]) -> None:
        if not vectors:
            raise ValueError("vectors cannot be empty")
        dim = len(vectors[0])
        matrix = self.np.array(vectors, dtype="float32")
        self.index = self.faiss.IndexFlatIP(dim)
        self.index.add(matrix)
        self.doc_ids = [doc.id for doc in documents]

    def search(self, query_vector: list[float], top_k: int = 5) -> list[VectorMatch]:
        if self.index is None:
            raise RuntimeError("FAISS index has not been built")
        query = self.np.array([query_vector], dtype="float32")
        scores, indices = self.index.search(query, top_k)
        matches: list[VectorMatch] = []
        for rank, (score, index) in enumerate(zip(scores[0], indices[0]), 1):
            if index < 0:
                continue
            matches.append(VectorMatch(self.doc_ids[int(index)], float(score), rank))
        return matches

    def save(self, path: str | Path) -> None:
        if self.index is None:
            raise RuntimeError("FAISS index has not been built")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.faiss.write_index(self.index, str(path))
        with open(path.with_suffix(path.suffix + ".meta.json"), "w", encoding="utf-8") as f:
            json.dump({"backend": "faiss", "doc_ids": self.doc_ids}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "FaissVectorStore":
        path = Path(path)
        store = cls()
        store.index = store.faiss.read_index(str(path))
        with open(path.with_suffix(path.suffix + ".meta.json"), "r", encoding="utf-8-sig") as f:
            store.doc_ids = [str(item) for item in json.load(f)["doc_ids"]]
        return store


def build_vector_store(backend: str):
    backend = backend.lower()
    if backend in {"memory", "in-memory", "inmemory"}:
        return InMemoryVectorStore()
    if backend == "faiss":
        return FaissVectorStore()
    raise ValueError(f"Unsupported vector store backend: {backend}")


def load_vector_store(path: str | Path, backend: str):
    backend = backend.lower()
    if backend in {"memory", "in-memory", "inmemory"}:
        return InMemoryVectorStore.load(path)
    if backend == "faiss":
        return FaissVectorStore.load(path)
    raise ValueError(f"Unsupported vector store backend: {backend}")
