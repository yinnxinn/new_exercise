from __future__ import annotations

from ..retrieval.vector import HashVectorizer


class HashEmbeddingModel:
    """Dependency-free baseline embedding for teaching and tests."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.vectorizer = HashVectorizer(dim=dim)

    def encode(self, text: str) -> list[float]:
        return self.vectorizer.encode(text)

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.encode(text) for text in texts]
