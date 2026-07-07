from __future__ import annotations

from typing import Protocol


class EmbeddingModel(Protocol):
    dim: int

    def encode(self, text: str) -> list[float]:
        """Encode one text into a dense vector."""
        ...

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple texts. Implementations may override for batching."""
        return [self.encode(text) for text in texts]
