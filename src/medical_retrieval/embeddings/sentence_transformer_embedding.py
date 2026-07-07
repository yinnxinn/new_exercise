from __future__ import annotations

from ..retrieval.vector import normalize


class SentenceTransformerEmbeddingModel:
    """Sentence-Transformers embedding backend.

    Install with: pip install sentence-transformers
    A small Chinese-capable model such as shibing624/text2vec-base-chinese
    is a good classroom default when network/model cache is available.
    """

    def __init__(self, model_name: str, device: str | None = None, normalize_embeddings: bool = True):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run `pip install sentence-transformers` or use backend='hash'."
            ) from exc

        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, text: str) -> list[float]:
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        result = [vector.astype(float).tolist() for vector in vectors]
        if self.normalize_embeddings:
            return [normalize(vector) for vector in result]
        return result
