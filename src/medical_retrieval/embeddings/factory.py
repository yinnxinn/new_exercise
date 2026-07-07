from __future__ import annotations

from .base import EmbeddingModel
from .hash_embedding import HashEmbeddingModel
from .sentence_transformer_embedding import SentenceTransformerEmbeddingModel


def build_embedding_model(config: dict) -> EmbeddingModel:
    embedding_cfg = config.get("embedding") or config.get("vector", {})
    backend = str(embedding_cfg.get("backend", "hash")).lower()

    if backend in {"hash", "hashing"}:
        return HashEmbeddingModel(dim=int(embedding_cfg.get("dim", 256)))

    if backend in {"sentence-transformer", "sentence_transformer", "sentence-transformers"}:
        model_name = str(embedding_cfg.get("model_name", "shibing624/text2vec-base-chinese"))
        return SentenceTransformerEmbeddingModel(
            model_name=model_name,
            device=embedding_cfg.get("device"),
            normalize_embeddings=bool(embedding_cfg.get("normalize", True)),
        )

    raise ValueError(f"Unsupported embedding backend: {backend}")
