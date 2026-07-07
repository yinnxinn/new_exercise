from __future__ import annotations

import argparse
from pathlib import Path

from ..data.io import load_documents, load_json
from ..embeddings import build_embedding_model
from ..retrieval.vector_store import build_vector_store

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "week10_vector.json"


def resolve_project_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return config_path.parent.parent / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build vector index for medical documents")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_json(config_path)
    data_path = resolve_project_path(config_path, config["data_path"])
    index_path = resolve_project_path(config_path, config["vector_store"]["index_path"])

    documents = load_documents(data_path)
    embedding_model = build_embedding_model(config)
    vectors = embedding_model.encode_batch([doc.full_text for doc in documents])

    store = build_vector_store(str(config["vector_store"].get("backend", "memory")))
    store.build(documents, vectors)
    store.save(index_path)

    print(f"documents={len(documents)}")
    print(f"embedding_backend={config.get('embedding', {}).get('backend', 'hash')}")
    print(f"vector_store={config['vector_store'].get('backend', 'memory')}")
    print(f"index_path={index_path}")


if __name__ == "__main__":
    main()
