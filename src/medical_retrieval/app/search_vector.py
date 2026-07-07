from __future__ import annotations

import argparse
from pathlib import Path

from ..data.io import load_documents, load_json
from ..embeddings import build_embedding_model
from ..retrieval.vector_store import load_vector_store

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "week10_vector.json"


def resolve_project_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return config_path.parent.parent / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Search vector index for medical documents")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_json(config_path)
    data_path = resolve_project_path(config_path, config["data_path"])
    index_path = resolve_project_path(config_path, config["vector_store"]["index_path"])

    documents = load_documents(data_path)
    docs_by_id = {doc.id: doc for doc in documents}
    embedding_model = build_embedding_model(config)
    store = load_vector_store(index_path, str(config["vector_store"].get("backend", "memory")))

    query_vector = embedding_model.encode(args.query)
    matches = store.search(query_vector, top_k=args.top_k)

    print(f"查询：{args.query}")
    for match in matches:
        doc = docs_by_id[match.doc_id]
        print(f"{match.rank}. [{doc.id}] {doc.title} | {doc.department} | vector_score={match.score:.3f}")
        print(f"   tags={','.join(doc.tags[:8])}")


if __name__ == "__main__":
    main()
