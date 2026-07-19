from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..data.io import load_json
from ..embeddings.hash_embedding import HashEmbeddingModel
from ..retrieval.milvus_schema import DEFAULT_COLLECTION_NAME, DEFAULT_EMBEDDING_DIM, create_collection


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def connect_collection(config_path: Path, drop_existing: bool = False):
    try:
        from pymilvus import connections
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Milvus loading requires `pip install -e .[database]`.") from exc

    config = load_json(config_path)
    milvus_cfg = config.get("milvus", {})
    host = milvus_cfg.get("host", "127.0.0.1")
    port = str(milvus_cfg.get("port", 19530))
    collection_name = milvus_cfg.get("collection", DEFAULT_COLLECTION_NAME)
    embedding_dim = int(milvus_cfg.get("embedding_dim", DEFAULT_EMBEDDING_DIM))
    connections.connect(alias="default", host=host, port=port)
    return create_collection(collection_name, embedding_dim, drop_existing=drop_existing), milvus_cfg


def load_rows_to_milvus(
    rows: list[dict[str, Any]],
    *,
    collection,
    embedding_dim: int,
    embedding_model: str,
    batch_size: int = 100,
) -> int:
    model = HashEmbeddingModel(dim=embedding_dim)
    inserted = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        vectors = model.encode_batch([str(row.get("embedding_text", "")) for row in batch])
        entities = [
            {
                "document_id": int(row["document_id"]),
                "document_external_id": str(row["document_external_id"]),
                "dialogue_id": int(row.get("dialogue_id") or 0),
                "department": str(row.get("department", "")),
                "department_tag": str(row.get("department_tag", "")),
                "source_category": str(row.get("source_category", "")),
                "category_tag": str(row.get("category_tag", "")),
                "tag_codes": [str(tag) for tag in row.get("tag_codes", [])][:64],
                "content_hash": str(row.get("content_hash", "")),
                "embedding_model": embedding_model,
                "is_active": bool(row.get("is_active", True)),
                "embedding": vectors[offset],
            }
            for offset, row in enumerate(batch)
        ]
        collection.insert(entities)
        inserted += len(entities)
    collection.flush()
    collection.load()
    return inserted


def load_milvus(input_path: Path, config_path: Path, *, drop_existing: bool = False) -> dict:
    collection, milvus_cfg = connect_collection(config_path, drop_existing=drop_existing)
    embedding_dim = int(milvus_cfg.get("embedding_dim", DEFAULT_EMBEDDING_DIM))
    embedding_model = str(milvus_cfg.get("embedding_model", "hash-embedding"))
    rows = read_jsonl(input_path)
    inserted = load_rows_to_milvus(
        rows,
        collection=collection,
        embedding_dim=embedding_dim,
        embedding_model=embedding_model,
    )
    return {
        "collection": collection.name,
        "input_rows": len(rows),
        "inserted": inserted,
        "num_entities": collection.num_entities,
        "embedding_dim": embedding_dim,
        "embedding_model": embedding_model,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Load converted medical dialogue vectors into Milvus")
    parser.add_argument("--input", default="data/dialogue/milvus_documents_mysql.jsonl")
    parser.add_argument("--config", default="configs/database.example.json")
    parser.add_argument("--drop-existing", action="store_true")
    args = parser.parse_args()

    result = load_milvus(Path(args.input), Path(args.config), drop_existing=args.drop_existing)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
