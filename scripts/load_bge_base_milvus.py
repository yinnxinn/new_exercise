from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
from sentence_transformers import SentenceTransformer


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def create_collection(name: str, dim: int, drop_existing: bool) -> Collection:
    connections.connect(alias="default", host="medical-milvus", port="19530")
    if utility.has_collection(name):
        if drop_existing:
            utility.drop_collection(name)
        else:
            collection = Collection(name)
            collection.load()
            return collection

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="document_id", dtype=DataType.INT64),
        FieldSchema(name="document_external_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="dialogue_id", dtype=DataType.INT64),
        FieldSchema(name="department", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="department_tag", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="source_category", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="category_tag", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(
            name="tag_codes",
            dtype=DataType.ARRAY,
            element_type=DataType.VARCHAR,
            max_capacity=64,
            max_length=255,
        ),
        FieldSchema(name="content_hash", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="embedding_model", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="is_active", dtype=DataType.BOOL),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    collection = Collection(name=name, schema=CollectionSchema(fields, description="BGE base Chinese medical embeddings"))
    collection.create_index(
        field_name="embedding",
        index_params={
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200},
        },
    )
    return collection


def batch_rows(rows: list[dict[str, Any]], size: int):
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def main() -> None:
    parser = argparse.ArgumentParser(description="Load BAAI/bge-base-zh-v1.5 vectors into Milvus")
    parser.add_argument("--input", default="data/dialogue/milvus_documents_mysql.jsonl")
    parser.add_argument("--collection", default="medical_document_embeddings_bge_base_zh")
    parser.add_argument("--model", default="BAAI/bge-base-zh-v1.5")
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--drop-existing", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input), limit=args.limit)
    collection = create_collection(args.collection, args.dim, drop_existing=args.drop_existing)
    model = SentenceTransformer(args.model)

    inserted = 0
    for batch in batch_rows(rows, args.batch_size):
        texts = [str(row.get("embedding_text", "")) for row in batch]
        vectors = model.encode(
            texts,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        entities = []
        for row, vector in zip(batch, vectors):
            entities.append(
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
                    "embedding_model": args.model,
                    "is_active": bool(row.get("is_active", True)),
                    "embedding": vector,
                }
            )
        collection.insert(entities)
        inserted += len(entities)
        print(json.dumps({"inserted": inserted, "total": len(rows)}, ensure_ascii=False), flush=True)

    collection.flush()
    collection.load()
    print(
        json.dumps(
            {
                "collection": collection.name,
                "model": args.model,
                "dim": args.dim,
                "input_rows": len(rows),
                "inserted": inserted,
                "num_entities": collection.num_entities,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
