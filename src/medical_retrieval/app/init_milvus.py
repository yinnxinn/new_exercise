from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..data.io import load_json
from ..retrieval.milvus_schema import DEFAULT_COLLECTION_NAME, DEFAULT_EMBEDDING_DIM, create_collection


def init_milvus(config_path: Path, drop_existing: bool = False) -> dict:
    try:
        from pymilvus import connections
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Milvus initialization requires `pip install -e .[database]`.") from exc

    config = load_json(config_path)
    milvus_cfg = config.get("milvus", {})
    host = milvus_cfg.get("host", "127.0.0.1")
    port = str(milvus_cfg.get("port", 19530))
    collection_name = milvus_cfg.get("collection", DEFAULT_COLLECTION_NAME)
    embedding_dim = int(milvus_cfg.get("embedding_dim", DEFAULT_EMBEDDING_DIM))

    connections.connect(alias="default", host=host, port=port)
    collection = create_collection(collection_name, embedding_dim, drop_existing=drop_existing)
    collection.load()
    return {
        "collection": collection.name,
        "embedding_dim": embedding_dim,
        "host": host,
        "port": port,
        "num_entities": collection.num_entities,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize Milvus collection for medical document embeddings")
    parser.add_argument("--config", default="configs/database.example.json")
    parser.add_argument("--drop-existing", action="store_true")
    args = parser.parse_args()

    result = init_milvus(Path(args.config), drop_existing=args.drop_existing)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
