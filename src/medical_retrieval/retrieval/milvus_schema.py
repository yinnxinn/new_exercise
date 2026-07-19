from __future__ import annotations


DEFAULT_COLLECTION_NAME = "medical_document_embeddings_bge_small_zh"
DEFAULT_EMBEDDING_DIM = 512


def build_collection_schema(embedding_dim: int = DEFAULT_EMBEDDING_DIM):
    try:
        from pymilvus import CollectionSchema, DataType, FieldSchema
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Milvus schema creation requires `pip install pymilvus`.") from exc

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
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
    ]
    return CollectionSchema(fields=fields, description="Chinese medical dialogue document embeddings")


def create_collection(
    name: str = DEFAULT_COLLECTION_NAME,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    *,
    drop_existing: bool = False,
):
    try:
        from pymilvus import Collection, utility
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Milvus collection creation requires `pip install pymilvus`.") from exc

    if utility.has_collection(name):
        if not drop_existing:
            return Collection(name)
        utility.drop_collection(name)
    collection = Collection(name=name, schema=build_collection_schema(embedding_dim))
    collection.create_index(
        field_name="embedding",
        index_params={
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200},
        },
    )
    return collection


def build_filter_expr(
    *,
    department_tag: str | None = None,
    category_tag: str | None = None,
    required_tag_codes: list[str] | None = None,
    embedding_model: str | None = None,
    active_only: bool = True,
) -> str:
    clauses: list[str] = []
    if active_only:
        clauses.append("is_active == true")
    if department_tag:
        clauses.append(f'department_tag == "{department_tag}"')
    if category_tag:
        clauses.append(f'category_tag == "{category_tag}"')
    if embedding_model:
        clauses.append(f'embedding_model == "{embedding_model}"')
    for tag_code in required_tag_codes or []:
        clauses.append(f'ARRAY_CONTAINS(tag_codes, "{tag_code}")')
    return " and ".join(clauses)
