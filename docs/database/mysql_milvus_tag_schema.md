# MySQL + Milvus Tag Schema

This project treats MySQL as the source of truth and Milvus as a rebuildable vector index.

## Data Ownership

- `medical_dialogues`: raw Chinese medical dialogue rows.
- `medical_documents`: normalized retrieval documents used by the Python retrieval layer.
- `medical_tags`: canonical tag dictionary.
- `medical_document_tags`: many-to-many document/tag links.
- Milvus collection: vector recall index with scalar fields for filtering.

Milvus must not be the only place where a tag exists. Every Milvus tag code should be traceable to `medical_tags.tag_code`.

## Tag Code Format

Tag codes are stable strings:

```text
dept:蹇冭绠″唴绉?cat:im_鍐呯
kw:楂樿鍘?ent:绯栧翱鐥?```

Tag types:

```text
department
category
keyword
entity
source
quality
```

The converter writes tags to:

```text
data/dialogue/mysql_tags.jsonl
data/dialogue/mysql_document_tags.jsonl
data/dialogue/milvus_documents.jsonl
```

## Milvus Filter Fields

Collection name:

```text
medical_document_embeddings_bge_small_zh
```

Fields:

```text
id                    INT64 auto primary key
document_id           INT64
document_external_id  VARCHAR(128)
dialogue_id           INT64
department            VARCHAR(128)
department_tag        VARCHAR(255)
source_category       VARCHAR(128)
category_tag          VARCHAR(255)
tag_codes             ARRAY<VARCHAR>
content_hash          VARCHAR(64)
embedding_model       VARCHAR(128)
is_active             BOOL
embedding             FLOAT_VECTOR(512)
```

Example filters:

```text
is_active == true and department_tag == "dept:蹇冭绠″唴绉?
is_active == true and ARRAY_CONTAINS(tag_codes, "kw:楂樿鍘?)
is_active == true and category_tag == "cat:im_鍐呯" and ARRAY_CONTAINS(tag_codes, "kw:鍙戠儹")
```

Use hard filters only when the user explicitly chooses a department or tag. For inferred tags, prefer broad vector recall followed by SQL-side reranking so cross-department questions are not lost.

## Conversion Outputs

Run:

```powershell
python -m medical_retrieval.data.convert_dialogue_dataset --input D:\datasets\Chinese-medical-dialogue-data --output-dir data\dialogue --limit 100
```

Outputs:

```text
medical_dialogue_docs.jsonl      Compatible with load_documents()
mysql_dialogues.jsonl            Raw dialogue import records
mysql_tags.jsonl                 Canonical tag import records
mysql_document_tags.jsonl        Document/tag link import records
milvus_documents.jsonl           Milvus scalar payload plus embedding_text
stats.json                       Conversion summary
```


## Local Services

Start MySQL and Milvus:

```powershell
docker compose up -d mysql etcd minio milvus
```

Install optional database dependencies:

```powershell
pip install -e ".[database]"
```

The compose file mounts `sql/` into MySQL init scripts for a fresh volume. To initialize an existing database while importing records:

```powershell
python -m medical_retrieval.data.mysql_loader --input-dir data\dialogue --config configs\database.example.json --init-schema
```

Dry-run an import without connecting to MySQL:

```powershell
python -m medical_retrieval.data.mysql_loader --input-dir data\dialogue --dry-run
```

Load converted records into MySQL:

```powershell
python -m medical_retrieval.data.mysql_loader --input-dir data\dialogue --config configs\database.example.json
```

After loading, the importer writes:

```text
data/dialogue/milvus_documents_mysql.jsonl
```

This file replaces temporary converter sequence ids with real MySQL `medical_documents.id` and `medical_dialogues.id` values.

Initialize the Milvus collection:

```powershell
python -m medical_retrieval.app.init_milvus --config configs\database.example.json
```

Load vectors into Milvus using the generated MySQL id payload:

```powershell
python -m medical_retrieval.app.load_milvus --input data\dialogue\milvus_documents_mysql.jsonl --config configs\database.example.json --drop-existing
```

For the local smoke test, `configs/database.example.json` uses a dependency-free hash embedding collection:

```text
medical_document_embeddings_hash_512
```

Create a separate BGE collection before loading real BGE vectors.
