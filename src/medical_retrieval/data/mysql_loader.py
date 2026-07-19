from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .io import load_json


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def json_value(value: Any) -> str:
    return json.dumps(value or {}, ensure_ascii=False)


def load_import_plan(input_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        "dialogues": read_jsonl(input_dir / "mysql_dialogues.jsonl"),
        "documents": read_jsonl(input_dir / "medical_dialogue_docs.jsonl"),
        "tags": read_jsonl(input_dir / "mysql_tags.jsonl"),
        "document_tags": read_jsonl(input_dir / "mysql_document_tags.jsonl"),
        "milvus": read_jsonl(input_dir / "milvus_documents.jsonl"),
    }


def summarize_plan(plan: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    return {name: len(rows) for name, rows in plan.items()}


def connect_mysql(config: dict):
    try:
        import pymysql
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("MySQL loading requires `pip install -e .[database]`.") from exc

    mysql_cfg = config.get("mysql", config)
    return pymysql.connect(
        host=mysql_cfg.get("host", "127.0.0.1"),
        port=int(mysql_cfg.get("port", 3306)),
        user=mysql_cfg.get("user", "medical"),
        password=mysql_cfg.get("password", "medical_dev_password"),
        database=mysql_cfg.get("database", "medical_retrieval"),
        charset="utf8mb4",
        autocommit=False,
        cursorclass=pymysql.cursors.DictCursor,
    )


def execute_schema(connection, schema_path: Path) -> None:
    sql_text = schema_path.read_text(encoding="utf-8")
    statements = [statement.strip() for statement in sql_text.split(";") if statement.strip()]
    with connection.cursor() as cursor:
        for statement in statements:
            cursor.execute(statement)
    connection.commit()


def insert_dialogues(connection, rows: list[dict[str, Any]]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    sql = """
        INSERT INTO medical_dialogues (
            external_id, source_dataset, source_category, source_file, source_row_number,
            department, title, question, answer, question_length, answer_length,
            content_hash, quality_status, quality_flags, raw_payload
        ) VALUES (
            %(external_id)s, %(source_dataset)s, %(source_category)s, %(source_file)s, %(source_row_number)s,
            %(department)s, %(title)s, %(question)s, %(answer)s, %(question_length)s, %(answer_length)s,
            %(content_hash)s, %(quality_status)s, %(quality_flags)s, %(raw_payload)s
        )
        ON DUPLICATE KEY UPDATE
            id = LAST_INSERT_ID(id),
            department = VALUES(department),
            title = VALUES(title),
            question = VALUES(question),
            answer = VALUES(answer),
            quality_status = VALUES(quality_status),
            updated_at = CURRENT_TIMESTAMP
    """
    with connection.cursor() as cursor:
        for row in rows:
            payload = dict(row)
            payload["quality_flags"] = json_value(row.get("quality_flags"))
            payload["raw_payload"] = json_value(row.get("raw_payload"))
            cursor.execute(sql, payload)
            mapping[str(row["external_id"])] = int(cursor.lastrowid)
    return mapping


def insert_documents(connection, rows: list[dict[str, Any]], dialogue_ids: dict[str, int]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    sql = """
        INSERT INTO medical_documents (
            external_id, dialogue_id, title, department, question, answer, content,
            source_type, source_dataset, source_ref, is_active, metadata, tag_codes_json
        ) VALUES (
            %(external_id)s, %(dialogue_id)s, %(title)s, %(department)s, %(question)s, %(answer)s, %(content)s,
            %(source_type)s, %(source_dataset)s, %(source_ref)s, %(is_active)s, %(metadata)s, %(tag_codes_json)s
        )
        ON DUPLICATE KEY UPDATE
            id = LAST_INSERT_ID(id),
            dialogue_id = VALUES(dialogue_id),
            title = VALUES(title),
            department = VALUES(department),
            question = VALUES(question),
            answer = VALUES(answer),
            content = VALUES(content),
            metadata = VALUES(metadata),
            tag_codes_json = VALUES(tag_codes_json),
            updated_at = CURRENT_TIMESTAMP
    """
    with connection.cursor() as cursor:
        for row in rows:
            external_id = str(row["id"])
            metadata = dict(row.get("metadata", {}))
            payload = {
                "external_id": external_id,
                "dialogue_id": dialogue_ids.get(external_id),
                "title": row["title"],
                "department": row["department"],
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "content": row["content"],
                "source_type": "dialogue",
                "source_dataset": metadata.get("source_dataset"),
                "source_ref": row.get("source", ""),
                "is_active": 1,
                "metadata": json_value(metadata),
                "tag_codes_json": json.dumps(row.get("tag_codes", []), ensure_ascii=False),
            }
            cursor.execute(sql, payload)
            mapping[external_id] = int(cursor.lastrowid)
    return mapping


def insert_tags(connection, rows: list[dict[str, Any]]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    sql = """
        INSERT INTO medical_tags (
            tag_code, tag_name, tag_type, normalized_name, source, metadata
        ) VALUES (
            %(tag_code)s, %(tag_name)s, %(tag_type)s, %(normalized_name)s, %(source)s, %(metadata)s
        )
        ON DUPLICATE KEY UPDATE
            id = LAST_INSERT_ID(id),
            tag_name = VALUES(tag_name),
            tag_type = VALUES(tag_type),
            source = VALUES(source)
    """
    with connection.cursor() as cursor:
        for row in rows:
            payload = {
                "tag_code": row["code"],
                "tag_name": row["name"],
                "tag_type": row["tag_type"],
                "normalized_name": row["name"],
                "source": row.get("source", "auto"),
                "metadata": json_value({}),
            }
            cursor.execute(sql, payload)
            mapping[str(row["code"])] = int(cursor.lastrowid)
    return mapping


def insert_document_tags(
    connection,
    rows: list[dict[str, Any]],
    document_ids: dict[str, int],
    tag_ids: dict[str, int],
) -> int:
    sql = """
        INSERT INTO medical_document_tags (
            document_id, tag_id, tag_code, tag_type, source, confidence
        ) VALUES (
            %(document_id)s, %(tag_id)s, %(tag_code)s, %(tag_type)s, %(source)s, %(confidence)s
        )
        ON DUPLICATE KEY UPDATE
            source = VALUES(source),
            confidence = VALUES(confidence)
    """
    inserted = 0
    with connection.cursor() as cursor:
        for row in rows:
            document_id = document_ids.get(str(row["document_external_id"]))
            tag_id = tag_ids.get(str(row["tag_code"]))
            if not document_id or not tag_id:
                continue
            cursor.execute(
                sql,
                {
                    "document_id": document_id,
                    "tag_id": tag_id,
                    "tag_code": row["tag_code"],
                    "tag_type": row["tag_type"],
                    "source": row.get("source", "auto"),
                    "confidence": row.get("confidence", 1.0),
                },
            )
            inserted += 1
    return inserted


def build_milvus_payload(
    rows: list[dict[str, Any]],
    document_ids: dict[str, int],
    dialogue_ids: dict[str, int],
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for row in rows:
        external_id = str(row["document_external_id"])
        document_id = document_ids.get(external_id)
        if not document_id:
            continue
        item = dict(row)
        item["document_id"] = document_id
        item["dialogue_id"] = dialogue_ids.get(str(row.get("dialogue_external_id", external_id)), 0)
        payload.append(item)
    return payload


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_to_mysql(
    input_dir: Path,
    config_path: Path,
    *,
    schema_path: Path | None = None,
    init_schema: bool = False,
) -> dict[str, int]:
    plan = load_import_plan(input_dir)
    config = load_json(config_path)
    connection = connect_mysql(config)
    try:
        if init_schema:
            if schema_path is None:
                raise ValueError("schema_path is required when init_schema=True")
            execute_schema(connection, schema_path)
        dialogue_ids = insert_dialogues(connection, plan["dialogues"])
        document_ids = insert_documents(connection, plan["documents"], dialogue_ids)
        tag_ids = insert_tags(connection, plan["tags"])
        document_tag_count = insert_document_tags(connection, plan["document_tags"], document_ids, tag_ids)
        milvus_payload = build_milvus_payload(plan["milvus"], document_ids, dialogue_ids)
        write_jsonl(input_dir / "milvus_documents_mysql.jsonl", milvus_payload)
        connection.commit()
        return {
            "dialogues": len(dialogue_ids),
            "documents": len(document_ids),
            "tags": len(tag_ids),
            "document_tags": document_tag_count,
            "milvus_payload": len(milvus_payload),
        }
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Load converted medical dialogue records into MySQL")
    parser.add_argument("--input-dir", default="data/dialogue")
    parser.add_argument("--config", default="configs/database.example.json")
    parser.add_argument("--schema", default="sql/001_mysql_dialogue_schema.sql")
    parser.add_argument("--init-schema", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if args.dry_run:
        print(json.dumps(summarize_plan(load_import_plan(input_dir)), ensure_ascii=False, indent=2))
        return

    stats = load_to_mysql(
        input_dir,
        Path(args.config),
        schema_path=Path(args.schema),
        init_schema=args.init_schema,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
