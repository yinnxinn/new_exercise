from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .tags import DocumentTag, build_dialogue_tags, make_tag_code


CSV_FIELD_ALIASES = {
    "department": ["department", "\u79d1\u5ba4"],
    "title": ["title", "\u6807\u9898"],
    "question": ["question", "\u95ee\u9898", "ask"],
    "answer": ["answer", "\u56de\u7b54", "doctor_answer"],
}


def stable_hash(*parts: str) -> str:
    text = "\n".join(part.strip() for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_csv_rows(path: Path) -> Iterable[dict[str, str]]:
    for encoding in ["utf-8-sig", "gb18030", "utf-8"]:
        try:
            with open(path, "r", encoding=encoding, newline="") as f:
                reader = csv.DictReader(f)
                yield from ({str(k).strip(): str(v or "").strip() for k, v in row.items()} for row in reader)
            return
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, f"unable to decode {path}")


def pick(row: dict[str, str], canonical: str) -> str:
    for key in CSV_FIELD_ALIASES[canonical]:
        if key in row and row[key]:
            return row[key].strip()
    return ""


def iter_dialogue_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(path for path in input_path.rglob("*.csv") if path.is_file())


def source_category(input_root: Path, csv_path: Path) -> str:
    if input_root.is_file():
        return csv_path.stem
    try:
        relative = csv_path.relative_to(input_root)
    except ValueError:
        return csv_path.parent.name
    return relative.parts[0] if len(relative.parts) > 1 else csv_path.stem


def normalize_record(
    *,
    row: dict[str, str],
    input_root: Path,
    csv_path: Path,
    row_number: int,
    sequence: int,
    keyword_limit: int,
) -> dict | None:
    department = pick(row, "department")
    title = pick(row, "title")
    question = pick(row, "question")
    answer = pick(row, "answer")
    if not question or not answer:
        return None

    category = source_category(input_root, csv_path)
    content_hash = stable_hash(department, title, question, answer)
    document_external_id = f"CMD{sequence:08d}"
    content = (
        f"\u6807\u9898\uff1a{title}\n"
        f"\u79d1\u5ba4\uff1a{department}\n"
        f"\u60a3\u8005\u95ee\u9898\uff1a{question}\n"
        f"\u533b\u751f\u56de\u7b54\uff1a{answer}"
    )
    tags = build_dialogue_tags(
        department=department,
        source_category=category,
        title=title,
        question=question,
        answer=answer,
        keyword_limit=keyword_limit,
    )
    tag_codes = [tag.code for tag in tags]
    department_tag = make_tag_code("department", department)
    category_tag = make_tag_code("category", category)
    source_ref = f"{csv_path.name}:{row_number}"

    return {
        "dialogue": {
            "external_id": document_external_id,
            "source_dataset": "Chinese-medical-dialogue-data",
            "source_category": category,
            "source_file": str(csv_path),
            "source_row_number": row_number,
            "department": department,
            "title": title,
            "question": question,
            "answer": answer,
            "question_length": len(question),
            "answer_length": len(answer),
            "content_hash": content_hash,
            "quality_status": "active",
            "quality_flags": {},
            "raw_payload": row,
        },
        "document": {
            "id": document_external_id,
            "title": title or question[:40],
            "department": department,
            "tags": [tag.name for tag in tags],
            "tag_codes": tag_codes,
            "content": content,
            "question": question,
            "answer": answer,
            "source": source_ref,
            "metadata": {
                "source_dataset": "Chinese-medical-dialogue-data",
                "source_category": category,
                "source_file": str(csv_path),
                "source_row_number": row_number,
                "content_hash": content_hash,
                "department_tag": department_tag,
                "category_tag": category_tag,
            },
        },
        "tags": [asdict(tag) for tag in tags],
        "document_tags": [
            {
                "document_external_id": document_external_id,
                "tag_code": tag.code,
                "tag_type": tag.tag_type,
                "source": tag.source,
                "confidence": tag.confidence,
            }
            for tag in tags
        ],
        "milvus": {
            "document_external_id": document_external_id,
            "document_id": sequence,
            "dialogue_external_id": document_external_id,
            "department": department,
            "department_tag": department_tag,
            "source_category": category,
            "category_tag": category_tag,
            "tag_codes": tag_codes,
            "content_hash": content_hash,
            "is_active": True,
            "embedding_text": content,
        },
    }


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def convert(input_path: Path, output_dir: Path, limit: int | None = None, keyword_limit: int = 12) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = iter_dialogue_files(input_path)
    seen_hashes: set[str] = set()
    documents: list[dict] = []
    dialogues: list[dict] = []
    tags_by_code: dict[str, dict] = {}
    document_tags: list[dict] = []
    milvus_rows: list[dict] = []
    skipped = 0
    duplicates = 0
    sequence = 0

    for csv_path in files:
        for row_number, row in enumerate(read_csv_rows(csv_path), 2):
            normalized = normalize_record(
                row=row,
                input_root=input_path,
                csv_path=csv_path,
                row_number=row_number,
                sequence=sequence + 1,
                keyword_limit=keyword_limit,
            )
            if normalized is None:
                skipped += 1
                continue
            content_hash = normalized["dialogue"]["content_hash"]
            if content_hash in seen_hashes:
                duplicates += 1
                continue
            seen_hashes.add(content_hash)
            sequence += 1
            dialogues.append(normalized["dialogue"])
            documents.append(normalized["document"])
            for tag in normalized["tags"]:
                tags_by_code.setdefault(tag["code"], tag)
            document_tags.extend(normalized["document_tags"])
            milvus_rows.append(normalized["milvus"])
            if limit and sequence >= limit:
                break
        if limit and sequence >= limit:
            break

    stats = {
        "source_files": len(files),
        "documents": len(documents),
        "dialogues": len(dialogues),
        "tags": len(tags_by_code),
        "document_tags": len(document_tags),
        "skipped_rows": skipped,
        "duplicate_rows": duplicates,
    }
    write_jsonl(output_dir / "medical_dialogue_docs.jsonl", documents)
    write_jsonl(output_dir / "mysql_dialogues.jsonl", dialogues)
    write_jsonl(output_dir / "mysql_tags.jsonl", sorted(tags_by_code.values(), key=lambda item: item["code"]))
    write_jsonl(output_dir / "mysql_document_tags.jsonl", document_tags)
    write_jsonl(output_dir / "milvus_documents.jsonl", milvus_rows)
    (output_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Chinese medical dialogue CSV files for MySQL and Milvus")
    parser.add_argument("--input", required=True, help="CSV file or Chinese-medical-dialogue-data root directory")
    parser.add_argument("--output-dir", default="data/dialogue")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--keyword-limit", type=int, default=12)
    args = parser.parse_args()

    stats = convert(Path(args.input), Path(args.output_dir), limit=args.limit, keyword_limit=args.keyword_limit)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
